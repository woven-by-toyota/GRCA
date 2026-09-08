// =============================================================================
// OptiX standalone driver
//
// Same inputs / outputs as GRCA_CUDA.cu but uses OptiX 7 BVH traversal instead
// of the hand-rolled GRCA Early/Late-Pass kernels.
//
// Build:
//   make          # builds optix_device.ptx + optix CPU binary
//   make clean    # removes build artefacts
//
// Usage (identical to grca_cuda):
//   ./optix <mesh.obj> [--hmin ..] [--hmax ..]
//               [--hnum ..] [--vmin ..] [--vmax ..] [--vnum ..]
//               [--rmin ..] [--rmax ..] [--px ..] [--py ..] [--pz ..]
//               [--out <output.csv>]
//
// Requires OptiX 7.4+ (uses optixModuleCreate).
// =============================================================================

#define OPTIX_STUBS_IMPLEMENTATION
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstddef>
#include <float.h>
#include "optix.h"
#include "optix_params.h"
#include "optix_custom_params.h"   // LaunchParamsCustom (optix-crti mode)
#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Per-triangle AABB for OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES.
// Computed on GPU from the vertex buffer — eliminates the CPU loop and the
// second H2D copy that would otherwise add ~67% bandwidth overhead on top of
// the vertex upload.
// ---------------------------------------------------------------------------
__global__ static void compute_aabbs_kernel(const float3* __restrict__ verts,
                                             OptixAabb* __restrict__ aabbs,
                                             unsigned int count)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float3 v0 = verts[i * 3 + 0];
    const float3 v1 = verts[i * 3 + 1];
    const float3 v2 = verts[i * 3 + 2];
    const float eps = 1e-4f;
    aabbs[i].minX = fminf(fminf(v0.x, v1.x), v2.x) - eps;
    aabbs[i].minY = fminf(fminf(v0.y, v1.y), v2.y) - eps;
    aabbs[i].minZ = fminf(fminf(v0.z, v1.z), v2.z) - eps;
    aabbs[i].maxX = fmaxf(fmaxf(v0.x, v1.x), v2.x) + eps;
    aabbs[i].maxY = fmaxf(fmaxf(v0.y, v1.y), v2.y) + eps;
    aabbs[i].maxZ = fmaxf(fmaxf(v0.z, v1.z), v2.z) + eps;
}

static void launch_compute_aabbs(const float3* d_verts, OptixAabb* d_aabbs, size_t triCount,
                                  cudaStream_t stream = 0)
{
    const unsigned int block = 256;
    const unsigned int grid  = (unsigned int)((triCount + block - 1) / block);
    compute_aabbs_kernel<<<grid, block, 0, stream>>>(d_verts, d_aabbs, (unsigned int)triCount);
}

enum BvhSyncMode { BVH_SYNC_REBUILD, BVH_SYNC_REFIT };

// Helper to compute BVH build flags.
//
// Pure refit (no periodic rebuild): PREFER_FAST_TRACE — init once with a high-quality
//   BVH, then only cheap in-place refits.
//
// Hybrid (refit + periodic rebuild) and pure rebuild: PREFER_FAST_BUILD — OptiX
//   requires that OPERATION_UPDATE passes the SAME buildFlags as the OPERATION_BUILD
//   that created the current structure.  After every periodic rebuild the structure
//   changes, so both init/rebuild AND subsequent refits must agree on FAST_BUILD.
static uint32_t make_bvh_flags(BvhSyncMode mode, bool compaction, bool hasPeriodicRebuild) {
    uint32_t flags = (mode == BVH_SYNC_REFIT && !hasPeriodicRebuild)
        ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE
        : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    if (compaction) flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    return flags;
}

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr,"CUDA error %s:%d  %s\n",                                \
              __FILE__,__LINE__,cudaGetErrorString(_e));                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while(0)

#define OPTIX_CHECK(call)                                                      \
  do {                                                                         \
    OptixResult _r = (call);                                                   \
    if (_r != OPTIX_SUCCESS) {                                                 \
      fprintf(stderr,"OptiX error %s:%d  %s\n",                               \
              __FILE__,__LINE__,optixGetErrorName(_r));                         \
      std::exit(1);                                                            \
    }                                                                          \
  } while(0)

#define OPTIX_CHECK_LOG(call)                                                  \
  do {                                                                         \
    char   _log[4096]; size_t _logSz = sizeof(_log);                          \
    OptixResult _r = (call);                                                   \
    if (_logSz > 1) fprintf(stderr,"[OptiX log] %s\n",_log);                  \
    if (_r != OPTIX_SUCCESS) {                                                 \
      fprintf(stderr,"OptiX error %s:%d  %s\n",                               \
              __FILE__,__LINE__,optixGetErrorName(_r));                         \
      std::exit(1);                                                            \
    }                                                                          \
  } while(0)

// ---------------------------------------------------------------------------
// SBT record helper — every record = aligned header + optional user data.
// ---------------------------------------------------------------------------
template<typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};
struct EmptyData {};
using RaygenRecord   = SbtRecord<EmptyData>;
using MissRecord     = SbtRecord<EmptyData>;
using HitGroupRecord = SbtRecord<EmptyData>;

// ---------------------------------------------------------------------------
// GPU helper: allocate + copy
// ---------------------------------------------------------------------------
template<typename T>
static CUdeviceptr uploadVector(const std::vector<T>& v)
{
    CUdeviceptr ptr = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ptr), v.data(),
                          v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

// ---------------------------------------------------------------------------
// Load PTX from file
// ---------------------------------------------------------------------------
static std::string loadPTX(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open PTX file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}


// ===========================================================================
// OptixState  –  persistent context for the game loop
// ===========================================================================
struct OptixState {
    OptixDeviceContext optixCtx    = nullptr;
    OptixPipeline      pipeline    = nullptr;
    OptixProgramGroup  pgRaygen   = nullptr;
    OptixProgramGroup  pgMiss     = nullptr;
    OptixProgramGroup  pgHitGroup = nullptr;

    OptixShaderBindingTable sbt          = {};
    CUdeviceptr             d_raygenRec  = 0;
    CUdeviceptr             d_missRec    = 0;
    CUdeviceptr             d_hitGroupRec= 0;

    // GAS
    CUdeviceptr            d_gasBuf    = 0;
    size_t                 gasBufBytes = 0;
    OptixTraversableHandle traversable = 0;
    bool                   allowUpdate          = false;
    BvhSyncMode            bvhSyncMode          = BVH_SYNC_REBUILD;
    uint32_t               bvhFlags             = 0;   // same flags for init, refit UPDATE, and rebuild BUILD
    int                    rebuildAfterFrames = 0; // -1=pure refit, 0=pure rebuild, N>0=hybrid (period N+1)
    int                    frameCount           = 0;

    // Pre-allocated update scratch (tempUpdateSizeInBytes from initial query)
    CUdeviceptr d_temp_update  = 0;
    size_t      tempUpdateBytes = 0;

    // Pre-allocated per-frame rebuild scratch and staging buffers
    CUdeviceptr d_temp_rebuild   = 0;   // reusable temp scratch for rebuild
    size_t      tempRebuildBytes = 0;
    CUdeviceptr d_build_output   = 0;   // pre-compaction staging (allowCompaction only)
    size_t      buildOutputBytes = 0;   // outputSizeInBytes from initial BVH query
    CUdeviceptr d_compSzBuf      = 0;   // compaction size query result buffer
    bool        allowCompaction  = false;

    // Vertex buffer on GPU — static portion uploaded once; dynamic updated per-frame
    CUdeviceptr  d_verts        = 0;
    unsigned int numVerts       = 0;   // total verts (static + dynamic)
    size_t       staticNumVerts = 0;   // static verts never re-uploaded
    unsigned int geomFlags      = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    // optix-crti mode
    bool        crti    = false;
    size_t      paramsSize     = sizeof(LaunchParams);  // sizeof(LaunchParamsCustom) when crti
    CUdeviceptr d_aabbs        = 0;    // AABB buffer (BVH build input, crti only)
    size_t      aabbBufBytes   = 0;

    // Pinned CPU staging for dynamic vertex upload and readback
    float3*     h_dynBuf            = nullptr;
    size_t      h_dynBufCount       = 0;
    float*      h_distances_pinned  = nullptr;
    float3*     h_directions_pinned = nullptr;

    // Per-lidar ray origin buffers
    RayOrigin* d_ray_origins        = nullptr;
    RayOrigin* h_ray_origins_pinned = nullptr;  // pinned — async-safe upload
    float*       d_distances   = nullptr;
    float3*      d_directions  = nullptr;
    size_t       total_rays    = 0;
    size_t       num_origins    = 0;

    // Launch params (traversable baked in at init; doesn't change on refit)
    CUdeviceptr d_params = 0;

    // CUDA stream and timing events
    cudaStream_t stream      = nullptr;
    cudaEvent_t evBvhStart  = nullptr, evBvhEnd   = nullptr;
    cudaEvent_t evTraceStart= nullptr, evTraceEnd  = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: flatten rays from all robots/lidars into a CPU-side ray_configs
// vector ready for GPU upload.  Also writes state->total_rays.
// ---------------------------------------------------------------------------
static void buildRayOrigins(const RunConfig& r, RayOrigin* out,
                               size_t& out_num_origins, size_t& out_total_rays)
{
    size_t li = 0, total = 0;
    for (const auto& robot : r.robots) {
        for (const auto& src : robot.lidars) {
            unsigned int H = (unsigned int)src.hnum;
            unsigned int V = (unsigned int)src.vnum;
            RayOrigin& lc = out[li++];
            lc.origin    = make_float3(src.w_pose.pos.x,     src.w_pose.pos.y,     src.w_pose.pos.z);
            lc.forward   = make_float3(src.w_pose.forward.x, src.w_pose.forward.y, src.w_pose.forward.z);
            lc.up        = make_float3(src.w_pose.up.x,      src.w_pose.up.y,      src.w_pose.up.z);
            lc.right     = make_float3(src.w_pose.right.x,   src.w_pose.right.y,   src.w_pose.right.z);
            lc.hmin      = src.hmin;
            lc.hStep     = (H > 1u) ? (src.hmax - src.hmin) / (float)(H - 1u) : 0.f;
            lc.vmin      = src.vmin;
            lc.vStep     = (V > 1u) ? (src.vmax - src.vmin) / (float)(V - 1u) : 0.f;
            lc.hNum      = H;
            lc.vNum      = V;
            lc.rayOffset = (unsigned int)total;
            lc.rangeMin  = src.range_min;
            lc.rangeMax  = src.range_max;
            total += (size_t)H * V;
        }
    }
    out_num_origins = li;
    out_total_rays = total;
}

// ---------------------------------------------------------------------------
// Helper: convert raw GPU distance/direction buffers into GrcaHit vector.
// ---------------------------------------------------------------------------
static void fillHits(const float*     h_distances,
                     const float3*    h_directions,
                     size_t           total,
                     const RunConfig& cfg,
                     std::vector<GrcaHit>& hits,
                     int& hitCount)
{
    hits.resize(total);
    hitCount = 0;
    size_t idx = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            const size_t num_rays = (size_t)lc.hnum * lc.vnum;
            for (size_t i = 0; i < num_rays; ++i, ++idx) {
                float dist  = h_distances[idx];
                float3 dir  = h_directions[idx];
                GrcaHit& h   = hits[idx];
                h.dx = dir.x;
                h.dy = dir.y;
                h.dz = dir.z;
                if (dist < FLT_MAX) {
                    h.dist = dist;
                    h.hx = lc.w_pose.pos.x + dir.x * dist;
                    h.hy = lc.w_pose.pos.y + dir.y * dist;
                    h.hz = lc.w_pose.pos.z + dir.z * dist;
                    ++hitCount;
                } else {
                    h.dist = lc.range_max;
                    h.hx = lc.w_pose.pos.x + dir.x * lc.range_max;
                    h.hy = lc.w_pose.pos.y + dir.y * lc.range_max;
                    h.hz = lc.w_pose.pos.z + dir.z * lc.range_max;
                }
            }
        }
    }
}

// ===========================================================================
// init_optix  –  one-time setup: context, pipeline, initial BVH, ray upload
// ===========================================================================
OptixState* init_optix(const RunConfig& r,
                       const std::vector<Tri3>& tris,
                       size_t staticTriCount,
                       FrameTimings& t)
{
    auto tInit = std::chrono::steady_clock::now();
    if (r.robots.empty()) {
        fprintf(stderr, "[OptiX] No robots configured.\n");
        return nullptr;
    }

    OptixState* s = new OptixState();
    s->crti = (r.backend == "optix-crti");

    // ---- load PTX ----------------------------------------------------------
    std::string ptxCode;
    try { ptxCode = loadPTX(s->crti ? "build/optix_custom_device.ptx"
                                           : "build/optix_device.ptx"); }
    catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        delete s; return nullptr;
    }

    // ---- CUDA + OptiX init -------------------------------------------------
    auto tGpu0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaFree(nullptr));
    CUcontext cuCtx = nullptr;
    cuCtxGetCurrent(&cuCtx);
    fprintf(stderr, "[OptiX] cuCtx = %p\n", (void*)cuCtx);
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction = [](unsigned int level, const char* tag,
                                     const char* msg, void*) {
        if (level <= 3) fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, msg);
    };
    ctxOpts.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &s->optixCtx));
    fprintf(stderr, "[OptiX] OptiX context created\n");

    // ---- build initial BVH (keep vertex buffer alive for per-frame refit) --
    // -1=pure refit → BVH_SYNC_REFIT; 0=pure rebuild → BVH_SYNC_REBUILD; N>0=hybrid → BVH_SYNC_REFIT
    BvhSyncMode bvhSyncMode = (r.optix_rebuild_after_frames == 0) ? BVH_SYNC_REBUILD : BVH_SYNC_REFIT;
    bool hasPeriodicRebuild = (r.optix_rebuild_after_frames > 0);
    uint32_t bvhFlags = make_bvh_flags(bvhSyncMode, r.optix_compaction, hasPeriodicRebuild);
    s->bvhFlags              = bvhFlags;
    s->bvhSyncMode           = bvhSyncMode;
    s->allowUpdate           = (bvhFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
    s->rebuildAfterFrames    = r.optix_rebuild_after_frames;

    s->numVerts       = (unsigned int)(tris.size() * 3);
    s->staticNumVerts = staticTriCount * 3;
    std::vector<float3> verts(s->numVerts);
    for (size_t i = 0; i < tris.size(); ++i)
        for (int k = 0; k < 3; ++k)
            verts[i*3+k] = make_float3(tris[i].v[k][0], tris[i].v[k][1], tris[i].v[k][2]);
    s->d_verts = uploadVector(verts);

    // crti: build AABB buffer from the already-uploaded d_verts on GPU.
    // This avoids a second CPU loop and a second H2D copy.
    if (s->crti) {
        s->aabbBufBytes = tris.size() * sizeof(OptixAabb);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_aabbs), s->aabbBufBytes));
        launch_compute_aabbs(reinterpret_cast<float3*>(s->d_verts),
                             reinterpret_cast<OptixAabb*>(s->d_aabbs), tris.size());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    OptixBuildInput bi = {};
    if (s->crti) {
        bi.type                                       = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers           = &s->d_aabbs;
        bi.customPrimitiveArray.numPrimitives         = (unsigned int)tris.size();
        bi.customPrimitiveArray.strideInBytes         = sizeof(OptixAabb);
        bi.customPrimitiveArray.numSbtRecords         = 1;
        bi.customPrimitiveArray.flags                 = &s->geomFlags;
    } else {
        bi.type                                 = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexFormat           = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.numVertices            = s->numVerts;
        bi.triangleArray.vertexBuffers          = &s->d_verts;
        bi.triangleArray.vertexStrideInBytes    = sizeof(float3);
        bi.triangleArray.numSbtRecords          = 1;
        bi.triangleArray.flags                  = &s->geomFlags;
    }

    OptixAccelBuildOptions buildOpts = {};
    buildOpts.buildFlags = bvhFlags;
    buildOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(s->optixCtx, &buildOpts, &bi, 1, &sizes));
    s->buildOutputBytes = sizes.outputSizeInBytes;
    s->tempRebuildBytes = sizes.tempSizeInBytes;
    s->allowCompaction  = (bvhFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

    CUdeviceptr d_temp = 0, d_output = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),   sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), sizes.outputSizeInBytes));

    cudaEvent_t evB0, evB1;
    CUDA_CHECK(cudaEventCreate(&evB0));
    CUDA_CHECK(cudaEventCreate(&evB1));
    CUDA_CHECK(cudaEventRecord(evB0, 0));

    bool allowCompaction = s->allowCompaction;
    if (allowCompaction) {
        CUdeviceptr d_compSz = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compSz), sizeof(uint64_t)));
        OptixAccelEmitDesc emit = {};
        emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = d_compSz;
        OPTIX_CHECK(optixAccelBuild(s->optixCtx, 0, &buildOpts, &bi, 1,
            d_temp, sizes.tempSizeInBytes,
            d_output, sizes.outputSizeInBytes,
            &s->traversable, &emit, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t compSz = 0;
        CUDA_CHECK(cudaMemcpy(&compSz, reinterpret_cast<void*>(d_compSz),
            sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_gasBuf), compSz));
        OPTIX_CHECK(optixAccelCompact(s->optixCtx, 0, s->traversable,
            s->d_gasBuf, compSz, &s->traversable));
        CUDA_CHECK(cudaDeviceSynchronize());
        s->gasBufBytes = compSz;
        cudaFree(reinterpret_cast<void*>(d_compSz));
        cudaFree(reinterpret_cast<void*>(d_output));
    } else {
        OPTIX_CHECK(optixAccelBuild(s->optixCtx, 0, &buildOpts, &bi, 1,
            d_temp, sizes.tempSizeInBytes,
            d_output, sizes.outputSizeInBytes,
            &s->traversable, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        s->d_gasBuf    = d_output;
        s->gasBufBytes = sizes.outputSizeInBytes;
    }

    CUDA_CHECK(cudaEventRecord(evB1, 0));
    CUDA_CHECK(cudaEventSynchronize(evB1));
    CUDA_CHECK(cudaEventDestroy(evB0));
    CUDA_CHECK(cudaEventDestroy(evB1));
    cudaFree(reinterpret_cast<void*>(d_temp));
    fprintf(stderr, "[OptiX] BVH built (traversable=0x%llx, gasBufBytes=%zu)\n",
            (unsigned long long)s->traversable, s->gasBufBytes);

    // Pre-allocate per-frame rebuild scratch (avoids malloc/free every frame).
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_temp_rebuild),
        s->tempRebuildBytes > 0 ? s->tempRebuildBytes : 1));
    if (s->allowCompaction) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_build_output), s->buildOutputBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_compSzBuf), sizeof(uint64_t)));
    }

    // Pre-allocate the update scratch buffer (tempUpdateSizeInBytes).
    if (s->allowUpdate) {
        OptixAccelBuildOptions updateOpts = buildOpts;
        updateOpts.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixAccelBufferSizes updateSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(s->optixCtx, &updateOpts, &bi, 1, &updateSizes));
        s->tempUpdateBytes = updateSizes.tempUpdateSizeInBytes;
        // Allocate at least 1 byte so the pointer is non-null.
        size_t allocSize = (s->tempUpdateBytes > 0) ? s->tempUpdateBytes : 1;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_temp_update), allocSize));
        fprintf(stderr, "[OptiX] update scratch = %zu bytes\n", s->tempUpdateBytes);
    }

    // ---- pipeline compile options (hardcoded) ------------------------------
    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur                   = false;
    pipeOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues                 = 1;
    pipeOpts.numAttributeValues               = 2;
    pipeOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "params";

    // ---- module from PTX (hardcoded) ---------------------------------------
    OptixModuleCompileOptions modOpts = {};
    modOpts.maxRegisterCount = 0;
    modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    OptixModule mod = nullptr;
    {
        char log[4096]; size_t logSz = sizeof(log);
        OPTIX_CHECK(optixModuleCreate(s->optixCtx, &modOpts, &pipeOpts,
            ptxCode.c_str(), ptxCode.size(), log, &logSz, &mod));
        if (logSz > 1) fprintf(stderr, "[module log] %s\n", log);
    }

    // ---- program groups + pipeline -----------------------------------------
    OptixProgramGroupOptions pgOpts = {};
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = mod;
        d.raygen.entryFunctionName = s->crti ? "__raygen__lidar_custom" : "__raygen__lidar";
        char log[4096]; size_t logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(s->optixCtx, &d, 1, &pgOpts, log, &logSz, &s->pgRaygen));
    }
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module = mod;
        d.miss.entryFunctionName = s->crti ? "__miss__lidar_custom" : "__miss__lidar";
        char log[4096]; size_t logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(s->optixCtx, &d, 1, &pgOpts, log, &logSz, &s->pgMiss));
    }
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleCH            = mod;
        d.hitgroup.entryFunctionNameCH = s->crti ? "__closesthit__lidar_custom"
                                                        : "__closesthit__lidar";
        if (s->crti) {
            d.hitgroup.moduleIS            = mod;
            d.hitgroup.entryFunctionNameIS = "__intersection__lidar_custom";
        }
        char log[4096]; size_t logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(s->optixCtx, &d, 1, &pgOpts, log, &logSz, &s->pgHitGroup));
    }
    {
        OptixPipelineLinkOptions linkOpts = {};
        linkOpts.maxTraceDepth = 1;
        OptixProgramGroup pgs[] = { s->pgRaygen, s->pgMiss, s->pgHitGroup };
        char log[4096]; size_t logSz = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(s->optixCtx, &pipeOpts, &linkOpts,
            pgs, 3, log, &logSz, &s->pipeline));
        OPTIX_CHECK(optixPipelineSetStackSize(s->pipeline, 0, 0, 2048, 1));
    }

    // ---- SBT ---------------------------------------------------------------
    // NOTE: pack SBT headers before destroying the module — some driver versions
    // still need the module data when computing the header bytes.
    RaygenRecord   h_rg  = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(s->pgRaygen,   &h_rg));
    MissRecord     h_ms  = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(s->pgMiss,     &h_ms));
    HitGroupRecord h_hg  = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(s->pgHitGroup, &h_hg));
    s->d_raygenRec   = uploadVector(std::vector<RaygenRecord>  {h_rg});
    s->d_missRec     = uploadVector(std::vector<MissRecord>    {h_ms});
    s->d_hitGroupRec = uploadVector(std::vector<HitGroupRecord>{h_hg});

    optixModuleDestroy(mod);   // safe to destroy after SBT headers are packed
    s->sbt.raygenRecord                = s->d_raygenRec;
    s->sbt.missRecordBase              = s->d_missRec;
    s->sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    s->sbt.missRecordCount             = 1;
    s->sbt.hitgroupRecordBase          = s->d_hitGroupRec;
    s->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    s->sbt.hitgroupRecordCount         = 1;
    fprintf(stderr, "[OptiX] SBT ready\n");

    // ---- flatten rays and upload to GPU ------------------------------------
    s->num_origins = 0;
    s->total_rays = 0;
    for (const auto& robot : r.robots)
        for (const auto& lc : robot.lidars) {
            ++s->num_origins;
            s->total_rays += (size_t)lc.hnum * lc.vnum;
        }
    CUDA_CHECK(cudaMalloc(&s->d_distances,    s->total_rays  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_directions,   s->total_rays  * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&s->d_ray_origins, s->num_origins * sizeof(RayOrigin)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s->h_ray_origins_pinned), s->num_origins * sizeof(RayOrigin)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s->h_distances_pinned),  s->total_rays * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s->h_directions_pinned), s->total_rays * sizeof(float3)));
    size_t dynVerts_init = s->numVerts - s->staticNumVerts;
    if (dynVerts_init > 0) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s->h_dynBuf), dynVerts_init * sizeof(float3)));
        s->h_dynBufCount = dynVerts_init;
    }
    size_t ignored;
    buildRayOrigins(r, s->h_ray_origins_pinned, s->num_origins, ignored);
    CUDA_CHECK(cudaMemcpy(s->d_ray_origins, s->h_ray_origins_pinned,
        s->num_origins * sizeof(RayOrigin), cudaMemcpyHostToDevice));
    if (s->crti) {
        LaunchParamsCustom params = {};
        params.ray_origins = s->d_ray_origins;
        params.num_origins = (unsigned int)s->num_origins;
        params.num_rays    = (unsigned int)s->total_rays;
        params.distances   = s->d_distances;
        params.directions  = s->d_directions;
        params.traversable = s->traversable;
        params.vertices    = reinterpret_cast<float3*>(s->d_verts);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_params), sizeof(LaunchParamsCustom)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s->d_params), &params,
            sizeof(LaunchParamsCustom), cudaMemcpyHostToDevice));
        s->paramsSize = sizeof(LaunchParamsCustom);
    } else {
        LaunchParams params = {};
        params.ray_origins = s->d_ray_origins;
        params.num_origins = (unsigned int)s->num_origins;
        params.num_rays    = (unsigned int)s->total_rays;
        params.distances   = s->d_distances;
        params.directions  = s->d_directions;
        params.traversable = s->traversable;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_params), sizeof(LaunchParams)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s->d_params), &params,
            sizeof(LaunchParams), cudaMemcpyHostToDevice));
        s->paramsSize = sizeof(LaunchParams);
    }

    // ---- reusable CUDA events ----------------------------------------------
    CUDA_CHECK(cudaStreamCreate(&s->stream));
    CUDA_CHECK(cudaEventCreate(&s->evBvhStart));
    CUDA_CHECK(cudaEventCreate(&s->evBvhEnd));
    CUDA_CHECK(cudaEventCreate(&s->evTraceStart));
    CUDA_CHECK(cudaEventCreate(&s->evTraceEnd));

    auto tGpu1 = std::chrono::steady_clock::now();
    t.ms_init_cpu = std::chrono::duration<double,std::milli>(tGpu1 - tInit).count();
    fprintf(stderr, "[OptiX] init: %zu tris, %zu rays, allowUpdate=%d  (%.2f ms)\n",
            tris.size(), s->total_rays, (int)s->allowUpdate, t.ms_init_cpu);
    return s;
}

// ===========================================================================
// run_optix_frame  –  per-frame: upload new vertex positions, refit, trace
// ===========================================================================
void run_optix_frame(OptixState*              s,
                     const RunConfig&         cfg,
                     const std::vector<Tri3>& tris,
                     std::vector<GrcaHit>&     hits,
                     FrameTimings&              t)
{
    // Decide once per frame whether to force a full rebuild in refit mode
    bool periodicRebuild = (s->rebuildAfterFrames > 0)
        && (s->frameCount % (s->rebuildAfterFrames + 1) == 0);
    s->frameCount++;

    // ---- update only the dynamic vertex sub-range on GPU ------------------
    // Static vertices were uploaded once at init and never change.
    auto tUp        = std::chrono::steady_clock::now();
    size_t totalVerts = tris.size() * 3;
    size_t dynVerts   = totalVerts - s->staticNumVerts;
    if (dynVerts > 0) {
        size_t base    = s->staticNumVerts / 3;   // first dynamic triangle index
        size_t dynTris = dynVerts / 3;
        // Tri3::float v[3][3] is layout-identical to packed float3*3
        memcpy(s->h_dynBuf, tris.data() + base, dynTris * sizeof(Tri3));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<float3*>(s->d_verts) + s->staticNumVerts,
            s->h_dynBuf, dynVerts * sizeof(float3), cudaMemcpyHostToDevice));
    }
    // ---- Re-build lidar configs from current world poses (per-lidar, not per-ray) ---
    size_t ignored_lidars, ignored_rays;
    buildRayOrigins(cfg, s->h_ray_origins_pinned, ignored_lidars, ignored_rays);
    CUDA_CHECK(cudaMemcpy(s->d_ray_origins, s->h_ray_origins_pinned,
        s->num_origins * sizeof(RayOrigin), cudaMemcpyHostToDevice));
    t.ms_dynamic_upload = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tUp).count();

    // BVH update/rebuild logic (always needed if there are dynamic objects)
    double msCompactSync = 0.0;  // compact sync+readback time (excluded from ms_bvh_rebuild_cpu)
    if (dynVerts > 0) {
        OptixBuildInput bi = {};
        if (s->crti) {
            bi.type                                       = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            bi.customPrimitiveArray.aabbBuffers           = &s->d_aabbs;
            bi.customPrimitiveArray.numPrimitives         = s->numVerts / 3;
            bi.customPrimitiveArray.strideInBytes         = sizeof(OptixAabb);
            bi.customPrimitiveArray.numSbtRecords         = 1;
            bi.customPrimitiveArray.flags                 = &s->geomFlags;
        } else {
            bi.type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.numVertices         = s->numVerts;
            bi.triangleArray.vertexBuffers       = &s->d_verts;
            bi.triangleArray.vertexStrideInBytes = sizeof(float3);
            bi.triangleArray.numSbtRecords       = 1;
            bi.triangleArray.flags               = &s->geomFlags;
        }

        auto tBvhCpu = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(s->evBvhStart, s->stream));

        // crti: compute AABBs on s->stream — serialized with the build and
        // bracketed by evBvhStart/evBvhEnd so GPU BVH time includes it.
        if (s->crti) {
            size_t base    = s->staticNumVerts / 3;
            size_t dynTris = dynVerts / 3;
            launch_compute_aabbs(
                reinterpret_cast<float3*>(s->d_verts) + s->staticNumVerts,
                reinterpret_cast<OptixAabb*>(s->d_aabbs) + base,
                dynTris, s->stream);
        }

        if (s->bvhSyncMode == BVH_SYNC_REFIT && !periodicRebuild) {
            // BVH refit — all buffers pre-allocated, stays fully async until event sync
            OptixAccelBuildOptions refitOpts = {};
            refitOpts.buildFlags = s->bvhFlags;
            refitOpts.operation  = OPTIX_BUILD_OPERATION_UPDATE;

            OPTIX_CHECK(optixAccelBuild(s->optixCtx, s->stream,
                &refitOpts, &bi, 1,
                s->d_temp_update, s->tempUpdateBytes,
                s->d_gasBuf,      s->gasBufBytes,
                &s->traversable, nullptr, 0));
            CUDA_CHECK(cudaEventRecord(s->evBvhEnd, s->stream));
        } else { // default: rebuild (pure-rebuild every frame, or hybrid periodic rebuild)
            OptixAccelBuildOptions buildOpts = {};
            buildOpts.buildFlags = s->bvhFlags;  // always matches init flags
            buildOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;


            if (s->allowCompaction) {
                OptixAccelEmitDesc emit = {};
                emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
                emit.result = s->d_compSzBuf;
                OPTIX_CHECK(optixAccelBuild(s->optixCtx, s->stream, &buildOpts, &bi, 1,
                    s->d_temp_rebuild, s->tempRebuildBytes,
                    s->d_build_output, s->buildOutputBytes,
                    &s->traversable, &emit, 1));
                // Sync + size readback: timed separately so ms_bvh_rebuild_cpu
                // only covers enqueue time, not GPU wait.
                auto tCompactSync = std::chrono::steady_clock::now();
                CUDA_CHECK(cudaStreamSynchronize(s->stream));
                msCompactSync = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tCompactSync).count();
                uint64_t compSz = 0;
                CUDA_CHECK(cudaMemcpy(&compSz, reinterpret_cast<void*>(s->d_compSzBuf),
                    sizeof(uint64_t), cudaMemcpyDeviceToHost));
                if (s->gasBufBytes != compSz) {
                    cudaFree(reinterpret_cast<void*>(s->d_gasBuf));
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s->d_gasBuf), compSz));
                    s->gasBufBytes = compSz;
                }
                OPTIX_CHECK(optixAccelCompact(s->optixCtx, s->stream, s->traversable,
                    s->d_gasBuf, compSz, &s->traversable));
            } else {
                OPTIX_CHECK(optixAccelBuild(s->optixCtx, s->stream, &buildOpts, &bi, 1,
                    s->d_temp_rebuild, s->tempRebuildBytes,
                    s->d_gasBuf,       s->gasBufBytes,
                    &s->traversable, nullptr, 0));
            }

            CUDA_CHECK(cudaEventRecord(s->evBvhEnd, s->stream));
        }

        double msBvhCpu = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tBvhCpu).count();
        if (s->bvhSyncMode == BVH_SYNC_REFIT && !periodicRebuild) t.ms_bvh_refit_cpu = msBvhCpu;
        else                                                      t.ms_bvh_rebuild_cpu = msBvhCpu - msCompactSync;
    }

    // Update traversable in d_params after any BVH rebuild/refit.
    // Must happen here (not before the BVH section) because a compaction rebuild
    // may reallocate d_gasBuf, giving s->traversable a new handle; updating
    // d_params beforehand would leave a dangling pointer on the device.
    // Use async on s->stream so the copy is queued after the BVH work and does
    // not implicitly synchronize the NULL stream with s->stream (which would
    // silently absorb the BVH rebuild wait into an unmeasured gap).
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(s->d_params) + offsetof(LaunchParams, traversable),
        &s->traversable, sizeof(OptixTraversableHandle),
        cudaMemcpyHostToDevice, s->stream));

    // ---- optixLaunch -------------------------------------------------------
    auto tTraceCpu0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(s->evTraceStart, s->stream));
    OPTIX_CHECK(optixLaunch(s->pipeline, s->stream,
        s->d_params, s->paramsSize, &s->sbt,
        (unsigned int)s->total_rays, 1, 1));
    CUDA_CHECK(cudaEventRecord(s->evTraceEnd, s->stream));
    t.ms_trace_cpu = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tTraceCpu0).count();
    auto tSync0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventSynchronize(s->evTraceEnd));
    t.ms_sync_readback = msCompactSync + std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tSync0).count();
    // Both evBvhEnd and evTraceEnd are complete — read all GPU timings now.
    if (dynVerts > 0) {
        float msBvh = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&msBvh, s->evBvhStart, s->evBvhEnd));
        if (s->bvhSyncMode == BVH_SYNC_REFIT && !periodicRebuild) t.ms_bvh_refit_gpu = (double)msBvh;
        else                                                       t.ms_bvh_rebuild_gpu = (double)msBvh;
    }
    float msTrace = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&msTrace, s->evTraceStart, s->evTraceEnd));
    t.ms_trace_gpu     = (double)msTrace;
    t.ms_lidar_sim_gpu = t.ms_bvh_refit_gpu + t.ms_bvh_rebuild_gpu + t.ms_trace_gpu;

    // ---- readback into pinned buffers --------------------------------------
    auto tRead = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(s->h_distances_pinned,  s->d_distances,
        s->total_rays * sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s->h_directions_pinned, s->d_directions,
        s->total_rays * sizeof(float3), cudaMemcpyDeviceToHost));
    t.ms_readback_cpu = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tRead).count();
    t.mb_readback  = (double)(s->total_rays * (sizeof(float) + sizeof(float3))) / (1024.0 * 1024.0);

    // ---- convert to GrcaHit -------------------------------------------------
    auto tFill0 = std::chrono::steady_clock::now();
    fillHits(s->h_distances_pinned, s->h_directions_pinned, s->total_rays, cfg, hits, t.hit_count);
    t.ms_fillhits = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tFill0).count();

}

// ===========================================================================
// destroy_optix  –  release all resources owned by state
// ===========================================================================
void destroy_optix(OptixState* s)
{
    if (!s) return;
    cudaStreamDestroy(s->stream);
    cudaEventDestroy(s->evTraceStart); cudaEventDestroy(s->evTraceEnd);
    cudaEventDestroy(s->evBvhStart);   cudaEventDestroy(s->evBvhEnd);
    cudaFree(reinterpret_cast<void*>(s->d_params));
    cudaFree(s->d_distances);
    cudaFree(s->d_directions);
    cudaFree(s->d_ray_origins);
    cudaFree(reinterpret_cast<void*>(s->d_verts));
    cudaFree(reinterpret_cast<void*>(s->d_temp_update));
    cudaFree(reinterpret_cast<void*>(s->d_temp_rebuild));
    cudaFree(reinterpret_cast<void*>(s->d_build_output));
    cudaFree(reinterpret_cast<void*>(s->d_compSzBuf));
    cudaFree(reinterpret_cast<void*>(s->d_gasBuf));
    cudaFree(reinterpret_cast<void*>(s->d_aabbs));
    if (s->h_ray_origins_pinned) cudaFreeHost(s->h_ray_origins_pinned);
    if (s->h_dynBuf)             cudaFreeHost(s->h_dynBuf);
    if (s->h_distances_pinned)  cudaFreeHost(s->h_distances_pinned);
    if (s->h_directions_pinned) cudaFreeHost(s->h_directions_pinned);
    cudaFree(reinterpret_cast<void*>(s->d_raygenRec));
    cudaFree(reinterpret_cast<void*>(s->d_missRec));
    cudaFree(reinterpret_cast<void*>(s->d_hitGroupRec));
    optixPipelineDestroy(s->pipeline);
    optixProgramGroupDestroy(s->pgRaygen);
    optixProgramGroupDestroy(s->pgMiss);
    optixProgramGroupDestroy(s->pgHitGroup);
    optixDeviceContextDestroy(s->optixCtx);
    delete s;
}

// ===========================================================================
// run_optix  –  legacy one-shot wrapper (init + single frame + destroy)
// ===========================================================================
void run_optix(const RunConfig& cfg,
               const std::vector<Tri3>& tris,
               std::vector<GrcaHit>& hits,
               FrameTimings& t)
{
    OptixState* state = init_optix(cfg, tris, 0, t);
    if (!state) return;
    run_optix_frame(state, cfg, tris, hits, t);
    destroy_optix(state);
}
