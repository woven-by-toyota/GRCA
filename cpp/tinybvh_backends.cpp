// =============================================================================
// tinybvh_backends.cpp
// TinyBVH CPU and GPU backends for GRCA.
//
// This is the ONE translation unit that defines TINYBVH_IMPLEMENTATION.
// Both the CPU backend (single-ray traversal on CPU) and the GPU backend
// (BVH built on CPU, rays traced via CUDA kernel) live here so the
// single-header library is only instantiated once.
//
// CPU backend ("tinybvh-cpu"):
//   Direct comparison with Embree single-threaded (packet_size=1).
//   Uses tinybvh::BVH with the same rebuild/refit hybrid logic as Embree.
//
// GPU backend ("tinybvh-gpu"):
//   BVH is built/refitted on CPU by tinybvh, then nodes and BVH-sorted
//   triangles are uploaded to GPU each frame.  Rays are traced by a custom
//   CUDA kernel (cuda/tinybvh_kernels.cu).
//
// Config fields (RunConfig):
//   tinybvh_cpu_rebuild_after_frames  — same semantics as embree_rebuild_after_frames
//   tinybvh_gpu_rebuild_after_frames  — same semantics
// =============================================================================

#define TINYBVH_IMPLEMENTATION
#include "../third_party/tiny_bvh.h"

#include "tinybvh_cpu.h"
#include "tinybvh_gpu.h"
#include "../include/grca_common.h"

#ifdef HAVE_CUDA
#  include "../cuda/tinybvh_kernels.h"
#  include <cuda_runtime_api.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tinybvh;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Convert Tri3 flat array to bvhvec4 (3 per triangle, w=0).
// The resulting vector is kept alive as long as the BVH is alive because
// tinybvh stores a pointer to the verts passed to Build().
static void fill_verts(const std::vector<Tri3>& tris,
                       std::vector<bvhvec4>& verts)
{
    verts.resize(tris.size() * 3);
    for (size_t i = 0; i < tris.size(); ++i) {
        for (int v = 0; v < 3; ++v) {
            verts[i*3+v].x = tris[i].v[v][0];
            verts[i*3+v].y = tris[i].v[v][1];
            verts[i*3+v].z = tris[i].v[v][2];
            verts[i*3+v].w = 0.f;
        }
    }
}

// Update only the dynamic portion of the verts array from current tris.
static void update_dyn_verts(const std::vector<Tri3>& tris,
                              std::vector<bvhvec4>& verts,
                              size_t staticTriCount)
{
    size_t dynCount = tris.size() - staticTriCount;
    for (size_t i = 0; i < dynCount; ++i) {
        size_t ti = staticTriCount + i;
        for (int v = 0; v < 3; ++v) {
            verts[ti*3+v].x = tris[ti].v[v][0];
            verts[ti*3+v].y = tris[ti].v[v][1];
            verts[ti*3+v].z = tris[ti].v[v][2];
        }
    }
}

// BVH hybrid mode decision: same logic as embree.cpp.
// Returns true if this frame should refit (false = full rebuild).
static bool decide_refit(int rebuildAfterFrames, int frameCount)
{
    if (rebuildAfterFrames == -1) return true;   // pure refit
    if (rebuildAfterFrames ==  0) return false;  // pure rebuild
    // hybrid: refit unless it's the periodic rebuild frame
    bool periodicRebuild = (frameCount % (rebuildAfterFrames + 1) == 0);
    return !periodicRebuild;
}

// Total ray count across all robots/lidars.
static size_t total_rays(const RunConfig& cfg)
{
    size_t n = 0;
    for (const auto& r : cfg.robots)
        for (const auto& l : r.lidars)
            n += (size_t)l.hnum * l.vnum;
    return n;
}

// ---------------------------------------------------------------------------
// Ray direction generation (matches optix_device.cu / embree.cpp exactly).
// ---------------------------------------------------------------------------
static inline void ray_dir(const LidarCfg& lc,
                            float cosH, float sinH,
                            float cosV, float sinV,
                            float& dx, float& dy, float& dz)
{
    dx = (cosH*lc.w_pose.forward.x + sinH*lc.w_pose.right.x)*cosV + sinV*lc.w_pose.up.x;
    dy = (cosH*lc.w_pose.forward.y + sinH*lc.w_pose.right.y)*cosV + sinV*lc.w_pose.up.y;
    dz = (cosH*lc.w_pose.forward.z + sinH*lc.w_pose.right.z)*cosV + sinV*lc.w_pose.up.z;
}

// =============================================================================
// CPU BACKEND
// =============================================================================

struct TinyBVHCPUState {
    BVH              bvh;
    std::vector<bvhvec4> verts;       // kept alive: tinybvh stores our pointer
    size_t staticTriCount = 0;
    size_t dynTriCount    = 0;
    int    rebuildAfterFrames = 0;
    int    frameCount         = 0;
};

TinyBVHCPUState* init_tinybvh_cpu(const RunConfig&          cfg,
                                   const std::vector<Tri3>& tris,
                                   size_t                   staticTriCount,
                                   FrameTimings&            t)
{
    auto tInit = clk::now();

    TinyBVHCPUState* s = new TinyBVHCPUState();
    s->staticTriCount     = staticTriCount;
    s->dynTriCount        = tris.size() - staticTriCount;
    s->rebuildAfterFrames = cfg.tinybvh_cpu_rebuild_after_frames;

    fill_verts(tris, s->verts);

    auto tBvh = clk::now();
    s->bvh.Build(s->verts.data(), (uint32_t)tris.size());
    t.ms_bvh_rebuild_cpu = ms(tBvh, clk::now());

    t.ms_init_cpu = ms(tInit, clk::now());
    fprintf(stderr, "[TinyBVH-CPU] init: %zu tris (%zus+%zud), %zu rays, "
            "rebuild_after=%d  (%.2f ms)\n",
            tris.size(), staticTriCount, s->dynTriCount,
            total_rays(cfg), s->rebuildAfterFrames, t.ms_init_cpu);
    return s;
}

void run_tinybvh_cpu_frame(TinyBVHCPUState*          s,
                            const RunConfig&          cfg,
                            const std::vector<Tri3>& tris,
                            std::vector<GrcaHit>&     hits,
                            FrameTimings&            t)
{
    // ---- dynamic vert upload -----------------------------------------------
    auto tUp = clk::now();
    if (s->dynTriCount > 0)
        update_dyn_verts(tris, s->verts, s->staticTriCount);
    t.ms_dynamic_upload = ms(tUp, clk::now());

    // ---- BVH refit / rebuild -----------------------------------------------
    if (s->dynTriCount > 0) {
        bool doRefit = decide_refit(s->rebuildAfterFrames, s->frameCount);
        s->frameCount++;
        auto tBvh = clk::now();
        if (doRefit) {
            s->bvh.Refit();
            t.ms_bvh_refit_cpu = ms(tBvh, clk::now());
        } else {
            s->bvh.Build(s->verts.data(), (uint32_t)tris.size());
            t.ms_bvh_rebuild_cpu = ms(tBvh, clk::now());
        }
    }

    // ---- Ray cast (single ray per call, same pattern as embree pkt=1) ------
    auto tTrace = clk::now();
    size_t numRays = total_rays(cfg);
    hits.resize(numRays);

    size_t offset = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            const unsigned hnum = (unsigned)lc.hnum;
            const unsigned vnum = (unsigned)lc.vnum;
            const float hstep = (hnum > 1u) ? (lc.hmax - lc.hmin) / (float)(hnum - 1u) : 0.f;
            const float vstep = (vnum > 1u) ? (lc.vmax - lc.vmin) / (float)(vnum - 1u) : 0.f;
            const float ox = lc.w_pose.pos.x, oy = lc.w_pose.pos.y, oz = lc.w_pose.pos.z;

            // Precompute trig tables (O(H+V) calls, not O(H*V)).
            std::vector<float> cosH(hnum), sinH(hnum);
            for (unsigned h = 0; h < hnum; ++h) {
                cosH[h] = cosf(lc.hmin + (float)h * hstep);
                sinH[h] = sinf(lc.hmin + (float)h * hstep);
            }

            for (unsigned vi = 0; vi < vnum; ++vi) {
                const float cosV = cosf(lc.vmin + (float)vi * vstep);
                const float sinV = sinf(lc.vmin + (float)vi * vstep);

                for (unsigned hi = 0; hi < hnum; ++hi) {
                    float dx, dy, dz;
                    ray_dir(lc, cosH[hi], sinH[hi], cosV, sinV, dx, dy, dz);

                    Ray ray;
                    ray.O  = bvhvec3(ox, oy, oz);
                    ray.D  = bvhvec3(dx, dy, dz);
                    ray.rD = bvhvec3(
                        fabsf(dx) > 1e-30f ? 1.f/dx : 1e30f,
                        fabsf(dy) > 1e-30f ? 1.f/dy : 1e30f,
                        fabsf(dz) > 1e-30f ? 1.f/dz : 1e30f);
                    ray.hit.t    = lc.range_max;
                    ray.hit.prim = 0xffffffffu;

                    s->bvh.Intersect(ray);

                    GrcaHit& g = hits[offset + (size_t)vi * hnum + hi];
                    g.dx   = dx; g.dy = dy; g.dz = dz;
                    g.dist = (ray.hit.t >= lc.range_min && ray.hit.t < lc.range_max)
                             ? ray.hit.t : lc.range_max;
                }
            }
            offset += (size_t)hnum * vnum;
        }
    }
    t.ms_trace_cpu = ms(tTrace, clk::now());

    // ---- fill hit world positions ------------------------------------------
    int hitCount = 0;
    auto tFill = clk::now();
    size_t fillOff = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            size_t n = (size_t)lc.hnum * lc.vnum;
            for (size_t i = fillOff; i < fillOff + n; ++i) {
                GrcaHit& g = hits[i];
                g.hx = lc.w_pose.pos.x + g.dx * g.dist;
                g.hy = lc.w_pose.pos.y + g.dy * g.dist;
                g.hz = lc.w_pose.pos.z + g.dz * g.dist;
                if (g.dist < lc.range_max) ++hitCount;
            }
            fillOff += n;
        }
    }
    t.ms_fillhits  = ms(tFill, clk::now());
    t.hit_count    = hitCount;
}

void destroy_tinybvh_cpu(TinyBVHCPUState* s)
{
    delete s;
}

// =============================================================================
// GPU BACKEND  (only compiled when CUDA is available)
// =============================================================================

#ifdef HAVE_CUDA

// GPU node buffer: 2 float4s per BVH node (32 bytes), matching BVH::BVHNode.
// GPU tri  buffer: 3 float4s per triangle (48 bytes), in BVH primIdx order.
struct TinyBVHGPUState {
    BVH              bvh;
    std::vector<bvhvec4> verts;         // host verts (kept alive for tinybvh ptr)
    std::vector<float4>  h_sortedTris;  // host sorted tris for GPU upload

    float4* d_nodes      = nullptr;
    float4* d_sortedTris = nullptr;
    float*  d_rays       = nullptr;   // [ox,oy,oz,dx,dy,dz] per ray
    float*  d_hitDist    = nullptr;

    int d_nodeCapacity = 0;
    int d_triCapacity  = 0;
    int d_rayCapacity  = 0;

    size_t staticTriCount = 0;
    size_t dynTriCount    = 0;
    int    rebuildAfterFrames = 0;
    int    frameCount         = 0;
};

// Ensure GPU buffer is at least newCap elements of elemSize bytes.
// Reallocates (free + malloc) if needed.
static void ensure_gpu_buf(void** ptr, int* cap, int newCap, size_t elemSize)
{
    if (newCap <= *cap) return;
    if (*ptr) cudaFree(*ptr);
    cudaMalloc(ptr, (size_t)newCap * elemSize);
    *cap = newCap;
}

// (Re)build sorted tri array from current bvh.primIdx and verts.
static void build_sorted_tris(const BVH& bvh,
                               const std::vector<bvhvec4>& verts,
                               std::vector<float4>& out)
{
    uint32_t n = bvh.triCount;
    out.resize((size_t)n * 3);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t orig = bvh.primIdx[i];
        out[i*3+0] = {verts[orig*3+0].x, verts[orig*3+0].y, verts[orig*3+0].z, 0.f};
        out[i*3+1] = {verts[orig*3+1].x, verts[orig*3+1].y, verts[orig*3+1].z, 0.f};
        out[i*3+2] = {verts[orig*3+2].x, verts[orig*3+2].y, verts[orig*3+2].z, 0.f};
    }
}

TinyBVHGPUState* init_tinybvh_gpu(const RunConfig&          cfg,
                                   const std::vector<Tri3>& tris,
                                   size_t                   staticTriCount,
                                   FrameTimings&            t)
{
    auto tInit = clk::now();

    TinyBVHGPUState* s = new TinyBVHGPUState();
    s->staticTriCount     = staticTriCount;
    s->dynTriCount        = tris.size() - staticTriCount;
    s->rebuildAfterFrames = cfg.tinybvh_gpu_rebuild_after_frames;

    fill_verts(tris, s->verts);

    // Build initial BVH on CPU.
    auto tBvh = clk::now();
    s->bvh.Build(s->verts.data(), (uint32_t)tris.size());
    t.ms_bvh_rebuild_cpu = ms(tBvh, clk::now());

    // Sort triangles in BVH leaf order and upload.
    build_sorted_tris(s->bvh, s->verts, s->h_sortedTris);

    int nodeCount = (int)s->bvh.newNodePtr;
    int triCount  = (int)s->bvh.triCount;
    int numRays   = (int)total_rays(cfg);

    cudaMalloc((void**)&s->d_nodes,      (size_t)nodeCount * 2 * sizeof(float4));
    cudaMalloc((void**)&s->d_sortedTris, (size_t)triCount  * 3 * sizeof(float4));
    cudaMalloc((void**)&s->d_rays,       (size_t)numRays   * 6 * sizeof(float));
    cudaMalloc((void**)&s->d_hitDist,    (size_t)numRays       * sizeof(float));
    s->d_nodeCapacity = nodeCount;
    s->d_triCapacity  = triCount;
    s->d_rayCapacity  = numRays;

    cudaMemcpy(s->d_nodes,      s->bvh.bvhNode,          (size_t)nodeCount * 2 * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(s->d_sortedTris, s->h_sortedTris.data(),  (size_t)triCount  * 3 * sizeof(float4), cudaMemcpyHostToDevice);

    t.ms_init_cpu = ms(tInit, clk::now());
    fprintf(stderr, "[TinyBVH-GPU] init: %zu tris (%zus+%zud), %d rays, "
            "rebuild_after=%d  (%.2f ms)\n",
            tris.size(), staticTriCount, s->dynTriCount,
            numRays, s->rebuildAfterFrames, t.ms_init_cpu);
    return s;
}

void run_tinybvh_gpu_frame(TinyBVHGPUState*          s,
                            const RunConfig&          cfg,
                            const std::vector<Tri3>& tris,
                            std::vector<GrcaHit>&     hits,
                            FrameTimings&            t)
{
    // ---- dynamic vert upload to CPU BVH ------------------------------------
    auto tUp = clk::now();
    if (s->dynTriCount > 0)
        update_dyn_verts(tris, s->verts, s->staticTriCount);
    t.ms_dynamic_upload = ms(tUp, clk::now());

    // ---- CPU BVH refit / rebuild -------------------------------------------
    if (s->dynTriCount > 0) {
        bool doRefit = decide_refit(s->rebuildAfterFrames, s->frameCount);
        s->frameCount++;
        auto tBvh = clk::now();
        if (doRefit) {
            s->bvh.Refit();
            t.ms_bvh_refit_cpu = ms(tBvh, clk::now());
        } else {
            s->bvh.Build(s->verts.data(), (uint32_t)tris.size());
            t.ms_bvh_rebuild_cpu = ms(tBvh, clk::now());
        }
    }

    // ---- Upload updated BVH nodes + sorted tris to GPU --------------------
    build_sorted_tris(s->bvh, s->verts, s->h_sortedTris);

    int nodeCount = (int)s->bvh.newNodePtr;
    int triCount  = (int)s->bvh.triCount;

    ensure_gpu_buf((void**)&s->d_nodes,      &s->d_nodeCapacity, nodeCount, 2 * sizeof(float4));
    ensure_gpu_buf((void**)&s->d_sortedTris, &s->d_triCapacity,  triCount,  3 * sizeof(float4));

    cudaMemcpy(s->d_nodes,      s->bvh.bvhNode,         (size_t)nodeCount * 2 * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(s->d_sortedTris, s->h_sortedTris.data(), (size_t)triCount  * 3 * sizeof(float4), cudaMemcpyHostToDevice);

    // ---- Build ray buffer on CPU and upload --------------------------------
    size_t numRays = total_rays(cfg);
    ensure_gpu_buf((void**)&s->d_rays,    &s->d_rayCapacity, (int)numRays, 6 * sizeof(float));
    // d_hitDist size never changes (numRays can only grow on new batch)
    if ((int)numRays > s->d_rayCapacity) {  // d_hitDist tracks rayCapacity
        if (s->d_hitDist) cudaFree(s->d_hitDist);
        cudaMalloc((void**)&s->d_hitDist, numRays * sizeof(float));
    }

    // Pack [ox, oy, oz, dx, dy, dz] per ray.
    std::vector<float> h_rays(numRays * 6);
    size_t ri = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            const unsigned hnum = (unsigned)lc.hnum;
            const unsigned vnum = (unsigned)lc.vnum;
            const float hstep = (hnum > 1u) ? (lc.hmax - lc.hmin) / (float)(hnum - 1u) : 0.f;
            const float vstep = (vnum > 1u) ? (lc.vmax - lc.vmin) / (float)(vnum - 1u) : 0.f;
            const float ox = lc.w_pose.pos.x, oy = lc.w_pose.pos.y, oz = lc.w_pose.pos.z;

            std::vector<float> cosH(hnum), sinH(hnum);
            for (unsigned h = 0; h < hnum; ++h) {
                cosH[h] = cosf(lc.hmin + (float)h * hstep);
                sinH[h] = sinf(lc.hmin + (float)h * hstep);
            }

            for (unsigned vi = 0; vi < vnum; ++vi) {
                const float cosV = cosf(lc.vmin + (float)vi * vstep);
                const float sinV = sinf(lc.vmin + (float)vi * vstep);
                for (unsigned hi = 0; hi < hnum; ++hi) {
                    float dx, dy, dz;
                    ray_dir(lc, cosH[hi], sinH[hi], cosV, sinV, dx, dy, dz);
                    h_rays[ri*6+0] = ox; h_rays[ri*6+1] = oy; h_rays[ri*6+2] = oz;
                    h_rays[ri*6+3] = dx; h_rays[ri*6+4] = dy; h_rays[ri*6+5] = dz;
                    ++ri;
                }
            }
        }
    }

    cudaMemcpy(s->d_rays, h_rays.data(), numRays * 6 * sizeof(float), cudaMemcpyHostToDevice);

    // ---- CUDA BVH traversal kernel -----------------------------------------
    // Use a per-frame tmin derived from the first lidar's range_min (all
    // lidars in a single batch have the same range_min by convention).
    float tmin = cfg.robots.empty() || cfg.robots[0].lidars.empty()
                 ? 0.f : cfg.robots[0].lidars[0].range_min;
    float tmax = cfg.robots.empty() || cfg.robots[0].lidars.empty()
                 ? 1e30f : cfg.robots[0].lidars[0].range_max;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    launch_tbvh_rays(s->d_nodes, s->d_sortedTris, s->d_rays, s->d_hitDist,
                     (int)numRays, tmin, tmax);

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float gpuMs = 0.f;
    cudaEventElapsedTime(&gpuMs, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    t.ms_trace_gpu = gpuMs;

    // ---- Readback hit distances and fill GrcaHit ----------------------------
    auto tRb = clk::now();
    std::vector<float> h_hitDist(numRays);
    cudaMemcpy(h_hitDist.data(), s->d_hitDist, numRays * sizeof(float), cudaMemcpyDeviceToHost);
    t.ms_readback_cpu = ms(tRb, clk::now());
    t.mb_readback     = (double)(numRays * sizeof(float)) / (1024.0 * 1024.0);

    auto tFill = clk::now();
    hits.resize(numRays);
    int hitCount = 0;
    size_t fillOff = 0;
    size_t rIdx    = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            size_t n = (size_t)lc.hnum * lc.vnum;
            for (size_t i = 0; i < n; ++i) {
                float dist = h_hitDist[rIdx];
                GrcaHit& g  = hits[fillOff + i];
                g.dx = h_rays[rIdx*6+3];
                g.dy = h_rays[rIdx*6+4];
                g.dz = h_rays[rIdx*6+5];
                g.dist = dist;
                g.hx = lc.w_pose.pos.x + g.dx * g.dist;
                g.hy = lc.w_pose.pos.y + g.dy * g.dist;
                g.hz = lc.w_pose.pos.z + g.dz * g.dist;
                if (dist < lc.range_max) ++hitCount;
                ++rIdx;
            }
            fillOff += n;
        }
    }
    t.ms_fillhits = ms(tFill, clk::now());
    t.hit_count   = hitCount;
}

void destroy_tinybvh_gpu(TinyBVHGPUState* s)
{
    if (!s) return;
    if (s->d_nodes)      cudaFree(s->d_nodes);
    if (s->d_sortedTris) cudaFree(s->d_sortedTris);
    if (s->d_rays)       cudaFree(s->d_rays);
    if (s->d_hitDist)    cudaFree(s->d_hitDist);
    delete s;
}

#endif // HAVE_CUDA
