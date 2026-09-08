// =============================================================================
// embree.cpp
// Intel Embree 4 backend for GRCA – single-threaded BVH ray casting.
//
// Mirrors the optix.cu init/frame/destroy pattern so it can be swapped in
// via --backend embree.  Intended for head-to-head comparison with GRCA_CPU.
//
// Ray direction formula matches optix_device.cu exactly:
//   dir = normalize((cosH * fwd + sinH * right) * cosV + sinV * up)
//
// Config (config.json "embree" section):
//   "bvh_rebuild_after_frames": -1  — pure refit (never rebuild after init)
//   "bvh_rebuild_after_frames":  0  — pure rebuild every frame (default)
//   "bvh_rebuild_after_frames":  N  — refit N frames, rebuild on frame N+1 (period N+1)
//
// Build requirements:
//   - Embree built with EMBREE_BACKFACE_CULLING=ON and EMBREE_FILTER_FUNCTION=OFF
//     (see README). Backface culling is handled natively in Embree's traversal
//     kernels; no filter callback is needed or registered.
//   - CMakeLists.txt detects embree and sets HAVE_EMBREE + links the lib
// =============================================================================

#include "embree.h"

#include <embree4/rtcore.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <float.h>
#include <numeric>
#include <nlohmann/json.hpp>
#include <vector>

// ---------------------------------------------------------------------------
// Error callback
// ---------------------------------------------------------------------------
static void embreeError(void* /*userPtr*/, RTCError code, const char* str)
{
    fprintf(stderr, "[Embree] Error %d: %s\n", (int)code, str ? str : "");
}

// ---------------------------------------------------------------------------
// EmbreeState
// ---------------------------------------------------------------------------
struct EmbreeState
{
    RTCDevice   device      = nullptr;
    RTCScene    scene       = nullptr;

    // Dynamic geometry handle kept alive so we can update its vertex buffer
    // each frame.  Static geometry is released to the scene after attach.
    RTCGeometry dyn_geom    = nullptr;
    float*      dyn_verts   = nullptr;   // points into Embree-owned buffer

    size_t staticTriCount = 0;
    size_t dynTriCount    = 0;

    int   rebuildAfterFrames = 0; // -1=pure refit, 0=pure rebuild, N>0=hybrid (period N+1)
    bool  compaction         = false;
    int   frameCount         = 0;
    int  packet_size            = 8;    // rays per BVH call: 1, 4, 8, or 16
};

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// init_embree
// ---------------------------------------------------------------------------
EmbreeState* init_embree(const RunConfig&          cfg,
                          const std::vector<Tri3>& tris,
                          size_t                   staticTriCount,
                          FrameTimings&              t)
{
    auto tInit = std::chrono::steady_clock::now();

    EmbreeState* s   = new EmbreeState();
    s->staticTriCount = staticTriCount;
    s->dynTriCount    = tris.size() - staticTriCount;
    s->rebuildAfterFrames = cfg.embree_rebuild_after_frames;
    s->compaction       = cfg.embree_compaction;
    s->packet_size      = cfg.embree_packet_size;
    // ---- Embree device + scene --------------------------------------------
    // Force fully single-threaded operation:
    //   threads=1      — limits Embree's internal task pool (BVH Rebuild, refit,
    //                    and ALL other internal parallel work) to 1 thread.
    //   start_threads=0 — don't pre-spawn even that 1 worker; work runs inline.
    // Ray traversal (rtcIntersect1) always executes on the calling thread and
    // never touches the task pool, so the serial loop below is already
    // single-threaded by construction.
    s->device = rtcNewDevice("threads=1,start_threads=0");
    if (!s->device) {
        fprintf(stderr, "[Embree] rtcNewDevice failed\n");
        delete s; return nullptr;
    }
    rtcSetDeviceErrorFunction(s->device, embreeError, nullptr);

    s->scene = rtcNewScene(s->device);
    {
        RTCSceneFlags flags = s->compaction ? RTC_SCENE_FLAG_COMPACT : RTC_SCENE_FLAG_NONE;
        if (s->dynTriCount > 0) {
            // Dynamic scenes need this flag for efficient repeated recommits.
            flags = (RTCSceneFlags)(flags | RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
            rtcSetSceneBuildQuality(s->scene, RTC_BUILD_QUALITY_LOW);
        }
        if (flags != RTC_SCENE_FLAG_NONE) rtcSetSceneFlags(s->scene, flags);
    }

    // ---- Static geometry (committed once, released to scene) --------------
    if (s->staticTriCount > 0) {
        RTCGeometry sg = rtcNewGeometry(s->device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetGeometryBuildQuality(sg, RTC_BUILD_QUALITY_HIGH);

        float* sv = (float*)rtcSetNewGeometryBuffer(
            sg, RTC_BUFFER_TYPE_VERTEX, 0,
            RTC_FORMAT_FLOAT3, 3 * sizeof(float), s->staticTriCount * 3);
        // Tri3 layout (float v[3][3]) is identical to Embree's packed float3
        // vertex buffer, so a single memcpy suffices.
        memcpy(sv, tris.data(), s->staticTriCount * sizeof(Tri3));

        unsigned int* si = (unsigned int*)rtcSetNewGeometryBuffer(
            sg, RTC_BUFFER_TYPE_INDEX, 0,
            RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), s->staticTriCount);
        std::iota(si, si + s->staticTriCount * 3, 0u);

        rtcCommitGeometry(sg);
        rtcAttachGeometry(s->scene, sg);
        rtcReleaseGeometry(sg);   // scene holds its own ref
    }

    // ---- Dynamic geometry (kept alive; vertex buffer updated per-frame) ---
    if (s->dynTriCount > 0) {
        s->dyn_geom = rtcNewGeometry(s->device, RTC_GEOMETRY_TYPE_TRIANGLE);
        // Pure refit: HIGH — init once, only cheap refits thereafter (mirrors OptiX PREFER_FAST_TRACE).
        // Hybrid / pure rebuild: LOW — must match the quality used on every periodic/per-frame
        // rebuild so the geometry quality is consistent throughout.
        rtcSetGeometryBuildQuality(s->dyn_geom,
            s->rebuildAfterFrames == -1 ? RTC_BUILD_QUALITY_HIGH : RTC_BUILD_QUALITY_LOW);

        s->dyn_verts = (float*)rtcSetNewGeometryBuffer(
            s->dyn_geom, RTC_BUFFER_TYPE_VERTEX, 0,
            RTC_FORMAT_FLOAT3, 3 * sizeof(float), s->dynTriCount * 3);
        memcpy(s->dyn_verts, tris.data() + staticTriCount,
               s->dynTriCount * sizeof(Tri3));

        unsigned int* di = (unsigned int*)rtcSetNewGeometryBuffer(
            s->dyn_geom, RTC_BUFFER_TYPE_INDEX, 0,
            RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), s->dynTriCount);
        std::iota(di, di + s->dynTriCount * 3, 0u);

        rtcCommitGeometry(s->dyn_geom);
        rtcAttachGeometry(s->scene, s->dyn_geom);
        // Do NOT release dyn_geom here — we need the handle for per-frame updates.
    }

    // ---- Initial BVH Rebuild ------------------------------------------------
    auto tBvh = std::chrono::steady_clock::now();
    rtcCommitScene(s->scene);
    t.ms_bvh_rebuild_cpu = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tBvh).count();

    size_t total_rays = 0;
    for (const auto& robot : cfg.robots)
        for (const auto& lc : robot.lidars)
            total_rays += (size_t)lc.hnum * lc.vnum;

    t.ms_init_cpu = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tInit).count();
    fprintf(stderr, "[Embree] init: %zu tris (%zus+%zud), %zu rays, rebuild_after=%d, compact=%d, pkt=%d  (%.2f ms)\n",
            tris.size(), s->staticTriCount, s->dynTriCount,
            total_rays, s->rebuildAfterFrames, (int)s->compaction, s->packet_size, t.ms_init_cpu);
    return s;
}

// ---------------------------------------------------------------------------
// run_embree_frame
// ---------------------------------------------------------------------------
void run_embree_frame(EmbreeState*              s,
                      const RunConfig&          cfg,
                      const std::vector<Tri3>& tris,
                      std::vector<GrcaHit>&     hits,
                      FrameTimings&              t)
{
    // ---- Dynamic data upload -----------------------------------------------
    auto tUp = std::chrono::steady_clock::now();

    if (s->dynTriCount > 0 && s->dyn_verts) {
        // Tri3 is float v[3][3] — identical layout to Embree's packed float3 buffer.
        memcpy(s->dyn_verts,
               tris.data() + s->staticTriCount,
               s->dynTriCount * sizeof(Tri3));
        rtcUpdateGeometryBuffer(s->dyn_geom, RTC_BUFFER_TYPE_VERTEX, 0);
        // Do NOT commit here — quality must be set first in the BVH section below.
    }
    t.ms_dynamic_upload = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tUp).count();

    // ---- BVH refit / rebuild ----------------------------------------------
    if (s->dynTriCount > 0) {
        // hybrid: rebuild every (rebuildAfterFrames+1) frames; refit all others
        bool periodicRebuild = (s->rebuildAfterFrames > 0)
            && (s->frameCount % (s->rebuildAfterFrames + 1) == 0);
        s->frameCount++;

        // -1 = pure refit; 0 = pure rebuild; N>0 = refit unless periodicRebuild
        bool doRefit = (s->rebuildAfterFrames == -1)
            || (s->rebuildAfterFrames > 0 && !periodicRebuild);
        rtcSetGeometryBuildQuality(s->dyn_geom,
            doRefit ? RTC_BUILD_QUALITY_REFIT : RTC_BUILD_QUALITY_LOW);
        rtcCommitGeometry(s->dyn_geom);

        auto tBvh = std::chrono::steady_clock::now();
        rtcCommitScene(s->scene);
        double msBvh = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tBvh).count();
        if (doRefit) t.ms_bvh_refit_cpu = msBvh;
        else         t.ms_bvh_rebuild_cpu = msBvh;
    }

    // ---- Ray cast (packet_size configurable: 1 / 4 / 8 / 16) ---------------
    auto tTrace = std::chrono::steady_clock::now();

    size_t total_rays = 0;
    for (const auto& robot : cfg.robots)
        for (const auto& lc : robot.lidars)
            total_rays += (size_t)lc.hnum * lc.vnum;
    hits.resize(total_rays);

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    // Nested vi/hi loops so cosV/sinV are computed once per vertical channel,
    // not once per ray.  The switch is outside the vi loop so there is no
    // per-ray branch.  The direction formula (cosH*fwd + sinH*right)*cosV +
    // sinV*up produces a unit vector for any orthonormal {fwd,right,up} basis,
    // so no normalise step is needed.

    size_t offset = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            const unsigned int hnum = (unsigned int)lc.hnum;
            const unsigned int vnum = (unsigned int)lc.vnum;
            const float hstep = (hnum > 1u) ? (lc.hmax - lc.hmin) / (float)(hnum - 1u) : 0.f;
            const float vstep = (vnum > 1u) ? (lc.vmax - lc.vmin) / (float)(vnum - 1u) : 0.f;
            const float ox = lc.w_pose.pos.x, oy = lc.w_pose.pos.y, oz = lc.w_pose.pos.z;

            // Precompute trig tables: O(H+V) calls instead of O(H*V).
            std::vector<float> cosH(hnum), sinH(hnum);
            for (unsigned int h = 0; h < hnum; ++h) {
                cosH[h] = cosf(lc.hmin + (float)h * hstep);
                sinH[h] = sinf(lc.hmin + (float)h * hstep);
            }

            switch (s->packet_size) {

            case 4: {
                static const int kV4[4] = {-1,-1,-1,-1};
                for (unsigned int vi = 0; vi < vnum; ++vi) {
                    const float cosV = cosf(lc.vmin + (float)vi * vstep);
                    const float sinV = sinf(lc.vmin + (float)vi * vstep);
                    const float ux = sinV*lc.w_pose.up.x, uy = sinV*lc.w_pose.up.y, uz = sinV*lc.w_pose.up.z;
                    unsigned int hi = 0;
                    for (; hi + 4 <= hnum; hi += 4) {
                        alignas(16) RTCRayHit4 rh;
                        for (int j = 0; j < 4; ++j) {
                            rh.ray.org_x[j] = ox; rh.ray.org_y[j] = oy; rh.ray.org_z[j] = oz;
                            rh.ray.dir_x[j] = (cosH[hi+j]*lc.w_pose.forward.x + sinH[hi+j]*lc.w_pose.right.x)*cosV + ux;
                            rh.ray.dir_y[j] = (cosH[hi+j]*lc.w_pose.forward.y + sinH[hi+j]*lc.w_pose.right.y)*cosV + uy;
                            rh.ray.dir_z[j] = (cosH[hi+j]*lc.w_pose.forward.z + sinH[hi+j]*lc.w_pose.right.z)*cosV + uz;
                            rh.ray.tnear[j] = lc.range_min; rh.ray.tfar[j] = lc.range_max;
                            rh.ray.time[j] = 0.f; rh.ray.mask[j] = 0xFFFFFFFF;
                            rh.ray.id[j] = 0; rh.ray.flags[j] = 0;
                            rh.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
                        }
                        rtcIntersect4(kV4, s->scene, &rh, &args);
                        const size_t base = offset + (size_t)vi * hnum + hi;
                        for (int j = 0; j < 4; ++j) {
                            GrcaHit& g = hits[base + j];
                            g.dx = rh.ray.dir_x[j]; g.dy = rh.ray.dir_y[j]; g.dz = rh.ray.dir_z[j];
                            g.dist = (rh.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar[j] : lc.range_max;
                        }
                    }
                    for (; hi < hnum; ++hi) {
                        RTCRayHit rh;
                        rh.ray.org_x = ox; rh.ray.org_y = oy; rh.ray.org_z = oz;
                        rh.ray.dir_x = (cosH[hi]*lc.w_pose.forward.x + sinH[hi]*lc.w_pose.right.x)*cosV + ux;
                        rh.ray.dir_y = (cosH[hi]*lc.w_pose.forward.y + sinH[hi]*lc.w_pose.right.y)*cosV + uy;
                        rh.ray.dir_z = (cosH[hi]*lc.w_pose.forward.z + sinH[hi]*lc.w_pose.right.z)*cosV + uz;
                        rh.ray.tnear = lc.range_min; rh.ray.tfar = lc.range_max;
                        rh.ray.time = 0.f; rh.ray.mask = 0xFFFFFFFF; rh.ray.id = 0; rh.ray.flags = 0;
                        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                        rtcIntersect1(s->scene, &rh, &args);
                        GrcaHit& g = hits[offset + (size_t)vi * hnum + hi];
                        g.dx = rh.ray.dir_x; g.dy = rh.ray.dir_y; g.dz = rh.ray.dir_z;
                        g.dist = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar : lc.range_max;
                    }
                }
                break;
            }

            case 8: {
                static const int kV8[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
                for (unsigned int vi = 0; vi < vnum; ++vi) {
                    const float cosV = cosf(lc.vmin + (float)vi * vstep);
                    const float sinV = sinf(lc.vmin + (float)vi * vstep);
                    const float ux = sinV*lc.w_pose.up.x, uy = sinV*lc.w_pose.up.y, uz = sinV*lc.w_pose.up.z;
                    unsigned int hi = 0;
                    for (; hi + 8 <= hnum; hi += 8) {
                        alignas(32) RTCRayHit8 rh;
                        for (int j = 0; j < 8; ++j) {
                            rh.ray.org_x[j] = ox; rh.ray.org_y[j] = oy; rh.ray.org_z[j] = oz;
                            rh.ray.dir_x[j] = (cosH[hi+j]*lc.w_pose.forward.x + sinH[hi+j]*lc.w_pose.right.x)*cosV + ux;
                            rh.ray.dir_y[j] = (cosH[hi+j]*lc.w_pose.forward.y + sinH[hi+j]*lc.w_pose.right.y)*cosV + uy;
                            rh.ray.dir_z[j] = (cosH[hi+j]*lc.w_pose.forward.z + sinH[hi+j]*lc.w_pose.right.z)*cosV + uz;
                            rh.ray.tnear[j] = lc.range_min; rh.ray.tfar[j] = lc.range_max;
                            rh.ray.time[j] = 0.f; rh.ray.mask[j] = 0xFFFFFFFF;
                            rh.ray.id[j] = 0; rh.ray.flags[j] = 0;
                            rh.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
                        }
                        rtcIntersect8(kV8, s->scene, &rh, &args);
                        const size_t base = offset + (size_t)vi * hnum + hi;
                        for (int j = 0; j < 8; ++j) {
                            GrcaHit& g = hits[base + j];
                            g.dx = rh.ray.dir_x[j]; g.dy = rh.ray.dir_y[j]; g.dz = rh.ray.dir_z[j];
                            g.dist = (rh.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar[j] : lc.range_max;
                        }
                    }
                    for (; hi < hnum; ++hi) {
                        RTCRayHit rh;
                        rh.ray.org_x = ox; rh.ray.org_y = oy; rh.ray.org_z = oz;
                        rh.ray.dir_x = (cosH[hi]*lc.w_pose.forward.x + sinH[hi]*lc.w_pose.right.x)*cosV + ux;
                        rh.ray.dir_y = (cosH[hi]*lc.w_pose.forward.y + sinH[hi]*lc.w_pose.right.y)*cosV + uy;
                        rh.ray.dir_z = (cosH[hi]*lc.w_pose.forward.z + sinH[hi]*lc.w_pose.right.z)*cosV + uz;
                        rh.ray.tnear = lc.range_min; rh.ray.tfar = lc.range_max;
                        rh.ray.time = 0.f; rh.ray.mask = 0xFFFFFFFF; rh.ray.id = 0; rh.ray.flags = 0;
                        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                        rtcIntersect1(s->scene, &rh, &args);
                        GrcaHit& g = hits[offset + (size_t)vi * hnum + hi];
                        g.dx = rh.ray.dir_x; g.dy = rh.ray.dir_y; g.dz = rh.ray.dir_z;
                        g.dist = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar : lc.range_max;
                    }
                }
                break;
            }

            case 16: {
                static const int kV16[16] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                for (unsigned int vi = 0; vi < vnum; ++vi) {
                    const float cosV = cosf(lc.vmin + (float)vi * vstep);
                    const float sinV = sinf(lc.vmin + (float)vi * vstep);
                    const float ux = sinV*lc.w_pose.up.x, uy = sinV*lc.w_pose.up.y, uz = sinV*lc.w_pose.up.z;
                    unsigned int hi = 0;
                    for (; hi + 16 <= hnum; hi += 16) {
                        alignas(64) RTCRayHit16 rh;
                        for (int j = 0; j < 16; ++j) {
                            rh.ray.org_x[j] = ox; rh.ray.org_y[j] = oy; rh.ray.org_z[j] = oz;
                            rh.ray.dir_x[j] = (cosH[hi+j]*lc.w_pose.forward.x + sinH[hi+j]*lc.w_pose.right.x)*cosV + ux;
                            rh.ray.dir_y[j] = (cosH[hi+j]*lc.w_pose.forward.y + sinH[hi+j]*lc.w_pose.right.y)*cosV + uy;
                            rh.ray.dir_z[j] = (cosH[hi+j]*lc.w_pose.forward.z + sinH[hi+j]*lc.w_pose.right.z)*cosV + uz;
                            rh.ray.tnear[j] = lc.range_min; rh.ray.tfar[j] = lc.range_max;
                            rh.ray.time[j] = 0.f; rh.ray.mask[j] = 0xFFFFFFFF;
                            rh.ray.id[j] = 0; rh.ray.flags[j] = 0;
                            rh.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
                            rh.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
                        }
                        rtcIntersect16(kV16, s->scene, &rh, &args);
                        const size_t base = offset + (size_t)vi * hnum + hi;
                        for (int j = 0; j < 16; ++j) {
                            GrcaHit& g = hits[base + j];
                            g.dx = rh.ray.dir_x[j]; g.dy = rh.ray.dir_y[j]; g.dz = rh.ray.dir_z[j];
                            g.dist = (rh.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar[j] : lc.range_max;
                        }
                    }
                    for (; hi < hnum; ++hi) {
                        RTCRayHit rh;
                        rh.ray.org_x = ox; rh.ray.org_y = oy; rh.ray.org_z = oz;
                        rh.ray.dir_x = (cosH[hi]*lc.w_pose.forward.x + sinH[hi]*lc.w_pose.right.x)*cosV + ux;
                        rh.ray.dir_y = (cosH[hi]*lc.w_pose.forward.y + sinH[hi]*lc.w_pose.right.y)*cosV + uy;
                        rh.ray.dir_z = (cosH[hi]*lc.w_pose.forward.z + sinH[hi]*lc.w_pose.right.z)*cosV + uz;
                        rh.ray.tnear = lc.range_min; rh.ray.tfar = lc.range_max;
                        rh.ray.time = 0.f; rh.ray.mask = 0xFFFFFFFF; rh.ray.id = 0; rh.ray.flags = 0;
                        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                        rtcIntersect1(s->scene, &rh, &args);
                        GrcaHit& g = hits[offset + (size_t)vi * hnum + hi];
                        g.dx = rh.ray.dir_x; g.dy = rh.ray.dir_y; g.dz = rh.ray.dir_z;
                        g.dist = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar : lc.range_max;
                    }
                }
                break;
            }

            default: {  // packet_size == 1
                for (unsigned int vi = 0; vi < vnum; ++vi) {
                    const float cosV = cosf(lc.vmin + (float)vi * vstep);
                    const float sinV = sinf(lc.vmin + (float)vi * vstep);
                    const float ux = sinV*lc.w_pose.up.x, uy = sinV*lc.w_pose.up.y, uz = sinV*lc.w_pose.up.z;
                    for (unsigned int hi = 0; hi < hnum; ++hi) {
                        RTCRayHit rh;
                        rh.ray.org_x = ox; rh.ray.org_y = oy; rh.ray.org_z = oz;
                        rh.ray.dir_x = (cosH[hi]*lc.w_pose.forward.x + sinH[hi]*lc.w_pose.right.x)*cosV + ux;
                        rh.ray.dir_y = (cosH[hi]*lc.w_pose.forward.y + sinH[hi]*lc.w_pose.right.y)*cosV + uy;
                        rh.ray.dir_z = (cosH[hi]*lc.w_pose.forward.z + sinH[hi]*lc.w_pose.right.z)*cosV + uz;
                        rh.ray.tnear = lc.range_min; rh.ray.tfar = lc.range_max;
                        rh.ray.time = 0.f; rh.ray.mask = 0xFFFFFFFF; rh.ray.id = 0; rh.ray.flags = 0;
                        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
                        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                        rtcIntersect1(s->scene, &rh, &args);
                        GrcaHit& g = hits[offset + (size_t)vi * hnum + hi];
                        g.dx = rh.ray.dir_x; g.dy = rh.ray.dir_y; g.dz = rh.ray.dir_z;
                        g.dist = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? rh.ray.tfar : lc.range_max;
                    }
                }
                break;
            }

            }  // switch packet_size

            offset += (size_t)hnum * vnum;
        }
    }

    t.ms_trace_cpu = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tTrace).count();

    // ---- Phase 2: fill hit positions from stored dist + direction ----------
    int hitCount = 0;
    auto tFill = std::chrono::steady_clock::now();

    size_t fillOffset = 0;
    for (const auto& robot : cfg.robots) {
        for (const auto& lc : robot.lidars) {
            const size_t num_rays = (size_t)lc.hnum * lc.vnum;
            for (size_t i = fillOffset; i < fillOffset + num_rays; ++i) {
                GrcaHit& ghit = hits[i];
                ghit.hx = lc.w_pose.pos.x + ghit.dx * ghit.dist;
                ghit.hy = lc.w_pose.pos.y + ghit.dy * ghit.dist;
                ghit.hz = lc.w_pose.pos.z + ghit.dz * ghit.dist;
                if (ghit.dist < lc.range_max) ++hitCount;
            }
            fillOffset += num_rays;
        }
    }

    t.ms_fillhits = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tFill).count();
    t.hit_count = hitCount;
}

// ---------------------------------------------------------------------------
// destroy_embree
// ---------------------------------------------------------------------------
void destroy_embree(EmbreeState* s)
{
    if (!s) return;
    if (s->dyn_geom) rtcReleaseGeometry(s->dyn_geom);  // drop our extra ref
    if (s->scene)    rtcReleaseScene(s->scene);
    if (s->device)   rtcReleaseDevice(s->device);
    delete s;
}
