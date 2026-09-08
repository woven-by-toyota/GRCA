// =============================================================================
// hybrid_gpu.cpp  –  Hybrid GPU backend: OptiX (static) + GRCA-CUDA (dynamic)
//
// Strategy
// --------
//   Static geometry  → OptiX RT cores.  Built at init with pure-refit mode
//                       (optix_rebuild_after_frames = -1) regardless of the
//                       caller's config, so OptiX uses OPTIX_BUILD_FLAG_ALLOW_UPDATE
//                       and the per-frame refit is a near-zero-cost no-op (vertices
//                       never change).  This gives the best static AS quality
//                       while avoiding any per-frame rebuild overhead.
//   Dynamic geometry → GRCA-CUDA two-pass angular filter on GPU.  Only the
//                       dynamic triangle slice is forwarded to GRCA each frame.
//   Merge            → Per-ray take the closer of the two hit distances.
//                       Done on the CPU after both readbacks complete.
// =============================================================================

#include "hybrid_gpu.h"
#include "../optix_cuda/optix.h"
#include "../cuda/grca_cuda.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>

using clk = std::chrono::steady_clock;

static double elapsed_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct HybridGpuState {
    OptixState* optix         = nullptr;
    CudaState*  grca           = nullptr;
    std::vector<Tri3> static_tris;    // permanent copy – passed to OptiX every frame
    std::vector<Tri3> dyn_init_tris;  // dynamic tris at init – used to size GRCA-CUDA buffers
    std::vector<Tri3> dyn_slice;      // reused per-frame buffer for dynamic tris
    size_t            staticTriCount  = 0;
    RunConfig         cfg_static;     // OptiX config with rebuild_after_frames = -1 (pure refit)
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

HybridGpuState* init_hybrid_gpu(const RunConfig&         cfg,
                                  const std::vector<Tri3>& static_tris,
                                  const std::vector<Tri3>& dyn_init_tris,
                                  FrameTimings&            t)
{
    auto* s = new HybridGpuState;
    s->staticTriCount = static_tris.size();

    s->static_tris   = static_tris;
    s->dyn_init_tris = dyn_init_tris;
    s->dyn_slice.resize(dyn_init_tris.size());

    // Force pure-refit mode for OptiX regardless of the caller's setting.
    // -1 = never rebuild, only refit.  Since static_tris never change, the
    // per-frame refit is a near-zero-cost no-op while still giving OptiX
    // OPTIX_BUILD_FLAG_ALLOW_UPDATE for best RT-core traversal performance.
    s->cfg_static = cfg;
    s->cfg_static.optix_rebuild_after_frames = -1;

    auto tInit = clk::now();

    // ------------------------------------------------------------------
    // OptiX: build a refit-capable AS over the static-only geometry.
    // staticTriCount == static_tris.size() → zero dynamic triangles.
    // ------------------------------------------------------------------
    FrameTimings optixT = {};
    s->optix = init_optix(s->cfg_static, s->static_tris, s->static_tris.size(), optixT);
    if (!s->optix) {
        fprintf(stderr, "[hybrid-gpu] init_optix failed\n");
        delete s;
        return nullptr;
    }

    // ------------------------------------------------------------------
    // GRCA-CUDA: allocate GPU buffers sized for the dynamic tris only.
    // staticTriCount=0 means GRCA-CUDA treats all its triangles as dynamic
    // and re-uploads them every frame (correct for moving objects).
    // ------------------------------------------------------------------
    FrameTimings grcaT = {};
    const size_t dynCount = s->dyn_init_tris.size();
    if (dynCount > 0) {
        s->grca = init_cuda(cfg, s->dyn_init_tris, 0, grcaT);
        if (!s->grca) {
            fprintf(stderr, "[hybrid-gpu] init_cuda failed\n");
            destroy_optix(s->optix);
            delete s;
            return nullptr;
        }
    }

    t.ms_init_cpu        = elapsed_ms(tInit, clk::now());
    t.ms_bvh_rebuild_gpu = optixT.ms_bvh_rebuild_gpu;  // OptiX AS build time
    fprintf(stderr,
            "[hybrid-gpu] init: %zu static tris → OptiX RT | "
            "%zu dynamic tris → GRCA-CUDA\n",
            s->staticTriCount, dynCount);
    return s;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void run_hybrid_gpu_frame(HybridGpuState*          state,
                           const RunConfig&         cfg,
                           const std::vector<Tri3>& tris,
                           std::vector<GrcaHit>&     hits,
                           FrameTimings&            t)
{
    // ------------------------------------------------------------------
    // Pass A: static geometry via OptiX RT cores
    //
    // Use current cfg (current robot poses) but override rebuild setting
    // to pure-refit.  We must NOT use state->cfg_static here: it is a
    // frozen snapshot from init and would carry stale robot poses in any
    // frame where robots have moved.
    // ------------------------------------------------------------------
    RunConfig optix_cfg = cfg;
    optix_cfg.optix_rebuild_after_frames = -1;  // force pure refit for static AS

    std::vector<GrcaHit> hits_static;
    FrameTimings tA = {};
    run_optix_frame(state->optix, optix_cfg, state->static_tris, hits_static, tA);

    // ------------------------------------------------------------------
    // Pass B: dynamic geometry via GRCA-CUDA
    // ------------------------------------------------------------------
    const size_t sc = state->staticTriCount;
    const size_t dc = tris.size() - sc;

    std::vector<GrcaHit> hits_dynamic;
    FrameTimings tB = {};

    if (dc > 0 && state->grca) {
        // Refresh dynamic slice with current-frame world-space positions.
        if (dc != state->dyn_slice.size()) state->dyn_slice.resize(dc);
        std::copy(tris.begin() + static_cast<ptrdiff_t>(sc), tris.end(),
                  state->dyn_slice.begin());
        run_cuda_frame(state->grca, cfg, state->dyn_slice, hits_dynamic, tB);
    } else {
        // No dynamic geometry – produce sentinel misses so the merge below
        // always selects the OptiX result.
        hits_dynamic.resize(hits_static.size());
        for (auto& h : hits_dynamic) {
            h.dx = 0.f; h.dy = 0.f; h.dz = 0.f;
            h.dist = 1e30f;
            h.hx = 0.f; h.hy = 0.f; h.hz = 0.f;
        }
    }

    // ------------------------------------------------------------------
    // Merge: per-ray keep the closer hit
    // ------------------------------------------------------------------
    const size_t n = hits_static.size();
    assert(n == hits_dynamic.size());
    hits.resize(n);
    for (size_t i = 0; i < n; ++i)
        hits[i] = (hits_dynamic[i].dist < hits_static[i].dist)
                      ? hits_dynamic[i]
                      : hits_static[i];

    // ------------------------------------------------------------------
    // Accumulate timings
    // ------------------------------------------------------------------
    t.ms_trace_gpu       = tA.ms_trace_gpu    + tB.ms_trace_gpu;
    t.ms_bvh_rebuild_gpu = tA.ms_bvh_rebuild_gpu;  // OptiX per-frame refit (none by default)
    t.ms_bvh_refit_gpu   = tA.ms_bvh_refit_gpu;
    t.ms_dynamic_upload  = tA.ms_dynamic_upload  + tB.ms_dynamic_upload;
    t.ms_readback_cpu    = tA.ms_readback_cpu    + tB.ms_readback_cpu;
    t.ms_sync_readback   = tA.ms_sync_readback   + tB.ms_sync_readback;
    t.mb_readback        = tA.mb_readback        + tB.mb_readback;
    t.ms_fillhits        = tA.ms_fillhits        + tB.ms_fillhits;
    t.ms_lidar_sim_gpu   = tA.ms_lidar_sim_gpu   + tB.ms_lidar_sim_gpu;
    t.num_bat            = tB.num_bat;
    t.num_early_t        = tB.num_early_t;
    t.num_rtic           = tB.num_rtic;
    // Approximate: overcounts rays that hit both static and dynamic geometry.
    t.hit_count = tA.hit_count + tB.hit_count;
}

// ---------------------------------------------------------------------------
// Destroy
// ---------------------------------------------------------------------------

void destroy_hybrid_gpu(HybridGpuState* state) {
    if (!state) return;
    if (state->optix) destroy_optix(state->optix);
    if (state->grca)   destroy_cuda(state->grca);
    delete state;
}
