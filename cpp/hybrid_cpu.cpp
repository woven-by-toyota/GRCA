// =============================================================================
// hybrid_cpu.cpp  –  Hybrid CPU backend: Embree (static) + GRCA-CPU (dynamic)
//
// Strategy
// --------
//   Static geometry  → Embree BVH.  Built at init with pure-refit mode
//                       (embree_rebuild_after_frames = -1) regardless of the
//                       caller's config, so Embree uses ALLOW_UPDATE flags and
//                       the per-frame refit is a near-zero-cost no-op (vertices
//                       never change).  This gives the best static BVH quality
//                       while avoiding any per-frame rebuild overhead.
//   Dynamic geometry → GRCA-CPU two-pass angular filter.  Only the dynamic
//                       triangle slice is forwarded to GRCA each frame.
//   Merge            → Per-ray take the closer of the two hit distances.
// =============================================================================

#include "hybrid_cpu.h"
#include "embree.h"
#include "cpu.h"

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

struct HybridCpuState {
    EmbreeState* embree        = nullptr;
    CpuState*    grca           = nullptr;
    std::vector<Tri3> static_tris;  // permanent copy – passed to Embree every frame
    std::vector<Tri3> dyn_slice;    // reused per-frame buffer for dynamic tris
    size_t            staticTriCount = 0;
    RunConfig         cfg_static;   // Embree config with rebuild_after_frames = -1 (pure refit)
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

HybridCpuState* init_hybrid_cpu(const RunConfig&         cfg,
                                  const std::vector<Tri3>& static_tris,
                                  FrameTimings&            t)
{
    auto* s = new HybridCpuState;
    s->staticTriCount = static_tris.size();
    s->static_tris    = static_tris;
    // dyn_slice is empty at init; run_hybrid_cpu_frame resizes it on first use.

    // Force pure-refit mode for Embree regardless of the caller's setting.
    // -1 = never rebuild, only refit.  Since static_tris never change, the
    // per-frame refit is a near-zero-cost no-op while still giving Embree
    // the ALLOW_UPDATE build flag for best ray-traversal performance.
    s->cfg_static = cfg;
    s->cfg_static.embree_rebuild_after_frames = -1;

    auto tInit = clk::now();

    // ------------------------------------------------------------------
    // Embree: build a refit-capable BVH over the static-only geometry.
    // staticTriCount == static_tris.size() → zero dynamic triangles.
    // ------------------------------------------------------------------
    FrameTimings embreeT = {};
    s->embree = init_embree(s->cfg_static, s->static_tris, s->static_tris.size(), embreeT);
    if (!s->embree) {
        fprintf(stderr, "[hybrid-cpu] init_embree failed\n");
        delete s;
        return nullptr;
    }

    // ------------------------------------------------------------------
    // GRCA-CPU: pre-compute per-lidar world-space ray-direction grids.
    // Triangles are not needed at init – they are passed per-frame.
    // ------------------------------------------------------------------
    FrameTimings grcaT = {};
    s->grca = init_cpu(cfg, "hybrid-grca-cpu", grcaT);
    if (!s->grca) {
        fprintf(stderr, "[hybrid-cpu] init_cpu failed\n");
        destroy_embree(s->embree);
        delete s;
        return nullptr;
    }

    t.ms_init_cpu = elapsed_ms(tInit, clk::now());
    fprintf(stderr,
            "[hybrid-cpu] init: %zu static tris → Embree BVH | "
            "dynamic tris → GRCA-CPU (per-frame)\n",
            s->staticTriCount);
    return s;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void run_hybrid_cpu_frame(HybridCpuState*          state,
                           const RunConfig&         cfg,
                           const std::vector<Tri3>& tris,
                           std::vector<GrcaHit>&     hits,
                           FrameTimings&            t)
{
    // ------------------------------------------------------------------
    // Pass A: static geometry via Embree BVH
    //
    // Use current cfg (current robot poses) but override rebuild setting
    // to pure-refit.  We must NOT use state->cfg_static here: it is a
    // frozen snapshot from init and would carry stale robot poses in any
    // frame where robots have moved.
    // ------------------------------------------------------------------
    RunConfig embree_cfg = cfg;
    embree_cfg.embree_rebuild_after_frames = -1;  // force pure refit for static BVH

    std::vector<GrcaHit> hits_static;
    FrameTimings tA = {};
    run_embree_frame(state->embree, embree_cfg, state->static_tris, hits_static, tA);

    // ------------------------------------------------------------------
    // Pass B: dynamic geometry via GRCA-CPU
    // ------------------------------------------------------------------
    const size_t sc = state->staticTriCount;
    const size_t dc = tris.size() - sc;

    std::vector<GrcaHit> hits_dynamic;
    FrameTimings tB = {};

    if (dc > 0) {
        // Refresh dynamic slice with current-frame world-space positions.
        if (dc != state->dyn_slice.size()) state->dyn_slice.resize(dc);
        std::copy(tris.begin() + static_cast<ptrdiff_t>(sc), tris.end(),
                  state->dyn_slice.begin());
        run_cpu_frame(state->grca, cfg, state->dyn_slice, hits_dynamic, tB);
    } else {
        // No dynamic geometry – produce sentinel misses so the merge below
        // always selects the Embree result.
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
    //   Embree reports its trace time in ms_trace_gpu (the "GPU-side"
    //   slot – Embree is CPU-side but uses that field for BVH tracing).
    //   GRCA-CPU reports in ms_trace_cpu.
    // ------------------------------------------------------------------
    t.ms_trace_cpu       = tB.ms_trace_cpu;
    t.ms_trace_gpu       = tA.ms_trace_gpu;       // Embree trace (cpu-side)
    t.ms_bvh_rebuild_cpu = tA.ms_bvh_rebuild_cpu + tB.ms_bvh_rebuild_cpu;
    t.ms_bvh_refit_cpu   = tA.ms_bvh_refit_cpu   + tB.ms_bvh_refit_cpu;
    t.ms_dynamic_upload  = tA.ms_dynamic_upload   + tB.ms_dynamic_upload;
    t.ms_lidar_sim_gpu   = tA.ms_lidar_sim_gpu;   // Embree wall-clock
    t.num_bat            = tB.num_bat;
    t.num_early_t        = tB.num_early_t;
    t.num_rtic           = tB.num_rtic;
    // Approximate: a ray that hits both static and dynamic geometry is counted
    // twice here; the merge picks the closer hit, so true hit_count ≤ this sum.
    t.hit_count = tA.hit_count + tB.hit_count;
}

// ---------------------------------------------------------------------------
// Destroy
// ---------------------------------------------------------------------------

void destroy_hybrid_cpu(HybridCpuState* state) {
    if (!state) return;
    if (state->embree) destroy_embree(state->embree);
    if (state->grca)    destroy_cpu(state->grca);
    delete state;
}
