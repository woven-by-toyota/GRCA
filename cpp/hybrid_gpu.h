#pragma once
#include "../include/grca_common.h"

// Opaque persistent hybrid-GPU state. Full definition lives in hybrid_gpu.cpp.
struct HybridGpuState;

// One-time init: builds an OptiX AS over the static scene geometry and
// initialises GRCA-CUDA for the dynamic geometry.
//
//   static_tris   → OptiX BVH / RT cores (built once, pure-refit mode)
//   dyn_init_tris → GRCA-CUDA buffer sizing (initial positions of dynamic tris)
//
// t receives ms_init_cpu (wall-clock) and ms_bvh_rebuild_gpu (OptiX AS build).
HybridGpuState* init_hybrid_gpu(const RunConfig&         cfg,
                                  const std::vector<Tri3>& static_tris,
                                  const std::vector<Tri3>& dyn_init_tris,
                                  FrameTimings&            t);

// Per-frame update:
//   Pass A — OptiX traces all rays against static geometry (RT cores).
//   Pass B — GRCA-CUDA traces all rays against dynamic geometry.
//   Merge  — per-ray, the closer hit wins.
//
// t receives combined timings from both sub-backends.
void run_hybrid_gpu_frame(HybridGpuState*          state,
                           const RunConfig&         cfg,
                           const std::vector<Tri3>& tris,
                           std::vector<GrcaHit>&     hits,
                           FrameTimings&            t);

// Release all OptiX, CUDA, and host resources owned by state.
void destroy_hybrid_gpu(HybridGpuState* state);
