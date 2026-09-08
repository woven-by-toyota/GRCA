#pragma once
#include "../include/grca_common.h"

// Opaque persistent hybrid-CPU state. Full definition lives in hybrid_cpu.cpp.
struct HybridCpuState;

// One-time init: builds an Embree BVH over the static scene geometry and
// initialises GRCA-CPU ray grids.  Only static_tris are passed; dynamic
// geometry is supplied per-frame via run_hybrid_cpu_frame.
//
// t receives ms_init_cpu (wall-clock of the entire init call).
HybridCpuState* init_hybrid_cpu(const RunConfig&         cfg,
                                 const std::vector<Tri3>& static_tris,
                                 FrameTimings&            t);

// Per-frame update:
//   Pass A — Embree traces all rays against static geometry.
//   Pass B — GRCA-CPU traces all rays against dynamic geometry (tris[staticTriCount..]).
//   Merge  — per-ray, the closer hit wins.
//
// t receives combined timings from both sub-backends.
void run_hybrid_cpu_frame(HybridCpuState*          state,
                          const RunConfig&         cfg,
                          const std::vector<Tri3>& tris,
                          std::vector<GrcaHit>&     hits,
                          FrameTimings&            t);

// Release all resources owned by state.
void destroy_hybrid_cpu(HybridCpuState* state);
