#pragma once
#include "../include/grca_common.h"

// Opaque persistent CPU state. Full definition lives in GRCA_CPU.cpp.
struct CpuState;

// One-time init: pre-computes per-lidar world-space ray direction grids.
// backendstr is used for log prefixes only.  t receives ms_ray_init, ms_init_cpu.
CpuState* init_cpu(const RunConfig&     cfg,
                   const std::string&   backendstr,
                   FrameTimings&        t);

// Per-frame GRCA: reset hit bins, process all triangles, build GrcaHit output.
// t receives ms_trace_cpu, hit_count.
void run_cpu_frame(CpuState*                state,
                   const RunConfig&          cfg,
                   const std::vector<Tri3>& tris,
                   std::vector<GrcaHit>&     hits,
                   FrameTimings&              t);

// Release all resources owned by state.
void destroy_cpu(CpuState* state);
