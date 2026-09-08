#pragma once
#include "../include/grca_common.h"

// Opaque persistent state. Full definition lives in tinybvh_backends.cpp.
struct TinyBVHCPUState;

// One-time init: builds initial BVH from all triangles.
// staticTriCount triangles are never moved; the rest are dynamic.
TinyBVHCPUState* init_tinybvh_cpu(const RunConfig&          cfg,
                                   const std::vector<Tri3>& tris,
                                   size_t                   staticTriCount,
                                   FrameTimings&            t);

// Per-frame: update dynamic verts, refit or rebuild BVH, cast all lidar rays.
void run_tinybvh_cpu_frame(TinyBVHCPUState*          state,
                            const RunConfig&          cfg,
                            const std::vector<Tri3>& tris,
                            std::vector<GrcaHit>&     hits,
                            FrameTimings&            t);

// Release all resources.
void destroy_tinybvh_cpu(TinyBVHCPUState* state);
