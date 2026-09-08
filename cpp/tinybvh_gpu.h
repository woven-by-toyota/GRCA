#pragma once
#include "../include/grca_common.h"

// Opaque persistent state. Full definition lives in tinybvh_backends.cpp.
struct TinyBVHGPUState;

// One-time init: builds BVH on CPU, uploads nodes + sorted triangles to GPU.
TinyBVHGPUState* init_tinybvh_gpu(const RunConfig&          cfg,
                                   const std::vector<Tri3>& tris,
                                   size_t                   staticTriCount,
                                   FrameTimings&            t);

// Per-frame: update dynamic verts, refit or rebuild BVH on CPU, re-upload to
// GPU, cast all lidar rays via CUDA kernel.
void run_tinybvh_gpu_frame(TinyBVHGPUState*          state,
                            const RunConfig&          cfg,
                            const std::vector<Tri3>& tris,
                            std::vector<GrcaHit>&     hits,
                            FrameTimings&            t);

// Release all CPU and GPU resources.
void destroy_tinybvh_gpu(TinyBVHGPUState* state);
