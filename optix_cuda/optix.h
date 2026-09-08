#pragma once
#include "../include/grca_common.h"

// Opaque persistent OptiX state. Full definition lives in optix.cu.
struct OptixState;

// One-time init: creates CUDA/OptiX context, compiles pipeline, uploads ray
// configs, and builds the initial BVH.  t receives all init-time timings.
// staticTriCount: number of triangles in the static (non-dynamic) part of tris;
//   these verts are uploaded once and never re-uploaded.
// Returns heap-allocated state; caller must eventually call destroy_optix.
OptixState* init_optix(const RunConfig&          cfg,
                       const std::vector<Tri3>&  tris,
                       size_t                    staticTriCount,
                       FrameTimings&               t);

// Per-frame update: uploads new vertex positions to the GPU, refits the BVH
// (if OPTIX_BUILD_FLAG_ALLOW_UPDATE was set), launches ray tracing, and
// reads results back.  t receives per-frame timings only.
void run_optix_frame(OptixState*              state,
                     const RunConfig&          cfg,
                     const std::vector<Tri3>& tris,
                     std::vector<GrcaHit>&     hits,
                     FrameTimings&              t);

// Release all OptiX / CUDA resources owned by state.
void destroy_optix(OptixState* state);

// Legacy one-shot wrapper: init + single frame + destroy.
// Used by non-loop callers (e.g. CPU/CUDA regression tests).
void run_optix(const RunConfig&          cfg,
               const std::vector<Tri3>& tris,
               std::vector<GrcaHit>&     hits,
               FrameTimings&              t);
