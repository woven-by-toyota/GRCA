#pragma once
#include "../include/grca_common.h"

// Opaque persistent Embree state. Full definition lives in embree.cpp.
struct EmbreeState;

// One-time init: creates Embree device/scene, uploads mesh geometry, builds
// initial BVH.  staticTriCount triangles are uploaded once and never touched
// again; the remainder are the dynamic triangles updated per-frame.
// t receives ms_cuda_init (device init), ms_bvh_rebuild_gpu (initial BVH).
EmbreeState* init_embree(const RunConfig&          cfg,
                          const std::vector<Tri3>& tris,
                          size_t                   staticTriCount,
                          FrameTimings&              t);

// Per-frame: copy new dynamic vertex positions into Embree's buffer,
// recommit the scene (refit or rebuild depending on config), then cast all
// lidar rays single-threaded and fill hits.
// t receives ms_dynamic_upload, ms_bvh_rebuild_gpu/ms_bvh_refit_gpu, ms_trace_gpu, hit_count.
void run_embree_frame(EmbreeState*              state,
                      const RunConfig&          cfg,
                      const std::vector<Tri3>& tris,
                      std::vector<GrcaHit>&     hits,
                      FrameTimings&              t);

// Release all Embree resources owned by state.
void destroy_embree(EmbreeState* state);
