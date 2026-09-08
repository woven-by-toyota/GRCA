#pragma once
#include "../include/grca_common.h"

// Opaque persistent CUDA state. Full definition lives in GRCA_CUDA.cu.
struct CudaState;

// One-time init: CUDA context, GPU buffer allocation, static data upload
// (lidar config + index buffer).  tris determines allocation size and must
// have the same triangle count for every subsequent run_cuda_frame call.
// staticTriCount: triangles in the static (non-dynamic) part of tris —
//   their vertex data is uploaded once at init and skipped each frame.
// t receives ms_cuda_init,  (static uploads), ms_dynamic_upload (dynamic mesh, if any).
CudaState* init_cuda(const RunConfig&          cfg,
                     const std::vector<Tri3>& tris,
                     size_t                   staticTriCount,
                     FrameTimings&              t);

// Per-frame: upload new vertex positions, run GRCA Early+Late passes,
// read back results.  t receives , ms_trace_gpu, ms_readback_cpu, ms_dynamic_upload.
void run_cuda_frame(CudaState*               state,
                    const RunConfig&          cfg,
                    const std::vector<Tri3>& tris,
                    std::vector<GrcaHit>&     hits,
                    FrameTimings&              t);

// Release all CUDA resources owned by state.
void destroy_cuda(CudaState* state);

// Legacy one-shot wrapper: init + single frame + destroy.
void run_cuda(const RunConfig&          cfg,
              const std::vector<Tri3>& tris,
              std::vector<GrcaHit>&     hits,
              FrameTimings&              t);
