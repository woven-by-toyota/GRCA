#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

// Launch the tinybvh BVH traversal kernel.
//
// d_nodes      – GPU BVH nodes: 2 float4s per node (32 bytes), nodeCount nodes.
//                Layout matches tinybvh::BVH::BVHNode:
//                  float4[0]: aabbMin.x/y/z, leftFirst (reinterp. as uint32)
//                  float4[1]: aabbMax.x/y/z, triCount  (reinterp. as uint32)
//
// d_sortedTris – GPU triangles in BVH-leaf order: 3 float4s per tri (48 bytes).
//                Leaf node leftFirst indexes directly into this array.
//
// d_rays       – packed ray data: [ox,oy,oz, dx,dy,dz] * numRays (6 floats/ray)
//
// d_hitDist    – output: closest hit distance per ray, or range_max on miss.
//
// tmin/tmax    – ray extents (hits outside [tmin,tmax] are ignored).
//
// stream       – CUDA stream to launch on (0 = default).
void launch_tbvh_rays(const float4* d_nodes,
                      const float4* d_sortedTris,
                      const float*  d_rays,
                      float*        d_hitDist,
                      int           numRays,
                      float         tmin,
                      float         tmax,
                      cudaStream_t  stream = 0);
