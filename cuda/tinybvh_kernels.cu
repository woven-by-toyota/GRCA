// =============================================================================
// tinybvh_kernels.cu
// CUDA BVH traversal kernel for the tinybvh-gpu backend.
//
// Node layout (32 bytes, matches tinybvh::BVH::BVHNode):
//   float4[0]: aabbMin.x, aabbMin.y, aabbMin.z, leftFirst  (uint32 bit-cast)
//   float4[1]: aabbMax.x, aabbMax.y, aabbMax.z, triCount   (uint32 bit-cast)
//
// Triangle layout (48 bytes, 3 float4s per tri in primIdx-sorted order):
//   float4[0]: v0.x, v0.y, v0.z, 0
//   float4[1]: v1.x, v1.y, v1.z, 0
//   float4[2]: v2.x, v2.y, v2.z, 0
//
// Ray layout: packed float array, 6 floats per ray [ox, oy, oz, dx, dy, dz].
//
// Output: one float per ray — closest hit distance in [tmin, tmax),
//         or tmax on miss.
// =============================================================================

#include "tinybvh_kernels.h"
#include <float.h>

// ---------------------------------------------------------------------------
// Device: slab test for one AABB.
// Returns the entry t, or FLT_MAX if the ray misses the box or exits behind
// the current best hit.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float aabb_intersect(
    float ox, float oy, float oz,
    float rdx, float rdy, float rdz,
    float aMinX, float aMinY, float aMinZ,
    float aMaxX, float aMaxY, float aMaxZ,
    float tMax)
{
    float tx1 = (aMinX - ox) * rdx, tx2 = (aMaxX - ox) * rdx;
    float ty1 = (aMinY - oy) * rdy, ty2 = (aMaxY - oy) * rdy;
    float tz1 = (aMinZ - oz) * rdz, tz2 = (aMaxZ - oz) * rdz;
    float tEnter = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
    float tLeave = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));
    if (tLeave < tEnter || tLeave < 0.f || tEnter >= tMax) return FLT_MAX;
    return tEnter;
}

// ---------------------------------------------------------------------------
// Device: Möller–Trumbore ray-triangle intersection (two-sided).
// Returns t > 0 on hit, or FLT_MAX on miss.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float tri_intersect(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z)
{
    const float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    const float hx = dy*e2z - dz*e2y;
    const float hy = dz*e2x - dx*e2z;
    const float hz = dx*e2y - dy*e2x;
    const float a  = e1x*hx + e1y*hy + e1z*hz;
    if (fabsf(a) < 1e-8f) return FLT_MAX;
    const float f  = 1.f / a;
    const float sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
    const float u  = f * (sx*hx + sy*hy + sz*hz);
    if (u < 0.f || u > 1.f) return FLT_MAX;
    const float qx = sy*e1z - sz*e1y;
    const float qy = sz*e1x - sx*e1z;
    const float qz = sx*e1y - sy*e1x;
    const float v  = f * (dx*qx + dy*qy + dz*qz);
    if (v < 0.f || u + v > 1.f) return FLT_MAX;
    return f * (e2x*qx + e2y*qy + e2z*qz);
}

// ---------------------------------------------------------------------------
// Kernel: one thread per ray, iterative BVH traversal with local stack.
// ---------------------------------------------------------------------------
__global__ void tbvh_trace_kernel(
    const float4* __restrict__ nodes,
    const float4* __restrict__ sortedTris,
    const float*  __restrict__ rays,
    float*                     hitDist,
    int   numRays,
    float tmin,
    float tmax)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    const float ox = rays[idx*6+0], oy = rays[idx*6+1], oz = rays[idx*6+2];
    const float dx = rays[idx*6+3], dy = rays[idx*6+4], dz = rays[idx*6+5];
    const float rdx = (fabsf(dx) > 1e-30f) ? 1.f/dx : 1e30f;
    const float rdy = (fabsf(dy) > 1e-30f) ? 1.f/dy : 1e30f;
    const float rdz = (fabsf(dz) > 1e-30f) ? 1.f/dz : 1e30f;

    float best_t = tmax;

    // Traversal stack — 32 entries covers all practical scene depths.
    uint32_t stack[32];
    int sptr = 0;
    stack[sptr++] = 0u; // root

    while (sptr > 0) {
        const uint32_t nodeIdx = stack[--sptr];
        const float4   n0      = nodes[nodeIdx * 2 + 0];
        const float4   n1      = nodes[nodeIdx * 2 + 1];

        // AABB test against the full node bounds.
        float tEntry = aabb_intersect(ox, oy, oz, rdx, rdy, rdz,
                                      n0.x, n0.y, n0.z,
                                      n1.x, n1.y, n1.z, best_t);
        if (tEntry == FLT_MAX) continue;

        const uint32_t leftFirst = __float_as_uint(n0.w);
        const uint32_t triCount  = __float_as_uint(n1.w);

        if (triCount > 0) {
            // Leaf: test each triangle.
            for (uint32_t i = 0; i < triCount; ++i) {
                const uint32_t ti = leftFirst + i;
                const float4   v0 = sortedTris[ti*3 + 0];
                const float4   v1 = sortedTris[ti*3 + 1];
                const float4   v2 = sortedTris[ti*3 + 2];
                float t = tri_intersect(ox, oy, oz, dx, dy, dz,
                                        v0.x, v0.y, v0.z,
                                        v1.x, v1.y, v1.z,
                                        v2.x, v2.y, v2.z);
                if (t > tmin && t < best_t) best_t = t;
            }
        } else {
            // Internal: push right child first so left is popped first.
            if (sptr < 31) {
                stack[sptr++] = leftFirst + 1u; // right
                stack[sptr++] = leftFirst;       // left
            }
        }
    }

    hitDist[idx] = best_t;
}

// ---------------------------------------------------------------------------
// Host launcher (called from cpp/tinybvh_backends.cpp via tinybvh_kernels.h)
// ---------------------------------------------------------------------------
void launch_tbvh_rays(const float4* d_nodes,
                      const float4* d_sortedTris,
                      const float*  d_rays,
                      float*        d_hitDist,
                      int           numRays,
                      float         tmin,
                      float         tmax,
                      cudaStream_t  stream)
{
    if (numRays <= 0) return;
    const int threads = 256;
    const int blocks  = (numRays + threads - 1) / threads;
    tbvh_trace_kernel<<<blocks, threads, 0, stream>>>(
        d_nodes, d_sortedTris, d_rays, d_hitDist, numRays, tmin, tmax);
}
