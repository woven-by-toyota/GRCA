// =============================================================================
// OptiX custom-primitive device programs
//
// Uses OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM so triangle intersection runs as a
// software Möller–Trumbore program on CUDA shader processors, not RT cores.
// BVH traversal still uses RT-core hardware acceleration.
//
// Useful for measuring how much of OptiX's advantage comes from RT-core
// triangle intersection vs. BVH traversal alone.
// =============================================================================

#include <optix_device.h>
#include <math.h>
#include <float.h>
#include "optix_custom_params.h"

// ---------------------------------------------------------------------------
// float3 helpers (optix_device.h does not provide operator overloads)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float3 f3sub(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __device__ __forceinline__ float3 f3cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}
static __device__ __forceinline__ float f3dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static __device__ __forceinline__ float3 f3norm(float3 v)
{
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_float3(inv*v.x, inv*v.y, inv*v.z);
}

extern "C" __constant__ LaunchParamsCustom params;

// ---------------------------------------------------------------------------
// Raygen — identical direction formula to optix_device.cu's __raygen__lidar
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__lidar_custom()
{
    const unsigned int rayId = optixGetLaunchIndex().x;

    unsigned int li = 0;
    while (li + 1 < params.num_origins &&
           rayId >= params.ray_origins[li + 1].rayOffset) ++li;

    const RayOrigin& lc = params.ray_origins[li];
    const unsigned int local = rayId - lc.rayOffset;
    const unsigned int h     = local % lc.hNum;
    const unsigned int v     = local / lc.hNum;

    float hAngle = lc.hmin + h * lc.hStep;
    float vAngle = lc.vmin + v * lc.vStep;

    float cosH = cosf(hAngle), sinH = sinf(hAngle);
    float cosV = cosf(vAngle), sinV = sinf(vAngle);

    float3 dir = f3norm(make_float3(
        (cosH * lc.forward.x + sinH * lc.right.x) * cosV + sinV * lc.up.x,
        (cosH * lc.forward.y + sinH * lc.right.y) * cosV + sinV * lc.up.y,
        (cosH * lc.forward.z + sinH * lc.right.z) * cosV + sinV * lc.up.z));

    params.directions[rayId] = dir;

    unsigned int p0 = __float_as_uint(lc.rangeMax);

    // OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES has no effect on custom
    // primitives; back-face culling is handled inside __intersection__lidar_custom.
    optixTrace(
        params.traversable,
        lc.origin, dir,
        lc.rangeMin,                    // tmin
        lc.rangeMax,                    // tmax
        0.f,                            // ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0u, 1u, 0u,
        p0);

    params.distances[rayId] = __uint_as_float(p0);
}

// ---------------------------------------------------------------------------
// Miss
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__lidar_custom()
{
    optixSetPayload_0(__float_as_uint(FLT_MAX));
}

// ---------------------------------------------------------------------------
// Closest-hit
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__lidar_custom()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}

// ---------------------------------------------------------------------------
// Intersection — Möller–Trumbore running on CUDA shader processors.
//
// Back-face culling: det < epsilon → skip (matches the triangle pipeline's
// OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES behaviour).
// ---------------------------------------------------------------------------
extern "C" __global__ void __intersection__lidar_custom()
{
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const float3 v0 = params.vertices[primIdx * 3 + 0];
    const float3 v1 = params.vertices[primIdx * 3 + 1];
    const float3 v2 = params.vertices[primIdx * 3 + 2];

    const float3 orig = optixGetWorldRayOrigin();
    const float3 dir  = optixGetWorldRayDirection();

    const float3 e1  = f3sub(v1, v0);
    const float3 e2  = f3sub(v2, v0);
    const float3 h   = f3cross(dir, e2);
    const float  det = f3dot(e1, h);

    // Back-face culling + parallel-ray reject
    if (det < 1e-7f) return;

    const float  inv_det = 1.f / det;
    const float3 s = f3sub(orig, v0);
    const float  u = inv_det * f3dot(s, h);
    if (u < 0.f || u > 1.f) return;

    const float3 q = f3cross(s, e1);
    const float  v = inv_det * f3dot(dir, q);
    if (v < 0.f || u + v > 1.f) return;

    const float t = inv_det * f3dot(e2, q);
    if (t < optixGetRayTmin() || t > optixGetRayTmax()) return;

    optixReportIntersection(t, 0u);
}
