// =============================================================================
// OptiX device programs
//
// Compiled to PTX by the build; loaded at runtime by optix.cu.
//
// Build:
//   make          # builds optix_device.ptx + optix CPU binary
// =============================================================================

#include <optix_device.h>
#include <math.h>
#include <float.h>
#include "optix_params.h"

static __device__ __forceinline__ float3 normalize(float3 v)
{
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_float3(inv*v.x, inv*v.y, inv*v.z);
}

// extern "C" prevents C++ name mangling so OptiX can locate "params" by name
// via pipelineLaunchParamsVariableName.  __constant__ is always module-static,
// so NVCC warns the extern is treated as a definition (suppressed via CMake).
extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
// Raygen — one invocation per ray (hId, vId).
//
// Replicates the direction formula from UpdateLidarRays in GRCA_CUDA.cu:
//   dir = normalize((cosH * forward + sinH * right) * cosV + sinV * up)
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__lidar()
{
    const unsigned int rayId = optixGetLaunchIndex().x;

    // Find which lidar owns this ray (linear scan; max ~32 lidars, fast in practice).
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

    float3 dir = normalize(
        make_float3(
            (cosH * lc.forward.x + sinH * lc.right.x) * cosV + sinV * lc.up.x,
            (cosH * lc.forward.y + sinH * lc.right.y) * cosV + sinV * lc.up.y,
            (cosH * lc.forward.z + sinH * lc.right.z) * cosV + sinV * lc.up.z
        ));

    // Store direction regardless of hit — same as GRCA_CUDA.cu's output.
    params.directions[rayId] = dir;

    // Payload word 0 carries the hit distance as raw float bits.
    // Initialised to rangeMax; overwritten by __closesthit__ on a hit,
    // overwritten to FLT_MAX by __miss__.
    unsigned int p0 = __float_as_uint(lc.rangeMax);

    unsigned int rayFlags = OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
                          | OPTIX_RAY_FLAG_DISABLE_ANYHIT;

    optixTrace(
        params.traversable,
        lc.origin, dir,
        lc.rangeMin,                    // tmin
        lc.rangeMax,                    // tmax
        0.f,                            // ray time (motion blur off)
        OptixVisibilityMask(1),
        rayFlags,
        0u,                             // SBT offset
        1u,                             // SBT stride
        0u,                             // miss SBT index
        p0);

    params.distances[rayId] = __uint_as_float(p0);
}

// ---------------------------------------------------------------------------
// Miss — ray escaped the scene, write FLT_MAX.
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__lidar()
{
    optixSetPayload_0(__float_as_uint(FLT_MAX));
}

// ---------------------------------------------------------------------------
// Closest-hit — write the intersection distance.
//
// optixGetRayTmax() returns the parametric t of the closest hit, which equals
// the Euclidean distance because the ray direction is unit length.
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__lidar()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}
