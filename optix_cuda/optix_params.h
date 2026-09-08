#pragma once
#include <optix.h>
#include <cuda_runtime.h>

// Shared between CPU (GRCA_OptiX.cu) and device (GRCA_OptiX_device.cu).
// Must be kept POD — no constructors, no std types.

// Per-lidar configuration — shared by all H×V rays of that lidar.
// The raygen shader derives hAngle/vAngle from the ray launch index.
struct RayOrigin {
    float3       origin;
    float3       forward, up, right;
    float        hmin,  hStep;
    float        vmin,  vStep;
    unsigned int hNum,  vNum;
    unsigned int rayOffset;   // index of first ray for this lidar in the launch
    float        rangeMin, rangeMax;
};

struct LaunchParams {
    RayOrigin* ray_origins;  // device pointer, size = num_origins
    unsigned int num_origins;
    unsigned int num_rays;

    // Output buffers — device pointers, each sized [num_rays]
    float*  distances;   // FLT_MAX on miss
    float3* directions;  // unit direction for every ray

    // Acceleration structure built from the full mesh.
    OptixTraversableHandle traversable;

};
