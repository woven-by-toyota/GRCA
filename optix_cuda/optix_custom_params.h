#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_params.h"   // RayOrigin

// Extended launch params for the custom-IS pipeline.
// Identical to LaunchParams but adds the flat triangle vertex buffer so
// the software Möller–Trumbore intersection program can read vertex data.
struct LaunchParamsCustom {
    RayOrigin*             ray_origins;
    unsigned int           num_origins;
    unsigned int           num_rays;
    float*                 distances;   // FLT_MAX on miss
    float3*                directions;  // unit direction per ray
    OptixTraversableHandle traversable;
    float3*                vertices;    // vertices[primIdx*3 + k], k=0..2
};
