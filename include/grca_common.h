#pragma once
#include <string>
#include <vector>

// =============================================================================
// Vec3  –  plain 3-float vector. Used in Pose so backends without float3
//          (e.g. CUDA, which has its own float3) can include grca_common.h
//          without a type conflict.
// =============================================================================
struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
};

// =============================================================================
// Pose  –  position and orientation (world-space or local-space).
//   No prefix = local space (relative to parent).
//   w_ prefix on the field that holds it = world space.
// =============================================================================
struct Pose {
    Vec3 pos     = {0.f, 0.f, 0.f};  // position
    Vec3 forward = {0.f, 0.f, 1.f};  // forward (unit)
    Vec3 up      = {0.f, 1.f, 0.f};  // up (unit)
    Vec3 right   = {1.f, 0.f, 0.f};  // right (unit)
};

// =============================================================================
// StaticMesh  –  .obj mesh placed at a fixed world-space position
// =============================================================================
struct StaticMesh {
    std::string file;
    Pose w_pose;             // world-space position (orientation unused for now)
    float scale = 1.0f;
    float color[3] = {0.7f, 0.7f, 0.7f}; // RGB, default gray
    bool color_override = false; // if true, color replaces per-triangle MTL colors
};

// =============================================================================
// DynamicMesh  –  .obj mesh with animation parameters
// =============================================================================
struct DynamicMesh {
    std::string file;
    Pose w_pose;             // world-space initial position (orientation unused for now)
    float velocity[3] = {0.f, 0.f, 0.f};
    float scale = 1.0f;
    float color[3] = {0.7f, 0.7f, 0.7f}; // RGB, default gray
    bool color_override = false; // if true, color replaces per-triangle MTL colors
};

// =============================================================================
// LidarCfg  –  per-lidar scan parameters, local-space and world-space pose.
// =============================================================================
struct LidarCfg {
    float hmin      = -3.14159265f, hmax      =  3.14159265f;
    int   hnum      =  3600;
    float vmin      = -0.5235987f,  vmax      =  0.5235987f;
    int   vnum      =  32;
    float range_min =  0.05f,       range_max =  100.0f;
    Pose l_pose;  // local-space pose (relative to parent robot, from config)
    Pose w_pose;  // computed world-space pose — set by main.cpp each frame
};

// =============================================================================
// RobotCfg  –  one robot carrying one or more lidars.
// =============================================================================
struct RobotCfg {
    std::vector<LidarCfg> lidars;
    Pose w_pose;  // world-space pose (from config "w_position/w_forward/w_up")
};

// =============================================================================
// RunConfig  –  unified configuration passed from main.cpp to every backend.
//
// CUDA limits (from GRCA_CUDA.cu defines):
//   MAX_ROBOT_COUNT     =  8  robots
//   MAX_LIDAR_COUNT     =  4  lidars per robot
//   MAX_LIDAR_RAY_COUNT = 524288  rays per robot (shared across all its lidars)
// =============================================================================
struct RunConfig {
    std::string output;                 // CSV prefix; files written as <output>_rR_lL.csv
    std::string backend = "grca-cpu";

    std::vector<RobotCfg> robots;       // all robots with their lidars
    std::vector<StaticMesh>  static_meshes;  // static .obj meshes (fixed position)
    std::vector<DynamicMesh> dynamic_meshes; // dynamic .obj meshes
    float dynamic_mesh_time = 0.0f; // time in seconds for dynamic mesh animation

    int seed = 0;
    bool random_dynamic_from_static_aabb = false;
    int   random_dynamic_pos_interpolate_frames   = 0;    // -1=disabled, 0=instant, N=lerp
    int   random_dynamic_rot_interpolate_frames   = 0;    // -1=disabled, 0=instant, N=lerp
    int   random_dynamic_scale_interpolate_frames = 0;    // -1=disabled, 0=instant, N=lerp
    float random_dynamic_scale_min = 1.0f;
    float random_dynamic_scale_max = 1.0f;
    int   random_tri_chaos = 0;  // per-triangle independent random pos/rot/scale: 0=off, 1=scene AABB, 2=mesh AABB
    bool  random_robots_share_pose = false;  // all robots share the same random pose each frame; lidar orientations are also randomised from cfg.seed

    // float scale = 1.0f; // (Removed: now per-mesh)

    bool        debug            = false;   // CUDA: print BAT/SAT dispatch stats
    int         cuda_sat_max_sweep_diff   = 64;  // CUDA: SAT sweep span threshold — exceeding defers to BAT
    int         cuda_sat_max_channel_diff = 64;  // CUDA: SAT channel span threshold — exceeding defers to BAT
    int         cpu_sat_max_sweep_diff    = 64;  // CPU:  SAT sweep span threshold — exceeding defers to BAT
    int         cpu_sat_max_channel_diff  = 64;  // CPU:  SAT channel span threshold — exceeding defers to BAT
    // -1 = pure refit, 0 = pure rebuild, N>0 = refit N frames then rebuild (period N+1)
    int         optix_rebuild_after_frames  = 0;
    bool        optix_compaction = false;
    int         embree_rebuild_after_frames = 0;
    bool        embree_compaction = false;
    int         embree_packet_size = 16;  // 1, 4, 8, or 16
    int         grca_cpu_packet_size = 16; // 1, 4, 8, or 16

    // TinyBVH backends — same semantics as embree_rebuild_after_frames
    int         tinybvh_cpu_rebuild_after_frames = 0;
    int         tinybvh_gpu_rebuild_after_frames = 0;

    int         bat_capacity = 800000;  // BAT buffer size (overridable via JSON "bat_capacity")

    // Visualizer (viewer) control
    bool visualizer_on  = false;          // true = open viewer window
    int  mesh_vis_on    = 0;              // 0 = off, 1 = wireframe, 2 = solid + wireframe
    bool cone_vis_on    = false;          // true = render lidar channel cones
};

// =============================================================================
// Tri3  –  common flat triangle type shared across all backends.
//          v[vertex_index][x/y/z]
// =============================================================================
struct Tri3 { float v[3][3]; };

// =============================================================================
// GrcaHit  –  per-ray result, one entry per lidar ray.
//
// Both hits and misses are always present:
//   hits:  dist = actual distance,  hx/hy/hz = hit world position
//   misses: dist = range_max,       hx/hy/hz = origin + dir * range_max
// =============================================================================
struct GrcaHit {
    float dx, dy, dz;   // ray direction (unit vector)
    float dist;         // hit distance, or range_max on miss
    float hx, hy, hz;   // hit world-space position
};

// =============================================================================
// FrameTimings  –  per-stage timing filled by main + each backend.
//
// main fills:  ms_total_cpu
// CPU fills:   ms_ray_init, ms_trace_cpu, hit_count
// CUDA fills:  ms_trace_gpu, ms_readback_cpu, mb_readback, hit_count
// Embree fills: ms_dynamic_upload, ms_bvh_rebuild_gpu/ms_bvh_refit_gpu, ms_trace_gpu, hit_count
// OptiX fills: ms_bvh_rebuild_gpu, ms_trace_gpu, ms_dynamic_upload
//              ms_readback_cpu, mb_readback, hit_count
//
// Zero means the stage was not applicable for this backend.
// =============================================================================
struct FrameTimings {
    double ms_lidar_sim_cpu    = 0;  // lidar sim wall-clock     (CPU, backend call)
    double ms_dynamic_upload   = 0;  // dynamic mesh + lidar pose upload (CPU, excludes BVH Rebuild/refit)
    double ms_bvh_rebuild_cpu  = 0;  // BVH Rebuild                (CPU, Embree/other)
    double ms_bvh_refit_cpu    = 0;  // BVH refit                (CPU, Embree/other)
    double ms_trace_cpu        = 0;  // CPU cost of submitting trace dispatches (wall-clock)
    double ms_sync_readback    = 0;  // CPU-GPU synchronization (cudaEventSynchronize/cudaDeviceSynchronize)
    double mb_readback         = 0;  // MB transferred in readback
    double ms_readback_cpu     = 0;  // GPU→CPU transfer         (CUDA/OptiX)
    double ms_fillhits         = 0;  // CPU-side hit conversion (CUDA/OptiX)
    double ms_lidar_sim_gpu    = 0;  // lidar sim GPU time       (BVH + trace, CUDA/OptiX)
    double ms_bvh_rebuild_gpu  = 0;  // BVH Rebuild                (GPU hw, OptiX)
    double ms_bvh_refit_gpu    = 0;  // BVH refit                (GPU hw, OptiX)
    double ms_trace_gpu        = 0;  // trace kernel GPU time    (CUDA events, CUDA/OptiX)
    double ms_init_cpu         = 0;  // Backend init wall-clock        (CPU, all backends)
    double ms_total_cpu        = 0;  // total frame wall-clock   (CPU)
    int    hit_count           = 0;  // number of rays that hit geometry
    int    num_bat             = -1; // BAT unique-triangle count this frame (-1 = not measured)
    int    num_early_t         = -1; // (robot,lidar,tri) triples passing early-T filter (-1 = not measured)
    int    num_rtic            = -1; // estimated ray-triangle intersection calls this frame (-1 = not measured)
    int    max_bat_capacity    = -1; // BAT buffer capacity (MAX_TRI_RAY_COUNT)
    // Per-frame performance metrics
    double sim_frame_rate      = 0;  // Actual simulation frame rate (Hz)
    double real_time_factor    = 0;  // Ratio of simulated time to real time
};
