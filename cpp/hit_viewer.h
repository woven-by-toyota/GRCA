
// hit_viewer.h  –  Live GLFW / OpenGL 3.3 core point-cloud visualiser
#pragma once
#include "grca_common.h"
#include <vector>

// Opaque handle returned by vis_init().
struct VisState;

// Create the GLFW window and set up per-lidar colour slots.
// Returns nullptr if GLFW / OpenGL init fails.
VisState* vis_init(const RunConfig& cfg);

// Upload this frame's hits and redraw.
// hits is the flat vector produced by the backend (same layout as main.cpp writeCSV).
// frame / sim_time are shown in the window title.
void vis_update(VisState* vs, const std::vector<GrcaHit>& hits,
                const RunConfig& cfg, int frame, float sim_time);

// Returns false once the user closes the window (main loop should exit).
bool vis_running(VisState* vs);

// Set camera eye position and orientation.
// eye[3]     — world-space position
// forward[3] — unit look direction (may be nullptr to keep current)
// up[3]      — unit up vector     (may be nullptr to keep current)
// Disables the auto-fit-on-first-frame so the camera stays put.
void vis_set_camera(VisState* vs, const float eye[3],
                    const float forward[3], const float up[3]);

// Returns true (and resets the flag) when the user pressed O in the window.
bool vis_capture_requested(VisState* vs);

// Request a PNG capture of the next rendered frame (same as pressing I).
void vis_request_image(VisState* vs);

// Runtime view settings; fields left at -1 keep their current value.
struct VisView {
    int mesh_mode      = -1;  // mesh_vis_on value: 0 = hide meshes, 1..4 = colouring (see RunConfig)
    int hits           = -1;  // 0/1: LiDAR hit points
    int aabbs          = -1;  // 0/1: bounding-box overlays
    int cones          = -1;  // 0/1: LiDAR channel cones
    int static_meshes  = -1;  // 0/1: static meshes
    int dynamic_meshes = -1;  // 0/1: dynamic meshes
    int depth_colormap = -1;  // heat-map hit points: 0 = rainbow (near red -> far blue), 1 = grayscale
    // Outside the heat map (mesh_vis_on 1..3), for every mesh; negative = keep. The viewer applies
    // wire_width; the caller bakes mesh_color / wire_color into the VisMesh data it uploads.
    float mesh_color[3] = {-1.f, -1.f, -1.f};  // fill colour (replaces MTL and JSON colours)
    float wire_color[3] = {-1.f, -1.f, -1.f};  // edge colour
    float wire_width    = -1.f;                // edge width in pixels (default 1)
    float cam_fov_deg   = -1.f;                // vertical field of view of the free camera (default 45)
};
void vis_apply_view(VisState* vs, const VisView& v);

// True if the last frame drew depth-coloured hit points (heat-map mode, points on); then
// near/far is their colour range and colormap 0 = rainbow over LiDAR range, 1 = grayscale
// linear camera depth (white = near, black = far).
bool vis_depth_mapping(const VisState* vs, float& near, float& far, int& colormap);

// Mesh colouring currently selected (mesh_vis_on semantics, 1..4), also while
// meshes are hidden; changes with the V key and vis_apply_view.
int vis_mesh_mode(const VisState* vs);

// Returns true once after the user pressed I: rgb receives the rendered frame
// (RGB8, top row first, w*h*3 bytes) exactly as shown in the window.
bool vis_take_image(VisState* vs, std::vector<unsigned char>& rgb, int& w, int& h);

// Write an RGB8 image (top row first) as an uncompressed PNG.  Returns false on I/O error.
bool vis_write_png(const std::string& path, int w, int h, const unsigned char* rgb);

// Destroy the window and free resources.
void vis_destroy(VisState* vs);

// One mesh's worth of wireframe data: pointer to triangles, count, and RGB color.
// If tri_colors is non-null it must point to count*3 floats (RGB per triangle)
// and overrides the flat color[3].  Default flat color is grey {0.7, 0.7, 0.7}.
struct VisMesh {
    const Tri3*  tris;
    size_t       count;
    float        color[3]    = {0.7f, 0.7f, 0.7f};
    const float* tri_colors  = nullptr; // per-triangle RGB (count*3 floats), or nullptr
    float        wire_color[3] = {-1.f, -1.f, -1.f}; // edge colour; negative = 30% of the fill colour
};

// Upload all mesh wireframes in one call (replaces previous wireframe data).
// Each VisMesh gets its own color baked per-vertex into the shared buffer.
void vis_set_meshes(VisState* vs, const std::vector<VisMesh>& meshes);

// Upload only the static meshes and remember how many wire-vertices they occupy.
// Call once at init; the buffer tail is reserved for dynamic meshes.
void vis_set_static_meshes(VisState* vs, const std::vector<VisMesh>& meshes);

// Overwrite only the dynamic portion of the wireframe buffer (after the static
// region written by vis_set_static_meshes).  Call every frame instead of
// vis_set_meshes to avoid re-uploading the static mesh each frame.
void vis_update_dynamic_meshes(VisState* vs, const std::vector<VisMesh>& meshes);

// Axis-aligned bounding box with a display color.
// aabb[6] = { xmin, ymin, zmin, xmax, ymax, zmax }
struct VisAABB {
    float aabb[6]  = {};
    float color[3] = {0.7f, 0.7f, 0.7f};
};

// Upload all AABBs (static + dynamic); replaces previous AABB data.
void vis_set_aabbs(VisState* vs, const std::vector<VisAABB>& aabbs);

// Same as vis_set_aabbs — use when only dynamic AABBs change each frame
// (static AABBs are re-submitted unchanged at the front of the list).
void vis_update_aabbs(VisState* vs, const std::vector<VisAABB>& aabbs);

// One lidar to visualise as a small cylinder + channel cones.
// pos[3]      — world-space apex (lidar origin)
// fwd[3]      — spin axis (up in ray formula, cylinder height axis)
// up[3]       — cross-section reference up
// color[3]    — RGB (cylinder and cones share this color)
// radius      — cylinder radius in world units
// height      — cylinder height in world units
// vmin/vmax   — vertical angle range in radians
// vnum        — number of vertical channels
// range_max   — cone length (ray range)
struct VisLidar {
    float pos[3]    = {};
    float fwd[3]    = {0,0,1};
    float up[3]     = {0,1,0};
    float color[3]  = {1.0f, 1.0f, 1.0f};
    float radius    = 0.15f;
    float height    = 0.15f;
    float vmin      = 0.0f;
    float vmax      = 0.0f;
    int   vnum      = 0;
    float range_max = 100.0f;
};

// Upload all lidar cylinders; replaces previous lidar geometry.
// Call every frame if robots can move.
void vis_set_lidars(VisState* vs, const std::vector<VisLidar>& lidars);

// Upload all lidar channel cones; replaces previous cone geometry.
// Call every frame if robots can move.
void vis_set_lidar_cones(VisState* vs, const std::vector<VisLidar>& lidars);
