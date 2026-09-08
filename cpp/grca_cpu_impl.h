#pragma once
// =============================================================================
// grca_cpu_impl.h  —  Internal shared types and helpers for GRCA_CPU.cpp and
//                    BMT_CPU.cpp.  Not part of the public API.
// =============================================================================

#include "../include/grca_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <tuple>
#ifdef __SSE4_1__
#  include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr float PI                = 3.14159265358979323846f;
static constexpr float FLT_MAX_VAL       = 3.402823466e+38f;
static constexpr float EPSILON           = 0.0001f;
static constexpr float MAX_RANGE         = 1000.0f;
static constexpr float MIN_APPARENT_AREA = 0.000001f;
static constexpr int   MAX_INTERSECTIONS = 6;
static constexpr int   MAX_ROBOT_LIDAR_COUNT = 32;
static constexpr int   MAX_TRI_RAY_COUNT               = 800000;


// ---------------------------------------------------------------------------
// Float3 math helpers
// ---------------------------------------------------------------------------
struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
    float3(Vec3 v) : x(v.x), y(v.y), z(v.z) {}   // implicit conversion from Vec3
};

inline float3 operator+(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline float3 operator-(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float3 operator*(float s, float3 a)  { return {s*a.x, s*a.y, s*a.z}; }
inline float3 operator*(float3 a, float s)  { return s * a; }
inline float3 operator/(float3 a, float s)  { return {a.x/s, a.y/s, a.z/s}; }
inline float3 operator-(float3 a)           { return {-a.x, -a.y, -a.z}; }

inline float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float3 cross(float3 a, float3 b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline float length(float3 v) { return std::sqrt(dot(v, v)); }
inline float3 normalize(float3 v) {
    float len = length(v);
    return (len > 1e-12f) ? (v / len) : float3{0,0,0};
}
inline float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline float  clamp(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
inline float  clamp01(float v)                   { return clamp(v, 0.0f, 1.0f); }

// ---------------------------------------------------------------------------
// Triangle — purely geometric, lidar-independent.
// Used by both GRCA_CPU.cpp and BMT_CPU.cpp.
// ---------------------------------------------------------------------------
struct Triangle {
    float3 Vertices[3];
    float3 Edge1;   // v1 - v0
    float3 Edge2;   // v2 - v0
    float3 Center;
    float3 Normal;
    float  Area;
};

// ---------------------------------------------------------------------------
// GRCA-specific per-lidar precomputed data (mirrors CUDA Late Pass shared mem:
// GRCAGacpDataSh, RayCastDataSh, GRCARspanDataSh).  Not used by BMT_CPU.
// ---------------------------------------------------------------------------
struct GrcaTriData {
    // ---- Mirrors GRCAGacpDataSh / rcShBuf ----
    float3 originToVert[3];   // Vertices[i] - L.pose.pos
    float  signdDist[3];      // dot(originToVert[i], L.pose.up)
    float  signdDistAngle[3]; // dot(normalize(originToVert[i]), L.pose.up)
    float  DdA[3];            // dot(edge_e, L.pose.up)                       — GACP c1/c2 term (negate for flipped)
    float  DdD[3];            // dot(edge_e, edge_e)                          — GACP c2 term
    float  P0dD[3];           // dot(originToVert[i0_e], edge_e)              — GACP c1 term
    float  P0dP0[3];          // dot(originToVert[i0_e], originToVert[i0_e]) — GACP c0 term
    // note: P0dA[e] = dot(originToVert[i0_e], up) = signdDist[e] — no separate field needed
    float  distCenterNormal;  // dot(Normal, Center - L.pose.pos)
    float  closestVertDist;   // closest distance from lidar origin to triangle surface
    bool   isBehindCone;
    bool   isBehindConeFlipped;
    bool   isTriIntersectConeDir;
    bool   isTriIntersectConeDirFlipped;
    // ---- Span prediction (mirrors TriRay.vFromIdToId + isAllCWFlags) ------
    uint32_t fromChannelId;
    uint32_t toChannelId;
    bool     isAllClockWise;
    uint32_t estimFromSweepId;
    uint32_t estimSweepDiff;
};

struct HitPoint {
    float3 pos  = {0,0,0};
    float  dist = FLT_MAX_VAL;
};

// ---------------------------------------------------------------------------
// Edge index table
// ---------------------------------------------------------------------------
static const int edgeIndices[3][2] = { {0, 1}, {1, 2}, {2, 0} };

// ---------------------------------------------------------------------------
// Per-lidar GRCA state
// ---------------------------------------------------------------------------
struct GrcaLidar {
    LidarCfg cfg;
    Pose     pose;

    uint32_t hnum;              // cached (uint32_t)cfg.hnum
    uint32_t vnum;              // cached (uint32_t)cfg.vnum
    uint32_t totalSampleCount;
    float    hAngleIncrement;
    float    vAngleIncrement;

    std::vector<float3>   worldDirs;
    std::vector<HitPoint> hitPoints;

    // SoA direction arrays — same data as worldDirs, split by component.
    // Used by the AVX2 inner-loop in GRCA_SAT_BAT for sequential ray loads.
    std::vector<float> wdx;
    std::vector<float> wdy;
    std::vector<float> wdz;

    std::vector<float> chanCosHalfAngle;
    std::vector<bool>  chanIsFlipped;
};

// ---------------------------------------------------------------------------
// Persistent CPU state (definition shared by GRCA_CPU.cpp and BMT_CPU.cpp)
// ---------------------------------------------------------------------------
struct SavedTri {
    float3   v[3];
    uint32_t lidarMask;                               // bit i set = lidar i passed GRCA_Early_T_Filter
};

struct CpuState {
    std::vector<GrcaLidar> lidars;
    std::vector<SavedTri> passedTris;  // reused across frames to avoid per-frame allocation
    int packet_size = 8;               // 1 = scalar, 8 = AVX2
};

// ===========================================================================
// Shared static inline helpers
// ===========================================================================

static inline Triangle BuildTriangle(float3 v0, float3 v1, float3 v2) {
    Triangle tri;
    tri.Vertices[0] = v0;
    tri.Vertices[1] = v1;
    tri.Vertices[2] = v2;
    tri.Edge1  = v1 - v0;
    tri.Edge2  = v2 - v0;
    tri.Center = (v0 + v1 + v2) / 3.0f;
    float3 cp  = cross(tri.Edge2, tri.Edge1);
    tri.Normal = normalize(cp);
    tri.Area   = 0.5f * length(cp);
    return tri;
}

static inline float3 ComputeRayDirection(const Pose& pose, float hAngle, float vAngle) {
    float cosH = std::cos(hAngle), sinH = std::sin(hAngle);
    float cosV = std::cos(vAngle), sinV = std::sin(vAngle);
    float3 d = (cosH * pose.forward + sinH * pose.right) * cosV + sinV * pose.up;
    return normalize(d);
}

static inline bool IsPointOnPlane(float3 checkPoint, float3 planePoint, float3 planeNormal) {
    return std::fabs(dot(planeNormal, checkPoint - planePoint)) < EPSILON;
}

static inline float ClosestDistanceToTriangle(float3 P, float3 v0, float3 v1, float3 v2, float3 normal) {
    float planeDist = dot(P - v0, normal);
    float3 proj = P - planeDist * normal;

    // Precompute all 3 edges and toP = proj - A (reused in both inside test and outside case).
    // Key identity: dot(P-A, edge) = dot(toP, edge)  (planeDist*normal ⊥ edge)
    // and |P - closest|² = |toP - t·edge|² + planeDist²  (planeDist component factors out)
    const float3 verts[3] = { v0, v1, v2 };
    float3 edges[3], toPs[3];
    for (int e = 0; e < 3; ++e) {
        edges[e] = verts[(e + 1) % 3] - verts[e];
        toPs[e]  = proj - verts[e];
    }

#if defined(__SSE4_1__)
    // --- Inside test: batch 3 cross(edge,toP)·normal ---
    __m128 ex = _mm_setr_ps(edges[0].x, edges[1].x, edges[2].x, 0.f);
    __m128 ey = _mm_setr_ps(edges[0].y, edges[1].y, edges[2].y, 0.f);
    __m128 ez = _mm_setr_ps(edges[0].z, edges[1].z, edges[2].z, 0.f);
    __m128 tx = _mm_setr_ps(toPs[0].x,  toPs[1].x,  toPs[2].x,  0.f);
    __m128 ty = _mm_setr_ps(toPs[0].y,  toPs[1].y,  toPs[2].y,  0.f);
    __m128 tz = _mm_setr_ps(toPs[0].z,  toPs[1].z,  toPs[2].z,  0.f);
    __m128 cx = _mm_sub_ps(_mm_mul_ps(ey, tz), _mm_mul_ps(ez, ty));
    __m128 cy = _mm_sub_ps(_mm_mul_ps(ez, tx), _mm_mul_ps(ex, tz));
    __m128 cz = _mm_sub_ps(_mm_mul_ps(ex, ty), _mm_mul_ps(ey, tx));
    __m128 d  = _mm_add_ps(_mm_add_ps(_mm_mul_ps(cx, _mm_set1_ps(normal.x)),
                                       _mm_mul_ps(cy, _mm_set1_ps(normal.y))),
                            _mm_mul_ps(cz, _mm_set1_ps(normal.z)));
    if ((_mm_movemask_ps(_mm_cmpge_ps(d, _mm_setzero_ps())) & 0x7) == 0x7)
        return std::fabs(planeDist);

    // --- Outside case: 3 t values + |toP - t·edge|² + planeDist² ---
    __m128 tpe = _mm_add_ps(_mm_add_ps(_mm_mul_ps(tx, ex), _mm_mul_ps(ty, ey)), _mm_mul_ps(tz, ez));
    __m128 ee  = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ex, ex), _mm_mul_ps(ey, ey)), _mm_mul_ps(ez, ez));
    __m128 t4  = _mm_min_ps(_mm_max_ps(_mm_div_ps(tpe, ee), _mm_setzero_ps()), _mm_set1_ps(1.0f));
    __m128 dx  = _mm_sub_ps(tx, _mm_mul_ps(t4, ex));
    __m128 dy  = _mm_sub_ps(ty, _mm_mul_ps(t4, ey));
    __m128 dz  = _mm_sub_ps(tz, _mm_mul_ps(t4, ez));
    __m128 dSq4 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
                                          _mm_mul_ps(dz, dz)),
                               _mm_set1_ps(planeDist * planeDist));
    alignas(16) float dsq[4];
    _mm_store_ps(dsq, dSq4);
    return std::sqrt(std::min(dsq[0], std::min(dsq[1], dsq[2])));
#else
    bool isInside = true;
    for (int e = 0; e < 3; ++e) {
        if (dot(cross(edges[e], toPs[e]), normal) < 0.0f) { isInside = false; break; }
    }
    if (isInside) return std::fabs(planeDist);

    float minDistSq = FLT_MAX_VAL;
    for (int e = 0; e < 3; ++e) {
        float t   = clamp01(dot(toPs[e], edges[e]) / dot(edges[e], edges[e]));
        float3 delta = toPs[e] - t * edges[e];
        float dSq = dot(delta, delta) + planeDist * planeDist;
        minDistSq = std::min(minDistSq, dSq);
    }
    return std::sqrt(minDistSq);
#endif
}

static inline bool GRCA_Early_T_Filter(const Pose& pose, const Triangle& tri) {
    // closestVertDist = FLT_MAX_VAL;
    if (IsPointOnPlane(pose.pos, tri.Center, tri.Normal)) { return false; }

    float3 toCenter = tri.Center - pose.pos;

    float dotNC = dot(toCenter, tri.Normal);
    if (dotNC <= 0.0f) { return false; }

    float distSqr = std::max(dot(toCenter, toCenter), EPSILON);
    float lhs = tri.Area * dotNC;
    if (lhs * lhs < MIN_APPARENT_AREA * MIN_APPARENT_AREA * distSqr * distSqr * distSqr) {
        return false;
    }
    // closestVertDist calculation and range check moved to run_cpu_frame
    return true;
}

static inline float M_T_RayTriangleIntersect(const Pose& pose, const float3 v0, const float3 edge1, const float3 edge2,
                                               float3 direction, float3& intersectPoint) {
    intersectPoint = {0,0,0};
    float3 tmp = cross(direction, edge2);
    float  inv_det = 1.0f / dot(edge1, tmp);

    float3 s = pose.pos - v0;
    float  u = inv_det * dot(s, tmp);
    if (u < 0.0f || u > 1.0f) return FLT_MAX_VAL;

    tmp = cross(s, edge1);
    float v = inv_det * dot(direction, tmp);
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_VAL;

    float t = inv_det * dot(edge2, tmp);
    if (t > EPSILON) {
        intersectPoint = pose.pos + direction * t;
        return t;
    }
    return FLT_MAX_VAL;
}

// Fills GrcaTriData for a Triangle from a specific lidar.
// Mirrors what CUDA Late Pass row-0 threads compute into shared memory
// (grcaGacpShBuf, rcShBuf) before the per-channel sweep begins.
// Span fields (fromChannelId … estimSweepDiff) are filled separately
// in GRCA_CPU.cpp after GRCA_CSpan_Predict / GRCA_Fast_RSpan_Predict.
static inline void FillGrcaTriData(const float3& v0, const float3& v1, const float3& v2,
                                                                  const float3& edge1, const float3& edge2, const float3& normal,
                                                                  const GrcaLidar& L, GrcaTriData& g) {
    g.originToVert[0] = v0 - L.pose.pos;
    g.originToVert[1] = v1 - L.pose.pos;
    g.originToVert[2] = v2 - L.pose.pos;

    float3 center = (v0 + v1 + v2) / 3.0f;

    for (int e = 0; e < 3; ++e) {
        int i0 = edgeIndices[e][0], i1 = edgeIndices[e][1];
        float3 D = g.originToVert[i1] - g.originToVert[i0]; // = edge_e
        g.DdA[e]            = dot(D, L.pose.up);
        g.DdD[e]            = dot(D, D);
        g.signdDist[e]      = dot(g.originToVert[e], L.pose.up);
        g.signdDistAngle[e] = dot(normalize(g.originToVert[e]), L.pose.up);
        g.P0dD[e]           = dot(g.originToVert[i0], D);
        g.P0dP0[e]          = dot(g.originToVert[i0], g.originToVert[i0]);
    }

    g.distCenterNormal    = dot(normal, center - L.pose.pos);
    g.isBehindCone        = (g.signdDist[0] < 0.0f && g.signdDist[1] < 0.0f && g.signdDist[2] < 0.0f);
    g.isBehindConeFlipped = (g.signdDist[0] > 0.0f && g.signdDist[1] > 0.0f && g.signdDist[2] > 0.0f);

    float3 ip;
    g.isTriIntersectConeDir        = (M_T_RayTriangleIntersect(L.pose, v0, edge1, edge2,  L.pose.up, ip) < FLT_MAX_VAL);
    g.isTriIntersectConeDirFlipped = (M_T_RayTriangleIntersect(L.pose, v0, edge1, edge2, -L.pose.up, ip) < FLT_MAX_VAL);
}

static inline void resetHits(GrcaLidar& L) {
    for (auto& h : L.hitPoints) { h.pos = {0,0,0}; h.dist = FLT_MAX_VAL; }
}

static inline void grca_init(GrcaLidar& L, const LidarCfg& cfg, const Pose& pose) {
    L.cfg  = cfg;
    L.pose = pose;
    L.hnum = (uint32_t)cfg.hnum;
    L.vnum = (uint32_t)cfg.vnum;

    if (L.cfg.range_max > MAX_RANGE)
        throw std::runtime_error("RangeMax exceeds maximum allowed range.");
    if (L.vnum < 1)
        throw std::runtime_error("VNumSamples must be >= 1.");

    L.vAngleIncrement = (L.vnum > 1)
        ? (L.cfg.vmax - L.cfg.vmin) / (L.vnum - 1)
        : 0.0f;
    L.hAngleIncrement = (L.cfg.hmax - L.cfg.hmin) / (L.hnum - 1);
    L.totalSampleCount = L.hnum * L.vnum;

    size_t total = (size_t)L.vnum * L.hnum;
    L.worldDirs.resize(total);
    L.hitPoints.resize(total);
    L.wdx.resize(total); L.wdy.resize(total); L.wdz.resize(total);

    L.chanCosHalfAngle.resize(L.vnum);
    L.chanIsFlipped.resize(L.vnum);
    for (uint32_t ch = 0; ch < L.vnum; ++ch) {
        float vAngle = L.cfg.vmin + ch * L.vAngleIncrement;
        L.chanIsFlipped[ch]    = (vAngle < 0.0f);
        L.chanCosHalfAngle[ch] = std::fabs(std::sin(vAngle));
    }

    // Precompute per-axis trig (O(H+V) calls instead of O(H*V)).
    // {fwd, right, up} is orthonormal so the direction is already unit-length —
    // no normalize needed.
    std::vector<float> cosH(L.hnum), sinH(L.hnum);
    for (uint32_t h = 0; h < L.hnum; ++h) {
        float hA = L.cfg.hmin + h * L.hAngleIncrement;
        cosH[h] = std::cos(hA); sinH[h] = std::sin(hA);
    }
    // vert-outer / hori-inner → sequential writes into worldDirs
    for (uint32_t v = 0; v < L.vnum; ++v) {
        float vA = L.cfg.vmin + v * L.vAngleIncrement;
        const float cv = std::cos(vA), sv = std::sin(vA);
        const float upVx = sv * L.pose.up.x;
        const float upVy = sv * L.pose.up.y;
        const float upVz = sv * L.pose.up.z;
        float3* row = L.worldDirs.data() + (size_t)v * L.hnum;
        for (uint32_t h = 0; h < L.hnum; ++h) {
            const float ch = cosH[h], sh = sinH[h];
            row[h] = float3{
                (ch * L.pose.forward.x + sh * L.pose.right.x) * cv + upVx,
                (ch * L.pose.forward.y + sh * L.pose.right.y) * cv + upVy,
                (ch * L.pose.forward.z + sh * L.pose.right.z) * cv + upVz
            };
        }
    }
    // Fill SoA direction arrays from the AoS worldDirs computed above.
    for (size_t i = 0; i < total; ++i) {
        L.wdx[i] = L.worldDirs[i].x;
        L.wdy[i] = L.worldDirs[i].y;
        L.wdz[i] = L.worldDirs[i].z;
    }
    resetHits(L);
}

// ---------------------------------------------------------------------------
// Shared frame helpers — used by both run_cpu_frame and run_bmt_frame
// ---------------------------------------------------------------------------

static inline void cpu_update_poses(CpuState& s, const RunConfig& r) {
    size_t li = 0;
    for (const auto& robot : r.robots) {
        for (const auto& lc : robot.lidars) {
            GrcaLidar& L = s.lidars[li++];
            L.pose = lc.w_pose;

            // Precompute trig tables: O(H+V) calls instead of O(H*V).
            std::vector<float> cosH(L.hnum), sinH(L.hnum);
            for (uint32_t h = 0; h < L.hnum; ++h) {
                float hA = L.cfg.hmin + h * L.hAngleIncrement;
                cosH[h] = std::cos(hA); sinH[h] = std::sin(hA);
            }

            // vert-outer / hori-inner → sequential writes (row-major worldDirs).
            for (uint32_t v = 0; v < L.vnum; ++v) {
                float vA = L.cfg.vmin + v * L.vAngleIncrement;
                const float cv = std::cos(vA), sv = std::sin(vA);
                // Per-row constants (broadcast into SIMD registers below).
                const float Ax = L.pose.forward.x * cv, Bx = L.pose.right.x * cv, Cx = sv * L.pose.up.x;
                const float Ay = L.pose.forward.y * cv, By = L.pose.right.y * cv, Cy = sv * L.pose.up.y;
                const float Az = L.pose.forward.z * cv, Bz = L.pose.right.z * cv, Cz = sv * L.pose.up.z;

                const size_t row = (size_t)v * L.hnum;
                float*  wdx = L.wdx.data() + row;
                float*  wdy = L.wdy.data() + row;
                float*  wdz = L.wdz.data() + row;
                float3* wds = L.worldDirs.data() + row;

                uint32_t h = 0;
#if defined(__AVX2__)
                {
                    const __m256 Ax8=_mm256_set1_ps(Ax), Bx8=_mm256_set1_ps(Bx), Cx8=_mm256_set1_ps(Cx);
                    const __m256 Ay8=_mm256_set1_ps(Ay), By8=_mm256_set1_ps(By), Cy8=_mm256_set1_ps(Cy);
                    const __m256 Az8=_mm256_set1_ps(Az), Bz8=_mm256_set1_ps(Bz), Cz8=_mm256_set1_ps(Cz);
                    for (; h + 8 <= L.hnum; h += 8) {
                        __m256 ch8 = _mm256_loadu_ps(cosH.data() + h);
                        __m256 sh8 = _mm256_loadu_ps(sinH.data() + h);
                        __m256 wx = _mm256_fmadd_ps(ch8, Ax8, _mm256_fmadd_ps(sh8, Bx8, Cx8));
                        __m256 wy = _mm256_fmadd_ps(ch8, Ay8, _mm256_fmadd_ps(sh8, By8, Cy8));
                        __m256 wz = _mm256_fmadd_ps(ch8, Az8, _mm256_fmadd_ps(sh8, Bz8, Cz8));
                        _mm256_storeu_ps(wdx + h, wx);
                        _mm256_storeu_ps(wdy + h, wy);
                        _mm256_storeu_ps(wdz + h, wz);
                    }
                }
#elif defined(__SSE4_1__)
                {
                    const __m128 Ax4=_mm_set1_ps(Ax), Bx4=_mm_set1_ps(Bx), Cx4=_mm_set1_ps(Cx);
                    const __m128 Ay4=_mm_set1_ps(Ay), By4=_mm_set1_ps(By), Cy4=_mm_set1_ps(Cy);
                    const __m128 Az4=_mm_set1_ps(Az), Bz4=_mm_set1_ps(Bz), Cz4=_mm_set1_ps(Cz);
                    for (; h + 4 <= L.hnum; h += 4) {
                        __m128 ch4 = _mm_loadu_ps(cosH.data() + h);
                        __m128 sh4 = _mm_loadu_ps(sinH.data() + h);
                        __m128 wx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ch4, Ax4), _mm_mul_ps(sh4, Bx4)), Cx4);
                        __m128 wy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ch4, Ay4), _mm_mul_ps(sh4, By4)), Cy4);
                        __m128 wz = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ch4, Az4), _mm_mul_ps(sh4, Bz4)), Cz4);
                        _mm_storeu_ps(wdx + h, wx);
                        _mm_storeu_ps(wdy + h, wy);
                        _mm_storeu_ps(wdz + h, wz);
                    }
                }
#endif
                // Scalar tail (also handles full loop when neither AVX2 nor SSE4.1).
                for (; h < L.hnum; ++h) {
                    wdx[h] = cosH[h] * Ax + sinH[h] * Bx + Cx;
                    wdy[h] = cosH[h] * Ay + sinH[h] * By + Cy;
                    wdz[h] = cosH[h] * Az + sinH[h] * Bz + Cz;
                }
                // Fill AoS worldDirs from SoA (sequential reads+writes, compiler vectorizes).
                for (uint32_t h2 = 0; h2 < L.hnum; ++h2)
                    wds[h2] = { wdx[h2], wdy[h2], wdz[h2] };
            }
        }
    }
}

static inline void cpu_reset_all_hits(CpuState& s) {
    for (auto& L : s.lidars) resetHits(L);
}

static inline void cpu_fill_hits(CpuState& s, const RunConfig& r,
                                  std::vector<GrcaHit>& hits, FrameTimings& t) {
    size_t totalHits = 0;
    size_t globalLidarIdx = 0;
    for (const auto& robot : r.robots)
        for (size_t li = 0; li < robot.lidars.size(); ++li, ++globalLidarIdx)
            totalHits += s.lidars[globalLidarIdx].hitPoints.size();
    hits.resize(totalHits);

    int    totalHitCount = 0;
    size_t hitIdx        = 0;
    globalLidarIdx       = 0;
    auto tFill0 = std::chrono::steady_clock::now();
    for (const auto& robot : r.robots) {
        for (size_t li = 0; li < robot.lidars.size(); ++li, ++globalLidarIdx) {
            GrcaLidar& entry = s.lidars[globalLidarIdx];
            size_t n = entry.hitPoints.size();
            for (size_t i = 0; i < n; ++i, ++hitIdx) {
                const HitPoint& h = entry.hitPoints[i];
                const float3&   d = entry.worldDirs[i];
                GrcaHit& gh = hits[hitIdx];
                gh.dx = d.x; gh.dy = d.y; gh.dz = d.z;
                if (h.dist < FLT_MAX_VAL) {
                    gh.dist = h.dist;
                    gh.hx   = h.pos.x;
                    gh.hy   = h.pos.y;
                    gh.hz   = h.pos.z;
                    ++totalHitCount;
                } else {
                    gh.dist = entry.cfg.range_max;
                    gh.hx   = entry.pose.pos.x + d.x * entry.cfg.range_max;
                    gh.hy   = entry.pose.pos.y + d.y * entry.cfg.range_max;
                    gh.hz   = entry.pose.pos.z + d.z * entry.cfg.range_max;
                }
            }
        }
    }
    t.ms_fillhits = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tFill0).count();
    t.hit_count = totalHitCount;
}
