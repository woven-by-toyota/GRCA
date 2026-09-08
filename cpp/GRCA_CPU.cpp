// =============================================================================
// GRCA_CPU.cpp
//
// Brute-force Möller–Trumbore reference path lives in BMT_CPU.cpp.
// Shared types, constants, and helpers live in grca_cpu_impl.h.
// =============================================================================

#include "cpu.h"
#include "grca_cpu_impl.h"
#include <immintrin.h>

// ===========================================================================
// GRCA-only static helpers
// ===========================================================================

struct RawTri { float3 v[3]; };

// ---------------------------------------------------------------------------
// AVX2 helpers for the early-T filter — process 8 triangles per batch.
// ---------------------------------------------------------------------------
#ifdef __AVX2__
static inline __m256 dot8(__m256 ax, __m256 ay, __m256 az,
                           __m256 bx, __m256 by, __m256 bz) {
    return _mm256_fmadd_ps(az, bz, _mm256_fmadd_ps(ay, by, _mm256_mul_ps(ax, bx)));
}

static inline void cross8(__m256 ax, __m256 ay, __m256 az,
                            __m256 bx, __m256 by, __m256 bz,
                            __m256& rx, __m256& ry, __m256& rz) {
    rx = _mm256_fmsub_ps(ay, bz, _mm256_mul_ps(az, by));
    ry = _mm256_fmsub_ps(az, bx, _mm256_mul_ps(ax, bz));
    rz = _mm256_fmsub_ps(ax, by, _mm256_mul_ps(ay, bx));
}

// Compute the center, unit normal, and area for 8 triangles in SoA layout.
// Mirrors BuildTriangle: cp = cross(Edge2, Edge1), normal = normalize(cp), area = 0.5*|cp|.
static inline void BuildTriangle8(__m256 v0x, __m256 v0y, __m256 v0z,
                                   __m256 v1x, __m256 v1y, __m256 v1z,
                                   __m256 v2x, __m256 v2y, __m256 v2z,
                                   __m256& cx,  __m256& cy,  __m256& cz,
                                   __m256& nx,  __m256& ny,  __m256& nz,
                                   __m256& area) {
    const __m256 k13 = _mm256_set1_ps(1.0f / 3.0f);
    cx = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0x, v1x), v2x), k13);
    cy = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0y, v1y), v2y), k13);
    cz = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v0z, v1z), v2z), k13);

    __m256 e1x = _mm256_sub_ps(v1x, v0x), e1y = _mm256_sub_ps(v1y, v0y), e1z = _mm256_sub_ps(v1z, v0z);
    __m256 e2x = _mm256_sub_ps(v2x, v0x), e2y = _mm256_sub_ps(v2y, v0y), e2z = _mm256_sub_ps(v2z, v0z);

    __m256 cpx, cpy, cpz;
    cross8(e2x, e2y, e2z, e1x, e1y, e1z, cpx, cpy, cpz);

    __m256 len = _mm256_sqrt_ps(dot8(cpx, cpy, cpz, cpx, cpy, cpz));
    area       = _mm256_mul_ps(_mm256_set1_ps(0.5f), len);

    // Safe normalise: degenerate triangles get a zero normal (will fail filter).
    __m256 inv = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
    __m256 ok  = _mm256_cmp_ps(len, _mm256_set1_ps(1e-12f), _CMP_GT_OQ);
    inv = _mm256_and_ps(inv, ok);
    nx  = _mm256_mul_ps(cpx, inv);
    ny  = _mm256_mul_ps(cpy, inv);
    nz  = _mm256_mul_ps(cpz, inv);
}

// Run GRCA_Early_T_Filter for 8 triangles against one lidar pose.
// The combined IsPointOnPlane + back-face condition reduces to dotNC >= EPSILON.
// Returns an 8-bit movemask: bit i set iff triangle i passes.
static inline int EarlyTFilter8(__m256 cx,   __m256 cy,   __m256 cz,
                                  __m256 nx,   __m256 ny,   __m256 nz,
                                  __m256 area, const Pose& pose) {
    const __m256 eps      = _mm256_set1_ps(EPSILON);
    const __m256 min_aa_sq = _mm256_set1_ps(MIN_APPARENT_AREA * MIN_APPARENT_AREA);

    __m256 tcx = _mm256_sub_ps(cx, _mm256_set1_ps(pose.pos.x));
    __m256 tcy = _mm256_sub_ps(cy, _mm256_set1_ps(pose.pos.y));
    __m256 tcz = _mm256_sub_ps(cz, _mm256_set1_ps(pose.pos.z));

    __m256 dotNC = dot8(tcx, tcy, tcz, nx, ny, nz);
    __m256 pass  = _mm256_cmp_ps(dotNC, eps, _CMP_GE_OQ);

    __m256 distSqr   = _mm256_max_ps(dot8(tcx, tcy, tcz, tcx, tcy, tcz), eps);
    __m256 lhs2      = _mm256_mul_ps(_mm256_mul_ps(area, dotNC), _mm256_mul_ps(area, dotNC));
    __m256 ds3       = _mm256_mul_ps(_mm256_mul_ps(distSqr, distSqr), distSqr);
    __m256 area_fail = _mm256_cmp_ps(lhs2, _mm256_mul_ps(min_aa_sq, ds3), _CMP_LT_OQ);

    return _mm256_movemask_ps(_mm256_andnot_ps(area_fail, pass));
}
#endif // __AVX2__

#if defined(__SSE4_1__)
static inline __m128 dot4(__m128 ax, __m128 ay, __m128 az,
                           __m128 bx, __m128 by, __m128 bz) {
    return _mm_add_ps(_mm_mul_ps(az, bz),
           _mm_add_ps(_mm_mul_ps(ay, by), _mm_mul_ps(ax, bx)));
}

static inline void cross4(__m128 ax, __m128 ay, __m128 az,
                            __m128 bx, __m128 by, __m128 bz,
                            __m128& rx, __m128& ry, __m128& rz) {
    rx = _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by));
    ry = _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz));
    rz = _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx));
}

static inline void BuildTriangle4(__m128 v0x, __m128 v0y, __m128 v0z,
                                   __m128 v1x, __m128 v1y, __m128 v1z,
                                   __m128 v2x, __m128 v2y, __m128 v2z,
                                   __m128& cx,  __m128& cy,  __m128& cz,
                                   __m128& nx,  __m128& ny,  __m128& nz,
                                   __m128& area) {
    const __m128 k13 = _mm_set1_ps(1.0f / 3.0f);
    cx = _mm_mul_ps(_mm_add_ps(_mm_add_ps(v0x, v1x), v2x), k13);
    cy = _mm_mul_ps(_mm_add_ps(_mm_add_ps(v0y, v1y), v2y), k13);
    cz = _mm_mul_ps(_mm_add_ps(_mm_add_ps(v0z, v1z), v2z), k13);

    __m128 e1x = _mm_sub_ps(v1x, v0x), e1y = _mm_sub_ps(v1y, v0y), e1z = _mm_sub_ps(v1z, v0z);
    __m128 e2x = _mm_sub_ps(v2x, v0x), e2y = _mm_sub_ps(v2y, v0y), e2z = _mm_sub_ps(v2z, v0z);

    __m128 cpx, cpy, cpz;
    cross4(e2x, e2y, e2z, e1x, e1y, e1z, cpx, cpy, cpz);

    __m128 len = _mm_sqrt_ps(dot4(cpx, cpy, cpz, cpx, cpy, cpz));
    area       = _mm_mul_ps(_mm_set1_ps(0.5f), len);

    __m128 inv = _mm_div_ps(_mm_set1_ps(1.0f), len);
    __m128 ok  = _mm_cmpgt_ps(len, _mm_set1_ps(1e-12f));
    inv = _mm_and_ps(inv, ok);
    nx  = _mm_mul_ps(cpx, inv);
    ny  = _mm_mul_ps(cpy, inv);
    nz  = _mm_mul_ps(cpz, inv);
}

static inline int EarlyTFilter4(__m128 cx,   __m128 cy,   __m128 cz,
                                  __m128 nx,   __m128 ny,   __m128 nz,
                                  __m128 area, const Pose& pose) {
    const __m128 eps       = _mm_set1_ps(EPSILON);
    const __m128 min_aa_sq = _mm_set1_ps(MIN_APPARENT_AREA * MIN_APPARENT_AREA);

    __m128 tcx = _mm_sub_ps(cx, _mm_set1_ps(pose.pos.x));
    __m128 tcy = _mm_sub_ps(cy, _mm_set1_ps(pose.pos.y));
    __m128 tcz = _mm_sub_ps(cz, _mm_set1_ps(pose.pos.z));

    __m128 dotNC = dot4(tcx, tcy, tcz, nx, ny, nz);
    __m128 pass  = _mm_cmpge_ps(dotNC, eps);

    __m128 distSqr   = _mm_max_ps(dot4(tcx, tcy, tcz, tcx, tcy, tcz), eps);
    __m128 lhs2      = _mm_mul_ps(_mm_mul_ps(area, dotNC), _mm_mul_ps(area, dotNC));
    __m128 ds3       = _mm_mul_ps(_mm_mul_ps(distSqr, distSqr), distSqr);
    __m128 area_fail = _mm_cmplt_ps(lhs2, _mm_mul_ps(min_aa_sq, ds3));

    return _mm_movemask_ps(_mm_andnot_ps(area_fail, pass));
}
#endif // __SSE4_1__

static float3 GetTriPlaneIntersectPoint(float3 vert0, float3 vert1,
                                         float signedDist0, float signedDist1) {
    return vert0 + (signedDist0 / (signedDist0 - signedDist1)) * (vert1 - vert0);
}

static bool GRCA_Fast_GACPT_IntersectionCheck(bool isFlipped, bool isTriIntersectConeDir,
                                              float3 /*direction*/, float cosHalfAngle,
                                              float rangeMax, const GrcaTriData& g) {
    if (std::fabs(cosHalfAngle) <= EPSILON) {
        bool e01 = (g.signdDist[0] * g.signdDist[1] < 0.0f);
        bool e12 = (g.signdDist[1] * g.signdDist[2] < 0.0f);
        bool e20 = (g.signdDist[2] * g.signdDist[0] < 0.0f);
        return (e01 && e12) || (e12 && e20) || (e20 && e01);
    }

    float cosTheta = cosHalfAngle;
    bool isAllVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] >= cosTheta && -g.signdDistAngle[1] >= cosTheta && -g.signdDistAngle[2] >= cosTheta)
        : ( g.signdDistAngle[0] >= cosTheta &&  g.signdDistAngle[1] >= cosTheta &&  g.signdDistAngle[2] >= cosTheta);
    if (isAllVertexInCone) { return false; }
    bool isNoVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] < cosTheta && -g.signdDistAngle[1] < cosTheta && -g.signdDistAngle[2] < cosTheta)
        : ( g.signdDistAngle[0] < cosTheta &&  g.signdDistAngle[1] < cosTheta &&  g.signdDistAngle[2] < cosTheta);
    if (!isNoVertexInCone) { return true; } // at least one vertex in cone volume → guaranteed edge crossing
    bool isCheckAll = isNoVertexInCone && isTriIntersectConeDir && g.closestVertDist <= rangeMax * cosTheta;
    if (isCheckAll) { return true; }

    float cos2 = cosTheta * cosTheta;
    float sign = isFlipped ? -1.0f : 1.0f;

    // direction = ±up, so dot(lerp(originToVert[i0], originToVert[i1], t), direction)
    // = sign*signdDist[i0_e] + t * sign*DdA[e] = p0dA + t*ddA  (both already computed below).
#if defined(__SSE4_1__)
    {
        // Process all 3 edges simultaneously, padded to 4 lanes (lane 3 mirrors lane 0).
        __m128 ddA4  = _mm_setr_ps(sign*g.DdA[0],       sign*g.DdA[1],       sign*g.DdA[2],       sign*g.DdA[0]);
        __m128 p0dA4 = _mm_setr_ps(sign*g.signdDist[0], sign*g.signdDist[1], sign*g.signdDist[2], sign*g.signdDist[0]);
        __m128 ddD4  = _mm_setr_ps(g.DdD[0],   g.DdD[1],   g.DdD[2],   g.DdD[0]);
        __m128 p0dD4 = _mm_setr_ps(g.P0dD[0],  g.P0dD[1],  g.P0dD[2],  g.P0dD[0]);
        __m128 p0dP4 = _mm_setr_ps(g.P0dP0[0], g.P0dP0[1], g.P0dP0[2], g.P0dP0[0]);
        __m128 cos24 = _mm_set1_ps(cos2);

        __m128 c2_4 = _mm_sub_ps(_mm_mul_ps(ddA4, ddA4), _mm_mul_ps(cos24, ddD4));
        __m128 c1_4 = _mm_mul_ps(_mm_set1_ps(2.0f),
                       _mm_sub_ps(_mm_mul_ps(ddA4, p0dA4), _mm_mul_ps(cos24, p0dD4)));
        __m128 c0_4 = _mm_sub_ps(_mm_mul_ps(p0dA4, p0dA4), _mm_mul_ps(cos24, p0dP4));

        __m128 discr4 = _mm_sub_ps(_mm_mul_ps(c1_4, c1_4),
                         _mm_mul_ps(_mm_set1_ps(4.0f), _mm_mul_ps(c2_4, c0_4)));

        // valid: |c2| >= EPSILON && discr >= 0 — only check 3 real edges (lane 3 is dummy)
        __m128 c2abs = _mm_andnot_ps(_mm_set1_ps(-0.0f), c2_4);
        __m128 valid = _mm_and_ps(_mm_cmpge_ps(c2abs, _mm_set1_ps(EPSILON)),
                                   _mm_cmpge_ps(discr4, _mm_setzero_ps()));
        if (!(_mm_movemask_ps(valid) & 0x7)) return false;

        // One sqrt for all 3 edges.
        __m128 sqrtD  = _mm_sqrt_ps(_mm_max_ps(discr4, _mm_setzero_ps()));
        __m128 inv2c2 = _mm_div_ps(_mm_set1_ps(0.5f), c2_4);
        __m128 nc1    = _mm_sub_ps(_mm_setzero_ps(), c1_4);
        __m128 t0_4   = _mm_mul_ps(_mm_sub_ps(nc1, sqrtD), inv2c2);
        __m128 t1_4   = _mm_mul_ps(_mm_add_ps(nc1, sqrtD), inv2c2);

        // check(t): t in [0,1] && p0dA + t*ddA >= 0
        __m128 zero4 = _mm_setzero_ps(), one4 = _mm_set1_ps(1.0f);
        auto hit4 = [&](__m128 t4) {
            __m128 inRange = _mm_and_ps(_mm_cmpge_ps(t4, zero4), _mm_cmple_ps(t4, one4));
            __m128 dir     = _mm_cmpge_ps(_mm_add_ps(p0dA4, _mm_mul_ps(t4, ddA4)), zero4);
            return _mm_movemask_ps(_mm_and_ps(_mm_and_ps(inRange, dir), valid)) & 0x7;
        };
        return (hit4(t0_4) | hit4(t1_4)) != 0;
    }
#else
    for (int e = 0; e < 3; ++e) {
        float ddA   = sign * g.DdA[e];
        float p0dA  = sign * g.signdDist[e];
        float ddD   = g.DdD[e];
        float p0dD  = g.P0dD[e];
        float p0dP0 = g.P0dP0[e];

        float c2 = ddA * ddA - cos2 * ddD;
        float c1 = 2.0f * (ddA * p0dA - cos2 * p0dD);
        float c0 = p0dA * p0dA - cos2 * p0dP0;

        float discr = c1*c1 - 4.0f*c2*c0;
        if (std::fabs(c2) < EPSILON || discr < 0.0f) continue;
        float invD  = 0.5f / c2;
        float sqrtD = std::sqrt(std::max(discr, 0.0f));
        float t0 = (-c1 - sqrtD) * invD;
        float t1 = (-c1 + sqrtD) * invD;

        // direction = ±up → dot(lerp(otv[i0],otv[i1],t), dir) = p0dA + t*ddA
        auto check = [&](float t) {
            return t >= 0.0f && t <= 1.0f && (p0dA + t * ddA) >= 0.0f;
        };
        if (check(t0) || check(t1)) return true;
    }
    return false;
#endif
}

static bool GRCA_GACP_T_IntersectionCheck(bool isFlipped, bool isTriIntersectConeDir,
                                          float3 /*direction*/, float cosHalfAngle,
                                          float rangeMax,
                                          const RawTri& tri, const GrcaTriData& g,
                                          float3 resultPoints[MAX_INTERSECTIONS],
                                          int&  count,
                                          bool& isCheckAll) {
    count = 0;
    isCheckAll = false;

    if (std::fabs(cosHalfAngle) <= EPSILON) {
        bool e01 = (g.signdDist[0] * g.signdDist[1] < 0.0f);
        bool e12 = (g.signdDist[1] * g.signdDist[2] < 0.0f);
        bool e20 = (g.signdDist[2] * g.signdDist[0] < 0.0f);

        if (e01 && e12) {
            resultPoints[0] = GetTriPlaneIntersectPoint(tri.v[0], tri.v[1], g.signdDist[0], g.signdDist[1]);
            resultPoints[1] = GetTriPlaneIntersectPoint(tri.v[1], tri.v[2], g.signdDist[1], g.signdDist[2]);
            return true;
        }
        if (e12 && e20) {
            resultPoints[0] = GetTriPlaneIntersectPoint(tri.v[1], tri.v[2], g.signdDist[1], g.signdDist[2]);
            resultPoints[1] = GetTriPlaneIntersectPoint(tri.v[2], tri.v[0], g.signdDist[2], g.signdDist[0]);
            return true;
        }
        if (e20 && e01) {
            resultPoints[0] = GetTriPlaneIntersectPoint(tri.v[2], tri.v[0], g.signdDist[2], g.signdDist[0]);
            resultPoints[1] = GetTriPlaneIntersectPoint(tri.v[0], tri.v[1], g.signdDist[0], g.signdDist[1]);
            return true;
        }
        return false;
    }

    float cosTheta = cosHalfAngle;
    bool isAllVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] >= cosTheta && -g.signdDistAngle[1] >= cosTheta && -g.signdDistAngle[2] >= cosTheta)
        : ( g.signdDistAngle[0] >= cosTheta &&  g.signdDistAngle[1] >= cosTheta &&  g.signdDistAngle[2] >= cosTheta);
    if (isAllVertexInCone) { return false; }
    bool isNoVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] < cosTheta && -g.signdDistAngle[1] < cosTheta && -g.signdDistAngle[2] < cosTheta)
        : ( g.signdDistAngle[0] < cosTheta &&  g.signdDistAngle[1] < cosTheta &&  g.signdDistAngle[2] < cosTheta);
    isCheckAll = isNoVertexInCone && isTriIntersectConeDir && g.closestVertDist <= rangeMax * cosTheta;
    if (isCheckAll) { return false; }

    float cos2 = cosTheta * cosTheta;
    float sign = isFlipped ? -1.0f : 1.0f;

    // direction = ±up → dot(lerp(otv[i0],otv[i1],t), dir) = p0dA + t*ddA.
#if defined(__SSE4_1__)
    {
        // Compute all 3 edge discriminants and direction checks in one SIMD pass.
        // Lane 3 is a harmless duplicate of lane 0 (pad to 4-wide).
        __m128 ddA4  = _mm_setr_ps(sign*g.DdA[0],       sign*g.DdA[1],       sign*g.DdA[2],       sign*g.DdA[0]);
        __m128 p0dA4 = _mm_setr_ps(sign*g.signdDist[0], sign*g.signdDist[1], sign*g.signdDist[2], sign*g.signdDist[0]);
        __m128 ddD4  = _mm_setr_ps(g.DdD[0],   g.DdD[1],   g.DdD[2],   g.DdD[0]);
        __m128 p0dD4 = _mm_setr_ps(g.P0dD[0],  g.P0dD[1],  g.P0dD[2],  g.P0dD[0]);
        __m128 p0dP4 = _mm_setr_ps(g.P0dP0[0], g.P0dP0[1], g.P0dP0[2], g.P0dP0[0]);
        __m128 cos24 = _mm_set1_ps(cos2);

        __m128 c2_4 = _mm_sub_ps(_mm_mul_ps(ddA4, ddA4), _mm_mul_ps(cos24, ddD4));
        __m128 c1_4 = _mm_mul_ps(_mm_set1_ps(2.0f),
                       _mm_sub_ps(_mm_mul_ps(ddA4, p0dA4), _mm_mul_ps(cos24, p0dD4)));
        __m128 c0_4 = _mm_sub_ps(_mm_mul_ps(p0dA4, p0dA4), _mm_mul_ps(cos24, p0dP4));
        __m128 discr4 = _mm_sub_ps(_mm_mul_ps(c1_4, c1_4),
                         _mm_mul_ps(_mm_set1_ps(4.0f), _mm_mul_ps(c2_4, c0_4)));

        __m128 c2abs = _mm_andnot_ps(_mm_set1_ps(-0.0f), c2_4);
        __m128 valid = _mm_and_ps(_mm_cmpge_ps(c2abs, _mm_set1_ps(EPSILON)),
                                   _mm_cmpge_ps(discr4, _mm_setzero_ps()));
        if (!(_mm_movemask_ps(valid) & 0x7)) return false;

        __m128 sqrtD  = _mm_sqrt_ps(_mm_max_ps(discr4, _mm_setzero_ps()));
        __m128 inv2c2 = _mm_div_ps(_mm_set1_ps(0.5f), c2_4);
        __m128 nc1    = _mm_sub_ps(_mm_setzero_ps(), c1_4);
        __m128 t0_4   = _mm_mul_ps(_mm_sub_ps(nc1, sqrtD), inv2c2);
        __m128 t1_4   = _mm_mul_ps(_mm_add_ps(nc1, sqrtD), inv2c2);

        __m128 zero4 = _mm_setzero_ps(), one4 = _mm_set1_ps(1.0f);
        auto hitMask = [&](__m128 t4) {
            __m128 inRange = _mm_and_ps(_mm_cmpge_ps(t4, zero4), _mm_cmple_ps(t4, one4));
            __m128 dir     = _mm_cmpge_ps(_mm_add_ps(p0dA4, _mm_mul_ps(t4, ddA4)), zero4);
            return _mm_movemask_ps(_mm_and_ps(_mm_and_ps(inRange, dir), valid)) & 0x7;
        };
        int m0 = hitMask(t0_4), m1 = hitMask(t1_4);

        if (__builtin_popcount(m0) + __builtin_popcount(m1) > 2) { isCheckAll = true; return false; }

        // Extract world-space result points for each hit bit (usually 0–2 iterations total).
        alignas(16) float t0a[4], t1a[4];
        _mm_store_ps(t0a, t0_4);
        _mm_store_ps(t1a, t1_4);
        auto addHits = [&](int mask, const float ta[4]) {
            while (mask) {
                int e = __builtin_ctz(mask);
                resultPoints[count++] = lerp(tri.v[edgeIndices[e][0]], tri.v[edgeIndices[e][1]], ta[e]);
                mask &= mask - 1;
            }
        };
        addHits(m0, t0a);
        addHits(m1, t1a);
    }
#else
    for (int e = 0; e < 3; ++e) {
        if (count > 2) break;
        int i0 = edgeIndices[e][0], i1 = edgeIndices[e][1];
        float ddA   = sign * g.DdA[e];
        float p0dA  = sign * g.signdDist[e];
        float ddD   = g.DdD[e];
        float p0dD  = g.P0dD[e];
        float p0dP0 = g.P0dP0[e];

        float c2 = ddA * ddA - cos2 * ddD;
        float c1 = 2.0f * (ddA * p0dA - cos2 * p0dD);
        float c0 = p0dA * p0dA - cos2 * p0dP0;

        float discr = c1*c1 - 4.0f*c2*c0;
        if (std::fabs(c2) < EPSILON || discr < 0.0f) continue;
        float invD  = 0.5f / c2;
        float sqrtD = std::sqrt(std::max(discr, 0.0f));
        float t0 = (-c1 - sqrtD) * invD;
        float t1 = (-c1 + sqrtD) * invD;

        // direction = ±up → dot(lerp(otv[i0],otv[i1],t), dir) = p0dA + t*ddA
        auto addPoint = [&](float t) {
            if (t < 0.0f || t > 1.0f) return;
            if ((p0dA + t * ddA) >= 0.0f)
                resultPoints[count++] = lerp(tri.v[i0], tri.v[i1], t);
        };
        addPoint(t0);
        addPoint(t1);
    }
#endif

    if (count == 1) {
        resultPoints[1] = resultPoints[0];
    } else if (count > 2) {
        isCheckAll = true;
        return false;
    } else if (count == 0) {
        return false;
    }
    return true;
}

// // -------------------------------------------------------------------------
// // Angular channel indexing (vertical / elevation axis)
// // -------------------------------------------------------------------------
// void GRCA_AngularChannelIndexing(float startAngle, float angularStep, uint32_t numSamples,
//                                  float3 planePos, float3 up, float3 checkPos,
//                                  uint32_t& closestRayId) const {
//     float3 dir = normalize(checkPos - planePos);
//     float  angle = std::asin(clamp(dot(dir, up), -1.0f, 1.0f));
//     int    id    = (int)std::round((angle - startAngle) / angularStep);
//     closestRayId = (uint32_t)std::max(0, std::min(id, (int)numSamples - 1));
// }

static uint32_t GRCA_AngularChannelIndexing(const GrcaLidar& L, float sa) {
    float a = std::asin(clamp(sa, -1.0f, 1.0f));
    int   id = (int)std::round((a - L.cfg.vmin) / L.vAngleIncrement);
    return (uint32_t)std::max(0, std::min(id, (int)L.vnum - 1));
}

#ifndef __SSE4_1__
static void GRCA_AngularSweepIndexing(float startAngle, float angularStep, uint32_t numSamples,
                                      float3 planePos, float3 right, float3 forward, float3 up,
                                      float3 checkPos, uint32_t& closestRayId) {
    // Original (for math reference):
    // float3 dir          = normalize(checkPos - planePos);
    // float3 horizontalDir = normalize(dir - dot(dir, up) * up);
    // float  angle = std::atan2(dot(horizontalDir, right), dot(horizontalDir, forward));
    // atan2 is scale-invariant so both normalize calls are unnecessary.
    float3 raw  = checkPos - planePos;
    float3 proj = raw - dot(raw, up) * up;
    float  angle = std::atan2(dot(proj, right), dot(proj, forward));
    int    id    = (int)std::round((angle - startAngle) / angularStep);
    closestRayId = (uint32_t)std::max(0, std::min(id, (int)numSamples - 1));
}
#endif // !__SSE4_1__

static uint32_t GetAngularRayDiff(uint32_t id0, uint32_t id1, uint32_t numSamples, bool isClockWise) {
    uint32_t fwd  = (id1 - id0 + numSamples) % numSamples;
    uint32_t back = (id0 - id1 + numSamples) % numSamples;
    return isClockWise ? fwd : back;
}

static uint32_t GetNextClosestRayId(uint32_t startId, uint32_t next, uint32_t numSamples, bool isClockWise) {
    if (isClockWise) return (startId + next) % numSamples;
    return (startId - next + numSamples) % numSamples;
}

static void GRCA_CSpan_Predict(const GrcaLidar& L, const GrcaTriData& g,
                               uint32_t& fromChannelId, uint32_t& toChannelId) {
    fromChannelId = L.vnum;
    toChannelId   = 0;

    uint32_t estimFrom = 0, estimTo = L.vnum - 1;

    bool allAbove = (g.signdDist[0] > 0.0f && g.signdDist[1] > 0.0f && g.signdDist[2] > 0.0f);
    bool allBelow = (g.signdDist[0] < 0.0f && g.signdDist[1] < 0.0f && g.signdDist[2] < 0.0f);

    if      (allAbove) estimFrom = L.vnum / 2;
    else if (allBelow) estimTo   = L.vnum / 2;

    uint32_t id0 = GRCA_AngularChannelIndexing(L, g.signdDistAngle[0]);
    uint32_t id1 = GRCA_AngularChannelIndexing(L, g.signdDistAngle[1]);
    uint32_t id2 = GRCA_AngularChannelIndexing(L, g.signdDistAngle[2]);

    uint32_t maxId  = std::max(id0, std::max(id1, id2));
    uint32_t minId  = std::min(id0, std::min(id1, id2));
    uint32_t centerVId = (uint32_t)std::ceil(0.5f * ((float)maxId + (float)minId));
    centerVId = std::min(estimTo, centerVId);

    int low  = (int)estimFrom;
    int high = (int)centerVId;
    while (low <= high) {
        int    mid        = low + ((high - low) >> 1);
        bool   isFlipped  = L.chanIsFlipped[mid];
        float  cosHalf    = L.chanCosHalfAngle[mid];
        float3 coneDir    = isFlipped ? -L.pose.up : L.pose.up;
        bool   isMirrored  = isFlipped ? g.isBehindConeFlipped : g.isBehindCone;
        bool   isIntersect = isFlipped ? g.isTriIntersectConeDirFlipped : g.isTriIntersectConeDir;

        if (!isMirrored && GRCA_Fast_GACPT_IntersectionCheck(isFlipped, isIntersect, coneDir, cosHalf, L.cfg.range_max, g)) {
            fromChannelId = (uint32_t)mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    low  = (int)std::min(fromChannelId, estimTo);
    high = (int)estimTo;
    while (low <= high) {
        int    mid        = low + ((high - low) >> 1);
        bool   isFlipped  = L.chanIsFlipped[mid];
        float  cosHalf    = L.chanCosHalfAngle[mid];
        float3 coneDir    = isFlipped ? -L.pose.up : L.pose.up;
        bool   isMirrored  = isFlipped ? g.isBehindConeFlipped : g.isBehindCone;
        bool   isIntersect = isFlipped ? g.isTriIntersectConeDirFlipped : g.isTriIntersectConeDir;

        if (!isMirrored && GRCA_Fast_GACPT_IntersectionCheck(isFlipped, isIntersect, coneDir, cosHalf, L.cfg.range_max, g)) {
            toChannelId = (uint32_t)mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
}

static bool GRCA_RSpan_Predict(const GrcaLidar& L,
                               bool isAllClockWise, bool isFlipped, bool isIntersect,
                               uint32_t channelId, float3 direction, float cosHalfAngle,
                               const RawTri& tri, const std::tuple<float3, float3, float3>& edges_normal, const GrcaTriData& g,
                               uint32_t& fromSweepId, uint32_t& sweepDiff, bool& isClockWise) {
    fromSweepId = 0;
    sweepDiff   = 0;
    isClockWise = true;

    float3 resultPoints[MAX_INTERSECTIONS];
    int  count    = 0;
    bool checkAll = false;

    if (!GRCA_GACP_T_IntersectionCheck(isFlipped, isIntersect, direction, cosHalfAngle,
                                           L.cfg.range_max, tri, g, resultPoints, count, checkAll)) {
        if (checkAll) {
            fromSweepId = 0;
            sweepDiff   = L.hnum;
            isClockWise = true;
            return true;
        }
        return false;
    }

    uint32_t closestRayId0, closestRayId1;
#if defined(__SSE4_1__)
    {
        const float3& pos = L.pose.pos;
        const float3& up  = L.pose.up;
        const float3& rt  = L.pose.right;
        const float3& fwd = L.pose.forward;
        __m128 rx = _mm_setr_ps(resultPoints[0].x - pos.x, resultPoints[1].x - pos.x, 0.f, 0.f);
        __m128 ry = _mm_setr_ps(resultPoints[0].y - pos.y, resultPoints[1].y - pos.y, 0.f, 0.f);
        __m128 rz = _mm_setr_ps(resultPoints[0].z - pos.z, resultPoints[1].z - pos.z, 0.f, 0.f);
        __m128 dru = _mm_add_ps(_mm_add_ps(_mm_mul_ps(rx, _mm_set1_ps(up.x)),
                                            _mm_mul_ps(ry, _mm_set1_ps(up.y))),
                                 _mm_mul_ps(rz, _mm_set1_ps(up.z)));
        __m128 px = _mm_sub_ps(rx, _mm_mul_ps(dru, _mm_set1_ps(up.x)));
        __m128 py = _mm_sub_ps(ry, _mm_mul_ps(dru, _mm_set1_ps(up.y)));
        __m128 pz = _mm_sub_ps(rz, _mm_mul_ps(dru, _mm_set1_ps(up.z)));
        __m128 dpr = _mm_add_ps(_mm_add_ps(_mm_mul_ps(px, _mm_set1_ps(rt.x)),
                                            _mm_mul_ps(py, _mm_set1_ps(rt.y))),
                                 _mm_mul_ps(pz, _mm_set1_ps(rt.z)));
        __m128 dpf = _mm_add_ps(_mm_add_ps(_mm_mul_ps(px, _mm_set1_ps(fwd.x)),
                                            _mm_mul_ps(py, _mm_set1_ps(fwd.y))),
                                 _mm_mul_ps(pz, _mm_set1_ps(fwd.z)));
        alignas(16) float r[4], f[4];
        _mm_store_ps(r, dpr); _mm_store_ps(f, dpf);
        auto toId = [&](float yr, float xf) -> uint32_t {
            int id = (int)std::round((std::atan2(yr, xf) - L.cfg.hmin) / L.hAngleIncrement);
            return (uint32_t)std::max(0, std::min(id, (int)L.hnum - 1));
        };
        closestRayId0 = toId(r[0], f[0]);
        closestRayId1 = toId(r[1], f[1]);
    }
#else
    GRCA_AngularSweepIndexing(L.cfg.hmin, L.hAngleIncrement, L.hnum,
                              L.pose.pos, L.pose.right, L.pose.forward, L.pose.up,
                              resultPoints[0], closestRayId0);
    GRCA_AngularSweepIndexing(L.cfg.hmin, L.hAngleIncrement, L.hnum,
                              L.pose.pos, L.pose.right, L.pose.forward, L.pose.up,
                              resultPoints[1], closestRayId1);
#endif

    fromSweepId        = std::min(closestRayId0, closestRayId1);
    uint32_t toSweepId = std::max(closestRayId0, closestRayId1);

    float vAngle = L.cfg.vmin + channelId * L.vAngleIncrement;

    if (fromSweepId != toSweepId) {
        if (isAllClockWise) {
            uint32_t stepsCW = GetAngularRayDiff(fromSweepId, toSweepId, L.hnum, true);
            sweepDiff   = stepsCW;
            isClockWise = true;
        } else {
            uint32_t stepsCW  = 1 + (toSweepId - fromSweepId + L.hnum) % L.hnum;
            uint32_t midIdCW  = (fromSweepId + stepsCW / 2) % L.hnum;
            float3   midDirCW = ComputeRayDirection(L.pose, L.cfg.hmin + midIdCW * L.hAngleIncrement, vAngle);

            uint32_t stepsCCW = 1 + (fromSweepId - toSweepId + L.hnum) % L.hnum;
            float3 midDirCCW = 2.0f * dot(midDirCW, L.pose.up) * L.pose.up - midDirCW;

            float denom = dot(std::get<2>(edges_normal), midDirCW);
            float dist  = g.distCenterNormal / denom;
            bool isCWHit  = (denom >= EPSILON && dist >= 0.0f);

            denom = dot(std::get<2>(edges_normal), midDirCCW);
            dist  = g.distCenterNormal / denom;
            bool isCCWHit = (denom >= EPSILON && dist >= 0.0f);

            if (isCWHit && !isCCWHit) {
                isClockWise = true;
            } else if (isCCWHit && !isCWHit) {
                isClockWise = false;
            } else {
                float3 ip;
                isClockWise = (M_T_RayTriangleIntersect(L.pose, tri.v[0], std::get<0>(edges_normal), std::get<1>(edges_normal), midDirCW, ip) < FLT_MAX_VAL);
            }
            sweepDiff = isClockWise ? stepsCW - 1 : stepsCCW - 1;
        }
    }
    return true;
}

static void GRCA_Fast_RSpan_Predict(const GrcaLidar& L,
                                    const RawTri& tri, const GrcaTriData& g,
                                    bool& isAllClockWise,
                                    uint32_t& estimFromSweepId, uint32_t& estimSweepDiff,
                                    int satMaxSweepDiff) {
    estimFromSweepId = 0;
    estimSweepDiff   = (uint32_t)satMaxSweepDiff;
    bool isCCWPossible = (L.cfg.hmax - L.cfg.hmin) > PI;
    if (isCCWPossible) {
        int midVId = (int)std::round(L.vnum / 2.0f);
        int midHId = (int)std::round(L.hnum / 2.0f);
        size_t midIndex = (size_t)midVId * L.hnum + (size_t)midHId;
        float3 midDir = L.worldDirs[midIndex];
        float3 worldRef = (std::abs(midDir.y) < 0.9f) ? float3{0.0f, 1.0f, 0.0f} : float3{1.0f, 0.0f, 0.0f};
        float3 up    = normalize(worldRef - dot(worldRef, midDir) * midDir);
        float3 right = normalize(cross(midDir, up));
#if defined(__SSE4_1__)
        (void)tri;  // tri.v[] is only needed in the non-SSE4.1 fallback path below
        // Batch 6 dot products (3x midDir, 3x right) in one SSE pass.
        // originToVert[i] = tri.v[i] - pose.pos (precomputed in GrcaTriData).
        __m128 ox = _mm_setr_ps(g.originToVert[0].x, g.originToVert[1].x, g.originToVert[2].x, 0.f);
        __m128 oy = _mm_setr_ps(g.originToVert[0].y, g.originToVert[1].y, g.originToVert[2].y, 0.f);
        __m128 oz = _mm_setr_ps(g.originToVert[0].z, g.originToVert[1].z, g.originToVert[2].z, 0.f);
        __m128 dFront = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ox, _mm_set1_ps(midDir.x)),
                                               _mm_mul_ps(oy, _mm_set1_ps(midDir.y))),
                                    _mm_mul_ps(oz, _mm_set1_ps(midDir.z)));
        __m128 dRight = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ox, _mm_set1_ps(right.x)),
                                               _mm_mul_ps(oy, _mm_set1_ps(right.y))),
                                    _mm_mul_ps(oz, _mm_set1_ps(right.z)));
        __m128 zero4 = _mm_setzero_ps();
        int frontMask = _mm_movemask_ps(_mm_cmpgt_ps(dFront, zero4)) & 0x7;
        int rightMask = _mm_movemask_ps(_mm_cmpgt_ps(dRight, zero4)) & 0x7;
        int  leftMask = _mm_movemask_ps(_mm_cmplt_ps(dRight, zero4)) & 0x7;
        isAllClockWise = (frontMask == 0x7) || (rightMask == 0x7) || (leftMask == 0x7);
#else
        bool isAllFront = (dot(g.originToVert[0], midDir) > 0.0f &&
                           dot(g.originToVert[1], midDir) > 0.0f &&
                           dot(g.originToVert[2], midDir) > 0.0f);
        bool isAllRight = (dot(g.originToVert[0], right) > 0.0f) && (dot(g.originToVert[1], right) > 0.0f) && (dot(g.originToVert[2], right) > 0.0f);
        bool isAllLeft  = (dot(g.originToVert[0], right) < 0.0f) && (dot(g.originToVert[1], right) < 0.0f) && (dot(g.originToVert[2], right) < 0.0f);
        isAllClockWise = isAllFront || isAllRight || isAllLeft;
#endif
    } else {
        isAllClockWise = true;
    }
    if (!isAllClockWise) return;

    uint32_t hId0, hId1, hId2;
#if defined(__SSE4_1__)
    {
        // Batch all 3 vertex projections in SoA SSE registers.
        // g.originToVert[i] = tri.v[i] - pose.pos (already in GrcaTriData — reuse).
        // Lane layout: [v0, v1, v2, pad].  atan2 is scalar (no HW intrinsic).
        const float3& up  = L.pose.up;
        const float3& rt  = L.pose.right;
        const float3& fwd = L.pose.forward;
        __m128 rx = _mm_setr_ps(g.originToVert[0].x, g.originToVert[1].x, g.originToVert[2].x, 0.f);
        __m128 ry = _mm_setr_ps(g.originToVert[0].y, g.originToVert[1].y, g.originToVert[2].y, 0.f);
        __m128 rz = _mm_setr_ps(g.originToVert[0].z, g.originToVert[1].z, g.originToVert[2].z, 0.f);
        // dot(raw, up) for all 3 vertices
        __m128 dru = _mm_add_ps(_mm_add_ps(_mm_mul_ps(rx, _mm_set1_ps(up.x)),
                                            _mm_mul_ps(ry, _mm_set1_ps(up.y))),
                                 _mm_mul_ps(rz, _mm_set1_ps(up.z)));
        // proj = raw - dru * up
        __m128 px = _mm_sub_ps(rx, _mm_mul_ps(dru, _mm_set1_ps(up.x)));
        __m128 py = _mm_sub_ps(ry, _mm_mul_ps(dru, _mm_set1_ps(up.y)));
        __m128 pz = _mm_sub_ps(rz, _mm_mul_ps(dru, _mm_set1_ps(up.z)));
        // dot(proj, right) and dot(proj, forward) for all 3 vertices
        __m128 dpr = _mm_add_ps(_mm_add_ps(_mm_mul_ps(px, _mm_set1_ps(rt.x)),
                                            _mm_mul_ps(py, _mm_set1_ps(rt.y))),
                                 _mm_mul_ps(pz, _mm_set1_ps(rt.z)));
        __m128 dpf = _mm_add_ps(_mm_add_ps(_mm_mul_ps(px, _mm_set1_ps(fwd.x)),
                                            _mm_mul_ps(py, _mm_set1_ps(fwd.y))),
                                 _mm_mul_ps(pz, _mm_set1_ps(fwd.z)));
        alignas(16) float r[4], f[4];
        _mm_store_ps(r, dpr); _mm_store_ps(f, dpf);
        auto toId = [&](float yr, float xf) -> uint32_t {
            int id = (int)std::round((std::atan2(yr, xf) - L.cfg.hmin) / L.hAngleIncrement);
            return (uint32_t)std::max(0, std::min(id, (int)L.hnum - 1));
        };
        hId0 = toId(r[0], f[0]);
        hId1 = toId(r[1], f[1]);
        hId2 = toId(r[2], f[2]);
    }
#else
    GRCA_AngularSweepIndexing(L.cfg.hmin, L.hAngleIncrement, L.hnum,
                              L.pose.pos, L.pose.right, L.pose.forward, L.pose.up,
                              tri.v[0], hId0);
    GRCA_AngularSweepIndexing(L.cfg.hmin, L.hAngleIncrement, L.hnum,
                              L.pose.pos, L.pose.right, L.pose.forward, L.pose.up,
                              tri.v[1], hId1);
    GRCA_AngularSweepIndexing(L.cfg.hmin, L.hAngleIncrement, L.hnum,
                              L.pose.pos, L.pose.right, L.pose.forward, L.pose.up,
                              tri.v[2], hId2);
#endif
    estimFromSweepId    = std::min(hId0, std::min(hId1, hId2));
    uint32_t estimToSweepId = std::max(hId0, std::max(hId1, hId2));
    estimSweepDiff      = GetAngularRayDiff(estimFromSweepId, estimToSweepId, L.hnum, true);
}

static void GRCA_SAT_BAT(const GrcaLidar& L, const RawTri& tri,
                         const std::tuple<float3, float3, float3>& edges_normal,
                         const GrcaTriData& g, HitPoint* hp, int packet_size,
                         int satMaxChannelDiff, int satMaxSweepDiff, int& rtic_count) {
    uint32_t fromChannelId = g.fromChannelId;
    uint32_t toChannelId   = g.toChannelId;
    if (fromChannelId > toChannelId || fromChannelId == L.vnum) { return; }

    bool     isAllClockWise   = g.isAllClockWise;
    uint32_t estimFromSweepId = g.estimFromSweepId;
    uint32_t estimSweepDiff   = g.estimSweepDiff;

    bool isSAT = isAllClockWise
                 && (toChannelId - fromChannelId) < (uint32_t)satMaxChannelDiff
                 && estimSweepDiff                < (uint32_t)satMaxSweepDiff;

    const float3& e1       = std::get<0>(edges_normal);
    const float3& e2       = std::get<1>(edges_normal);
    const float3& pose_pos = L.pose.pos;

    // Scalar intersection helper — used for wraparound rows and tails.
    auto intersect1 = [&](size_t fi, const float3& dir) {
        float3 ipt;
        float d = M_T_RayTriangleIntersect(L.pose, tri.v[0], e1, e2, dir, ipt);
        if (d < FLT_MAX_VAL && d < hp[fi].dist) { hp[fi].dist = d; hp[fi].pos = ipt; }
    };

#if defined(__AVX2__)
    if (packet_size >= 8) {
        // ---- Precompute per-triangle Möller-Trumbore constants (shared across all rays) --
        // s = pose.pos - v0  (constant for this triangle)
        // sce1 = cross(s, e1)  (constant vector)
        // e2dsc = dot(e2, sce1)  (constant scalar — used as the numerator for t)
        // Result: inv_det * e2dsc gives t directly; cross(s, e1) is never recomputed per ray.
        const float3 s     = pose_pos - tri.v[0];
        const float3 sce1  = cross(s, e1);
        const float  e2dsc = dot(e2, sce1);

        const __m256 e1x8 = _mm256_set1_ps(e1.x),  e1y8 = _mm256_set1_ps(e1.y),  e1z8 = _mm256_set1_ps(e1.z);
        const __m256 e2x8 = _mm256_set1_ps(e2.x),  e2y8 = _mm256_set1_ps(e2.y),  e2z8 = _mm256_set1_ps(e2.z);
        const __m256 sx8  = _mm256_set1_ps(s.x),   sy8  = _mm256_set1_ps(s.y),   sz8  = _mm256_set1_ps(s.z);
        const __m256 sc1x8= _mm256_set1_ps(sce1.x),sc1y8= _mm256_set1_ps(sce1.y),sc1z8= _mm256_set1_ps(sce1.z);
        const __m256 ed8  = _mm256_set1_ps(e2dsc);
        const __m256 eps8 = _mm256_set1_ps(EPSILON);
        const __m256 zer8 = _mm256_setzero_ps();
        const __m256 one8 = _mm256_set1_ps(1.0f);
        // Möller-Trumbore for 8 rays loaded from SoA (non-wrapping, sequential floats).
        // Writes valid hits directly to hp[base + 0..7].
        auto intersect8 = [&](const float* xp, const float* yp, const float* zp, size_t base) {
            __m256 dx = _mm256_loadu_ps(xp), dy = _mm256_loadu_ps(yp), dz = _mm256_loadu_ps(zp);

            // tmp = cross(dir, e2)
            __m256 tx = _mm256_fmsub_ps(dy, e2z8, _mm256_mul_ps(dz, e2y8));
            __m256 ty = _mm256_fmsub_ps(dz, e2x8, _mm256_mul_ps(dx, e2z8));
            __m256 tz = _mm256_fmsub_ps(dx, e2y8, _mm256_mul_ps(dy, e2x8));

            __m256 inv = _mm256_div_ps(one8, dot8(e1x8, e1y8, e1z8, tx, ty, tz));
            __m256 u   = _mm256_mul_ps(inv, dot8(sx8, sy8, sz8, tx, ty, tz));
            __m256 vld = _mm256_and_ps(_mm256_cmp_ps(u, zer8, _CMP_GE_OQ),
                                        _mm256_cmp_ps(u, one8, _CMP_LE_OQ));
            __m256 v   = _mm256_mul_ps(inv, dot8(dx, dy, dz, sc1x8, sc1y8, sc1z8));
            vld        = _mm256_and_ps(vld, _mm256_cmp_ps(v, zer8, _CMP_GE_OQ));
            vld        = _mm256_and_ps(vld, _mm256_cmp_ps(_mm256_add_ps(u, v), one8, _CMP_LE_OQ));
            __m256 t8  = _mm256_mul_ps(inv, ed8);
            vld        = _mm256_and_ps(vld, _mm256_cmp_ps(t8, eps8, _CMP_GT_OQ));

            int bits = _mm256_movemask_ps(vld);
            if (bits) {
                alignas(32) float ta[8], xa[8], ya[8], za[8];
                _mm256_store_ps(ta, t8); _mm256_store_ps(xa, dx);
                _mm256_store_ps(ya, dy); _mm256_store_ps(za, dz);
                while (bits) {
                    int lane = __builtin_ctz(bits);
                    size_t fi = base + lane;
                    float tv = ta[lane];
                    if (tv < hp[fi].dist) {
                        hp[fi].dist = tv;
                        hp[fi].pos  = { pose_pos.x + xa[lane]*tv,
                                        pose_pos.y + ya[lane]*tv,
                                        pose_pos.z + za[lane]*tv };
                    }
                    bits &= bits - 1;
                }
            }
        };

        // 16-wide variant: two independent AVX2 chains interleaved for ILP.
        // Both batches share the same broadcast constants; no data dependency between A and B.
        auto intersect16 = [&](const float* xp, const float* yp, const float* zp, size_t base) {
            __m256 dxA=_mm256_loadu_ps(xp  ), dyA=_mm256_loadu_ps(yp  ), dzA=_mm256_loadu_ps(zp  );
            __m256 dxB=_mm256_loadu_ps(xp+8), dyB=_mm256_loadu_ps(yp+8), dzB=_mm256_loadu_ps(zp+8);

            __m256 txA=_mm256_fmsub_ps(dyA,e2z8,_mm256_mul_ps(dzA,e2y8)), tyA=_mm256_fmsub_ps(dzA,e2x8,_mm256_mul_ps(dxA,e2z8)), tzA=_mm256_fmsub_ps(dxA,e2y8,_mm256_mul_ps(dyA,e2x8));
            __m256 txB=_mm256_fmsub_ps(dyB,e2z8,_mm256_mul_ps(dzB,e2y8)), tyB=_mm256_fmsub_ps(dzB,e2x8,_mm256_mul_ps(dxB,e2z8)), tzB=_mm256_fmsub_ps(dxB,e2y8,_mm256_mul_ps(dyB,e2x8));

            __m256 invA=_mm256_div_ps(one8,dot8(e1x8,e1y8,e1z8,txA,tyA,tzA)), invB=_mm256_div_ps(one8,dot8(e1x8,e1y8,e1z8,txB,tyB,tzB));
            __m256 uA=_mm256_mul_ps(invA,dot8(sx8,sy8,sz8,txA,tyA,tzA)), uB=_mm256_mul_ps(invB,dot8(sx8,sy8,sz8,txB,tyB,tzB));

            __m256 vldA=_mm256_and_ps(_mm256_cmp_ps(uA,zer8,_CMP_GE_OQ),_mm256_cmp_ps(uA,one8,_CMP_LE_OQ));
            __m256 vldB=_mm256_and_ps(_mm256_cmp_ps(uB,zer8,_CMP_GE_OQ),_mm256_cmp_ps(uB,one8,_CMP_LE_OQ));

            __m256 vvA=_mm256_mul_ps(invA,dot8(dxA,dyA,dzA,sc1x8,sc1y8,sc1z8)), vvB=_mm256_mul_ps(invB,dot8(dxB,dyB,dzB,sc1x8,sc1y8,sc1z8));
            vldA=_mm256_and_ps(vldA,_mm256_and_ps(_mm256_cmp_ps(vvA,zer8,_CMP_GE_OQ),_mm256_cmp_ps(_mm256_add_ps(uA,vvA),one8,_CMP_LE_OQ)));
            vldB=_mm256_and_ps(vldB,_mm256_and_ps(_mm256_cmp_ps(vvB,zer8,_CMP_GE_OQ),_mm256_cmp_ps(_mm256_add_ps(uB,vvB),one8,_CMP_LE_OQ)));

            __m256 tA=_mm256_mul_ps(invA,ed8), tB=_mm256_mul_ps(invB,ed8);
            vldA=_mm256_and_ps(vldA,_mm256_cmp_ps(tA,eps8,_CMP_GT_OQ));
            vldB=_mm256_and_ps(vldB,_mm256_cmp_ps(tB,eps8,_CMP_GT_OQ));

            int bA = _mm256_movemask_ps(vldA), bB = _mm256_movemask_ps(vldB);
            if (bA) {
                alignas(32) float ta[8],xa[8],ya[8],za[8];
                _mm256_store_ps(ta,tA); _mm256_store_ps(xa,dxA); _mm256_store_ps(ya,dyA); _mm256_store_ps(za,dzA);
                while (bA) { int l=__builtin_ctz(bA); size_t fi=base+l; float tv=ta[l];
                    if (tv<hp[fi].dist){hp[fi].dist=tv; hp[fi].pos={pose_pos.x+xa[l]*tv,pose_pos.y+ya[l]*tv,pose_pos.z+za[l]*tv};}
                    bA&=bA-1; }
            }
            if (bB) {
                alignas(32) float tb[8],xb[8],yb[8],zb[8];
                _mm256_store_ps(tb,tB); _mm256_store_ps(xb,dxB); _mm256_store_ps(yb,dyB); _mm256_store_ps(zb,dzB);
                while (bB) { int l=__builtin_ctz(bB); size_t fi=base+8+l; float tv=tb[l];
                    if (tv<hp[fi].dist){hp[fi].dist=tv; hp[fi].pos={pose_pos.x+xb[l]*tv,pose_pos.y+yb[l]*tv,pose_pos.z+zb[l]*tv};}
                    bB&=bB-1; }
            }
        };

        // Sweep one channel: non-wrapping → vectorised; wrapping → scalar.
        auto doSweep = [&](uint32_t fromSweep, uint32_t sweepDiff, bool isClockWise, uint32_t ch) {
            const bool     noWrap  = isClockWise ? (fromSweep + sweepDiff < L.hnum)
                                                 : (fromSweep >= sweepDiff);
            const uint32_t minRay  = isClockWise ? fromSweep : (fromSweep - sweepDiff);
            const uint32_t count   = sweepDiff + 1;
            const size_t   row     = (size_t)ch * L.hnum;
            rtic_count += (int)count;

            if (noWrap) {
                const float* xp = L.wdx.data() + row + minRay;
                const float* yp = L.wdy.data() + row + minRay;
                const float* zp = L.wdz.data() + row + minRay;
                uint32_t k = 0;
                if (packet_size >= 16)
                    for (; k + 16 <= count; k += 16) intersect16(xp+k, yp+k, zp+k, row+minRay+k);
                for (; k + 8 <= count; k += 8)  intersect8(xp+k, yp+k, zp+k, row+minRay+k);
                for (; k < count; ++k)           intersect1(row+minRay+k, L.worldDirs[row+minRay+k]);
            } else {
                for (uint32_t i = 0; i <= sweepDiff; ++i) {
                    uint32_t rayId = GetNextClosestRayId(fromSweep, i, L.hnum, isClockWise);
                    intersect1(row + rayId, L.worldDirs[row + rayId]);
                }
            }
        };

        if (isSAT) {
            for (uint32_t ch = fromChannelId; ch <= toChannelId; ++ch)
                doSweep(estimFromSweepId, estimSweepDiff, isAllClockWise, ch);
        } else {
            for (uint32_t ch = fromChannelId; ch <= toChannelId; ++ch) {
                bool   isFlipped   = L.chanIsFlipped[ch];
                float  cosHalf     = L.chanCosHalfAngle[ch];
                float3 dir         = isFlipped ? -L.pose.up : L.pose.up;
                bool   isMirrored  = isFlipped ? g.isBehindConeFlipped : g.isBehindCone;
                bool   isIntersect = isFlipped ? g.isTriIntersectConeDirFlipped : g.isTriIntersectConeDir;
                if (isMirrored) continue;
                uint32_t fromSweepId, sweepDiff;
                bool     isClockWise;
                if (GRCA_RSpan_Predict(L, isAllClockWise, isFlipped, isIntersect, ch,
                                       dir, cosHalf, tri, edges_normal, g,
                                       fromSweepId, sweepDiff, isClockWise))
                    doSweep(fromSweepId, sweepDiff, isClockWise, ch);
            }
        }
        return;
    }
#endif // __AVX2__

#if defined(__SSE4_1__)
    if (packet_size == 4) {
        const float3 s     = pose_pos - tri.v[0];
        const float3 sce1  = cross(s, e1);
        const float  e2dsc = dot(e2, sce1);

        const __m128 e1x4=_mm_set1_ps(e1.x), e1y4=_mm_set1_ps(e1.y), e1z4=_mm_set1_ps(e1.z);
        const __m128 e2x4=_mm_set1_ps(e2.x), e2y4=_mm_set1_ps(e2.y), e2z4=_mm_set1_ps(e2.z);
        const __m128 sx4 =_mm_set1_ps(s.x),  sy4 =_mm_set1_ps(s.y),  sz4 =_mm_set1_ps(s.z);
        const __m128 sc1x4=_mm_set1_ps(sce1.x),sc1y4=_mm_set1_ps(sce1.y),sc1z4=_mm_set1_ps(sce1.z);
        const __m128 ed4=_mm_set1_ps(e2dsc), eps4=_mm_set1_ps(EPSILON);
        const __m128 zer4=_mm_setzero_ps(), one4=_mm_set1_ps(1.0f);

        auto intersect4 = [&](const float* xp, const float* yp, const float* zp, size_t base) {
            __m128 dx=_mm_loadu_ps(xp), dy=_mm_loadu_ps(yp), dz=_mm_loadu_ps(zp);
            __m128 tx=_mm_sub_ps(_mm_mul_ps(dy,e2z4),_mm_mul_ps(dz,e2y4));
            __m128 ty=_mm_sub_ps(_mm_mul_ps(dz,e2x4),_mm_mul_ps(dx,e2z4));
            __m128 tz=_mm_sub_ps(_mm_mul_ps(dx,e2y4),_mm_mul_ps(dy,e2x4));
            __m128 inv=_mm_div_ps(one4, dot4(e1x4,e1y4,e1z4,tx,ty,tz));
            __m128 u  =_mm_mul_ps(inv, dot4(sx4,sy4,sz4,tx,ty,tz));
            __m128 vld=_mm_and_ps(_mm_cmpge_ps(u,zer4),_mm_cmple_ps(u,one4));
            __m128 v  =_mm_mul_ps(inv, dot4(dx,dy,dz,sc1x4,sc1y4,sc1z4));
            vld=_mm_and_ps(vld,_mm_and_ps(_mm_cmpge_ps(v,zer4),_mm_cmple_ps(_mm_add_ps(u,v),one4)));
            __m128 t4 =_mm_mul_ps(inv, ed4);
            vld=_mm_and_ps(vld, _mm_cmpgt_ps(t4, eps4));
            int bits = _mm_movemask_ps(vld);
            if (bits) {
                alignas(16) float ta[4],xa[4],ya[4],za[4];
                _mm_store_ps(ta,t4); _mm_store_ps(xa,dx); _mm_store_ps(ya,dy); _mm_store_ps(za,dz);
                while (bits) {
                    int lane=__builtin_ctz(bits); size_t fi=base+lane; float tv=ta[lane];
                    if (tv<hp[fi].dist){hp[fi].dist=tv; hp[fi].pos={pose_pos.x+xa[lane]*tv,pose_pos.y+ya[lane]*tv,pose_pos.z+za[lane]*tv};}
                    bits&=bits-1;
                }
            }
        };

        auto doSweep = [&](uint32_t fromSweep, uint32_t sweepDiff, bool isClockWise, uint32_t ch) {
            const bool     noWrap = isClockWise ? (fromSweep+sweepDiff<L.hnum) : (fromSweep>=sweepDiff);
            const uint32_t minRay = isClockWise ? fromSweep : (fromSweep-sweepDiff);
            const uint32_t count  = sweepDiff + 1;
            const size_t   row    = (size_t)ch * L.hnum;
            rtic_count += (int)count;
            if (noWrap) {
                const float* xp=L.wdx.data()+row+minRay, *yp=L.wdy.data()+row+minRay, *zp=L.wdz.data()+row+minRay;
                uint32_t k = 0;
                for (; k+4<=count; k+=4) intersect4(xp+k, yp+k, zp+k, row+minRay+k);
                for (; k<count; ++k)     intersect1(row+minRay+k, L.worldDirs[row+minRay+k]);
            } else {
                for (uint32_t i=0; i<=sweepDiff; ++i) {
                    uint32_t rayId=GetNextClosestRayId(fromSweep,i,L.hnum,isClockWise);
                    intersect1(row+rayId, L.worldDirs[row+rayId]);
                }
            }
        };

        if (isSAT) {
            for (uint32_t ch=fromChannelId; ch<=toChannelId; ++ch)
                doSweep(estimFromSweepId, estimSweepDiff, isAllClockWise, ch);
        } else {
            for (uint32_t ch=fromChannelId; ch<=toChannelId; ++ch) {
                bool   isFlipped  =L.chanIsFlipped[ch];
                float  cosHalf    =L.chanCosHalfAngle[ch];
                float3 dir        =isFlipped ? -L.pose.up : L.pose.up;
                bool   isMirrored =isFlipped ? g.isBehindConeFlipped : g.isBehindCone;
                bool   isIntersect=isFlipped ? g.isTriIntersectConeDirFlipped : g.isTriIntersectConeDir;
                if (isMirrored) continue;
                uint32_t fromSweepId, sweepDiff; bool isClockWise;
                if (GRCA_RSpan_Predict(L,isAllClockWise,isFlipped,isIntersect,ch,
                                       dir,cosHalf,tri,edges_normal,g,fromSweepId,sweepDiff,isClockWise))
                    doSweep(fromSweepId, sweepDiff, isClockWise, ch);
            }
        }
        return;
    }
#endif // __SSE4_1__

    // ---- Scalar fallback (packet_size == 1) ----------------------------------------
    if (isSAT) {
        for (uint32_t channelId = fromChannelId; channelId <= toChannelId; ++channelId) {
            rtic_count += (int)(estimSweepDiff + 1);
            for (uint32_t i = 0; i <= estimSweepDiff; ++i) {
                uint32_t rayId = GetNextClosestRayId(estimFromSweepId, i, L.hnum, isAllClockWise);
                size_t flatIdx = (size_t)channelId * L.hnum + rayId;
                intersect1(flatIdx, L.worldDirs[flatIdx]);
            }
        }
    } else {
        for (uint32_t channelId = fromChannelId; channelId <= toChannelId; ++channelId) {
            bool   isFlipped   = L.chanIsFlipped[channelId];
            float  cosHalf     = L.chanCosHalfAngle[channelId];
            float3 dir         = isFlipped ? -L.pose.up : L.pose.up;
            bool   isMirrored  = isFlipped ? g.isBehindConeFlipped : g.isBehindCone;
            bool   isIntersect = isFlipped ? g.isTriIntersectConeDirFlipped : g.isTriIntersectConeDir;
            if (isMirrored) continue;
            uint32_t fromSweepId, sweepDiff;
            bool     isClockWise;
            if (GRCA_RSpan_Predict(L, isAllClockWise, isFlipped, isIntersect, channelId,
                                   dir, cosHalf, tri, edges_normal, g,
                                   fromSweepId, sweepDiff, isClockWise)) {
                rtic_count += (int)(sweepDiff + 1);
                for (uint32_t i = 0; i <= sweepDiff; ++i) {
                    uint32_t rayId = GetNextClosestRayId(fromSweepId, i, L.hnum, isClockWise);
                    size_t flatIdx = (size_t)channelId * L.hnum + rayId;
                    intersect1(flatIdx, L.worldDirs[flatIdx]);
                }
            }
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

CpuState* init_cpu(const RunConfig& r,
                   const std::string& backendstr,
                   FrameTimings& t)
{
    auto tInit = std::chrono::steady_clock::now();
    CpuState* s = new CpuState();
    for (const auto& robot : r.robots) {
        for (const auto& lc : robot.lidars) {
            s->lidars.emplace_back();
            grca_init(s->lidars.back(), lc, lc.w_pose);
            fprintf(stderr, "[%s]  %ux%u rays  pos(%.1f,%.1f,%.1f)\n",
                    backendstr.c_str(),
                    (uint32_t)lc.hnum, (uint32_t)lc.vnum,
                    lc.w_pose.pos.x, lc.w_pose.pos.y, lc.w_pose.pos.z);
        }
    }
    s->packet_size = r.grca_cpu_packet_size;
    t.ms_init_cpu = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tInit).count();
    fprintf(stderr, "[%s] Backend initialized in %.2f ms\n", backendstr.c_str(), t.ms_init_cpu);
    return s;
}

void run_cpu_frame(CpuState*                s,
                   const RunConfig&          r,
                   const std::vector<Tri3>& tris,
                   std::vector<GrcaHit>&     hits,
                   FrameTimings&              t)
{
    auto tTrace = std::chrono::steady_clock::now();

    cpu_update_poses(*s, r);
    cpu_reset_all_hits(*s);

    const size_t nLidars = s->lidars.size();
    if (nLidars == 0) { cpu_fill_hits(*s, r, hits, t); return; }

    // Two-phase GRCA execution
    std::vector<SavedTri>& passedTris = s->passedTris;
    passedTris.clear();
    passedTris.reserve(r.bat_capacity);

    // Early T filter — fused with passedTris packing, dispatched by s->packet_size.
    // Geometry (BuildTriangle) is computed once per SIMD batch; the two old separate
    // passes (filter + pack) are merged to avoid the intermediate trilidarMasks vector.
    {
        const size_t nTris = tris.size();
        const size_t nLi   = std::min(nLidars, (size_t)MAX_ROBOT_LIDAR_COUNT);

        // Scatter the per-lidar pass bit into the 8/4 per-triangle mask accumulators.
        auto scatter_bits = [](uint32_t* masks, int bits, int li) {
            while (bits) { int i = __builtin_ctz(bits); masks[i] |= (1u << li); bits &= bits - 1; }
        };

#if defined(__AVX2__)
        // ---- 16-wide: two independent AVX2 batches per iteration (better ILP) ----------
        if (s->packet_size >= 16) {
            alignas(32) float tmpA[9][8], tmpB[9][8];

            auto load8 = [&](float tmp[][8], size_t base, int count) {
                for (int i = 0; i < count; ++i) {
                    const Tri3& t = tris[base + i];
                    tmp[0][i]=t.v[0][0]; tmp[1][i]=t.v[0][1]; tmp[2][i]=t.v[0][2];
                    tmp[3][i]=t.v[1][0]; tmp[4][i]=t.v[1][1]; tmp[5][i]=t.v[1][2];
                    tmp[6][i]=t.v[2][0]; tmp[7][i]=t.v[2][1]; tmp[8][i]=t.v[2][2];
                }
                for (int i = count; i < 8; ++i)
                    for (int c = 0; c < 9; ++c) tmp[c][i] = tmp[c][0];
            };

            // Full 16-tri batches.
            size_t base = 0;
            for (; base + 16 <= nTris; base += 16) {
                load8(tmpA, base,     8);
                load8(tmpB, base + 8, 8);

                __m256 Acx,Acy,Acz,Anx,Any,Anz,Aarea;
                __m256 Bcx,Bcy,Bcz,Bnx,Bny,Bnz,Barea;
                BuildTriangle8(_mm256_load_ps(tmpA[0]),_mm256_load_ps(tmpA[1]),_mm256_load_ps(tmpA[2]),
                                _mm256_load_ps(tmpA[3]),_mm256_load_ps(tmpA[4]),_mm256_load_ps(tmpA[5]),
                                _mm256_load_ps(tmpA[6]),_mm256_load_ps(tmpA[7]),_mm256_load_ps(tmpA[8]),
                                Acx,Acy,Acz,Anx,Any,Anz,Aarea);
                BuildTriangle8(_mm256_load_ps(tmpB[0]),_mm256_load_ps(tmpB[1]),_mm256_load_ps(tmpB[2]),
                                _mm256_load_ps(tmpB[3]),_mm256_load_ps(tmpB[4]),_mm256_load_ps(tmpB[5]),
                                _mm256_load_ps(tmpB[6]),_mm256_load_ps(tmpB[7]),_mm256_load_ps(tmpB[8]),
                                Bcx,Bcy,Bcz,Bnx,Bny,Bnz,Barea);

                uint32_t maskA[8]={}, maskB[8]={};
                for (size_t li = 0; li < nLi; ++li) {
                    scatter_bits(maskA, EarlyTFilter8(Acx,Acy,Acz,Anx,Any,Anz,Aarea,s->lidars[li].pose),(int)li);
                    scatter_bits(maskB, EarlyTFilter8(Bcx,Bcy,Bcz,Bnx,Bny,Bnz,Barea,s->lidars[li].pose),(int)li);
                }
                for (int i = 0; i < 8; ++i) if (maskA[i]) {
                    SavedTri sv; sv.v[0]={tmpA[0][i],tmpA[1][i],tmpA[2][i]};
                    sv.v[1]={tmpA[3][i],tmpA[4][i],tmpA[5][i]}; sv.v[2]={tmpA[6][i],tmpA[7][i],tmpA[8][i]};
                    sv.lidarMask=maskA[i]; passedTris.push_back(sv);
                }
                for (int i = 0; i < 8; ++i) if (maskB[i]) {
                    SavedTri sv; sv.v[0]={tmpB[0][i],tmpB[1][i],tmpB[2][i]};
                    sv.v[1]={tmpB[3][i],tmpB[4][i],tmpB[5][i]}; sv.v[2]={tmpB[6][i],tmpB[7][i],tmpB[8][i]};
                    sv.lidarMask=maskB[i]; passedTris.push_back(sv);
                }
            }
            // Tail: up to 15 remaining triangles processed 8-at-a-time.
            for (; base < nTris; base += 8) {
                const int count = (int)std::min((size_t)8, nTris - base);
                load8(tmpA, base, count);
                __m256 cx,cy,cz,nx,ny,nz,area;
                BuildTriangle8(_mm256_load_ps(tmpA[0]),_mm256_load_ps(tmpA[1]),_mm256_load_ps(tmpA[2]),
                                _mm256_load_ps(tmpA[3]),_mm256_load_ps(tmpA[4]),_mm256_load_ps(tmpA[5]),
                                _mm256_load_ps(tmpA[6]),_mm256_load_ps(tmpA[7]),_mm256_load_ps(tmpA[8]),
                                cx,cy,cz,nx,ny,nz,area);
                uint32_t masks[8]={};
                for (size_t li = 0; li < nLi; ++li)
                    scatter_bits(masks, EarlyTFilter8(cx,cy,cz,nx,ny,nz,area,s->lidars[li].pose),(int)li);
                for (int i = 0; i < count; ++i) if (masks[i]) {
                    SavedTri sv; sv.v[0]={tmpA[0][i],tmpA[1][i],tmpA[2][i]};
                    sv.v[1]={tmpA[3][i],tmpA[4][i],tmpA[5][i]}; sv.v[2]={tmpA[6][i],tmpA[7][i],tmpA[8][i]};
                    sv.lidarMask=masks[i]; passedTris.push_back(sv);
                }
            }
        }
        // ---- 8-wide AVX2 ---------------------------------------------------------------
        else if (s->packet_size == 8) {
            alignas(32) float tmp[9][8];
            for (size_t base = 0; base < nTris; base += 8) {
                const int count = (int)std::min((size_t)8, nTris - base);
                for (int i = 0; i < count; ++i) {
                    const Tri3& t = tris[base + i];
                    tmp[0][i]=t.v[0][0]; tmp[1][i]=t.v[0][1]; tmp[2][i]=t.v[0][2];
                    tmp[3][i]=t.v[1][0]; tmp[4][i]=t.v[1][1]; tmp[5][i]=t.v[1][2];
                    tmp[6][i]=t.v[2][0]; tmp[7][i]=t.v[2][1]; tmp[8][i]=t.v[2][2];
                }
                for (int i = count; i < 8; ++i)
                    for (int c = 0; c < 9; ++c) tmp[c][i] = tmp[c][0];

                __m256 cx,cy,cz,nx,ny,nz,area;
                BuildTriangle8(_mm256_load_ps(tmp[0]),_mm256_load_ps(tmp[1]),_mm256_load_ps(tmp[2]),
                                _mm256_load_ps(tmp[3]),_mm256_load_ps(tmp[4]),_mm256_load_ps(tmp[5]),
                                _mm256_load_ps(tmp[6]),_mm256_load_ps(tmp[7]),_mm256_load_ps(tmp[8]),
                                cx,cy,cz,nx,ny,nz,area);
                uint32_t masks[8]={};
                for (size_t li = 0; li < nLi; ++li)
                    scatter_bits(masks, EarlyTFilter8(cx,cy,cz,nx,ny,nz,area,s->lidars[li].pose),(int)li);
                for (int i = 0; i < count; ++i) if (masks[i]) {
                    SavedTri sv; sv.v[0]={tmp[0][i],tmp[1][i],tmp[2][i]};
                    sv.v[1]={tmp[3][i],tmp[4][i],tmp[5][i]}; sv.v[2]={tmp[6][i],tmp[7][i],tmp[8][i]};
                    sv.lidarMask=masks[i]; passedTris.push_back(sv);
                }
            }
        }
        else
#endif // __AVX2__
#if defined(__SSE4_1__)
        // ---- 4-wide SSE4.1 -------------------------------------------------------------
        if (s->packet_size == 4) {
            alignas(16) float tmp[9][4];
            for (size_t base = 0; base < nTris; base += 4) {
                const int count = (int)std::min((size_t)4, nTris - base);
                for (int i = 0; i < count; ++i) {
                    const Tri3& t = tris[base + i];
                    tmp[0][i]=t.v[0][0]; tmp[1][i]=t.v[0][1]; tmp[2][i]=t.v[0][2];
                    tmp[3][i]=t.v[1][0]; tmp[4][i]=t.v[1][1]; tmp[5][i]=t.v[1][2];
                    tmp[6][i]=t.v[2][0]; tmp[7][i]=t.v[2][1]; tmp[8][i]=t.v[2][2];
                }
                for (int i = count; i < 4; ++i)
                    for (int c = 0; c < 9; ++c) tmp[c][i] = tmp[c][0];

                __m128 cx,cy,cz,nx,ny,nz,area;
                BuildTriangle4(_mm_load_ps(tmp[0]),_mm_load_ps(tmp[1]),_mm_load_ps(tmp[2]),
                                _mm_load_ps(tmp[3]),_mm_load_ps(tmp[4]),_mm_load_ps(tmp[5]),
                                _mm_load_ps(tmp[6]),_mm_load_ps(tmp[7]),_mm_load_ps(tmp[8]),
                                cx,cy,cz,nx,ny,nz,area);
                uint32_t masks[4]={};
                for (size_t li = 0; li < nLi; ++li)
                    scatter_bits(masks, EarlyTFilter4(cx,cy,cz,nx,ny,nz,area,s->lidars[li].pose),(int)li);
                for (int i = 0; i < count; ++i) if (masks[i]) {
                    SavedTri sv; sv.v[0]={tmp[0][i],tmp[1][i],tmp[2][i]};
                    sv.v[1]={tmp[3][i],tmp[4][i],tmp[5][i]}; sv.v[2]={tmp[6][i],tmp[7][i],tmp[8][i]};
                    sv.lidarMask=masks[i]; passedTris.push_back(sv);
                }
            }
        }
        else
#endif // __SSE4_1__
        // ---- Scalar fallback (packet_size == 1) ----------------------------------------
        {
            for (const auto& raw : tris) {
                float3 v0 = {raw.v[0][0], raw.v[0][1], raw.v[0][2]};
                float3 v1 = {raw.v[1][0], raw.v[1][1], raw.v[1][2]};
                float3 v2 = {raw.v[2][0], raw.v[2][1], raw.v[2][2]};
                Triangle builtTri = BuildTriangle(v0, v1, v2);
                uint32_t mask = 0;
                for (size_t li = 0; li < nLi; ++li)
                    if (GRCA_Early_T_Filter(s->lidars[li].pose, builtTri))
                        mask |= (1u << li);
                if (mask) {
                    SavedTri sv; sv.v[0]=v0; sv.v[1]=v1; sv.v[2]=v2; sv.lidarMask=mask;
                    passedTris.push_back(sv);
                }
            }
        }
    }

    // GRCA Core: per-lidar outer loop so L.worldDirs / L.hitPoints stay hot in cache
    int early_t_count = 0, bat_count = 0, rtic_count = 0;
    bool loggedFirst = false;
    for (size_t li = 0; li < nLidars; ++li) {
        GrcaLidar& L  = s->lidars[li];
        uint32_t bit = 1u << li;
        for (const auto& sv : passedTris) {
            if (!(sv.lidarMask & bit)) continue;
            RawTri tri = {sv.v[0], sv.v[1], sv.v[2]};
            // Compute normal first for cheap range check before the full FillGrcaTriData.
            // This avoids cross+normalize (and all the dot products) for out-of-range triangles.
            const float3 e1 = tri.v[1] - tri.v[0];
            const float3 e2 = tri.v[2] - tri.v[0];
            const float3 n  = normalize(cross(e2, e1));
            auto edges_normal = std::make_tuple(e1, e2, n);
            const float closestDist = ClosestDistanceToTriangle(L.pose.pos, tri.v[0], tri.v[1], tri.v[2], n);
            if (closestDist < L.cfg.range_min || closestDist > L.cfg.range_max) continue;
            GrcaTriData g;
            FillGrcaTriData(tri.v[0], tri.v[1], tri.v[2], e1, e2, n, L, g);
            g.closestVertDist = closestDist;
            if (L.vnum > 1) {
                GRCA_CSpan_Predict(L, g, g.fromChannelId, g.toChannelId);
                if (!loggedFirst) {
                    std::printf("[GRCA] li=%zu first tri cspan: from=%u to=%u (vnum=%u)\n",
                        li, g.fromChannelId, g.toChannelId, L.vnum);
                    loggedFirst = true;
                }
                if (g.fromChannelId > g.toChannelId || g.fromChannelId == L.vnum)
                    continue;
            } else {
                g.fromChannelId = 0;
                g.toChannelId   = 0;
            }
            GRCA_Fast_RSpan_Predict(L, tri, g, g.isAllClockWise, g.estimFromSweepId, g.estimSweepDiff, r.cpu_sat_max_sweep_diff);
            ++early_t_count;
            bool isSAT = g.isAllClockWise
                      && (g.toChannelId - g.fromChannelId) < (uint32_t)r.cpu_sat_max_channel_diff
                      && g.estimSweepDiff                  < (uint32_t)r.cpu_sat_max_sweep_diff;
            if (!isSAT) ++bat_count;
            GRCA_SAT_BAT(L, tri, edges_normal, g, L.hitPoints.data(), s->packet_size, r.cpu_sat_max_channel_diff, r.cpu_sat_max_sweep_diff, rtic_count);
        }
    }
    t.num_bat          = bat_count;
    t.num_early_t      = early_t_count;
    t.num_rtic         = rtic_count;
    t.max_bat_capacity = r.bat_capacity;
    t.ms_trace_cpu = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tTrace).count();

    cpu_fill_hits(*s, r, hits, t);
}

void destroy_cpu(CpuState* s) { delete s; }
