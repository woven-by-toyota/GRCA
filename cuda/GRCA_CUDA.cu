// =============================================================================
// GRCA_CUDA.cu
//
// Ported from:
//   SSP/GeometryPreProcessShader.compute  -> GRCA_Early_Pass
//   SSP/GeometryPostProcessShader.compute -> GRCA_Late_Pass, RayCastOutputAsFloat
//   SSP/SensorUtilityShader.compute       -> InitLidarRays, UpdateLidarRays (SetupLidarRays) 
//
// Excluded: mesh skinning, bone matrices, texture sampling, draw passes.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>   // __half, __float2half, __half2float
#include <math_constants.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Constants (mirrors Common.hlsl)
// ---------------------------------------------------------------------------
#define PI                              3.14159265358979323846f
#define UNITY_HALF_PI                   1.57079632679f
#define FLT_MAX_VAL                     3.402823466e+38f
#define EPSILON                         0.0001f
#define MAX_ROBOT_COUNT                 8
#define MAX_LIDAR_COUNT                 4
#define MAX_ROBOT_LIDAR_COUNT           32   // MAX_ROBOT_COUNT * MAX_LIDAR_COUNT
#define MAX_ROBOT_LIDAR_COUNT_HALF      16
#define MAX_REFLECTIVE_OBJ_TYPE         3
#define MAX_LIDAR_RAY_COUNT             524288
#define MAX_TRI_RAY_COUNT               800000
#define PER_THREAD_RAY_BATCH            64
#define MIN_APPARENT_AREA               0.000001f
#define MIN16UINT_MAX                   65535u

// GRCA_Early_Pass flag bit positions
#define GRCA_FLAGS_IS_BEHIND_CONE                0
#define GRCA_FLAGS_IS_BEHIND_CONE_FLIPPED        1
#define GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR     2
#define GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR_FL  3

// RayOrigin.flags bit positions
#define RAYORIGIN_FLAG_CCW_POSSIBLE             28

// edge index table (constant memory)
__constant__ int edgeIndices[3][2] = { {0, 1}, {1, 2}, {2, 0} };

// ---------------------------------------------------------------------------cuda_sat_max_sweep_diff
// Float3 math helpers  (CUDA float3 is a POD struct with .x .y .z)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 make_f3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_f3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_f3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 operator*(float s, float3 a) { return make_f3(s*a.x, s*a.y, s*a.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return s * a; }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return make_f3(a.x/s, a.y/s, a.z/s); }
__device__ __forceinline__ float3 neg(float3 a) { return make_f3(-a.x, -a.y, -a.z); }

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_f3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ __forceinline__ float3 normalize(float3 v) {
    float inv = rsqrtf(dot(v, v));
    return v * inv;
}
__device__ __forceinline__ float3 lerp(float3 a, float3 b, float t) {
    return a + (t * (b - a));
}
__device__ __forceinline__ float saturate(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}
__device__ __forceinline__ float3 zerof3() { return make_f3(0,0,0); }

// ---------------------------------------------------------------------------
// Bit flag helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool Is32BitSet(unsigned int flags, int pos) {
    return (flags & (1u << pos)) != 0u;
}
__device__ __forceinline__ void Set32BitFlag(unsigned int& flags, int pos) {
    flags |= (1u << pos);
}

// ---------------------------------------------------------------------------
// Pack / Unpack 16-bit pairs into a 32-bit uint
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int PackTwo16Bits(unsigned int a, unsigned int b) {
    return (b << 16) | (a & 0xFFFFu);
}
__device__ __forceinline__ unsigned int Unpack16BitsLower(unsigned int packed) {
    return packed & 0xFFFFu;
}
__device__ __forceinline__ unsigned int Unpack16BitsUpper(unsigned int packed) {
    return (packed >> 16) & 0xFFFFu;
}

// ---------------------------------------------------------------------------
// f16 <-> f32 packing (mirrors HLSL f32tof16 / f16tof32)
// Stored in lower or upper 16 bits of a uint32.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int f32tof16(float v) {
    __half h = __float2half(v);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(uint16_t));
    return (unsigned int)bits;
}
__device__ __forceinline__ float f16tof32(unsigned int bits16) {
    uint16_t b = (uint16_t)(bits16 & 0xFFFFu);
    __half h;
    memcpy(&h, &b, sizeof(uint16_t));
    return __half2float(h);
}

// ---------------------------------------------------------------------------
// Sortable float encoding (mirrors asUINT / asFLOAT from Common.hlsl)
// Used for atomic min on range values.
// Reference: https://www.jeremyong.com/graphics/2023/09/05/f32-interlocked-min-max-hlsl/
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int asUINT_sortable(float value) {
    unsigned int uv = __float_as_uint(value);
    unsigned int mask = (unsigned int)(-(int)(uv >> 31)) | 0x80000000u;
    return uv ^ mask;
}
__device__ __forceinline__ float asFLOAT_sortable(unsigned int value) {
    unsigned int mask = ((value >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(value ^ mask);
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection (mirrors Common.hlsl)
// edge1 = v1-v0, edge2 = v2-v0  (standard convention).
// Returns distance or FLT_MAX_VAL if no hit.
// ---------------------------------------------------------------------------
__device__ float RayTriangleIntersect(float3 direction, float3 origin,
                                       float3 v0, float3 edge1, float3 edge2) {
    float3 tmp = cross(direction, edge2);
    float inv_det = 1.0f / dot(edge1, tmp);

    float3 s = origin - v0;
    float u = inv_det * dot(s, tmp);
    if (u < 0.0f || u > 1.0f) return FLT_MAX_VAL;

    tmp = cross(s, edge1);
    float v = inv_det * dot(direction, tmp);
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_VAL;

    float t = inv_det * dot(edge2, tmp);
    return (t > EPSILON) ? t : FLT_MAX_VAL;
}

// ---------------------------------------------------------------------------
// Angular indexing helpers (mirrors Common.hlsl)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int GetAngularRayDiff(unsigned int id0, unsigned int id1,
                                                           unsigned int numSamples, bool isClockWise) {
    unsigned int fwd  = (id1 - id0 + numSamples) % numSamples;
    unsigned int back = (id0 - id1 + numSamples) % numSamples;
    return isClockWise ? fwd : back;
}

__device__ __forceinline__ unsigned int GetNextClosestRayId(unsigned int startId, unsigned int next,
                                                             unsigned int numSamples, bool isClockWise) {
    if (isClockWise) return (startId + next) % numSamples;
    else             return (startId - next + numSamples) % numSamples;
}

// Takes pre-computed sine of elevation angle (signdDistAngle component).
// Old: normalize(checkPos - planePos) -> dot(dir, up) -> asinf
__device__ unsigned int GRCA_AngularChannelIndexing(float sa, float startAngle, float angularStep, unsigned int numSamples) {
    float angle = asinf(sa);
    int id = (int)roundf((angle - startAngle) / angularStep);
    return (unsigned int)max(0, min(id, (int)numSamples - 1));
}

// Projects a 3-D point onto the horizontal sweep axis and returns the
// nearest ray index.  No noise — one-to-one match with the non-noisy path
// from Common.hlsl / GeometryPreProcessShader.compute.
__device__ void GRCA_AngularSweepIndexing(float startAngle, float angularStep, unsigned int numSamples,
                                          float3 planePos, float3 right, float3 forward, float3 up,
                                          float3 checkPos,
                                          unsigned int& closestRayId) {
    // Original (for math reference):
    // float3 dir = normalize(checkPos - planePos);
    // dir = normalize(dir - dot(dir, up) * up);
    // float angle = atan2f(dot(dir, right), dot(dir, forward));
    // atan2 is scale-invariant so both normalize calls are unnecessary.
    float3 raw  = checkPos - planePos;
    float3 proj = raw - dot(raw, up) * up;
    float angle = atan2f(dot(proj, right), dot(proj, forward));
    int id = (int)roundf((angle - startAngle) / angularStep);
    closestRayId = (unsigned int)max(0, min(id, (int)numSamples - 1));
}

// ---------------------------------------------------------------------------
// Closest point on triangle to P (mirrors GeometryPreProcessShader.compute)
// Convex edge test then edge-clamped distance.
// ---------------------------------------------------------------------------
__device__ float ClosestDistanceToTriangle(float3 v0, float3 v1, float3 v2, float3 n, float3 P) {
    float planeDist = dot(P - v0, n);
    float3 proj = P - planeDist * n;

    // Precompute all 3 edges and toP = proj - A (reused in both inside test and outside case).
    // Key identity: dot(P-A, edge) = dot(toP, edge)  (planeDist*n ⊥ edge)
    // and |P - closest|² = |toP - t·edge|² + planeDist²
    const float3 verts[3] = { v0, v1, v2 };
    float3 edges[3], toPs[3];
    for (int e = 0; e < 3; ++e) {
        edges[e] = verts[(e + 1) % 3] - verts[e];
        toPs[e]  = proj - verts[e];
    }

    bool isInside = true;
    for (int e = 0; e < 3 && isInside; ++e) {
        if (dot(cross(edges[e], toPs[e]), n) < 0.0f) isInside = false;
    }
    if (isInside) return fabsf(planeDist);

    float minDistSq = FLT_MAX_VAL;
    float pd2 = planeDist * planeDist;
    for (int e = 0; e < 3; ++e) {
        float  t     = saturate(dot(toPs[e], edges[e]) / dot(edges[e], edges[e]));
        float3 delta = toPs[e] - t * edges[e];
        minDistSq = fminf(minDistSq, dot(delta, delta) + pd2);
    }
    return sqrtf(minDistSq);
}

// ---------------------------------------------------------------------------
// Early-pass triangle filter (mirrors GRCA_Early_T_Filter)
// ---------------------------------------------------------------------------
__device__ bool GRCA_Early_T_Filter(float3 v0, float3 v1, float3 v2, float3 normal, float3 center,
                                    float area, float3 origin, float minRange, float maxRange,
                                    float& closestVertDist) {
    closestVertDist = FLT_MAX_VAL;
    // reject degenerate (origin on the triangle plane)
    if (fabsf(dot(normal, origin - center)) < EPSILON) return false;

    float3 toCenter = center - origin;
    float dotNC = dot(toCenter, normal);
    if (dotNC <= 0.0f) return false;

    float distSqr = fmaxf(dot(toCenter, toCenter), EPSILON);
    float lhs = area * dotNC;
    if (lhs * lhs < MIN_APPARENT_AREA * MIN_APPARENT_AREA * distSqr * distSqr * distSqr) return false;

    closestVertDist = ClosestDistanceToTriangle(v0, v1, v2, normal, origin);
    if (closestVertDist < minRange || closestVertDist > maxRange) return false;
    return true;
}

// ---------------------------------------------------------------------------
// GRCA_Fast_GACP_T_IntersectionCheck (early pass — local data, no shared mem)
// ---------------------------------------------------------------------------
__device__ bool GRCA_Fast_GACP_T_IntersectionCheck(
    float signdDist[3], float signdDistAngle[3],
    float3 originToVert[3], float3 triVerts[3],
    float3 up, float3 position,
    bool isFlipped, bool isTriIntersectConeDir,
    float3 direction, float cosHalfAngle, int rayOrigflagOffset,
    float closestVertDist, float maxRange)
{
    if (fabsf(cosHalfAngle) <= EPSILON) {
        // flat-plane case
        bool e01 = (signdDist[0] * signdDist[1] < 0.0f);
        bool e12 = (signdDist[1] * signdDist[2] < 0.0f);
        bool e20 = (signdDist[2] * signdDist[0] < 0.0f);
        return (e01 && e12) || (e12 && e20) || (e20 && e01);
    }

    float cosTheta = cosHalfAngle;
    bool isAllVertexInCone = isFlipped
        ? (-signdDistAngle[0] >= cosTheta && -signdDistAngle[1] >= cosTheta && -signdDistAngle[2] >= cosTheta)
        : ( signdDistAngle[0] >= cosTheta &&  signdDistAngle[1] >= cosTheta &&  signdDistAngle[2] >= cosTheta);
    if (isAllVertexInCone) { return false; }
    bool isNoVertexInCone = isFlipped
        ? (-signdDistAngle[0] < cosTheta && -signdDistAngle[1] < cosTheta && -signdDistAngle[2] < cosTheta)
        : (signdDistAngle[0] < cosTheta  &&  signdDistAngle[1] < cosTheta &&  signdDistAngle[2] < cosTheta);
    if (!isNoVertexInCone) { return true; } // at least one vertex in cone volume → guaranteed edge crossing
    bool isCheckAll = isNoVertexInCone && isTriIntersectConeDir && closestVertDist <= maxRange * cosTheta;
    if (isCheckAll) return true;

    float cos2 = cosTheta * cosTheta;
    // float3 apex = position; // removed unused variable

    for (int e = 0; e < 3; ++e) {
        int i0 = edgeIndices[e][0];
        int i1 = edgeIndices[e][1];

        // D = edge vector from v[i0] to v[i1]
        float3 D  = originToVert[i1] - originToVert[i0];
        float DdD = dot(D, D);
        float DdA = dot(triVerts[i1], up) - dot(triVerts[i0], up);
        float P0dA = dot(triVerts[i0], up) - dot(position, up);
        float P0dD = dot(originToVert[i0], D);
        float P0dP0 = dot(originToVert[i0], originToVert[i0]);

        // c2/c1/c0 are identical for flipped and non-flipped ((-x)²=x²).
        // Only the direction check changes sign.
        float c2 = DdA * DdA - cos2 * DdD;
        float c1 = 2.0f * (DdA * P0dA - cos2 * P0dD);
        float c0 = P0dA * P0dA - cos2 * P0dP0;

        float discr = c1 * c1 - 4.0f * c2 * c0;
        if (fabsf(c2) < EPSILON || discr < 0.0f) continue;

        discr = fmaxf(discr, 0.0f);
        float invDenom = 0.5f / c2;
        float sqrtD    = sqrtf(discr);
        float t0 = (-c1 - sqrtD) * invDenom;
        float t1 = (-c1 + sqrtD) * invDenom;

        // direction = ±up → dot(lerp(v[i0],v[i1],t) - apex, dir) = sign*(P0dA + t*DdA)
        // Original: float3 pt = lerp(triVerts[i0], triVerts[i1], t); if (dot(pt - apex, direction) >= 0) return true;
        float sign = isFlipped ? -1.0f : 1.0f;
        if (t0 >= 0.0f && t0 <= 1.0f && sign * (P0dA + t0 * DdA) >= 0.0f) return true;
        if (t1 >= 0.0f && t1 <= 1.0f && sign * (P0dA + t1 * DdA) >= 0.0f) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Data structures (mirrors Common.hlsl + per-shader local structs)
// ---------------------------------------------------------------------------

struct RayOrigin {
    float3 position;
    float3 forward;
    float3 up;
    float3 right;
    float  hAngularStep;
    float  hStartAngle;
    float  hEndAngle;
    float  vAngularStep;
    float  vStartAngle;
    float  vEndAngle;
    float  minRange;
    float  maxRange;
    unsigned int contiguousGlobalRayIdOffset;
    unsigned int localRayIdOffset;
    unsigned int hNumSamples;
    unsigned int vNumSamples;
    unsigned int flags;         // bit28=isCCWPossible
    float3 hitPointColor;
};

struct RayOriginData {
    unsigned int rayOriginCount;
    unsigned int totalRayCount;
    RayOrigin    rayOrigin[MAX_LIDAR_COUNT];
};

struct RayData {
    float3       direction;
    unsigned int rangeAsUint;
};

struct TriRay {
    unsigned int triIndex;
    unsigned int rayOriginFlags;
    unsigned int triConeApexHit_P_Flags;
    unsigned int triConeApexHit_N_Flags;
    unsigned int isAllCWFlags;
    unsigned int packd_closestVertDist[MAX_ROBOT_LIDAR_COUNT_HALF];
    unsigned int vFromIdToId[MAX_ROBOT_LIDAR_COUNT];
};

struct UniformData {
    unsigned int rayCastDispatchGroupX;
    unsigned int rayCastDispatchGroupY;
    unsigned int rayCastDispatchGroupZ;
    unsigned int rayCastDispatchTriangleCount;  // BAT triangles (sent to Late Pass)
    unsigned int rayCastMaxChannelDiff;
    unsigned int rayCastMaxSweepDiff;
    unsigned int batCapacity;
};

struct RayCastOutput {
    float3 direction;
    float  distance;
};

// Vertex layout: tightly packed float3 world position (x, y, z)
#define VERTEX_STRIDE 12u
__device__ __forceinline__ float3 LoadWorldPos(unsigned int v, const unsigned char* vertexBuffer) {
    const float* p = (const float*)(vertexBuffer + v * VERTEX_STRIDE);
    return make_f3(p[0], p[1], p[2]);
}

// ---------------------------------------------------------------------------
// Shared-memory structs for GRCA_Early_Pass
// ---------------------------------------------------------------------------
struct RayOriginCache {
    float3       position, up, right, forward;
    unsigned int vNumSamples, hNumSamples, localRayIdOffset;
    float        hStartAngle, hAngularStep;
    float        vStartAngle, vAngularStep;
    float        minRange, maxRange;
    unsigned int flags;
};

struct TriTemp {
    float3 verts[3];
    float3 normal;
};

// ---------------------------------------------------------------------------
// Shared-memory structs for GRCA_Late_Pass
// ---------------------------------------------------------------------------
struct TriRayLimit {
    unsigned int hFromIdSweepDiff;
    unsigned int flags;
};

struct RayCastDataSh {
    float3       triVerts[3];
    float3       pos;
    float3       edge1, edge2;
    float        closestVertDist;
    float        maxRange;
    unsigned int reflectiveObjType;
    unsigned int flags;
    unsigned int globalRayIdOffset;
    unsigned int hNumSamples;
    unsigned int vFromId, vToId;
};

struct GRCARspanDataSh {
    float3       toCenter, normal, up;
    float3       midRayDirCW, midRayDirCCW;
    float        denom, distCW, distCCW;
    bool         isCWHit, isCCWHit;
    unsigned int stepsCW, midRayIdCW, stepsCCW;
};

struct GRCAIndexDataSh {
    float3       up, right, forward;
    float        hStartAngle, hAngularStep;
    float        vStartAngle, vAngularStep;
    unsigned int closestRayIds[2];
    unsigned int flags;
};

struct GRCAGacpDataSh {
    float  DdD[3], DdA[3], P0dA[3], P0dD[3], P0dP0[3];
    float  signdDist[3];
    float3 temp;
    float  cosTheta;
    bool   isAllVertexInCone;
    bool   isNoVertexInCone;
    float  cos2;
    int    count, i0, i1, e;
    float  c0, c1, c2, discr, invDenom, sqrtD, t0, t1;
    float  signdDistAngle[3];
    bool   isTriIntersectConeDir, isTriIntersectConeDirFlipped;
    bool   edge01Intersects, edge12Intersects, edge20Intersects;
};

// ---------------------------------------------------------------------------
// GRCA_GACP_T_IntersectionCheck for the Late Pass (uses shared memory structs)
// ---------------------------------------------------------------------------
__device__ bool GRCA_GACP_T_IntersectionCheck(
    bool isFlipped, bool isTriIntersectConeDir,
    float3 direction, float cosHalfAngle, int shId,
    float3 resultPoints[3], bool& isCheckAll,
    RayCastDataSh*   rcSh,
    GRCAGacpDataSh*   gacpSh,
    GRCAIndexDataSh*  indexSh)
{
    isCheckAll = false;
    GRCAGacpDataSh& g = gacpSh[shId];
    RayCastDataSh& r = rcSh[shId];

    if (fabsf(cosHalfAngle) <= EPSILON) {
        g.edge01Intersects = (g.signdDist[0] * g.signdDist[1] < 0.0f);
        g.edge12Intersects = (g.signdDist[1] * g.signdDist[2] < 0.0f);
        g.edge20Intersects = (g.signdDist[2] * g.signdDist[0] < 0.0f);

        auto planeIntersect = [&](float3& va, float3& vb, float da, float db) -> float3 {
            return va + (da / (da - db)) * (vb - va);
        };
        if (g.edge01Intersects && g.edge12Intersects) {
            resultPoints[0] = planeIntersect(r.triVerts[0], r.triVerts[1], g.signdDist[0], g.signdDist[1]);
            resultPoints[1] = planeIntersect(r.triVerts[1], r.triVerts[2], g.signdDist[1], g.signdDist[2]);
            return true;
        }
        if (g.edge12Intersects && g.edge20Intersects) {
            resultPoints[0] = planeIntersect(r.triVerts[1], r.triVerts[2], g.signdDist[1], g.signdDist[2]);
            resultPoints[1] = planeIntersect(r.triVerts[2], r.triVerts[0], g.signdDist[2], g.signdDist[0]);
            return true;
        }
        if (g.edge20Intersects && g.edge01Intersects) {
            resultPoints[0] = planeIntersect(r.triVerts[2], r.triVerts[0], g.signdDist[2], g.signdDist[0]);
            resultPoints[1] = planeIntersect(r.triVerts[0], r.triVerts[1], g.signdDist[0], g.signdDist[1]);
            return true;
        }
        return false;
    }

    // Cone-edge intersection (GWT reference)
    g.cosTheta = cosHalfAngle;
    g.isAllVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] >= g.cosTheta && -g.signdDistAngle[1] >= g.cosTheta && -g.signdDistAngle[2] >= g.cosTheta)
        : ( g.signdDistAngle[0] >= g.cosTheta &&  g.signdDistAngle[1] >= g.cosTheta &&  g.signdDistAngle[2] >= g.cosTheta);
    if (g.isAllVertexInCone) { return false; }
    g.isNoVertexInCone = isFlipped
        ? (-g.signdDistAngle[0] < g.cosTheta && -g.signdDistAngle[1] < g.cosTheta && -g.signdDistAngle[2] < g.cosTheta)
        : ( g.signdDistAngle[0] < g.cosTheta &&  g.signdDistAngle[1] < g.cosTheta &&  g.signdDistAngle[2] < g.cosTheta);
    isCheckAll = g.isNoVertexInCone && isTriIntersectConeDir && r.closestVertDist <= r.maxRange * g.cosTheta;
    if (isCheckAll) return false;

    g.cos2    = g.cosTheta * g.cosTheta;
    g.count   = 0;
    g.e       = 0;
    g.c0 = g.c1 = g.c2 = g.discr = g.invDenom = g.sqrtD = g.t0 = g.t1 = 0.0f;

    for (g.e = 0; g.e < 3; ++g.e) {
        if (g.count > 2) break;
        g.i0 = edgeIndices[g.e][0];
        g.i1 = edgeIndices[g.e][1];

        // c2/c1/c0 are identical for flipped and non-flipped ((-x)²=x²).
        // Only the direction check changes sign.
        g.c2 = g.DdA[g.e]  * g.DdA[g.e]  - g.cos2 * g.DdD[g.e];
        g.c1 = 2.0f * (g.DdA[g.e] * g.P0dA[g.e] - g.cos2 * g.P0dD[g.e]);
        g.c0 = g.P0dA[g.e] * g.P0dA[g.e] - g.cos2 * g.P0dP0[g.e];

        g.discr = g.c1 * g.c1 - 4.0f * g.c2 * g.c0;
        if (fabsf(g.c2) < EPSILON || g.discr < 0.0f) continue;

        g.discr    = fmaxf(g.discr, 0.0f);
        g.invDenom = 0.5f / g.c2;
        g.sqrtD    = sqrtf(g.discr);
        g.t0       = (-g.c1 - g.sqrtD) * g.invDenom;
        g.t1       = (-g.c1 + g.sqrtD) * g.invDenom;

        // direction = ±up → dot(lerp(v[i0],v[i1],t) - pos, dir) = sign*(P0dA + t*DdA)
        // Original: g.temp = lerp(...); if (dot(g.temp - r.pos, direction) >= 0) resultPoints[g.count++] = g.temp;
        float sP0dA = isFlipped ? -g.P0dA[g.e] : g.P0dA[g.e];
        float sDdA  = isFlipped ? -g.DdA[g.e]  : g.DdA[g.e];
        if (g.t0 >= 0.0f && g.t0 <= 1.0f && (sP0dA + g.t0 * sDdA) >= 0.0f)
            resultPoints[g.count++] = lerp(r.triVerts[g.i0], r.triVerts[g.i1], g.t0);
        if (g.t1 >= 0.0f && g.t1 <= 1.0f && (sP0dA + g.t1 * sDdA) >= 0.0f)
            resultPoints[g.count++] = lerp(r.triVerts[g.i0], r.triVerts[g.i1], g.t1);
    }
    if (g.count == 1)       { resultPoints[1] = resultPoints[0]; return true; }
    if (g.count > 2)        { isCheckAll = true; return false; }
    if (g.count == 0)       { return false; }
    return true;
}

// ---------------------------------------------------------------------------
// GRCA_RSpan_Predict (Late Pass)
// ---------------------------------------------------------------------------
__device__ void GRCA_RSpan_Predict(
    int shId, bool isAllClockWise, bool isCCWPossible,
    unsigned int fromSweepId, unsigned int toSweepId,
    unsigned int rayIdOffset, float distCenterNormal,
    bool& isClockWise, unsigned int& sweepDiff,
    RayCastDataSh*  rcSh, GRCARspanDataSh* rspanSh,
    const RayData*  rayDataBuf)
{
    GRCARspanDataSh& rs = rspanSh[shId];
    RayCastDataSh&  r  = rcSh[shId];

    if (fromSweepId != toSweepId) {
        if (isAllClockWise) {
            rs.stepsCW = GetAngularRayDiff(fromSweepId, toSweepId, r.hNumSamples, true);
            sweepDiff  = rs.stepsCW;
            isClockWise = true;
        } else {
            rs.stepsCW     = 1 + (toSweepId   - fromSweepId + r.hNumSamples) % r.hNumSamples;
            rs.midRayIdCW  = (fromSweepId + rs.stepsCW / 2) % r.hNumSamples;
            rs.midRayDirCW = rayDataBuf[rs.midRayIdCW + rayIdOffset].direction;
            rs.stepsCCW    = 1 + (fromSweepId - toSweepId   + r.hNumSamples) % r.hNumSamples;
            // mirror the CW mid-ray across the up-plane
            rs.midRayDirCCW = 2.0f * dot(rs.midRayDirCW, rs.up) * rs.up - rs.midRayDirCW;

            rs.denom   = dot(rs.normal, rs.midRayDirCW);
            rs.distCW  = distCenterNormal / rs.denom;
            rs.isCWHit = (rs.denom >= EPSILON) && (rs.distCW >= 0.0f);

            rs.denom    = dot(rs.normal, rs.midRayDirCCW);
            rs.distCCW  = distCenterNormal / rs.denom;
            rs.isCCWHit = (rs.denom >= EPSILON) && (rs.distCCW >= 0.0f);

            if       ( rs.isCWHit && !rs.isCCWHit)  isClockWise = true;
            else if  (!rs.isCWHit &&  rs.isCCWHit)  isClockWise = false;
            else     isClockWise = (RayTriangleIntersect(rs.midRayDirCW, r.pos,
                                        r.triVerts[0], r.edge1, r.edge2) < FLT_MAX_VAL);

            sweepDiff = isClockWise ? rs.stepsCW - 1 : rs.stepsCCW - 1;
        }
    } else {
        isClockWise = true;
        sweepDiff   = 0;
    }
}

// ---------------------------------------------------------------------------
// GRCA_CSpan_Predict (Early Pass — mirrors GeometryPreProcessShader.compute)
// Returns false if there is no channel overlap (triangle should be skipped).
// ---------------------------------------------------------------------------
__device__ bool GRCA_CSpan_Predict(
    float signdDist[3], float signdDistAngle[3],
    float3 originToVert[3], float3 triVerts[3],
    unsigned int& grcaFlags, bool isTICD, bool isTICDF,
    const RayOriginCache& oc,
    unsigned int estimFromChannelId, unsigned int estimToChannelId,
    unsigned int& fromChannelId, unsigned int& toChannelId,
    float closestVertDist)
{
    // All-above / all-below half-plane classification
    bool above0 = signdDist[0] > 0.0f;
    bool above1 = signdDist[1] > 0.0f;
    bool above2 = signdDist[2] > 0.0f;
    if  ( above0 &&  above1 &&  above2) estimFromChannelId = oc.vNumSamples / 2;
    else if (!above0 && !above1 && !above2) estimToChannelId  = oc.vNumSamples / 2;

    bool isBehindCone        = signdDist[0] < 0.0f && signdDist[1] < 0.0f && signdDist[2] < 0.0f;
    bool isBehindConeFlipped = -signdDist[0] < 0.0f && -signdDist[1] < 0.0f && -signdDist[2] < 0.0f;
    if (isBehindCone)        Set32BitFlag(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE);
    if (isBehindConeFlipped) Set32BitFlag(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE_FLIPPED);

    unsigned int id0 = GRCA_AngularChannelIndexing(signdDistAngle[0], oc.vStartAngle, oc.vAngularStep, oc.vNumSamples);
    unsigned int id1 = GRCA_AngularChannelIndexing(signdDistAngle[1], oc.vStartAngle, oc.vAngularStep, oc.vNumSamples);
    unsigned int id2 = GRCA_AngularChannelIndexing(signdDistAngle[2], oc.vStartAngle, oc.vAngularStep, oc.vNumSamples);
    unsigned int closestCenterVId =
        (unsigned int)ceilf(0.5f * (float)(max(id0, max(id1, id2)) + min(id0, min(id1, id2))));
    closestCenterVId = min(estimToChannelId, closestCenterVId);

    fromChannelId = oc.vNumSamples;
    toChannelId   = 0;

    // Binary search — lower bound
    int low = (int)estimFromChannelId, high = (int)closestCenterVId;
    while (low <= high) {
        int   mid      = low + ((high - low) >> 1);
        float vAngle   = oc.vStartAngle + mid * oc.vAngularStep;
        bool   flip       = (vAngle < 0.0f);
        float  cosHalf    = sinf(fabsf(vAngle));
        float3 coneDir    = flip ? neg(oc.up) : oc.up;
        bool   isMirr     = flip ? Is32BitSet(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE_FLIPPED)
                                 : Is32BitSet(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE);
        bool   isIntersect = flip ? isTICDF : isTICD;
        if (!isMirr && GRCA_Fast_GACP_T_IntersectionCheck(
                signdDist, signdDistAngle, originToVert, triVerts,
                oc.up, oc.position, flip, isIntersect, coneDir, cosHalf, 0,
                closestVertDist, oc.maxRange))
            { fromChannelId = (unsigned int)mid; high = mid - 1; }
        else
            { low = mid + 1; }
    }

    // Binary search — upper bound
    low  = (int)min(fromChannelId, estimToChannelId);
    high = (int)estimToChannelId;
    while (low <= high) {
        int   mid      = low + ((high - low) >> 1);
        float vAngle   = oc.vStartAngle + mid * oc.vAngularStep;
        bool   flip       = (vAngle < 0.0f);
        float  cosHalf    = sinf(fabsf(vAngle));
        float3 coneDir    = flip ? neg(oc.up) : oc.up;
        bool   isMirr     = flip ? Is32BitSet(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE_FLIPPED)
                                 : Is32BitSet(grcaFlags, GRCA_FLAGS_IS_BEHIND_CONE);
        bool   isIntersect = flip ? isTICDF : isTICD;
        if (!isMirr && GRCA_Fast_GACP_T_IntersectionCheck(
                signdDist, signdDistAngle, originToVert, triVerts,
                oc.up, oc.position, flip, isIntersect, coneDir, cosHalf, 0,
                closestVertDist, oc.maxRange))
            { toChannelId = (unsigned int)mid; low = mid + 1; }
        else
            { high = mid - 1; }
    }

    return !(fromChannelId > toChannelId || fromChannelId == oc.vNumSamples);
}

// ---------------------------------------------------------------------------
// GRCA_Fast_RSpan_Predict (Early Pass — mirrors GeometryPreProcessShader.compute)
// ---------------------------------------------------------------------------
__device__ void GRCA_Fast_RSpan_Predict(
    float3 originToVert[3], float3 triVerts[3],
    unsigned int robotId, const RayOriginCache& oc,
    const RayData* rayDataBuf,
    unsigned int maxSweepCount,
    bool& isAllClockWise, unsigned int& estimFromSweepId, unsigned int& estimSweepDiff)
{
    estimFromSweepId = 0;
    estimSweepDiff   = maxSweepCount;

    bool isCCWPossible = Is32BitSet(oc.flags, RAYORIGIN_FLAG_CCW_POSSIBLE);
    if (isCCWPossible) {
        unsigned int midVId = (unsigned int)roundf(oc.vNumSamples / 2.0f);
        unsigned int midHId = (unsigned int)roundf(oc.hNumSamples / 2.0f);
        unsigned int gOff   = robotId * MAX_LIDAR_RAY_COUNT;
        unsigned int rOff   = gOff + oc.localRayIdOffset + midVId * oc.hNumSamples;
        unsigned int midId  = rOff + midHId;
        float3 midDir = rayDataBuf[midId].direction;
        // All triangles in front of the lidar should be clockwise
        bool isAllFront = (dot(originToVert[0], midDir) > 0.0f) &&
                          (dot(originToVert[1], midDir) > 0.0f) &&
                          (dot(originToVert[2], midDir) > 0.0f);
        // Build right vector perpendicular to midDir via Gram-Schmidt.
        float3 worldRef = (fabsf(midDir.y) < 0.9f) ? make_float3(0.0f, 1.0f, 0.0f)
                                                    : make_float3(1.0f, 0.0f, 0.0f);
        float3 up    = normalize(worldRef - dot(worldRef, midDir) * midDir);
        float3 right = normalize(cross(midDir, up));
        // Triangle fully left or right of midDir — no seam wrap possible.
        bool isAllRight = (dot(originToVert[0], right) > 0.0f) &&
                        (dot(originToVert[1], right) > 0.0f) &&
                        (dot(originToVert[2], right) > 0.0f);
        bool isAllLeft  = (dot(originToVert[0], right) < 0.0f) &&
                        (dot(originToVert[1], right) < 0.0f) &&
                        (dot(originToVert[2], right) < 0.0f);
        isAllClockWise = isAllFront || isAllRight || isAllLeft;
    } else {
        isAllClockWise = true;
    }

    if (isAllClockWise) {
        unsigned int hId0, hId1, hId2;
        GRCA_AngularSweepIndexing(oc.hStartAngle, oc.hAngularStep, oc.hNumSamples,
                                 oc.position, oc.right, oc.forward, oc.up, triVerts[0], hId0);
        GRCA_AngularSweepIndexing(oc.hStartAngle, oc.hAngularStep, oc.hNumSamples,
                                 oc.position, oc.right, oc.forward, oc.up, triVerts[1], hId1);
        GRCA_AngularSweepIndexing(oc.hStartAngle, oc.hAngularStep, oc.hNumSamples,
                                 oc.position, oc.right, oc.forward, oc.up, triVerts[2], hId2);

        estimFromSweepId = min(hId0, min(hId1, hId2));
        unsigned int estimToSweepId = max(hId0, max(hId1, hId2));
        estimSweepDiff = GetAngularRayDiff(estimFromSweepId, estimToSweepId,
                                           oc.hNumSamples, true);
    }
}

// ===========================================================================
// Kernel: SetupLidarRays
// Dispatch: (robotCount, ceil(totalRayCount/1024), 1) @ blockDim(32,32)
// Finds each ray's origin, computes its direction from spherical angles, and
// initialises rangeAsUint to FLT_MAX (combines the old Init + Update passes).
// ===========================================================================
__global__ void SetupLidarRays(RayData* rayDataBuf, const RayOriginData* rayOriginBuf) {
    unsigned int robotId    = blockIdx.x;
    // threadIdx.y*32+threadIdx.x gives sequential ray IDs per warp → coalesced writes
    unsigned int localRayId = blockIdx.y * 1024u + threadIdx.y * 32u + threadIdx.x;

    unsigned int totalRayCount = rayOriginBuf[robotId].totalRayCount;
    if (localRayId >= totalRayCount) return;

    unsigned int rayOriginCount = rayOriginBuf[robotId].rayOriginCount;
    unsigned int globalRayId    = robotId * MAX_LIDAR_RAY_COUNT + localRayId;

    // Find which lidar origin this ray belongs to.
    unsigned int rayOriginId = 0;
    for (; rayOriginId < rayOriginCount; ++rayOriginId) {
        const RayOrigin& ro = rayOriginBuf[robotId].rayOrigin[rayOriginId];
        unsigned int startId = ro.localRayIdOffset;
        unsigned int endId   = startId + ro.hNumSamples * ro.vNumSamples;
        if (localRayId >= startId && localRayId < endId) break;
    }

    const RayOrigin& ro = rayOriginBuf[robotId].rayOrigin[rayOriginId];
    unsigned int rayId  = localRayId - ro.localRayIdOffset;
    unsigned int hRayId = rayId % ro.hNumSamples;
    unsigned int vRayId = rayId / ro.hNumSamples;

    float hAngle = ro.hStartAngle + hRayId * ro.hAngularStep;
    float vAngle = ro.vStartAngle + vRayId * ro.vAngularStep;

    // clockwise Unity convention — matches C# ComputeRayDirection (+sinH)
    // sincosf computes both in one instruction; normalize is redundant because
    // forward/right/up are orthonormal so the spherical formula already yields a unit vector.
    float cosH, sinH, cosV, sinV;
    sincosf(hAngle, &sinH, &cosH);
    sincosf(vAngle, &sinV, &cosV);
    float3 dir = (cosH * ro.forward + sinH * ro.right) * cosV + sinV * ro.up;

    RayData ray;
    ray.direction     = dir;
    ray.rangeAsUint   = asUINT_sortable(FLT_MAX_VAL);
    rayDataBuf[globalRayId] = ray;
}

// ===========================================================================
// Kernel: GRCA_Early_Pass
// Dispatch: (ceil(triCount/512), 1, 1) @ blockDim(32,16,1)
//
// For each triangle:
//   1. Filter: facing, apparent area, distance  (GRCA_Early_T_Filter)
//   2. Channel-span prediction                  (GRCA_CSpan_Predict)
//   3. Sweep-span prediction                    (GRCA_Fast_RSpan_Predict)
//   4. If small enough span: RTI ray tests inline (GRCA_SAT_RTI)
//   5. Otherwise: emit TriRay to BAT list        (GRCA_Late_Pass input)
//
// Inputs
//   vertexBuf    — packed vertex buffer (world-pos at byte offset 36)
//   indexBuf     — triangle index buffer (uint32 triplets)
//   rayOriginBuf — per-robot lidar descriptors
//   rayDataBuf   — per-ray RayData (rangeAsUint updated atomically)
// Outputs
//   triRayCastBuf — BAT triangle list fed into GRCA_Late_Pass
//   uniformBuf    — updated dispatch args / bounds
// ===========================================================================
__global__ void GRCA_Early_Pass(
    const unsigned char* vertexBuf,
    const unsigned int*  indexBuf,
    const RayOriginData* rayOriginBuf,
    RayData*             rayDataBuf,
    TriRay*              triRayCastBuf,
    UniformData*         uniformBuf,
    unsigned int         robotCount,
    unsigned int         maxLidarCount,
    unsigned int         triCount,
    unsigned int         maxSweepCount,
    unsigned int         maxChannelCount,
    unsigned int*        d_debug)        // [0]=earlyFilterTCount [1]=rticCount; null when !debug
{
    __shared__ RayOriginCache rayOriginCache[MAX_ROBOT_LIDAR_COUNT];
    __shared__ TriTemp        triTempCache[512];

    // --- Load lidar descriptors into shared memory (row 0 only, 32 threads) ---
    if (threadIdx.y == 0) {
        unsigned int rId = threadIdx.x / MAX_LIDAR_COUNT;
        unsigned int lId = threadIdx.x % MAX_LIDAR_COUNT;
        if (rId < robotCount) {
            const RayOrigin& ro = rayOriginBuf[rId].rayOrigin[lId];
            RayOriginCache& c   = rayOriginCache[threadIdx.x];
            c.position       = ro.position;
            c.up             = ro.up;
            c.right          = ro.right;
            c.forward        = ro.forward;
            c.vNumSamples    = ro.vNumSamples;
            c.hNumSamples    = ro.hNumSamples;
            c.localRayIdOffset = ro.localRayIdOffset;
            c.hStartAngle    = ro.hStartAngle;
            c.hAngularStep   = ro.hAngularStep;
            c.vStartAngle    = ro.vStartAngle;
            c.vAngularStep   = ro.vAngularStep;
            c.minRange       = ro.minRange;
            c.maxRange       = ro.maxRange;
            c.flags          = ro.flags;
        }
    }
    __syncthreads();

    unsigned int localIndex = threadIdx.y * 32u + threadIdx.x;
    unsigned int triIndex   = blockIdx.x * 512u + localIndex;
    if (triIndex >= triCount) return;

    // --- Load triangle from vertex/index buffers ---
    unsigned int base = triIndex * 3u;
    unsigned int vi0  = indexBuf[base + 0];
    unsigned int vi1  = indexBuf[base + 1];
    unsigned int vi2  = indexBuf[base + 2];

    triTempCache[localIndex].verts[0] = LoadWorldPos(vi0, vertexBuf);
    triTempCache[localIndex].verts[1] = LoadWorldPos(vi1, vertexBuf);
    triTempCache[localIndex].verts[2] = LoadWorldPos(vi2, vertexBuf);

    float3 center = (triTempCache[localIndex].verts[0] +
                     triTempCache[localIndex].verts[1] +
                     triTempCache[localIndex].verts[2]) / 3.0f;
     // Edge vectors — CW winding: e1 = v1-v0, e2 = v2-v0
    float3 edge1 = triTempCache[localIndex].verts[1] - triTempCache[localIndex].verts[0];
    float3 edge2 = triTempCache[localIndex].verts[2] - triTempCache[localIndex].verts[0];

    float3 tmp  = cross(edge2, edge1);  // CW winding for normal (consistent with Unity)
    float  area = 0.5f * sqrtf(dot(tmp, tmp));
    triTempCache[localIndex].normal = normalize(tmp);

    // intensity / material id from vertex UV2.y  (packed as float bits -> uint)
    // (Simplified: use 0 when not loading full UV layout)
    // unsigned int intensityId = 0; // removed unused variable
    unsigned int nonUniformLidarFlag = 0;
    unsigned int maxSweepDiff   = 0;
    unsigned int maxChannelDiff = 0;
    unsigned int maxHNumSamples = 0;

    unsigned int triRayCastIndex = 0;
    bool         isTriWritten    = false;
    unsigned int isValidTriangleFlags = 0;
    float closestVertDistArr[MAX_ROBOT_LIDAR_COUNT];
    for (unsigned int i = 0; i < MAX_ROBOT_LIDAR_COUNT; ++i) closestVertDistArr[i] = FLT_MAX_VAL;

    for (unsigned int robotId = 0; robotId < robotCount; ++robotId) {
            unsigned int lidarCount = rayOriginBuf[robotId].rayOriginCount;

            for (unsigned int lidarId = 0; lidarId < lidarCount; ++lidarId) {
                unsigned int rayOrigflagOffset = robotId * MAX_LIDAR_COUNT + lidarId;
                // --- Triangle filter ---
                if (GRCA_Early_T_Filter(
                        triTempCache[localIndex].verts[0],
                        triTempCache[localIndex].verts[1],
                        triTempCache[localIndex].verts[2],
                        triTempCache[localIndex].normal,
                        center, area,
                        rayOriginBuf[robotId].rayOrigin[lidarId].position,
                        rayOriginBuf[robotId].rayOrigin[lidarId].minRange,
                        rayOriginBuf[robotId].rayOrigin[lidarId].maxRange,
                        closestVertDistArr[rayOrigflagOffset])) {
                            Set32BitFlag(isValidTriangleFlags, rayOrigflagOffset);
                        }
            }
    }

    for (unsigned int robotId = 0; robotId < robotCount; ++robotId) {
        unsigned int lidarCount = rayOriginBuf[robotId].rayOriginCount;

        for (unsigned int lidarId = 0; lidarId < lidarCount; ++lidarId) {
            unsigned int rayOrigflagOffset = robotId * MAX_LIDAR_COUNT + lidarId;

            if (!Is32BitSet(isValidTriangleFlags, rayOrigflagOffset)) continue;

            const RayOriginCache& oc = rayOriginCache[rayOrigflagOffset];

            maxHNumSamples = max(oc.hNumSamples, maxHNumSamples);


            // --- Compute per-vertex signed distances & angle-distances ---
            float3 originToVert[3];
            originToVert[0] = triTempCache[localIndex].verts[0] - oc.position;
            originToVert[1] = triTempCache[localIndex].verts[1] - oc.position;
            originToVert[2] = triTempCache[localIndex].verts[2] - oc.position;

            float signdDist[3], signdDistAngle[3];
            signdDist[0] = dot(originToVert[0], oc.up);
            signdDist[1] = dot(originToVert[1], oc.up);
            signdDist[2] = dot(originToVert[2], oc.up);
            signdDistAngle[0] = dot(normalize(originToVert[0]), oc.up);
            signdDistAngle[1] = dot(normalize(originToVert[1]), oc.up);
            signdDistAngle[2] = dot(normalize(originToVert[2]), oc.up);

            unsigned int grcaFlags = 0;
            bool isTICD  = RayTriangleIntersect( oc.up, oc.position, triTempCache[localIndex].verts[0], edge1, edge2) < FLT_MAX_VAL;
            bool isTICDF = RayTriangleIntersect(neg(oc.up), oc.position, triTempCache[localIndex].verts[0], edge1, edge2) < FLT_MAX_VAL;
            if (isTICD)  Set32BitFlag(grcaFlags, GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR);
            if (isTICDF) Set32BitFlag(grcaFlags, GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR_FL);

            // --- GRCA_CSpan_Predict ---
            unsigned int estimFromChannelId = 0;
            unsigned int estimToChannelId   = oc.vNumSamples - 1;
            unsigned int fromChannelId = oc.vNumSamples;
            unsigned int toChannelId   = 0;
            if (oc.vNumSamples > 1) {
                if (!GRCA_CSpan_Predict(signdDist, signdDistAngle, originToVert,
                                       triTempCache[localIndex].verts,
                                       grcaFlags, isTICD, isTICDF, oc,
                                       estimFromChannelId, estimToChannelId,
                                       fromChannelId, toChannelId,
                                       closestVertDistArr[rayOrigflagOffset])) continue;
            }

            unsigned int channelDiff = toChannelId - fromChannelId;

            // Count (robot,lidar,tri) triples that survive to the SAT/BAT decision.
            if (d_debug) atomicAdd(&d_debug[0], 1u);

            // --- GRCA_Fast_RSpan_Predict ---
            bool         isAllClockWise  = false;
            unsigned int estimFromSweepId = 0;
            unsigned int estimSweepDiff   = maxSweepCount;
            GRCA_Fast_RSpan_Predict(originToVert, triTempCache[localIndex].verts,
                                   robotId, oc, rayDataBuf,
                                   maxSweepCount,
                                   isAllClockWise, estimFromSweepId, estimSweepDiff);

            // Pack output flags / dist
            unsigned int outFlags = 0;
            if (Is32BitSet(grcaFlags, GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR))    outFlags |= (1u << 0);
            if (Is32BitSet(grcaFlags, GRCA_FLAGS_IS_TRI_INTERSECT_CONE_DIR_FL)) outFlags |= (1u << 1);
            if (isAllClockWise)                                                 outFlags |= (1u << 2);
            unsigned int outDistAndFlags = PackTwo16Bits(f32tof16(closestVertDistArr[rayOrigflagOffset]), outFlags);
            unsigned int outVFromIdToId  = PackTwo16Bits(fromChannelId, toChannelId);

            bool isBAT = (!isAllClockWise ||
                          channelDiff >= maxChannelCount ||
                          estimSweepDiff >= maxSweepCount);

            if (isBAT) {
                // Store triangle to BAT list for GRCA_Late_Pass
                maxChannelDiff = max(maxChannelDiff, channelDiff);
                Set32BitFlag(nonUniformLidarFlag, rayOrigflagOffset);

                if (!isTriWritten) {
                    // Warp-ballot: one atomicAdd per warp instead of one per thread.
                    unsigned int warpMask  = __ballot_sync(__activemask(), true);
                    unsigned int laneId    = threadIdx.x & 31u;
                    unsigned int laneRank  = __popc(warpMask & ((1u << laneId) - 1u));
                    unsigned int warpCount = __popc(warpMask);

                    unsigned int warpBase = 0;
                    if (laneRank == 0)
                        warpBase = atomicAdd(&uniformBuf->rayCastDispatchTriangleCount, warpCount);
                    warpBase = __shfl_sync(warpMask, warpBase, __ffs((int)warpMask) - 1);

                    unsigned int myIndex = warpBase + laneRank;
                    if (myIndex >= uniformBuf->batCapacity) return;
                    triRayCastIndex = myIndex;
                    isTriWritten    = true;

                    triRayCastBuf[triRayCastIndex].triIndex = triIndex;
                    triRayCastBuf[triRayCastIndex].rayOriginFlags           = 0;
                    triRayCastBuf[triRayCastIndex].triConeApexHit_P_Flags   = 0;
                    triRayCastBuf[triRayCastIndex].triConeApexHit_N_Flags   = 0;
                    triRayCastBuf[triRayCastIndex].isAllCWFlags              = 0;
                    for (unsigned int k = 0; k < MAX_ROBOT_LIDAR_COUNT / 2; ++k)
                        triRayCastBuf[triRayCastIndex].packd_closestVertDist[k] = 0;
                }

                Set32BitFlag(triRayCastBuf[triRayCastIndex].rayOriginFlags, (int)rayOrigflagOffset);
                if (outFlags & (1u << 0)) Set32BitFlag(triRayCastBuf[triRayCastIndex].triConeApexHit_P_Flags, (int)rayOrigflagOffset);
                if (outFlags & (1u << 1)) Set32BitFlag(triRayCastBuf[triRayCastIndex].triConeApexHit_N_Flags, (int)rayOrigflagOffset);
                if (outFlags & (1u << 2)) Set32BitFlag(triRayCastBuf[triRayCastIndex].isAllCWFlags,           (int)rayOrigflagOffset);

                unsigned int distF16 = Unpack16BitsLower(outDistAndFlags);
                unsigned int idx = rayOrigflagOffset / 2;
                unsigned int packed = triRayCastBuf[triRayCastIndex].packd_closestVertDist[idx];
                if ((rayOrigflagOffset % 2) == 0)
                    triRayCastBuf[triRayCastIndex].packd_closestVertDist[idx] = PackTwo16Bits(distF16, Unpack16BitsUpper(packed));
                else
                    triRayCastBuf[triRayCastIndex].packd_closestVertDist[idx] = PackTwo16Bits(Unpack16BitsLower(packed), distF16);

                triRayCastBuf[triRayCastIndex].vFromIdToId[rayOrigflagOffset] = outVFromIdToId;
            } else {
                if (d_debug)
                    atomicAdd(&d_debug[1],
                              (toChannelId - fromChannelId + 1u) * (estimSweepDiff + 1u));
                unsigned int globalRayIdBase = robotId * MAX_LIDAR_RAY_COUNT;
                for (unsigned int chId = fromChannelId; chId <= toChannelId; ++chId) {
                    unsigned int rayRowOff = globalRayIdBase + oc.localRayIdOffset +
                                            chId * oc.hNumSamples;
                    for (unsigned int i = 0; i <= estimSweepDiff; ++i) {
                        unsigned int globalRayId =
                            GetNextClosestRayId(estimFromSweepId, i, oc.hNumSamples, isAllClockWise)
                            + rayRowOff;
                        // commented out because cause more misses than hits
                        // if (closestVertDist > asFLOAT_sortable(rayDataBuf[globalRayId].rangeAsUint))
                        //     continue;
                        float dist = RayTriangleIntersect(
                            rayDataBuf[globalRayId].direction, oc.position,
                            triTempCache[localIndex].verts[0], edge1, edge2);
                        if (dist >= FLT_MAX_VAL) continue;

                        unsigned int newDist = asUINT_sortable(dist);
                        unsigned int prevDist = atomicMin(&rayDataBuf[globalRayId].rangeAsUint, newDist);
                    }
                }
            }
        }
    }

    if (isTriWritten) {
        maxSweepDiff = maxHNumSamples - 1;
        // Use local values — reading back uniformBuf here races with other threads
        // still writing their atomicMax/atomicAdd results.
        unsigned int myGroupX = (triRayCastIndex + 32u) / 32u;
        unsigned int myMaxCh  = maxChannelDiff + 1u;
        unsigned int myMaxSw  = maxSweepDiff   + 1u;
        unsigned int myGroupY = (unsigned int)ceilf((float)myMaxSw / 32.0f / (float)PER_THREAD_RAY_BATCH);
        unsigned int myGroupZ = (unsigned int)ceilf((float)myMaxCh / (float)PER_THREAD_RAY_BATCH)
                                * robotCount * maxLidarCount;

        // Warp-level max reduction before hitting global atomics (~32x less contention).
        // __shfl_xor_sync returns the caller's own value if the partner lane is inactive,
        // so inactive lanes in the warp don't corrupt the result.
        unsigned int activeMask = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            myGroupX = max(myGroupX, __shfl_xor_sync(activeMask, myGroupX, offset));
            myMaxCh  = max(myMaxCh,  __shfl_xor_sync(activeMask, myMaxCh,  offset));
            myMaxSw  = max(myMaxSw,  __shfl_xor_sync(activeMask, myMaxSw,  offset));
            myGroupY = max(myGroupY, __shfl_xor_sync(activeMask, myGroupY, offset));
            myGroupZ = max(myGroupZ, __shfl_xor_sync(activeMask, myGroupZ, offset));
        }

        // Only the lowest active lane in the warp does the global atomics.
        if (threadIdx.x == (unsigned int)(__ffs((int)activeMask) - 1)) {
            atomicMax(&uniformBuf->rayCastDispatchGroupX, myGroupX);
            atomicMax(&uniformBuf->rayCastMaxChannelDiff, myMaxCh);
            atomicMax(&uniformBuf->rayCastMaxSweepDiff,   myMaxSw);
            atomicMax(&uniformBuf->rayCastDispatchGroupY, myGroupY);
            atomicMax(&uniformBuf->rayCastDispatchGroupZ, myGroupZ);
        }
    }
}

// ===========================================================================
// Kernel: GRCA_Late_Pass
// Dispatch: (ceil(triCount/32), ceil(maxSweepDiff/32), totalChannelBatches*robotCount*lidarCount)
//           @ blockDim(32,32,1)
//
// Handles BAT triangles left by the Early Pass:
//   1. Precompute GACP cone-intersection data per channel in thread-row 0
//   2. All 32 threads in y process their sweep slice and run RTI
// ===========================================================================
__global__ void GRCA_Late_Pass(
    RayCastOutput*       rayCastOutputBuf,
    RayData*             rayDataBuf,
    const UniformData*   uniformBuf,
    const RayOriginData* rayOriginBuf,
    const TriRay*        triBuf,
    const unsigned char* vertexBuf,
    const unsigned int*  indexBuf,
    unsigned int         robotCount,
    unsigned int*        d_debug)
{
    __shared__ TriRayLimit    trlShBuf[PER_THREAD_RAY_BATCH * 32];
    __shared__ RayCastDataSh  rcShBuf[32];
    __shared__ GRCAGacpDataSh  grcaGacpShBuf[32];
    __shared__ GRCAIndexDataSh grcaIndexShBuf[32];
    __shared__ GRCARspanDataSh grcaRspanShBuf[32];

    unsigned int shId    = threadIdx.x;
    unsigned int totalChannelBatches = (unsigned int)ceilf(
        (float)uniformBuf->rayCastMaxChannelDiff / (float)PER_THREAD_RAY_BATCH);
    unsigned int channelBatchId = blockIdx.z % totalChannelBatches;
    unsigned int robotLidarId   = blockIdx.z / totalChannelBatches;
    unsigned int robotId        = robotLidarId % robotCount;
    unsigned int localOriginId  = robotLidarId / robotCount;
    unsigned int rayOriginCount = rayOriginBuf[robotId].rayOriginCount;

    // --- Row 0: load per-triangle data into shared memory ---
    if (threadIdx.y == 0) {
        unsigned int rayOrigflagOffset = robotId * MAX_LIDAR_COUNT + localOriginId;
        unsigned int uniqueTriId       = blockIdx.x * 32u + shId;

        rcShBuf[shId].hNumSamples = rayOriginBuf[robotId].rayOrigin[localOriginId].hNumSamples;
        rcShBuf[shId].vFromId     = Unpack16BitsLower(triBuf[uniqueTriId].vFromIdToId[rayOrigflagOffset]);
        rcShBuf[shId].vToId       = Unpack16BitsUpper(triBuf[uniqueTriId].vFromIdToId[rayOrigflagOffset]);
        unsigned int triIndex = triBuf[uniqueTriId].triIndex;
        unsigned int vi0 = indexBuf[triIndex * 3 + 0];
        unsigned int vi1 = indexBuf[triIndex * 3 + 1];
        unsigned int vi2 = indexBuf[triIndex * 3 + 2];
        rcShBuf[shId].triVerts[0] = LoadWorldPos(vi0, vertexBuf);
        rcShBuf[shId].triVerts[1] = LoadWorldPos(vi1, vertexBuf);
        rcShBuf[shId].triVerts[2] = LoadWorldPos(vi2, vertexBuf);
        rcShBuf[shId].edge1       = rcShBuf[shId].triVerts[1] - rcShBuf[shId].triVerts[0];
        rcShBuf[shId].edge2       = rcShBuf[shId].triVerts[2] - rcShBuf[shId].triVerts[0];
        grcaRspanShBuf[shId].normal = normalize(cross(rcShBuf[shId].edge2, rcShBuf[shId].edge1));

        // Unpack closest vertex distance (f16 stored in packed uint)
        unsigned int idx        = rayOrigflagOffset / 2;
        unsigned int packedHalf = triBuf[uniqueTriId].packd_closestVertDist[idx];
        float unpackedDist      = ((rayOrigflagOffset % 2) == 0)
            ? f16tof32(Unpack16BitsLower(packedHalf))
            : f16tof32(Unpack16BitsUpper(packedHalf));
        rcShBuf[shId].closestVertDist   = unpackedDist;
        rcShBuf[shId].maxRange          = rayOriginBuf[robotId].rayOrigin[localOriginId].maxRange;

        rcShBuf[shId].flags = 0;
        if (Is32BitSet(triBuf[uniqueTriId].rayOriginFlags, (int)rayOrigflagOffset))
            Set32BitFlag(rcShBuf[shId].flags, 0);
        if (Is32BitSet(triBuf[uniqueTriId].isAllCWFlags, (int)rayOrigflagOffset))
            Set32BitFlag(rcShBuf[shId].flags, 2);

        rcShBuf[shId].pos = rayOriginBuf[robotId].rayOrigin[localOriginId].position;
        rcShBuf[shId].globalRayIdOffset =
            robotId * MAX_LIDAR_RAY_COUNT +
            rayOriginBuf[robotId].rayOrigin[localOriginId].localRayIdOffset;

        grcaIndexShBuf[shId].up      = rayOriginBuf[robotId].rayOrigin[localOriginId].up;
        grcaIndexShBuf[shId].right   = rayOriginBuf[robotId].rayOrigin[localOriginId].right;
        grcaIndexShBuf[shId].forward = rayOriginBuf[robotId].rayOrigin[localOriginId].forward;
        grcaIndexShBuf[shId].hStartAngle = rayOriginBuf[robotId].rayOrigin[localOriginId].hStartAngle;
        grcaIndexShBuf[shId].hAngularStep= rayOriginBuf[robotId].rayOrigin[localOriginId].hAngularStep;
        grcaIndexShBuf[shId].vStartAngle = rayOriginBuf[robotId].rayOrigin[localOriginId].vStartAngle;
        grcaIndexShBuf[shId].vAngularStep= rayOriginBuf[robotId].rayOrigin[localOriginId].vAngularStep;
        grcaIndexShBuf[shId].flags       = rayOriginBuf[robotId].rayOrigin[localOriginId].flags;

        // Pre-compute per-edge GACP dot-products (used in cone-edge intersection)
        float3 originToVert[3] = {
            rcShBuf[shId].triVerts[0] - rcShBuf[shId].pos,
            rcShBuf[shId].triVerts[1] - rcShBuf[shId].pos,
            rcShBuf[shId].triVerts[2] - rcShBuf[shId].pos
        };
        float planeDotVerts[3] = {
            dot(rcShBuf[shId].triVerts[0], grcaIndexShBuf[shId].up),
            dot(rcShBuf[shId].triVerts[1], grcaIndexShBuf[shId].up),
            dot(rcShBuf[shId].triVerts[2], grcaIndexShBuf[shId].up)
        };
        float planeDotOrigin = dot(rcShBuf[shId].pos, grcaIndexShBuf[shId].up);

        for (int e = 0; e < 3; ++e) {
            int i0 = edgeIndices[e][0];
            int i1 = edgeIndices[e][1];
            float3 D = originToVert[i1] - originToVert[i0];
            grcaGacpShBuf[shId].DdD[e]  = dot(D, D);
            grcaGacpShBuf[shId].DdA[e]  = planeDotVerts[i1] - planeDotVerts[i0];
            grcaGacpShBuf[shId].P0dA[e] = planeDotVerts[i0] - planeDotOrigin;
            grcaGacpShBuf[shId].P0dD[e] = dot(originToVert[i0], D);
            grcaGacpShBuf[shId].P0dP0[e]= dot(originToVert[i0], originToVert[i0]);
            grcaGacpShBuf[shId].signdDist[e]      = dot(originToVert[e], grcaIndexShBuf[shId].up);
            grcaGacpShBuf[shId].signdDistAngle[e] = dot(normalize(originToVert[e]), grcaIndexShBuf[shId].up);
        }

        grcaGacpShBuf[shId].isTriIntersectConeDir        = Is32BitSet(triBuf[uniqueTriId].triConeApexHit_P_Flags, (int)rayOrigflagOffset);
        grcaGacpShBuf[shId].isTriIntersectConeDirFlipped  = Is32BitSet(triBuf[uniqueTriId].triConeApexHit_N_Flags, (int)rayOrigflagOffset);

        grcaRspanShBuf[shId].toCenter =
            (rcShBuf[shId].triVerts[0] + rcShBuf[shId].triVerts[1] + rcShBuf[shId].triVerts[2]) / 3.0f
            - rcShBuf[shId].pos;
        grcaRspanShBuf[shId].up = grcaIndexShBuf[shId].up;
    }
    __syncthreads();

    unsigned int uniqueTriId = blockIdx.x * 32u + shId;
    bool isVisible = (uniqueTriId < min(uniformBuf->rayCastDispatchTriangleCount,
                                        uniformBuf->batCapacity))
                  && (localOriginId < rayOriginCount)
                  && Is32BitSet(rcShBuf[shId].flags, 0);

    // --- Row 0: compute per-channel sweep bounds ---
    if (threadIdx.y == 0 && isVisible) {
        bool isAllClockWise = Is32BitSet(rcShBuf[shId].flags, 2);
        float distCenterNormal = dot(grcaRspanShBuf[shId].normal, grcaRspanShBuf[shId].toCenter);

        for (unsigned int channelIdOffset = 0; channelIdOffset < PER_THREAD_RAY_BATCH; ++channelIdOffset) {
            unsigned int trlShId  = channelIdOffset * 32u + shId;  // transposed for coalesced access
            trlShBuf[trlShId].hFromIdSweepDiff = PackTwo16Bits(0, 0);
            trlShBuf[trlShId].flags = 0;

            unsigned int channelId = rcShBuf[shId].vFromId
                                   + channelBatchId * PER_THREAD_RAY_BATCH
                                   + channelIdOffset;
            if (channelId > rcShBuf[shId].vToId) break;

            float  vAngle    = grcaIndexShBuf[shId].vStartAngle + channelId * grcaIndexShBuf[shId].vAngularStep;
            bool   isFlipped = (vAngle < 0.0f);
            float  cosHalf   = sinf(fabsf(vAngle));
            float3 coneDir   = isFlipped ? neg(grcaIndexShBuf[shId].up) : grcaIndexShBuf[shId].up;

            bool isMirrored = isFlipped
                ? (-grcaGacpShBuf[shId].signdDist[0] < 0.0f &&
                   -grcaGacpShBuf[shId].signdDist[1] < 0.0f &&
                   -grcaGacpShBuf[shId].signdDist[2] < 0.0f)
                : ( grcaGacpShBuf[shId].signdDist[0] < 0.0f &&
                    grcaGacpShBuf[shId].signdDist[1] < 0.0f &&
                    grcaGacpShBuf[shId].signdDist[2] < 0.0f);
            bool isIntersect = isFlipped ? grcaGacpShBuf[shId].isTriIntersectConeDirFlipped
                                         : grcaGacpShBuf[shId].isTriIntersectConeDir;
            if (isMirrored) continue;

            float3 intersectPoints[3] = { zerof3(), zerof3(), zerof3() };
            bool   isCheckAll         = false;

            if (!GRCA_GACP_T_IntersectionCheck(isFlipped, isIntersect, coneDir, cosHalf,
                                               (int)shId, intersectPoints, isCheckAll,
                                               rcShBuf, grcaGacpShBuf, grcaIndexShBuf)) {
                if (isCheckAll) {
                    trlShBuf[trlShId].hFromIdSweepDiff = PackTwo16Bits(0, rcShBuf[shId].hNumSamples);
                    Set32BitFlag(trlShBuf[trlShId].flags, 30);
                    Set32BitFlag(trlShBuf[trlShId].flags, 31);
                }
                continue;
            }

            unsigned int rayIdOffset = rcShBuf[shId].globalRayIdOffset
                                     + channelId * rcShBuf[shId].hNumSamples;

            // GRCA_AngularSweepIndexing — project intersection points to sweep IDs
            GRCA_AngularSweepIndexing(grcaIndexShBuf[shId].hStartAngle, grcaIndexShBuf[shId].hAngularStep,
                                     rcShBuf[shId].hNumSamples,
                                     rcShBuf[shId].pos, grcaIndexShBuf[shId].right,
                                     grcaIndexShBuf[shId].forward, grcaIndexShBuf[shId].up,
                                     intersectPoints[0], grcaIndexShBuf[shId].closestRayIds[0]);
            GRCA_AngularSweepIndexing(grcaIndexShBuf[shId].hStartAngle, grcaIndexShBuf[shId].hAngularStep,
                                     rcShBuf[shId].hNumSamples,
                                     rcShBuf[shId].pos, grcaIndexShBuf[shId].right,
                                     grcaIndexShBuf[shId].forward, grcaIndexShBuf[shId].up,
                                     intersectPoints[1], grcaIndexShBuf[shId].closestRayIds[1]);

            unsigned int fromSweepId = min(grcaIndexShBuf[shId].closestRayIds[0],
                                           grcaIndexShBuf[shId].closestRayIds[1]);
            unsigned int toSweepId   = max(grcaIndexShBuf[shId].closestRayIds[0],
                                           grcaIndexShBuf[shId].closestRayIds[1]);

            Set32BitFlag(trlShBuf[trlShId].flags, 31);

            bool         isClockWise = true;
            unsigned int sweepDiff   = 0;
            bool isCCWPossible = Is32BitSet(grcaIndexShBuf[shId].flags, RAYORIGIN_FLAG_CCW_POSSIBLE);
            GRCA_RSpan_Predict((int)shId, isAllClockWise, isCCWPossible, fromSweepId, toSweepId,
                               rayIdOffset, distCenterNormal,
                               isClockWise, sweepDiff,
                               rcShBuf, grcaRspanShBuf, rayDataBuf);

            if (isClockWise) Set32BitFlag(trlShBuf[trlShId].flags, 30);
            trlShBuf[trlShId].hFromIdSweepDiff = PackTwo16Bits(fromSweepId, sweepDiff);
        }
    }
    __syncthreads();

    if (!isVisible) return;

    // --- All 32x32 threads: run Moller-Trumbore RTI over assigned sweep/channel slices ---
    // unsigned int reflectiveObjType = rcShBuf[shId].reflectiveObjType; // unused
    unsigned int channelIdOffset   = 0;

    for (channelIdOffset = 0; channelIdOffset < PER_THREAD_RAY_BATCH; ++channelIdOffset) {
        unsigned int trlShId  = channelIdOffset * 32u + shId;  // transposed for coalesced access
        unsigned int channelId = rcShBuf[shId].vFromId
                               + channelBatchId * PER_THREAD_RAY_BATCH
                               + channelIdOffset;
        if (channelId > rcShBuf[shId].vToId) break;
        if (!Is32BitSet(trlShBuf[trlShId].flags, 31)) continue;

        bool         isClockWise = Is32BitSet(trlShBuf[trlShId].flags, 30);
        unsigned int hFromId     = Unpack16BitsLower(trlShBuf[trlShId].hFromIdSweepDiff);
        unsigned int sweepDiff   = Unpack16BitsUpper(trlShBuf[trlShId].hFromIdSweepDiff);

        for (unsigned int sweepIdOffset = 0; sweepIdOffset < PER_THREAD_RAY_BATCH; ++sweepIdOffset) {
            unsigned int sweepId = (blockIdx.y * 32u + threadIdx.y) * PER_THREAD_RAY_BATCH + sweepIdOffset;  // blockIdx.y partitions the sweep range across Y-blocks
            if (sweepId > sweepDiff) break;

            unsigned int globalRayId =
                GetNextClosestRayId(hFromId, sweepId, rcShBuf[shId].hNumSamples, isClockWise)
                + (rcShBuf[shId].globalRayIdOffset + channelId * rcShBuf[shId].hNumSamples);

            // commented out because cause more misses than hits
            // if (rcShBuf[shId].closestVertDist > asFLOAT_sortable(atomicAdd(&rayDataBuf[globalRayId].rangeAsUint, 0u)))
            //     continue;

            float dist = RayTriangleIntersect(
                rayDataBuf[globalRayId].direction, rcShBuf[shId].pos,
                rcShBuf[shId].triVerts[0], rcShBuf[shId].edge1, rcShBuf[shId].edge2);
            if (d_debug) atomicAdd(&d_debug[1], 1u);
            if (dist >= FLT_MAX_VAL) continue;

            unsigned int newDist  = asUINT_sortable(dist);
            unsigned int prevDist = atomicMin(&rayDataBuf[globalRayId].rangeAsUint, newDist);
        }
    }
}

// ===========================================================================
// Kernel: RayCastOutputAsFloat
// Dispatch: (robotCount, ceil(maxRayCount/1024), 1) @ blockDim(32,32,1)
//
// Decodes the sortable-uint range back to float and writes to output buffer.
// ===========================================================================
__global__ void RayCastOutputAsFloat(RayCastOutput* outputBuf, const RayData* rayDataBuf, const RayOriginData* rayOriginBuf) {
    unsigned int robotId     = blockIdx.x;
    // threadIdx.y*32+threadIdx.x gives sequential ray IDs per warp → coalesced reads/writes
    unsigned int uniqueRayId = blockIdx.y * 1024u + threadIdx.y * 32u + threadIdx.x;
    unsigned int totalRayCount = rayOriginBuf[robotId].totalRayCount;
    if (uniqueRayId >= totalRayCount) return;
    unsigned int globalRayId = robotId * MAX_LIDAR_RAY_COUNT + uniqueRayId;
    unsigned int contiguousGlobalRayId = rayOriginBuf[robotId].rayOrigin[0].contiguousGlobalRayIdOffset + uniqueRayId;
    outputBuf[contiguousGlobalRayId].direction = rayDataBuf[globalRayId].direction;
    outputBuf[contiguousGlobalRayId].distance  = asFLOAT_sortable(rayDataBuf[globalRayId].rangeAsUint);
}

// ===========================================================================
// Kernel: DispatchLatePass  (indirect dispatch via Dynamic Parallelism)
//
// Launched with <<<1,1>>> after GRCA_Early_Pass.  Reads the dispatch
// dimensions computed by the Early Pass directly from uniformBuf on the
// GPU — no CPU-side cudaMemcpy / if-check required.
//
// Requires: nvcc -rdc=true and linking against cudadevrt.
// ===========================================================================
__global__ void DispatchLatePass(
    RayCastOutput*       rayCastOutputBuf,
    RayData*             rayDataBuf,
    const UniformData*   uniformBuf,
    const RayOriginData* rayOriginBuf,
    const TriRay*        triBuf,
    const unsigned char* vertexBuf,
    const unsigned int*  indexBuf,
    unsigned int         robotCount,
    unsigned int*        d_debug)
{
    unsigned int X = uniformBuf->rayCastDispatchGroupX;
    unsigned int Y = uniformBuf->rayCastDispatchGroupY;
    unsigned int Z = uniformBuf->rayCastDispatchGroupZ;
    if (X == 0u || Y == 0u || Z == 0u) return;
    GRCA_Late_Pass<<<dim3(X, Y, Z), dim3(32, 32, 1)>>>(
        rayCastOutputBuf, rayDataBuf, uniformBuf,
        rayOriginBuf, triBuf, vertexBuf, indexBuf, robotCount, d_debug);
}

// ===========================================================================
// Host-side launch helpers  (optional convenience wrappers)
// ===========================================================================
void launchSetupLidarRays(RayData* d_rays, const RayOriginData* d_rayOrigins,
                           unsigned int robotCount, unsigned int maxRayCount,
                           cudaStream_t stream = 0) {
    dim3 block(32, 32);
    unsigned int gridY = (maxRayCount + 1023u) / 1024u;
    dim3 grid(robotCount, gridY);
    SetupLidarRays<<<grid, block, 0, stream>>>(d_rays, d_rayOrigins);
}

void launchGRCA_EarlyPass(
    const unsigned char* d_vertexBuf, const unsigned int* d_indexBuf,
    const RayOriginData* d_rayOrigins, RayData* d_rays,
    TriRay* d_triRayCast, UniformData* d_uniform,
    unsigned int triCount, unsigned int robotCount, unsigned int maxLidarCount,
    unsigned int maxSweepCount, unsigned int maxChannelCount,
    unsigned int* d_debug, cudaStream_t stream = 0)
{
    unsigned int groups = (triCount + 511u) / 512u;
    dim3 block(32, 16);
    dim3 grid(groups);
    GRCA_Early_Pass<<<grid, block, 0, stream>>>(
        d_vertexBuf, d_indexBuf, d_rayOrigins, d_rays,
        d_triRayCast, d_uniform,
        robotCount, maxLidarCount, triCount,
        maxSweepCount, maxChannelCount, d_debug);
}

void launchGRCA_LatePass(
    RayCastOutput* d_output, RayData* d_rays,
    const UniformData* d_uniform, const RayOriginData* d_rayOrigins,
    const TriRay* d_triRayCast,
    const unsigned char* d_vertexBuf, const unsigned int* d_indexBuf,
    unsigned int  rayCastDispatchGroupX, unsigned int rayCastDispatchGroupY,
    unsigned int  rayCastDispatchGroupZ, unsigned int robotCount,
    unsigned int* d_debug, cudaStream_t stream = 0)
{
    dim3 block(32, 32);
    dim3 grid(rayCastDispatchGroupX, rayCastDispatchGroupY, rayCastDispatchGroupZ);
    GRCA_Late_Pass<<<grid, block, 0, stream>>>(
        d_output, d_rays, d_uniform, d_rayOrigins,
        d_triRayCast, d_vertexBuf, d_indexBuf, robotCount, d_debug);
}

void launchRayCastOutputAsFloat(RayCastOutput* d_output, const RayData* d_rays, const RayOriginData* d_rayOrigins,
                                 unsigned int robotCount, unsigned int maxRayCount,
                                 cudaStream_t stream = 0) {
    dim3 block(32, 32);
    unsigned int gridY = (maxRayCount + 1023u) / 1024u;
    dim3 grid(robotCount, gridY);
    RayCastOutputAsFloat<<<grid, block, 0, stream>>>(d_output, d_rays, d_rayOrigins);
}
// =============================================================================
// GRCA CUDA backend — called from main.cpp via run_cuda().
// Receives pre-loaded triangles (Tri3), runs the full GRCA pipeline, and
// returns results as std::vector<GrcaHit> with timing in FrameTimings.
//
// Options (all optional, shown with defaults):
//   --hmin <rad>   HAngleMin     default: -3.141593
//   --hmax <rad>   HAngleMax     default:  3.124139
//   --hnum <int>   HNumSamples   default:  3600
//   --vmin <rad>   VAngleMin     default: -0.523599
//   --vmax <rad>   VAngleMax     default:  0.523599
//   --vnum <int>   VNumSamples   default:  32
//   --rmin <m>     RangeMin      default:  0.05
//   --rmax <m>     RangeMax      default:  100.0
//   --px/--py/--pz Lidar origin  default:  0 0 0
//   --out <file>   Output CSV    default:  ../hits_grca_cu.csv
// =============================================================================

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

#include "grca_cuda.h"

// ---------------------------------------------------------------------------
// CHECK_CUDA — prints error and aborts on failure.
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d  %s\n",                                \
              __FILE__, __LINE__, cudaGetErrorString(_e));                       \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// run_cuda  –  CUDA backend entry point called from main.cpp
// ---------------------------------------------------------------------------
// ===========================================================================
// CudaState  –  persistent GPU context for the game loop
// ===========================================================================
struct CudaState {
    cudaStream_t   stream      = nullptr;
    cudaEvent_t    fenceSetup  = nullptr;
    cudaEvent_t    fenceOutput = nullptr;

    // GPU buffers (allocated once at init, freed at destroy)
    RayData*       d_rays   = nullptr;
    RayOriginData* d_rod    = nullptr;  // lidar config — static
    RayCastOutput* d_out    = nullptr;
    UniformData*   d_unif   = nullptr;
    TriRay*        d_triRay = nullptr;
    unsigned char* d_verts  = nullptr;  // vertex positions — updated per frame
    unsigned int*  d_index  = nullptr;  // sequential indices — static

    unsigned int ROBOT_COUNT         = 0;
    unsigned int LIDAR_COUNT         = 0;
    size_t       RAY_ELEMS           = 0;
    size_t       N_VERTS             = 0;
    size_t       N_TRIS              = 0;
    size_t       staticN_VERTS       = 0;  // static verts — pre-filled, never re-uploaded

    std::vector<unsigned char> h_vbuf;  // static vert staging (uploaded once at init)
    unsigned int totalGlobalRayCount = 0;
    unsigned int maxTotalRays        = 0;

    RayOriginData* h_rods_pinned = nullptr;  // pinned lidar config — async-safe upload + hit fill

    // Pinned CPU buffers — async-safe, no internal CUDA staging copy
    unsigned char* h_dynVbuf_pinned = nullptr;  // dynamic vertex staging
    RayCastOutput* h_out_pinned     = nullptr;  // readback buffer

    unsigned int maxPreprocessSweepCount = 0;
    unsigned int maxPreprocessChannelCount = 0;
    unsigned int batCapacity = MAX_TRI_RAY_COUNT;

    bool           debug   = false;
    unsigned int*  d_debug = nullptr;  // [0]=earlyFilterTCount [1]=rticCount; null when !debug
};

// ---------------------------------------------------------------------------
// Helper: populate h_rods from RunConfig.
// ---------------------------------------------------------------------------
static void buildRods(const RunConfig& r,
                      RayOriginData* h_rods,
                      unsigned int& maxTotalRays,
                      unsigned int& totalGlobalRayCount)
{
    const unsigned int ROBOT_COUNT = (unsigned int)r.robots.size();
    maxTotalRays = 0;
    totalGlobalRayCount = 0;

    for (unsigned int ri = 0; ri < ROBOT_COUNT; ++ri) {
        const auto& robot = r.robots[ri];
        RayOriginData& rod = h_rods[ri];
        memset(&rod, 0, sizeof(rod));
        rod.rayOriginCount = (unsigned int)robot.lidars.size();

        unsigned int localOffset = 0;
        for (unsigned int li = 0; li < (unsigned int)robot.lidars.size(); ++li) {
            const LidarCfg& lc = robot.lidars[li];
            const unsigned int H = (unsigned int)lc.hnum;
            const unsigned int V = (unsigned int)lc.vnum;

            float hStep = (H > 1u) ? (lc.hmax - lc.hmin) / (float)(H - 1u) : 0.f;
            float vStep = (V > 1u) ? (lc.vmax - lc.vmin) / (float)(V - 1u) : 0.f;

            RayOrigin& ro = rod.rayOrigin[li];
            ro.position     = make_float3(lc.w_pose.pos.x,      lc.w_pose.pos.y,      lc.w_pose.pos.z);
            ro.forward      = make_float3(lc.w_pose.forward.x,  lc.w_pose.forward.y,  lc.w_pose.forward.z);
            ro.up           = make_float3(lc.w_pose.up.x,       lc.w_pose.up.y,       lc.w_pose.up.z);
            ro.right        = make_float3(lc.w_pose.right.x,    lc.w_pose.right.y,    lc.w_pose.right.z);
            ro.hStartAngle  = lc.hmin;
            ro.hAngularStep = hStep;
            ro.hEndAngle    = lc.hmin + (float)(H - 1u) * hStep;
            ro.vStartAngle  = lc.vmin;
            ro.vAngularStep = vStep;
            ro.vEndAngle    = lc.vmin + (float)(V - 1u) * vStep;
            ro.hNumSamples  = H;
            ro.vNumSamples  = V;
            ro.localRayIdOffset            = localOffset;
            ro.contiguousGlobalRayIdOffset = totalGlobalRayCount + localOffset;
            ro.minRange = lc.range_min;
            ro.maxRange = lc.range_max;
            ro.flags = ((ro.hEndAngle - ro.hStartAngle) > PI)
                       ? (1u << RAYORIGIN_FLAG_CCW_POSSIBLE) : 0u;
            localOffset += H * V;
        }

        if (localOffset > MAX_LIDAR_RAY_COUNT)
            throw std::runtime_error("[CUDA] Robot " + std::to_string(ri)
                                     + " ray budget exceeded: " + std::to_string(localOffset)
                                     + " > " + std::to_string(MAX_LIDAR_RAY_COUNT));
        rod.totalRayCount += localOffset;
        maxTotalRays = std::max(maxTotalRays, localOffset);
        totalGlobalRayCount += localOffset;
    }
}

// ---------------------------------------------------------------------------
// Helper: convert readback buffer to GrcaHit vector.
// ---------------------------------------------------------------------------
static void cudaFillHits(const RayCastOutput*  h_out,
                         size_t                total,
                         const RayOriginData*  h_rods,
                         unsigned int ROBOT_COUNT,
                         std::vector<GrcaHit>& hits,
                         int& totalHitCount)
{
    totalHitCount = 0;
    hits.resize(total);
    size_t contiguousIdx = 0;
    for (unsigned int ri = 0; ri < ROBOT_COUNT; ++ri) {
        const RayOriginData& rod = h_rods[ri];
        for (unsigned int li = 0; li < rod.rayOriginCount; ++li) {
            const RayOrigin& ro = rod.rayOrigin[li];
            unsigned int nRays  = ro.hNumSamples * ro.vNumSamples;
            float posX = ro.position.x, posY = ro.position.y, posZ = ro.position.z;
            float rangeMax = ro.maxRange;
            for (unsigned int rayIdx = 0; rayIdx < nRays; ++rayIdx, ++contiguousIdx) {
                const RayCastOutput& out = h_out[contiguousIdx];
                GrcaHit& h = hits[contiguousIdx];
                h.dx = out.direction.x;
                h.dy = out.direction.y;
                h.dz = out.direction.z;
                if (out.distance < FLT_MAX_VAL) {
                    h.dist = out.distance;
                    h.hx = posX + out.direction.x * out.distance;
                    h.hy = posY + out.direction.y * out.distance;
                    h.hz = posZ + out.direction.z * out.distance;
                    ++totalHitCount;
                } else {
                    h.dist = rangeMax;
                    h.hx = posX + out.direction.x * rangeMax;
                    h.hy = posY + out.direction.y * rangeMax;
                    h.hz = posZ + out.direction.z * rangeMax;
                }
            }
        }
    }
}

// ===========================================================================
// init_cuda  –  one-time setup: alloc GPU buffers, upload static data
// ===========================================================================
CudaState* init_cuda(const RunConfig& r,
                     const std::vector<Tri3>& tris,
                     size_t staticTriCount,
                     FrameTimings& t)
{
    if (r.robots.empty()) {
        fprintf(stderr, "[CUDA] No robots configured.\n");
        return nullptr;
    }

    CudaState* s = new CudaState();
    auto tInitStart = std::chrono::steady_clock::now();
    s->maxPreprocessSweepCount   = (unsigned int)r.cuda_sat_max_sweep_diff;
    s->maxPreprocessChannelCount = (unsigned int)r.cuda_sat_max_channel_diff;
    s->batCapacity = (unsigned int)r.bat_capacity;
    s->debug = r.debug;

    s->ROBOT_COUNT = (unsigned int)r.robots.size();
    s->LIDAR_COUNT = 0;
    for (const auto& robot : r.robots)
        s->LIDAR_COUNT = std::max(s->LIDAR_COUNT, (unsigned int)robot.lidars.size());

    if (s->ROBOT_COUNT > MAX_ROBOT_COUNT)
        throw std::runtime_error("[CUDA] Too many robots: " + std::to_string(s->ROBOT_COUNT));
    if (s->LIDAR_COUNT > MAX_LIDAR_COUNT)
        throw std::runtime_error("[CUDA] Too many lidars/robot: " + std::to_string(s->LIDAR_COUNT));

    // Allocate pinned rods buffer before buildRods so it can write directly into it
    CHECK_CUDA(cudaMallocHost(&s->h_rods_pinned, s->ROBOT_COUNT * sizeof(RayOriginData)));
    buildRods(r, s->h_rods_pinned, s->maxTotalRays, s->totalGlobalRayCount);

    s->RAY_ELEMS     = (size_t)s->ROBOT_COUNT * MAX_LIDAR_RAY_COUNT;
    s->N_TRIS        = tris.size();
    s->N_VERTS       = s->N_TRIS * 3u;
    s->staticN_VERTS = staticTriCount * 3u;

    // ---- CUDA alloc --------------------------------------------------------
    auto tInit0 = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaStreamCreate(&s->stream));
    CHECK_CUDA(cudaEventCreate(&s->fenceSetup));
    CHECK_CUDA(cudaEventCreate(&s->fenceOutput));

    CHECK_CUDA(cudaMalloc(&s->d_rays,   s->RAY_ELEMS                 * sizeof(RayData)));
    CHECK_CUDA(cudaMalloc(&s->d_rod,    s->ROBOT_COUNT               * sizeof(RayOriginData)));
    CHECK_CUDA(cudaMalloc(&s->d_out,    s->RAY_ELEMS                 * sizeof(RayCastOutput)));
    CHECK_CUDA(cudaMalloc(&s->d_unif,                                   sizeof(UniformData)));
    CHECK_CUDA(cudaMalloc(&s->d_triRay, (size_t)s->batCapacity       * sizeof(TriRay)));
    CHECK_CUDA(cudaMalloc(&s->d_verts,  s->N_VERTS                   * VERTEX_STRIDE));
    CHECK_CUDA(cudaMalloc(&s->d_index,  s->N_TRIS * 3u               * sizeof(unsigned int)));
    if (s->debug)
        CHECK_CUDA(cudaMalloc(&s->d_debug, 2 * sizeof(unsigned int)));

    auto tInit1 = std::chrono::steady_clock::now();
    t.ms_init_cpu = std::chrono::duration<double,std::milli>(tInit1 - tInit0).count();

    // ---- Upload static data: lidar config + sequential index buffer --------
    auto tUp0 = std::chrono::steady_clock::now();

    // Sequential index buffer — filled with std::iota, no manual loop needed.
    std::vector<unsigned int> h_ibuf(s->N_TRIS * 3u);
    std::iota(h_ibuf.begin(), h_ibuf.end(), 0u);

    CHECK_CUDA(cudaMemcpy(s->d_rod,   s->h_rods_pinned, s->ROBOT_COUNT * sizeof(RayOriginData), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(s->d_index, h_ibuf.data(),    s->N_TRIS * 3u * sizeof(unsigned int),  cudaMemcpyHostToDevice));

    // Static vertices: Tri3 layout (float v[3][3]) is identical to the packed
    // float3 vertex buffer (VERTEX_STRIDE=12), so a single memcpy suffices.
    if (staticTriCount > 0) {
        s->h_vbuf.resize(s->staticN_VERTS * VERTEX_STRIDE);
        memcpy(s->h_vbuf.data(), tris.data(), staticTriCount * sizeof(Tri3));
        CHECK_CUDA(cudaMemcpy(s->d_verts, s->h_vbuf.data(),
            s->staticN_VERTS * VERTEX_STRIDE, cudaMemcpyHostToDevice));
        // Static data is now on the GPU; release the CPU staging buffer.
        s->h_vbuf.clear();
        s->h_vbuf.shrink_to_fit();
    }
    // cudaMemcpy above is synchronous — no extra DeviceSynchronize needed.

    auto tUp1 = std::chrono::steady_clock::now();
    t.ms_dynamic_upload = std::chrono::duration<double,std::milli>(tUp1 - tUp0).count();

    // Pinned readback buffer
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&s->h_out_pinned),
        s->totalGlobalRayCount * sizeof(RayCastOutput)));
    // Pinned dynamic vertex staging (only if there are dynamic verts)
    size_t dynVerts_init = s->N_VERTS - s->staticN_VERTS;
    if (dynVerts_init > 0)
        CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&s->h_dynVbuf_pinned),
            dynVerts_init * VERTEX_STRIDE));

    t.ms_init_cpu = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tInitStart).count();
    fprintf(stderr, "[CUDA] init: %zu tris, %zu rays, %u robot(s)  (%.2f ms)\n",
            tris.size(), (size_t)s->totalGlobalRayCount, s->ROBOT_COUNT, t.ms_init_cpu);
    return s;
}

// ===========================================================================
// run_cuda_frame  –  per-frame: upload verts, GRCA passes, readback
// ===========================================================================
void run_cuda_frame(CudaState*               s,
                    const RunConfig&          r,
                    const std::vector<Tri3>& tris,
                    std::vector<GrcaHit>&     hits,
                    FrameTimings&              t)
{
    // ---- update only the dynamic vertex suffix and upload it ---------------
    // Static vertices are already in d_verts from init — skip them. 
    auto tUp0       = std::chrono::steady_clock::now();

    size_t dynVerts = s->N_VERTS - s->staticN_VERTS;
    if (dynVerts > 0) {
        // Tri3 layout matches the packed vertex buffer — one memcpy per frame.
        size_t baseTri = s->staticN_VERTS / 3;
        memcpy(s->h_dynVbuf_pinned, tris.data() + baseTri,
               (dynVerts / 3) * sizeof(Tri3));
        CHECK_CUDA(cudaMemcpy(
            s->d_verts + s->staticN_VERTS * VERTEX_STRIDE,
            s->h_dynVbuf_pinned,
            dynVerts * VERTEX_STRIDE,
            cudaMemcpyHostToDevice));
    }

    // ---- Re-build lidar ray data from current world poses (robots teleport each frame) ---
    buildRods(r, s->h_rods_pinned, s->maxTotalRays, s->totalGlobalRayCount);
    CHECK_CUDA(cudaMemcpy(s->d_rod, s->h_rods_pinned,
        s->ROBOT_COUNT * sizeof(RayOriginData), cudaMemcpyHostToDevice));

    auto tUp1 = std::chrono::steady_clock::now();
    t.ms_dynamic_upload = std::chrono::duration<double,std::milli>(tUp1 - tUp0).count();

    // ---- GPU passes — single stream, no inter-stream waits needed ----------
    auto tTraceCpu0 = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(s->fenceSetup, s->stream));
    launchSetupLidarRays(s->d_rays, s->d_rod, s->ROBOT_COUNT, MAX_LIDAR_RAY_COUNT, s->stream);

    {
        UniformData h_init{};
        h_init.batCapacity = s->batCapacity;
        CHECK_CUDA(cudaMemcpyAsync(s->d_unif, &h_init, sizeof(UniformData), cudaMemcpyHostToDevice, s->stream));
    }
    if (s->d_debug)
        CHECK_CUDA(cudaMemsetAsync(s->d_debug, 0, 2 * sizeof(unsigned int), s->stream));
    launchGRCA_EarlyPass(s->d_verts, s->d_index, s->d_rod, s->d_rays,
                        s->d_triRay, s->d_unif, (unsigned int)s->N_TRIS,
                        s->ROBOT_COUNT, s->LIDAR_COUNT,
                        s->maxPreprocessSweepCount, s->maxPreprocessChannelCount,
                        s->d_debug, s->stream);
    DispatchLatePass<<<1, 1, 0, s->stream>>>(s->d_out, s->d_rays, s->d_unif,
                                             s->d_rod, s->d_triRay, s->d_verts, s->d_index, s->ROBOT_COUNT, s->d_debug);

    launchRayCastOutputAsFloat(s->d_out, s->d_rays, s->d_rod,
                               s->ROBOT_COUNT, MAX_LIDAR_RAY_COUNT, s->stream);
    CHECK_CUDA(cudaEventRecord(s->fenceOutput, s->stream));
    t.ms_trace_cpu = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tTraceCpu0).count();

    auto tSync0 = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventSynchronize(s->fenceOutput));
    t.ms_sync_readback = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tSync0).count();

    float msGpu = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&msGpu, s->fenceSetup, s->fenceOutput));
    t.ms_trace_gpu = (double)msGpu;
    t.ms_lidar_sim_gpu = t.ms_trace_gpu;

    {
        UniformData h_unif{};
        CHECK_CUDA(cudaMemcpy(&h_unif, s->d_unif, sizeof(UniformData), cudaMemcpyDeviceToHost));
        t.num_bat          = (int)h_unif.rayCastDispatchTriangleCount;
        t.max_bat_capacity = (int)s->batCapacity;
        if (s->d_debug) {
            unsigned int h_dbg[2] = {};
            CHECK_CUDA(cudaMemcpy(h_dbg, s->d_debug, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            t.num_early_t = (int)h_dbg[0];
            t.num_rtic    = (int)h_dbg[1];
        }
    }

    // ---- readback into pinned buffer ----------------------------------------
    auto tRead0 = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaMemcpy(s->h_out_pinned, s->d_out,
        s->totalGlobalRayCount * sizeof(RayCastOutput), cudaMemcpyDeviceToHost));
    t.ms_readback_cpu = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - tRead0).count();
    t.mb_readback  = (double)(s->totalGlobalRayCount * sizeof(RayCastOutput)) / (1024.0 * 1024.0);

    // ---- convert to GrcaHit -------------------------------------------------
    int hitCount = 0;
    auto tFill0 = std::chrono::steady_clock::now();
    cudaFillHits(s->h_out_pinned, s->totalGlobalRayCount, s->h_rods_pinned, s->ROBOT_COUNT, hits, hitCount);
    t.ms_fillhits = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tFill0).count();
    t.hit_count = hitCount;
}

// ===========================================================================
// destroy_cuda  –  release all CUDA resources
// ===========================================================================
void destroy_cuda(CudaState* s)
{
    if (!s) return;
    cudaFree(s->d_rays);   cudaFree(s->d_rod);
    cudaFree(s->d_out);    cudaFree(s->d_unif);
    cudaFree(s->d_triRay); cudaFree(s->d_verts); cudaFree(s->d_index);
    if (s->d_debug) cudaFree(s->d_debug);
    if (s->h_rods_pinned)    cudaFreeHost(s->h_rods_pinned);
    if (s->h_dynVbuf_pinned) cudaFreeHost(s->h_dynVbuf_pinned);
    if (s->h_out_pinned)     cudaFreeHost(s->h_out_pinned);
    cudaEventDestroy(s->fenceSetup);
    cudaEventDestroy(s->fenceOutput);
    cudaStreamDestroy(s->stream);
    delete s;
}

// ===========================================================================
// run_cuda  –  legacy one-shot wrapper (init + single frame + destroy)
// ===========================================================================
void run_cuda(const RunConfig& r,
              const std::vector<Tri3>& tris,
              std::vector<GrcaHit>& hits,
              FrameTimings& t)
{
    FrameTimings initT = {};
    CudaState* s = init_cuda(r, tris, 0, initT);
    if (!s) return;
    t.ms_init_cpu = initT.ms_init_cpu;
    t.ms_dynamic_upload = initT.ms_dynamic_upload;
    run_cuda_frame(s, r, tris, hits, t);
    destroy_cuda(s);
}
