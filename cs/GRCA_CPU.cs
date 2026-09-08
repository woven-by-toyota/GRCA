using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;

//Examples:

// (2D LiDAR: 1957 rays):
// {
//   "HAngleMin":-3.141590118408203,
//   "HAngleMax":3.141590118408203,
//   "HNumSamples":1957,
//   "RangeMin":0.05000000074505806,
//   "RangeMax":15.0
// }

// (3D LiDAR: 32 * 3600 rays):
// {
//   "HAngleMin":-3.141593,
//   "HAngleMax":3.124139,
//   "HNumSamples":3600,
//   "VAngleMin":-1.5708,
//   "VAngleMax":1.5708,
//   "VNumSamples":32, 
//   "RangeMin":0.05000000074505806,
//   "RangeMax":100.0,
//   "HitPointColor": {
//     "r":1.0,
//     "g":0.0,
//     "b":0.0,
//     "a":1.0
//   }
// }


public class GrcaLidarCpu : MonoBehaviour {
  public float HAngleMin;
  public float HAngleMax;
  public uint HNumSamples;

  public float VAngleMin;
  public float VAngleMax;
  public uint VNumSamples = 1;
  public float RangeMin;
  public float RangeMax;
  //For Debugging:
  public Color HitPointColor = Color.white;

  private uint _totalSampleCount;
  private float _hAngleIncrement;
  private float _vAngleIncrement;
  private Vector3[,] _localDirs;
  private Vector3[,] _worldDirs;
  public struct HitPoint {
    public Vector3 Pos;
    public float Dist;
  }
  private HitPoint[,] _hitPoints;
  private const int _maxIntersections = 6;
  private const float _epsilon = 0.0001f;
  private List<MeshRenderer> _allMeshRenderers = new List<MeshRenderer>();
  private static int[,] _edgeIndices = new int[3, 2] { { 0, 1 }, { 1, 2 }, { 2, 0 } };
  private const float _maxRange = 1000f;
  private const float _min_apparent_area = 0.000001f; //1mm^2
  private bool _isShowHitpoints = false;
  // SAT/BAT classification thresholds — must match GPU MAX_PREPROCESS_SWEEP_COUNT / MAX_PREPROCESS_CHANNEL_COUNT
  public const int SatMaxChannelDiff = 64;
  public const int SatMaxSweepDiff   = 64;

  public class Triangle {
    public Vector3[] Vertices = new Vector3[3];
    public Vector3 Edge1;
    public Vector3 Edge2;
    public Vector3 Center;
    public Vector3 Normal;
    public float[] DdD = new float[3];
    public float Area;
  };

  public Triangle BuildTriangle(Vector3 v0, Vector3 v1, Vector3 v2) {
    var tri = new GrcaLidarCpu.Triangle();
    tri.Vertices[0] = v0; tri.Vertices[1] = v1; tri.Vertices[2] = v2;
    tri.Edge1  = v1 - v0;
    tri.Edge2  = v2 - v0;
    tri.Center = (v0 + v1 + v2) / 3f;
    var cross  = Vector3.Cross(tri.Edge2, tri.Edge1);
    tri.Normal = cross.normalized;
    tri.Area   = 0.5f * cross.magnitude;
    for (int e = 0; e < 3; ++e) {
      var d = tri.Vertices[_edgeIndices[e, 1]] - tri.Vertices[_edgeIndices[e, 0]];
      tri.DdD[e] = Vector3.Dot(d, d);
    }
    return tri;
  }

  public void Awake() {
    if (RangeMax > _maxRange) {
      throw new InvalidOperationException($"Lidars can't have RangeMax that's more than {_maxRange}.");
    }
    if (VNumSamples < 1) {
      throw new InvalidOperationException("VNumSamples should be 1 for a 2D lidar and more than 1 for 3D lidar!");
    } else if (VNumSamples > 1) {
      _vAngleIncrement = (VAngleMax - VAngleMin) / (VNumSamples - 1);
    }
    _hAngleIncrement = (HAngleMax - HAngleMin) / (HNumSamples - 1);
    _totalSampleCount = HNumSamples * VNumSamples;
    _localDirs = new Vector3[VNumSamples, HNumSamples];
    for (uint vert = 0; vert < VNumSamples; ++vert) {
      for (uint hori = 0; hori < HNumSamples; ++hori) {
        float vertAngle = VAngleMin + vert * _vAngleIncrement;
        float horiAngle = HAngleMin + hori * _hAngleIncrement;
        _localDirs[vert, hori] = ComputeRayDirection(horiAngle, vertAngle);
      }
    }
    _worldDirs = new Vector3[VNumSamples, HNumSamples];
    _hitPoints = new HitPoint[VNumSamples, HNumSamples];
    _allMeshRenderers = GameObject.FindObjectsOfType<MeshRenderer>(true)
                          .Where(x => ((1 << x.gameObject.layer) & LayerMask.GetMask("Building", "Ceiling", "Default", "NoCollision", "Robot", "Facility", "Tray")) != 0)
                          .ToList();
    ResetOutput();
  }

  private void ResetOutput() {
    for (uint vert = 0; vert < VNumSamples; ++vert) {
      for (uint hori = 0; hori < HNumSamples; ++hori) {
        _worldDirs[vert, hori] = transform.TransformDirection(_localDirs[vert, hori]);
        _hitPoints[vert, hori].Pos = Vector3.zero;
        _hitPoints[vert, hori].Dist = float.MaxValue;
      }
    }
  }

  private void OnDrawGizmos() {
    var keyboard = Keyboard.current;
    var triCount = 0;

    triCount = 0;
    //GRCA test (Press 'G' key in Game view)
    if (keyboard != null && keyboard.gKey.wasPressedThisFrame) {
      ResetOutput();
      System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
      sw.Start();
      foreach (var x in _allMeshRenderers) {
        if (x == null || !x.enabled) { continue; }

        Mesh mesh = null;

        if (x.TryGetComponent(out MeshFilter mf)) {
          mesh = mf.sharedMesh;
          if (mesh == null) { mesh = mf.mesh; }
        }
        // else if (x.TryGetComponent(out SkinnedMeshRenderer skinned)) {
        //   mesh = new Mesh();
        //   skinned.BakeMesh(mesh);
        // }

        if (mesh == null) { continue; }

        var verts = mesh.vertices;
        var tris = mesh.triangles;

        for (int i = 0; i < tris.Length; i += 3) {
          triCount++;
          // Debug.Log($"Start processing triangle no {triCount} in {sw.ElapsedMilliseconds} ms.");
          Vector3 v0 = x.transform.TransformPoint(verts[tris[i]]);
          Vector3 v1 = x.transform.TransformPoint(verts[tris[i + 1]]);
          Vector3 v2 = x.transform.TransformPoint(verts[tris[i + 2]]);
          GRCA(v0, v1, v2);
        }
      }
      sw.Stop();
      Debug.LogError($"GRCA Done! Processed: {_allMeshRenderers.Count} renderers, {triCount} triangles, {_totalSampleCount} rays in {sw.ElapsedMilliseconds} ms.");
      WriteHitsCSV(Path.Combine(Application.dataPath, "hits_grca.csv"));
    }

    //Show hitpoints (Press 'H' key in Game view)
    if (keyboard != null && keyboard.hKey.wasPressedThisFrame) { _isShowHitpoints = !_isShowHitpoints; }
    if (_isShowHitpoints) {
      int visibleCount = 0;
      for (uint vert = 0; vert < VNumSamples; ++vert) {
        for (uint hori = 0; hori < HNumSamples; ++hori) {
          if (_hitPoints[vert, hori].Dist < float.MaxValue) {
            Gizmos.color = HitPointColor;
            Gizmos.DrawSphere(_hitPoints[vert, hori].Pos, 0.02f);
            //Gizmos.DrawLine(transform.position, transform.position + _worldDirs[vert, hori] * RangeMax);
            visibleCount++;
          }
        }
      }
      Debug.Log($"Total intersecting triangles: {visibleCount}");
      // UnityEditor.SceneView.RepaintAll();
    }
  }

  private void WriteHitsCSV(string path) {
    using var sw = new StreamWriter(path);
    sw.WriteLine("dx,dy,dz,dist,hx,hy,hz");
    for (uint vert = 0; vert < VNumSamples; ++vert) {
      for (uint hori = 0; hori < HNumSamples; ++hori) {
        var d = _worldDirs[vert, hori];
        var h = _hitPoints[vert, hori];
        if (h.Dist < float.MaxValue) {
          sw.WriteLine($"{d.x},{d.y},{d.z},{h.Dist},{h.Pos.x},{h.Pos.y},{h.Pos.z}");
        } else {
          var origin = transform.position;
          var hx = origin.x + d.x * RangeMax;
          var hy = origin.y + d.y * RangeMax;
          var hz = origin.z + d.z * RangeMax;
          sw.WriteLine($"{d.x},{d.y},{d.z},{RangeMax},{hx},{hy},{hz}");
        }
      }
    }
    Debug.Log($"Wrote {path}");
  }

  bool IsPointOnPlane(Vector3 checkPoint, Vector3 planePoint, Vector3 planeNormal) {
    return Mathf.Abs(Vector3.Dot(planeNormal, checkPoint - planePoint)) < _epsilon;
  }

  float ClosestDistanceToTriangle(Triangle tri) {
    Vector3 P = transform.position;
    float planeDist = Vector3.Dot(P - tri.Vertices[0], tri.Normal);
    Vector3 proj = P - planeDist * tri.Normal;

    // Precompute all 3 edges and toP = proj - A (reused in both inside test and outside case).
    // Key identity: dot(P-A, edge) = dot(toP, edge)  (planeDist*normal ⊥ edge)
    // and |P - closest|² = |toP - t·edge|² + planeDist²
    var edges = new Vector3[3];
    var toPs  = new Vector3[3];
    for (int e = 0; e < 3; ++e) {
      edges[e] = tri.Vertices[(e + 1) % 3] - tri.Vertices[e];
      toPs[e]  = proj - tri.Vertices[e];
    }

    bool isInside = true;
    for (int e = 0; e < 3; ++e) {
      if (Vector3.Dot(Vector3.Cross(edges[e], toPs[e]), tri.Normal) < 0) { isInside = false; break; }
    }
    if (isInside) return Mathf.Abs(planeDist);

    float minDistSq = float.MaxValue;
    float pd2 = planeDist * planeDist;
    for (int e = 0; e < 3; ++e) {
      float t = Mathf.Clamp01(Vector3.Dot(toPs[e], edges[e]) / Vector3.Dot(edges[e], edges[e]));
      Vector3 delta = toPs[e] - t * edges[e];
      minDistSq = Mathf.Min(minDistSq, Vector3.Dot(delta, delta) + pd2);
    }
    return Mathf.Sqrt(minDistSq);
  }
            
  public bool GRCA_Early_T_Filter(Triangle tri) {
    closestVertDist = float.MaxValue;
    //reject triangles that are on the triangle plane
    if (IsPointOnPlane(transform.position, tri.Center, tri.Normal))                   { return false; }
    Vector3 toCenter = tri.Center - transform.position;
    // 1-1: facing angle — sign of dot product is sufficient, no normalize needed
    // Old: Vector3 toCenterNorm = toCenter.normalized;
    //      float faceAngl = Vector3.Dot(-toCenterNorm, tri.Normal);
    //      if (faceAngl >= 0.0) return false;
    float dotNC = Vector3.Dot(toCenter, tri.Normal);
    if (dotNC <= 0.0f)                                                                { return false; }
    // 1-2: apparent area — squared comparison avoids sqrt entirely
    // Original: tri.Area * cosTheta / distSqr >= _min_apparent_area
    //   where cosTheta = dotNC / sqrt(distSqr)
    // Squared: tri.Area² * dotNC² >= _min_apparent_area² * distSqr³
    float distSqr = Mathf.Max(Vector3.Dot(toCenter, toCenter), _epsilon);
    float lhs = tri.Area * dotNC;
    if (lhs * lhs < _min_apparent_area * _min_apparent_area * distSqr * distSqr * distSqr) { return false; }
    // // 1-3: range
    // closestVertDist = ClosestDistanceToTriangle(tri);
    // if (closestVertDist < RangeMin || closestVertDist > RangeMax)                     { return false; }
    return true;
  }

  public Vector3 ComputeRayDirection(float hAngle, float vAngle) {
    //clockwise Unity
    float cosH = Mathf.Cos(hAngle), sinH = Mathf.Sin(hAngle), cosV = Mathf.Cos(vAngle), sinV = Mathf.Sin(vAngle); //+ve sinV makes 0th channel point downwards
    //counter-clockwise ROS
    // float cosH = Mathf.Cos(hAngle), sinH = -Mathf.Sin(hAngle), cosV = Mathf.Cos(vAngle), sinV = Mathf.Sin(vAngle);
    // Compute the rotated forward vector using yaw (horizontal rotation) and pitch (vertical rotation)
    Vector3 direction = (cosH * transform.forward + sinH * transform.right) * cosV + sinV * transform.up;
    return direction.normalized;
  }

  // Old: in float halfAngleRad — boundary check was Mathf.Abs(halfAngleRad - PI/2) <= _epsilon, cosTheta = Mathf.Cos(halfAngleRad)
  bool GRCA_Fast_GACPT_IntersectionCheck(in bool isFlipped, in bool isTriIntersectConeDir, in Vector3 apex, in Vector3 direction, in float cosHalfAngle, in Triangle tri, in float[] signdDist, in float[] signdDistAngle, float closestVertDist) {
    if (Mathf.Abs(cosHalfAngle) <= _epsilon) {
      // Perform intersection with the plane (cone surface becomes a plane)
      // Determine which edges intersect the plane
      bool edge01Intersects = (signdDist[0] * signdDist[1] < 0);
      bool edge12Intersects = (signdDist[1] * signdDist[2] < 0);
      bool edge20Intersects = (signdDist[2] * signdDist[0] < 0);

      // Select the intersection points from intersecting edges
      return (edge01Intersects && edge12Intersects) || (edge12Intersects && edge20Intersects) || (edge20Intersects && edge01Intersects);
    } else {
      float cosTheta = cosHalfAngle;

      bool isAllVertexInCone = isFlipped
                        ? -signdDistAngle[0] >= cosTheta && -signdDistAngle[1] >= cosTheta && -signdDistAngle[2] >= cosTheta
                        : signdDistAngle[0] >= cosTheta && signdDistAngle[1] >= cosTheta && signdDistAngle[2] >= cosTheta;
      if (isAllVertexInCone) { return false; }

      bool isNoVertexInCone = isFlipped
                        ? -signdDistAngle[0] < cosTheta && -signdDistAngle[1] < cosTheta && -signdDistAngle[2] < cosTheta
                        : signdDistAngle[0] < cosTheta && signdDistAngle[1] < cosTheta && signdDistAngle[2] < cosTheta;
      if (!isNoVertexInCone) { return true; } // at least one vertex in cone volume → guaranteed edge crossing
      if (isNoVertexInCone && isTriIntersectConeDir) { return closestVertDist <= RangeMax * cosTheta; }

      {
        float cos2 = cosTheta * cosTheta;
        int i0 = 0, i1 = 0, e = 0;
        float c0 = 0, c1 = 0, c2 = 0, discr = 0, invDenom = 0, sqrtD = 0, t0 = 0, t1 = 0;

        for (; e < 3; ++e) {
          i0 = _edgeIndices[e, 0];
          i1 = _edgeIndices[e, 1];

          Vector3 p0 = tri.Vertices[i0] - apex;
          Vector3 p1 = tri.Vertices[i1] - apex;
          Vector3 d = p1 - p0;                   // Edge direction

          float ddA = Vector3.Dot(d, direction);
          float p0dA = Vector3.Dot(p0, direction);
          float ddD = Vector3.Dot(d, d);
          float p0dD = Vector3.Dot(p0, d);
          float p0dP0 = Vector3.Dot(p0, p0);

          c2 = (ddA * ddA) - cos2 * ddD;
          c1 = 2 * (ddA * p0dA - cos2 * p0dD);
          c0 = (p0dA * p0dA) - cos2 * p0dP0;

          // Solve quadratic c2 * t^2 + c1 * t + c0 = 0
          discr = c1 * c1 - 4.0f * c2 * c0;

          // Skip degenerate / nearly linear cases
          if (Mathf.Abs(c2) < _epsilon || discr < 0) { continue; }
          discr = Mathf.Max(discr, 0);
          invDenom = 0.5f / c2;
          sqrtD = Mathf.Sqrt(discr);
          t0 = (-c1 - sqrtD) * invDenom;
          t1 = (-c1 + sqrtD) * invDenom;

          // direction = ±up → dot(lerp(v[i0],v[i1],t) - apex, dir) = p0dA + t*ddA
          // Original: var temp = Vector3.Lerp(..., Clamp(t,0,1)); if (Dot(temp-apex, dir) >= 0) return true;
          if (t0 >= 0 && t0 <= 1 && p0dA + t0 * ddA >= 0) return true;
          if (t1 >= 0 && t1 <= 1 && p0dA + t1 * ddA >= 0) return true;
        }
        return false;
      }
    }
  }

  // Old: in float halfAngleRad — boundary check was Mathf.Abs(halfAngleRad - PI/2) <= _epsilon, cosTheta = Mathf.Cos(halfAngleRad)
  bool GRCA_GACP_T_IntersectionCheck(in bool isFlipped, in bool isTriIntersectConeDir, in Vector3 apex, in Vector3 direction, in float cosHalfAngle, in Triangle tri, in float[] signdDist, in float[] signdDistAngle, float closestVertDist, out Vector3[] resultPoints, out int count, out bool isCheckAll) {
    count = 0;
    isCheckAll = false;
    resultPoints = new Vector3[_maxIntersections];
    if (Mathf.Abs(cosHalfAngle) <= _epsilon) {
      // Perform intersection with the plane (cone surface becomes a plane)
      // Determine which edges intersect the plane
      bool edge01Intersects = (signdDist[0] * signdDist[1] < 0);
      bool edge12Intersects = (signdDist[1] * signdDist[2] < 0);
      bool edge20Intersects = (signdDist[2] * signdDist[0] < 0);

      if (edge01Intersects && edge12Intersects) {
        resultPoints[0] = GetTriPlaneIntersectPoint(tri.Vertices[0], tri.Vertices[1], signdDist[0], signdDist[1]);
        resultPoints[1] = GetTriPlaneIntersectPoint(tri.Vertices[1], tri.Vertices[2], signdDist[1], signdDist[2]);
        return true;
      }
      if (edge12Intersects && edge20Intersects) {
        resultPoints[0] = GetTriPlaneIntersectPoint(tri.Vertices[1], tri.Vertices[2], signdDist[1], signdDist[2]);
        resultPoints[1] = GetTriPlaneIntersectPoint(tri.Vertices[2], tri.Vertices[0], signdDist[2], signdDist[0]);
        return true;
      }
      if (edge20Intersects && edge01Intersects) {
        resultPoints[0] = GetTriPlaneIntersectPoint(tri.Vertices[2], tri.Vertices[0], signdDist[2], signdDist[0]);
        resultPoints[1] = GetTriPlaneIntersectPoint(tri.Vertices[0], tri.Vertices[1], signdDist[0], signdDist[1]);
        return true;
      }
    } else {
      float cosTheta = cosHalfAngle;

      bool isAllVertexInCone = isFlipped
                        ? -signdDistAngle[0] >= cosTheta && -signdDistAngle[1] >= cosTheta && -signdDistAngle[2] >= cosTheta
                        : signdDistAngle[0] >= cosTheta && signdDistAngle[1] >= cosTheta && signdDistAngle[2] >= cosTheta;
      if (isAllVertexInCone) { return false; }

      bool isNoVertexInCone = isFlipped
                        ? -signdDistAngle[0] < cosTheta && -signdDistAngle[1] < cosTheta && -signdDistAngle[2] < cosTheta
                        : signdDistAngle[0] < cosTheta && signdDistAngle[1] < cosTheta && signdDistAngle[2] < cosTheta;
      if (!isNoVertexInCone) { return true; } // at least one vertex in cone volume → guaranteed edge crossing
      isCheckAll = isNoVertexInCone && isTriIntersectConeDir && closestVertDist <= RangeMax * cosTheta;

      if (isCheckAll) {
        return false; // caller handles isCheckAll=true as a full-sweep
      } else {
        float cos2 = cosTheta * cosTheta;
        int i0 = 0, i1 = 0, e = 0;
        float c0 = 0, c1 = 0, c2 = 0, discr = 0, invDenom = 0, sqrtD = 0, t0 = 0, t1 = 0;

        for (; e < 3; ++e) {
          if (count > 2) { break; }
          i0 = _edgeIndices[e, 0];
          i1 = _edgeIndices[e, 1];

          Vector3 p0 = tri.Vertices[i0] - apex;
          Vector3 p1 = tri.Vertices[i1] - apex;
          Vector3 d = p1 - p0;                   // Edge direction

          float ddA = Vector3.Dot(d, direction);
          float p0dA = Vector3.Dot(p0, direction);
          float ddD = Vector3.Dot(d, d);
          float p0dD = Vector3.Dot(p0, d);
          float p0dP0 = Vector3.Dot(p0, p0);

          c2 = (ddA * ddA) - cos2 * ddD;
          c1 = 2 * (ddA * p0dA - cos2 * p0dD);
          c0 = (p0dA * p0dA) - cos2 * p0dP0;

          // Solve quadratic c2 * t^2 + c1 * t + c0 = 0
          discr = c1 * c1 - 4.0f * c2 * c0;

          // Skip degenerate / nearly linear cases
          if (Mathf.Abs(c2) < _epsilon || discr < 0) { continue; }
          discr = Mathf.Max(discr, 0);
          invDenom = 0.5f / c2;
          sqrtD = Mathf.Sqrt(discr);
          t0 = (-c1 - sqrtD) * invDenom;
          t1 = (-c1 + sqrtD) * invDenom;

          // direction = ±up → dot(lerp(v[i0],v[i1],t) - apex, dir) = p0dA + t*ddA
          // Original: var temp = Vector3.Lerp(..., Clamp(t,0,1)); if (Dot(temp-apex, dir) >= 0) resultPoints[count++] = temp;
          if (t0 >= 0 && t0 <= 1 && p0dA + t0 * ddA >= 0)
            resultPoints[count++] = Vector3.Lerp(tri.Vertices[i0], tri.Vertices[i1], t0);
          if (t1 >= 0 && t1 <= 1 && p0dA + t1 * ddA >= 0)
            resultPoints[count++] = Vector3.Lerp(tri.Vertices[i0], tri.Vertices[i1], t1);
        }
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
    }
    return false;
  }

  public void GRCA_CSpan_Predict(bool isBehindCone, bool isBehindConeFlipped, bool isTriIntersectConeDir, bool isTriIntersectConeDirFlipped, Triangle tri, float[] signdDist, float[] signdDistAngle, float closestVertDist, ref uint fromChannelId, ref uint toChannelId) {
    fromChannelId = VNumSamples; // Invalid
    toChannelId = 0;             // Invalid
    uint estimFromChannelId = 0, estimToChannelId = VNumSamples - 1;
    // Compute sign flags
    bool above0 = signdDist[0] > 0;
    bool above1 = signdDist[1] > 0;
    bool above2 = signdDist[2] > 0;

    // Classification
    bool allAbove = above0 && above1 && above2;
    bool allBelow = !above0 && !above1 && !above2;

    if (allAbove) {
      estimFromChannelId = VNumSamples / 2;
      //Debug.LogWarning("isAbove");
    } else if (allBelow) {
      estimToChannelId = VNumSamples / 2;
      //Debug.LogWarning("isBelow");
    } else {
      //Debug.LogWarning("isIntersecting");
    }

    uint id0 = GRCA_AngularChannelIndexing(signdDistAngle[0]);
    uint id1 = GRCA_AngularChannelIndexing(signdDistAngle[1]);
    uint id2 = GRCA_AngularChannelIndexing(signdDistAngle[2]);
    var closestCenterVId = (uint)Mathf.CeilToInt(0.5f * (Math.Max(id0, Math.Max(id1, id2)) + Math.Min(id0, Math.Min(id1, id2))));
    closestCenterVId = (uint)Mathf.Min(estimToChannelId, closestCenterVId);

    // Binary search for lowest valid channel
    // Old: float angle = Mathf.Acos(Mathf.Sin(vAngle));
    //      bool isFlipped = angle > Mathf.PI / 2;
    //      if (isFlipped) { angle = Mathf.PI - angle; coneDir = -coneDir; }
    //      float cosHalf = Mathf.Cos(angle);   // = Mathf.Abs(Mathf.Sin(vAngle))
    int low = (int)estimFromChannelId;
    int high = (int)closestCenterVId;
    int mid;

    while (low <= high) {
      mid = low + ((high - low) >> 1);
      float vAngle = VAngleMin + mid * _vAngleIncrement;
      bool isFlipped = vAngle < 0;
      float cosHalf = Mathf.Abs(Mathf.Sin(vAngle));
      var coneDir = isFlipped ? -transform.up : transform.up;
      bool isMirrored = isFlipped ? isBehindConeFlipped : isBehindCone;
      bool isIntersect = isFlipped ? isTriIntersectConeDirFlipped : isTriIntersectConeDir;
      if (!isMirrored && GRCA_Fast_GACPT_IntersectionCheck(isFlipped, isIntersect, transform.position, coneDir, cosHalf, tri, signdDist, signdDistAngle, closestVertDist)) {
        fromChannelId = (uint)mid;
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    low = (int)Math.Min(fromChannelId, estimToChannelId);
    high = (int)estimToChannelId;

    // Old: (same Acos/Sin/Cos chain as lower-bound search above)
    while (low <= high) {
      mid = low + ((high - low) >> 1);
      float vAngle = VAngleMin + mid * _vAngleIncrement;
      bool isFlipped = vAngle < 0;
      float cosHalf = Mathf.Abs(Mathf.Sin(vAngle));
      var coneDir = isFlipped ? -transform.up : transform.up;
      bool isMirrored = isFlipped ? isBehindConeFlipped : isBehindCone;
      bool isIntersect = isFlipped ? isTriIntersectConeDirFlipped : isTriIntersectConeDir;
      if (!isMirrored && GRCA_Fast_GACPT_IntersectionCheck(isFlipped, isIntersect, transform.position, coneDir, cosHalf, tri, signdDist, signdDistAngle, closestVertDist)) {
        toChannelId = (uint)mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
  }

  // Old: float halfAngleRad — passed through to GRCA_GACP_T_IntersectionCheck
  public bool GRCA_RSpan_Predict(bool isAllClockWise, bool isFlipped, bool isIntersect, uint channelId, Vector3 apex, Vector3 direction, float cosHalfAngle, Triangle tri, float distCenterNormal, in float[] signdDist, float[] signdDistAngle, float closestVertDist, out uint fromSweepId, out uint sweepDiff, out bool isClockWise) {
    fromSweepId = 0;
    sweepDiff = 0;
    isClockWise = true;
    if (!GRCA_GACP_T_IntersectionCheck(isFlipped, isIntersect, apex, direction, cosHalfAngle, tri, signdDist, signdDistAngle, closestVertDist, out Vector3[] intersectionPoints, out int count, out bool isCheckAll)) {
      if (isCheckAll) {
        fromSweepId = 0;
        sweepDiff = HNumSamples;
        isClockWise = true;
        return true;
      }
      // if (count == 0) {
      //   if (isCheckAll) {
      //     Debug.LogWarning($"All Check! channelId {channelId} possibly has all rays intersecting due to triangle completely overlapping the it!");
      //   } else {
      //     Debug.LogWarning($"No Check! channelId {channelId} has no intersections!");
      //   }
      // } else if (count > 2) {
      //   Debug.LogWarning($"All Check! channelId {channelId} has more than two: {count} intersections!");
      // } else {
      //   Debug.LogError($"This shouldn't happen! channelId {channelId}, {count} intersections!");
      // }
      return false;
    }

    GRCA_AngularSweepIndexing(HAngleMin, _hAngleIncrement, HNumSamples, transform.position, transform.right, transform.forward, transform.up, intersectionPoints[0], out uint closestRayId0);
    GRCA_AngularSweepIndexing(HAngleMin, _hAngleIncrement, HNumSamples, transform.position, transform.right, transform.forward, transform.up, intersectionPoints[1], out uint closestRayId1);

    fromSweepId = Math.Min(closestRayId0, closestRayId1);
    uint toSweepId = Math.Max(closestRayId0, closestRayId1);
    float vAngle = VAngleMin + channelId * _vAngleIncrement;
    Vector3 dirA = ComputeRayDirection(HAngleMin + fromSweepId * _hAngleIncrement, vAngle);
    Vector3 dirB = ComputeRayDirection(HAngleMin + toSweepId * _hAngleIncrement, vAngle);
    Vector3 midDir = (dirA + dirB).normalized;
    Vector3 intersectionMid = (intersectionPoints[0] + intersectionPoints[1]) / 2.0f;

    // Gizmos.color = Color.red;
    // Gizmos.DrawLine(transform.position, transform.position + dirA);
    // Gizmos.DrawLine(transform.position, transform.position + dirB);
    // Gizmos.DrawLine(transform.position, transform.position + midDir);
    // Gizmos.DrawSphere(intersectionPoints[0], 0.1f);
    // Gizmos.DrawSphere(intersectionPoints[1], 0.1f);
    // Gizmos.DrawLine(intersectionPoints[0], intersectionPoints[1]);

    if (fromSweepId != toSweepId) {
      if (isAllClockWise) {
        uint stepsCW = GetAngularRayDiff(fromSweepId, toSweepId, HNumSamples, true);
        sweepDiff = stepsCW;
        isClockWise = true;
      } else {
        uint stepsCW = 1 + ((uint)toSweepId - (uint)fromSweepId + HNumSamples) % HNumSamples;
        uint midRayIdCW = ((uint)fromSweepId + stepsCW / 2) % HNumSamples;
        Vector3 midRayDirCW = ComputeRayDirection(HAngleMin + midRayIdCW * _hAngleIncrement, vAngle);
        uint stepsCCW = 1 + ((uint)fromSweepId - (uint)toSweepId + HNumSamples) % HNumSamples;
        // uint midRayIdCCW = (fromSweepId - stepsCCW / 2 + HNumSamples) % HNumSamples;
        // Vector3 midRayDirCCW = ComputeRayDirection(HAngleMin + midRayIdCCW * _hAngleIncrement, vAngle);
        Vector3 midRayDirCCW = 2 * Vector3.Dot(midRayDirCW, transform.up) * transform.up - midRayDirCW;

        float denom = Vector3.Dot(tri.Normal, midRayDirCW);
        float dist = distCenterNormal / denom;
        bool isCWHit = denom >= _epsilon && dist >= 0;
        denom = Vector3.Dot(tri.Normal, midRayDirCCW);
        dist = distCenterNormal / denom;
        bool isCCWHit = denom >= _epsilon && dist >= 0;

        if (isCWHit && !isCCWHit) {
          isClockWise = true;
          // M_T_RayTriangleIntersect(transform.position, tri, midRayDirCW, out var intersectPointCW);
          // Gizmos.DrawSphere(intersectPointCW, 0.1f);
          // Gizmos.DrawLine(transform.position, intersectPointCW);
        } else if (isCCWHit && !isCWHit) {
          isClockWise = false;
          // M_T_RayTriangleIntersect(transform.position, tri, midRayDirCCW, out var intersectPointCCW);
          // Gizmos.DrawSphere(intersectPointCCW, 0.1f);
          // Gizmos.DrawLine(transform.position, intersectPointCCW);
        } else {
          // Gizmos.DrawSphere(intersectionMid, 0.1f);
          // Gizmos.DrawLine(transform.position, intersectionMid);
          isClockWise = M_T_RayTriangleIntersect(transform.position, tri, midRayDirCW, out var intersectPointCW) < float.MaxValue;
        }
        sweepDiff = isClockWise ? stepsCW - 1 : stepsCCW - 1;
      }
    }
    return true;
  }

  // Takes pre-computed sine of elevation angle (signdDistAngle component).
  // Old: Vector3 direction = (checkPos - planePos).normalized; float angle = Mathf.Asin(Vector3.Dot(direction, up));
  private uint GRCA_AngularChannelIndexing(float sa) {
    float angle = Mathf.Asin(sa);
    return Math.Clamp((uint)Mathf.RoundToInt((angle - VAngleMin) / _vAngleIncrement), 0, VNumSamples - 1);
  }

  private void GRCA_AngularSweepIndexing(in float startAngle, in float angularStep, in uint numSamples, in Vector3 planePos, in Vector3 right, in Vector3 forward, in Vector3 up, in Vector3 checkPos, out uint closestRayId) {
    // Original (for math reference):
    // Vector3 direction = (checkPos - planePos).normalized;
    // Vector3 horizontalDir = (direction - Vector3.Dot(direction, up) * up).normalized;
    // float angle = Mathf.Atan2(Vector3.Dot(horizontalDir, right), Vector3.Dot(horizontalDir, forward)); //clockwise Unity
    // float angle = Mathf.Atan2(Vector3.Dot(horizontalDir, -right), Vector3.Dot(horizontalDir, forward)); //counter-clockwise ROS
    // Atan2 is scale-invariant so both .normalized calls are unnecessary.
    Vector3 raw  = checkPos - planePos;
    Vector3 proj = raw - Vector3.Dot(raw, up) * up;
    float angle = Mathf.Atan2(Vector3.Dot(proj, right), Vector3.Dot(proj, forward)); //clockwise Unity
    // float angle = Mathf.Atan2(Vector3.Dot(proj, -right), Vector3.Dot(proj, forward)); //counter-clockwise ROS
    closestRayId = Math.Clamp((uint)Mathf.RoundToInt((angle - startAngle) / angularStep), 0, numSamples - 1);
  }

  private uint GetAngularRayDiff(uint id0, uint id1, uint numSamples, in bool isClockWise) {
    uint forwardSteps = (id1 - id0 + numSamples) % numSamples;
    uint backwardSteps = (id0 - id1 + numSamples) % numSamples;

    if (isClockWise) {
      return forwardSteps;
    } else {
      return backwardSteps;
    }
  }

  private uint GetNextClosestRayId(uint startId, uint next, uint numSamples, bool isClockWise) {
    if (isClockWise) {
      return (startId + next) % numSamples;
    } else {
      return (startId - next + numSamples) % numSamples;
    }
  }

  public void GRCA_Fast_RSpan_Predict(Triangle tri, Vector3[] originToVert, out bool isAllClockWise, out uint estimFromSweepId, out uint estimSweepDiff) {
    estimFromSweepId = 0;
    estimSweepDiff   = (uint)SatMaxSweepDiff; // default = MAX_PREPROCESS_SWEEP_COUNT (force BAT if not all-CW)
    bool isCCWPossible = HAngleMax - HAngleMin > Mathf.PI;
    if (isCCWPossible) {
      int midVId = Mathf.RoundToInt(VNumSamples / 2.0f);
      int midHId = Mathf.RoundToInt(HNumSamples / 2.0f);
      // Use the precomputed world-space ray directions (matches GPU / Burst behavior)
      var midDir = _worldDirs[midVId, midHId];
      // All triangles in front of the lidar should be clockwise
      bool isAllFront = (Vector3.Dot(originToVert[0], midDir) > 0f) &&
            (Vector3.Dot(originToVert[1], midDir) > 0f) &&
            (Vector3.Dot(originToVert[2], midDir) > 0f);
      // Build right vector perpendicular to midDir via Gram-Schmidt.
      var worldRef = (Mathf.Abs(midDir.y) < 0.9f) ? Vector3.up : Vector3.right;
      var up = (worldRef - Vector3.Dot(worldRef, midDir) * midDir).normalized;
      var right = Vector3.Cross(midDir, up).normalized;
      // Triangle fully left or right of midDir — no seam wrap possible.
      bool isAllRight = (Vector3.Dot(originToVert[0], right) > 0f) &&
            (Vector3.Dot(originToVert[1], right) > 0f) &&
            (Vector3.Dot(originToVert[2], right) > 0f);
      bool isAllLeft  = (Vector3.Dot(originToVert[0], right) < 0f) &&
            (Vector3.Dot(originToVert[1], right) < 0f) &&
            (Vector3.Dot(originToVert[2], right) < 0f);
      isAllClockWise = isAllFront || isAllRight || isAllLeft;
    } else {
      isAllClockWise = true;
    }
    if (!isAllClockWise) { return; } // estimSweepDiff stays at SatMaxSweepDiff → forces BAT
    // Vertex-based horizontal sweep extent
    GRCA_AngularSweepIndexing(HAngleMin, _hAngleIncrement, HNumSamples, transform.position, transform.right, transform.forward, transform.up, tri.Vertices[0], out uint hId0);
    GRCA_AngularSweepIndexing(HAngleMin, _hAngleIncrement, HNumSamples, transform.position, transform.right, transform.forward, transform.up, tri.Vertices[1], out uint hId1);
    GRCA_AngularSweepIndexing(HAngleMin, _hAngleIncrement, HNumSamples, transform.position, transform.right, transform.forward, transform.up, tri.Vertices[2], out uint hId2);
    estimFromSweepId  = Math.Min(hId0, Math.Min(hId1, hId2));
    uint estimToSweepId = Math.Max(hId0, Math.Max(hId1, hId2));
    estimSweepDiff    = GetAngularRayDiff(estimFromSweepId, estimToSweepId, HNumSamples, true);
  }

  Vector3 GetTriPlaneIntersectPoint(Vector3 vert0, Vector3 vert1, float signedDist0, float signedDist1) {
    return vert0 + signedDist0 / (signedDist0 - signedDist1) * (vert1 - vert0);
  }

  public float M_T_RayTriangleIntersect(Vector3 origin, Triangle tri, Vector3 direction, out Vector3 intersectPoint) {
    intersectPoint = Vector3.zero;
    Vector3 tmp = Vector3.Cross(direction, tri.Edge2); //ray_cross_e2
    float inv_det = 1 / Vector3.Dot(tri.Edge1, tmp);  //inv_det;

    // Check if parallel  
    // if (abs(det) < _epsilon)                 { return FLT_MAX; } //ToDo: Only in use for pre-processed triangles, so commented out

    Vector3 s = origin - tri.Vertices[0];
    float u = inv_det * Vector3.Dot(s, tmp);

    if (u < 0 || u > 1) { return float.MaxValue; }

    tmp = Vector3.Cross(s, tri.Edge1); //s_cross_e1
    float tmpflt = inv_det * Vector3.Dot(direction, tmp); //v

    if (tmpflt < 0 || u + tmpflt > 1) { return float.MaxValue; }

    // At this stage we can compute t to find out where the intersection point is on the line.
    tmpflt = inv_det * Vector3.Dot(tri.Edge2, tmp); //t
    if (tmpflt > _epsilon) {
      intersectPoint = transform.position + direction * tmpflt;
    }
    // ray intersection distance
    return tmpflt > _epsilon ? tmpflt : float.MaxValue;
  }

  private void GRCA(Vector3 v0, Vector3 v1, Vector3 v2) {
    Triangle tri = BuildTriangle(v0, v1, v2);
    var toCenter = tri.Center - transform.position;
    if (!GRCA_Early_T_Filter(tri, out float closestVertDist)) { return; }

    Vector3[] originToVert = new Vector3[3]{tri.Vertices[0] - transform.position
                              , tri.Vertices[1] - transform.position
                              , tri.Vertices[2] - transform.position};
    float[] signdDist = new float[3] {Vector3.Dot(originToVert[0], transform.up)
                              , Vector3.Dot(originToVert[1], transform.up)
                              , Vector3.Dot(originToVert[2], transform.up)};
    float[] signdDistAngle = new float[3] {Vector3.Dot(originToVert[0].normalized, transform.up)
                              , Vector3.Dot(originToVert[1].normalized, transform.up)
                              , Vector3.Dot(originToVert[2].normalized, transform.up)};
    // var estimChannelDiff = Mathf.CeilToInt(area / (_vAngleIncrement * _vAngleIncrement));
    // estimChannelDiff = Mathf.CeilToInt(estimChannelDiff * 0.5f);
    // Debug.LogWarning(estimChannelDiff);
    float distCenterNormal = Vector3.Dot(tri.Normal, toCenter);
    bool isBehindCone = signdDist[0] < 0 && signdDist[1] < 0 && signdDist[2] < 0;
    bool isBehindConeFlipped = -signdDist[0] < 0 && -signdDist[1] < 0 && -signdDist[2] < 0;
    bool isTriIntersectConeDir = M_T_RayTriangleIntersect(transform.position, tri, transform.up, out _) < float.MaxValue;
    bool isTriIntersectConeDirFlipped = M_T_RayTriangleIntersect(transform.position, tri, -transform.up, out _) < float.MaxValue;

    uint fromChannelId = 0, toChannelId = 0;
    if (VNumSamples > 1) {
      GRCA_CSpan_Predict(isBehindCone, isBehindConeFlipped, isTriIntersectConeDir, isTriIntersectConeDirFlipped, tri, signdDist, signdDistAngle, closestVertDist, ref fromChannelId, ref toChannelId);
      if (fromChannelId > toChannelId || fromChannelId == VNumSamples) {
        // Debug.LogError("Can't find channel span!");
        return; // No valid range found
      }
    }
    //Debug.LogWarning($"fromChannelId: {fromChannelId}, toChannelId: {toChannelId}, channelDiff: {toChannelId - fromChannelId}");

    // ── SAT / BAT classification (matches GPU GRCA_Early_Pass_Core) ──
    GRCA_Fast_RSpan_Predict(tri, originToVert, out bool isAllClockWise, out uint estimFromSweepId, out uint estimSweepDiff);
    uint channelDiff = toChannelId - fromChannelId;
    bool isSAT = isAllClockWise && channelDiff < SatMaxChannelDiff && estimSweepDiff < SatMaxSweepDiff;

    if (isSAT) {
      for (uint channelId = fromChannelId; channelId <= toChannelId; ++channelId) {
        for (uint i = 0; i <= estimSweepDiff; ++i) {
          var rayId = GetNextClosestRayId(estimFromSweepId, i, HNumSamples, isAllClockWise);
          // // partial occlusion using closest vertex distance heuristic
          // if (closestVertDist > _hitPoints[channelId, rayId].Dist) { continue; }
          var dist = M_T_RayTriangleIntersect(transform.position, tri, _worldDirs[channelId, rayId], out var intersectPoint);
          if (dist < float.MaxValue && dist < _hitPoints[channelId, rayId].Dist) {
            _hitPoints[channelId, rayId].Dist = dist;
            _hitPoints[channelId, rayId].Pos = intersectPoint;
          }
        }
      }
    } else {
      for (uint channelId = fromChannelId; channelId <= toChannelId; ++channelId) {
        // Old: float angle = Mathf.Acos(Mathf.Sin(vAngle));
        //      bool isFlipped = angle > Mathf.PI / 2;
        //      if (isFlipped) angle = Mathf.PI - angle;
        //      float cosHalf = Mathf.Cos(angle);   // = Mathf.Abs(Mathf.Sin(vAngle))
        float vAngle  = VAngleMin + channelId * _vAngleIncrement;
        bool isFlipped = vAngle < 0;
        float cosHalf  = Mathf.Abs(Mathf.Sin(vAngle));
        var dir = isFlipped ? -transform.up : transform.up;
        bool isMirrored = isFlipped ? isBehindConeFlipped : isBehindCone;
        bool isIntersect = isFlipped ? isTriIntersectConeDirFlipped : isTriIntersectConeDir;
        if (isMirrored) { continue; }
        if (GRCA_RSpan_Predict(isAllClockWise, isFlipped, isIntersect, channelId, transform.position, dir, cosHalf, tri, distCenterNormal, signdDist, signdDistAngle, closestVertDist, out uint fromSweepId, out uint sweepDiff, out bool isClockWise)) {
          for (uint i = 0; i <= sweepDiff; ++i) {
            var rayId = GetNextClosestRayId(fromSweepId, i, HNumSamples, isClockWise);
            // // partial occlusion using closest vertex distance heuristic
            // if (closestVertDist > _hitPoints[channelId, rayId].Dist) { continue; }
            var dist = M_T_RayTriangleIntersect(transform.position, tri, _worldDirs[channelId, rayId], out var intersectPoint);
            if (dist < float.MaxValue && dist < _hitPoints[channelId, rayId].Dist) {
              _hitPoints[channelId, rayId].Dist = dist;
              _hitPoints[channelId, rayId].Pos = intersectPoint;
            }
          }
          //Debug.LogWarning($"channelId {channelId} has intersections! sweepDiff {sweepDiff}, isClockWise; {isClockWise}, fromSweepId: {fromSweepId}, rays: {rayIdStr}");
        }
        // Debug.LogWarning($"CW: {cwCounter}, CCW: {ccwCounter}, isUpDownIntersect: {isTriIntersectConeDir || isTriIntersectConeDirFlipped}");
      }
    }
  }
}
