using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Attach to any GameObject in the scene.
/// Left-click in the Game view to select a triangle under the mouse pointer.
/// Press 'D' in the Game view to run both Raw M-T and GRCA on the currently selected triangle
/// with full per-step debug output to the Console.
/// Visualizes the selected triangle, its vertices, and all hit/miss points.
/// </summary>
public class GRCATriDebugger : ARSim.MonoBehaviour {

  [Header("Selection & References")]
  public Color HighlightColor = Color.yellow;
  public float HighlightNormalLength = 0.1f;
  public LayerMask SelectionLayers = ~0; // All layers by default
  public GrcaLidarCpu LidarCpu;           // source of all LiDAR parameters + transform

  [Header("Hit Visualization")]
  public Color RmtHitColor   = Color.green;
  public Color GrcaHitColor   = Color.red;
  public Color GrcaMissColor  = new Color(1f, 0.5f, 0f);  // orange — rays tested by GRCA but no hit
  public float HitSphereRadius = 0.03f;
  public bool  DrawRayLines    = false;  // draw ray lines from sensor to hit point

  // ---- public read-only result ----
  [Header("Last Selected Triangle (read-only)")]
  public string SelectedRendererName = "";
  public string SelectedMeshName = "";
  public int SelectedTriangleIndex = -1;   // index into mesh.triangles / 3
  public Vector3 SelectedVertex0;
  public Vector3 SelectedVertex1;
  public Vector3 SelectedVertex2;

  // ---- private state ----
  private bool _hasSelection = false;
  private List<MeshRenderer> _allMeshRenderers = new List<MeshRenderer>();
  private Camera _mirrorCam;   // Game-view camera that mirrors the Scene view camera

  // stored results for gizmo drawing
  private readonly List<Vector3> _rmtHits  = new List<Vector3>();
  private readonly List<Vector3> _grcaHits  = new List<Vector3>();
  private readonly List<Vector3> _grcaMisses = new List<Vector3>(); // ray origin end-points for misses

  private struct HitResult {
    public MeshRenderer Renderer;
    public Mesh Mesh;
    public int TriangleIndex;
    public Vector3 V0, V1, V2;
    public float Distance;
  }

  private static readonly int[,] _edgeIdx = { { 0, 1 }, { 1, 2 }, { 2, 0 } };
  private const float _eps = 0.0001f;

  // -----------------------------------------------------------------------
  // Unity lifecycle
  // -----------------------------------------------------------------------

  private void Awake() {
    _allMeshRenderers = FindObjectsOfType<MeshRenderer>(true)
      .Where(x => ((1 << x.gameObject.layer) & SelectionLayers) != 0)
      .ToList();
    if (_allMeshRenderers.Count == 0) {
      Debug.LogWarning("[GRCATriDebugger] Awake: No MeshRenderers found matching SelectionLayers. Selection will always miss.");
    } else {
      Debug.Log($"[GRCATriDebugger] Awake: Registered {_allMeshRenderers.Count} MeshRenderer(s) for triangle selection.");
    }

    var mirrorGO = new GameObject("[GRCATriDebugger] MirrorCam");
    mirrorGO.transform.SetParent(transform, false);
    _mirrorCam = mirrorGO.AddComponent<Camera>();
    _mirrorCam.enabled = true;
    _mirrorCam.depth = -100;
    _mirrorCam.clearFlags = CameraClearFlags.Nothing;
    _mirrorCam.cullingMask = 0;
    if (mirrorGO.TryGetComponent<AudioListener>(out var al)) { Destroy(al); }
    Debug.Log("[GRCATriDebugger] Awake: MirrorCam created.");
  }

  private void Update() {
#if UNITY_EDITOR
    var sceneView = UnityEditor.SceneView.lastActiveSceneView;
    if (sceneView != null && sceneView.camera != null && _mirrorCam != null) {
      Camera sv = sceneView.camera;
      _mirrorCam.transform.position   = sv.transform.position;
      _mirrorCam.transform.rotation   = sv.transform.rotation;
      _mirrorCam.fieldOfView          = sv.fieldOfView;
      _mirrorCam.aspect               = sv.aspect;
      _mirrorCam.nearClipPlane        = sv.nearClipPlane;
      _mirrorCam.farClipPlane         = sv.farClipPlane;
      _mirrorCam.orthographic         = sv.orthographic;
      _mirrorCam.orthographicSize     = sv.orthographicSize;
    }
#endif
    if (Input.GetMouseButtonDown(0)) {
      if (_mirrorCam == null) {
        Debug.LogWarning("[GRCATriDebugger] Update: _mirrorCam is null, cannot cast ray.");
        return;
      }
      Vector2 mousePos = Input.mousePosition;
      Vector3 viewportPoint = new Vector3(mousePos.x / Screen.width, mousePos.y / Screen.height, 0f);
      Ray ray = _mirrorCam.ViewportPointToRay(viewportPoint);
      Debug.Log($"[GRCATriDebugger] Update: Click at {mousePos}, viewport={viewportPoint:F3}, ray origin={ray.origin:F3}, dir={ray.direction:F3}");
      TrySelectTriangle(ray);
    }
  }

  // -----------------------------------------------------------------------
  // Selection logic
  // -----------------------------------------------------------------------

  private void TrySelectTriangle(Ray ray) {
    HitResult best = new HitResult { Distance = float.MaxValue, TriangleIndex = -1 };
    if (_allMeshRenderers.Count == 0) {
      Debug.LogWarning("[GRCATriDebugger] TrySelectTriangle: Renderer list is empty — call Awake() or check SelectionLayers.");
      return;
    }
    int skippedNull = 0, skippedNoFilter = 0, skippedNoMesh = 0;
    foreach (var renderer in _allMeshRenderers) {
      if (renderer == null || !renderer.enabled) { skippedNull++; continue; }
      if (!renderer.TryGetComponent<MeshFilter>(out var mf)) { skippedNoFilter++; Debug.LogWarning($"[GRCATriDebugger] '{renderer.gameObject.name}' has no MeshFilter — skipped."); continue; }
      Mesh mesh = mf.sharedMesh ?? mf.mesh;
      if (mesh == null) { skippedNoMesh++; Debug.LogWarning($"[GRCATriDebugger] '{renderer.gameObject.name}' MeshFilter has no mesh — skipped."); continue; }
      Vector3[] verts = mesh.vertices;
      int[] tris = mesh.triangles;
      for (int i = 0; i < tris.Length; i += 3) {
        Vector3 v0 = renderer.transform.TransformPoint(verts[tris[i]]);
        Vector3 v1 = renderer.transform.TransformPoint(verts[tris[i + 1]]);
        Vector3 v2 = renderer.transform.TransformPoint(verts[tris[i + 2]]);
        var tri = LidarCpu.BuildTriangle(v0, v1, v2);
        float dist = LidarCpu.M_T_RayTriangleIntersect(ray.origin, tri, ray.direction, out var intersectPoint);
        if (dist < best.Distance) {
          best.Distance    = dist;
          best.Renderer    = renderer;
          best.Mesh        = mesh;
          best.TriangleIndex = i / 3;
          best.V0 = v0; best.V1 = v1; best.V2 = v2;
        }
      }
    }
    if (skippedNull > 0)     { Debug.LogWarning($"[GRCATriDebugger] TrySelectTriangle: {skippedNull} renderer(s) were null or disabled."); }
    if (skippedNoFilter > 0) { Debug.LogWarning($"[GRCATriDebugger] TrySelectTriangle: {skippedNoFilter} renderer(s) had no MeshFilter."); }
    if (skippedNoMesh > 0)   { Debug.LogWarning($"[GRCATriDebugger] TrySelectTriangle: {skippedNoMesh} renderer(s) had a null mesh."); }
    if (best.TriangleIndex >= 0) {
      ApplySelection(best.Renderer, best.Mesh, best.TriangleIndex, best.V0, best.V1, best.V2, best.Distance);
    } else {
      _hasSelection = false;
      Debug.LogWarning("[GRCATriDebugger] TrySelectTriangle: No triangle hit. Ray may not intersect any visible geometry.");
    }
  }

  private void ApplySelection(MeshRenderer renderer, Mesh mesh, int triangleIndex, Vector3 v0, Vector3 v1, Vector3 v2, float distance) {
    _hasSelection = true;
    SelectedRendererName  = renderer != null ? renderer.gameObject.name : "<none>";
    SelectedMeshName      = mesh != null ? mesh.name : "<none>";
    SelectedTriangleIndex = triangleIndex;
    SelectedVertex0       = v0;
    SelectedVertex1       = v1;
    SelectedVertex2       = v2;
    Vector3 center = (v0 + v1 + v2) / 3f;
    Vector3 edge1  = v1 - v0;
    Vector3 edge2  = v0 - v2;
    Vector3 normal = -Vector3.Cross(edge2, edge1).normalized;
    Debug.Log(
      $"[GRCATriDebugger] Hit!\n" +
      $"  Renderer  : {SelectedRendererName}\n" +
      $"  Mesh      : {SelectedMeshName}\n" +
      $"  Triangle# : {SelectedTriangleIndex}  (triangles[{SelectedTriangleIndex * 3}..{SelectedTriangleIndex * 3 + 2}])\n" +
      $"  Distance  : {distance:F4} m\n" +
      $"  V0 (world): {v0}\n" +
      $"  V1 (world): {v1}\n" +
      $"  V2 (world): {v2}\n" +
      $"  Center    : {center}\n" +
      $"  Normal    : {normal}");
  }

#if UNITY_EDITOR
  private void OnDrawGizmos() {
    // Draw selected triangle and vertices
    if (_hasSelection) {
      Vector3 v0 = SelectedVertex0;
      Vector3 v1 = SelectedVertex1;
      Vector3 v2 = SelectedVertex2;
      Gizmos.color = HighlightColor;
      Gizmos.DrawLine(v0, v1);
      Gizmos.DrawLine(v1, v2);
      Gizmos.DrawLine(v2, v0);
      float r = 0.02f;
      Gizmos.DrawSphere(v0, r);
      Gizmos.DrawSphere(v1, r);
      Gizmos.DrawSphere(v2, r);
      UnityEditor.Handles.color = HighlightColor;
      UnityEditor.Handles.Label(v0 + Vector3.up * 0.04f, "V0");
      UnityEditor.Handles.Label(v1 + Vector3.up * 0.04f, "V1");
      UnityEditor.Handles.Label(v2 + Vector3.up * 0.04f, "V2");
      Vector3 center = (v0 + v1 + v2) / 3f;
      Vector3 edge1  = v1 - v0;
      Vector3 edge2  = v0 - v2;
      Vector3 normal = -Vector3.Cross(edge2, edge1).normalized;
      Gizmos.color = Color.cyan;
      Gizmos.DrawLine(center, center + normal * HighlightNormalLength);
      UnityEditor.Handles.color = new Color(HighlightColor.r, HighlightColor.g, HighlightColor.b, 0.25f);
      UnityEditor.Handles.DrawAAConvexPolygon(v0, v1, v2);
      UnityEditor.Handles.color = HighlightColor;
      UnityEditor.Handles.Label(center + Vector3.up * 0.05f,
        $"Tri#{SelectedTriangleIndex}\n{SelectedRendererName}");
    }
    // Draw hitpoints (if any)
    if (LidarCpu != null) { DrawHitpoints(); }

    // ── D key: run GRCA + Raw MT on the selected triangle ──
    var kb = Keyboard.current;
    if (kb != null && kb.dKey.wasPressedThisFrame) {
      if (!_hasSelection) { Debug.LogError("[GRCATriDebugger] No triangle selected — left-click a triangle in the Game view first."); return; }
      if (LidarCpu == null)   { Debug.LogError("[GRCATriDebugger] LidarCpu is not assigned."); return; }
      Vector3 v0 = SelectedVertex0, v1 = SelectedVertex1, v2 = SelectedVertex2;
      Debug.Log($"[GRCATriDebugger] ══════════════════════════════════════════\n" +
                $"  Selected triangle #{SelectedTriangleIndex} on '{SelectedRendererName}'\n" +
                $"  V0={v0}  V1={v1}  V2={v2}\n" +
                $"══════════════════════════════════════════");
      _rmtHits.Clear(); _grcaHits.Clear(); _grcaMisses.Clear();
      RunRawMT(v0, v1, v2);
      RunGRCA(v0, v1, v2);
    }
  }
#endif

  // Draw hitpoints and ray lines
  private void DrawHitpoints() {
    Vector3 lidarPos = LidarCpu != null ? LidarCpu.transform.position : Vector3.zero;
    Gizmos.color = RmtHitColor;
    foreach (var pt in _rmtHits) {
      Gizmos.DrawSphere(pt, HitSphereRadius);
      if (DrawRayLines) { Gizmos.DrawLine(lidarPos, pt); }
    }
    Gizmos.color = GrcaHitColor;
    foreach (var pt in _grcaHits) {
      Gizmos.DrawSphere(pt, HitSphereRadius);
      if (DrawRayLines) { Gizmos.DrawLine(lidarPos, pt); }
    }
    Gizmos.color = GrcaMissColor;
    foreach (var pt in _grcaMisses) {
      Gizmos.DrawWireSphere(pt, HitSphereRadius * 0.6f);
      if (DrawRayLines) { Gizmos.DrawLine(lidarPos, pt); }
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  RAW MÖLLER–TRUMBORE  (using GRCA edge convention, matching GRCA_CPU.cs)
  // ══════════════════════════════════════════════════════════════════════════
  private void RunRawMT(Vector3 v0, Vector3 v1, Vector3 v2) {
    Debug.Log("[GRCATriDebugger] ── RAW M-T ──────────────────────────────");
    var tri = LidarCpu.BuildTriangle(v0, v1, v2);
    LogTriSetup(tri);
    float hInc = HInc();
    float vInc = VInc();
    int hitCount = 0;
    var sb = new StringBuilder();
    for (uint vert = 0; vert < LidarCpu.VNumSamples; ++vert) {
      for (uint hori = 0; hori < LidarCpu.HNumSamples; ++hori) {
        float hAngle = LidarCpu.HAngleMin + hori * hInc;
        float vAngle = LidarCpu.VAngleMin + vert * vInc;
        Vector3 dir = LidarCpu.ComputeRayDirection(hAngle, vAngle);
        float dist = LidarCpu.M_T_RayTriangleIntersect(LidarCpu.transform.position, tri, dir, out Vector3 ipt);
        if (dist < float.MaxValue) {
          hitCount++;
          _rmtHits.Add(ipt);
          sb.AppendLine($"    HIT  vert={vert} hori={hori}  dist={dist:F4}  pt={ipt}  dir={dir:F3}");
        }
      }
    }
    Debug.Log($"[GRCATriDebugger] Raw M-T total hits on this triangle: {hitCount}\n{sb}");
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  GRCA  – full pipeline with per-step debug output
  // ══════════════════════════════════════════════════════════════════════════
  private void RunGRCA(Vector3 v0, Vector3 v1, Vector3 v2) {
    Debug.Log("[GRCATriDebugger] ── GRCA ───────────────────────────────────");
    var tri = LidarCpu.BuildTriangle(v0, v1, v2);
    LogTriSetup(tri);
    bool pass = LidarCpu.GRCA_Early_T_Filter(tri, out float closestDist);
    Debug.Log($"[GRCATriDebugger] GRCA_Early_T_Filter → {(pass ? "PASS" : "REJECT")}  closestDist={closestDist:F4}");
    if (!pass) { Debug.Log("[GRCATriDebugger] Triangle rejected by early filter — no GRCA rays tested."); return; }
    Vector3 lidarPos = LidarCpu.transform.position;
    Vector3 up       = LidarCpu.transform.up;
    Vector3[] otv = {
      tri.Vertices[0] - lidarPos,
      tri.Vertices[1] - lidarPos,
      tri.Vertices[2] - lidarPos
    };
    float[] sd = {
      Vector3.Dot(otv[0], up),
      Vector3.Dot(otv[1], up),
      Vector3.Dot(otv[2], up)
    };
    float[] sda = {
      Vector3.Dot(otv[0].normalized, up),
      Vector3.Dot(otv[1].normalized, up),
      Vector3.Dot(otv[2].normalized, up)
    };
    float distCN = Vector3.Dot(tri.Normal, tri.Center - lidarPos);
    bool behindCone         = sd[0]  < 0 && sd[1]  < 0 && sd[2]  < 0;
    bool behindConeFlipped  = -sd[0] < 0 && -sd[1] < 0 && -sd[2] < 0;
    bool intersectUp        = LidarCpu.M_T_RayTriangleIntersect(LidarCpu.transform.position, tri, up,  out _) < float.MaxValue;
    bool intersectDown      = LidarCpu.M_T_RayTriangleIntersect(LidarCpu.transform.position, tri, -up, out _) < float.MaxValue;
    Debug.Log($"[GRCATriDebugger] signdDist        = [{sd[0]:F4}, {sd[1]:F4}, {sd[2]:F4}]\n" +
              $"               signdDistAngle   = [{sda[0]:F4}, {sda[1]:F4}, {sda[2]:F4}]\n" +
              $"               distCenterNormal = {distCN:F4}\n" +
              $"               isBehindCone={behindCone}  isBehindConeFlipped={behindConeFlipped}\n" +
              $"               isTriIntersectUp={intersectUp}  isTriIntersectDown={intersectDown}");
    float vInc   = VInc();
    float hInc   = HInc();
    uint fromCh = LidarCpu.VNumSamples, toCh = 0;
    if (LidarCpu.VNumSamples > 1) {
      LidarCpu.GRCA_CSpan_Predict(behindCone, behindConeFlipped, intersectUp, intersectDown, tri, sd, sda, ref fromCh, ref toCh);
      if (fromCh > toCh || fromCh == LidarCpu.VNumSamples) {
        Debug.Log($"[GRCATriDebugger] CSpan result: INVALID (fromCh={fromCh}, toCh={toCh}) — no rays tested.");
        return;
      }
    }
    Debug.Log($"[GRCATriDebugger] CSpan result: fromChannelId={fromCh}  toChannelId={toCh}  span={toCh-fromCh+1} channels");
    bool ccwPossible = LidarCpu.HAngleMax - LidarCpu.HAngleMin > Mathf.PI;
    bool allCW;
    if (ccwPossible) {
      int midV = Mathf.RoundToInt(LidarCpu.VNumSamples / 2.0f);
      int midH = Mathf.RoundToInt(LidarCpu.HNumSamples / 2.0f);
      Vector3 midDir = LidarCpu.ComputeRayDirection(LidarCpu.HAngleMin + midH * hInc, LidarCpu.VAngleMin + midV * vInc);
      allCW = Vector3.Dot(otv[0], midDir) > 0 && Vector3.Dot(otv[1], midDir) > 0 && Vector3.Dot(otv[2], midDir) > 0;
    } else {
      allCW = true;
    }
    Debug.Log($"[GRCATriDebugger] isAllClockWise={allCW}  isCCWPossible={ccwPossible}");
    int totalHits = 0;
    var hitSb = new StringBuilder();
    for (uint ch = fromCh; ch <= toCh; ++ch) {
      float vAngle  = LidarCpu.VAngleMin + ch * vInc;
      float angle   = Mathf.Acos(Mathf.Sin(vAngle));
      bool flipped  = angle > Mathf.PI / 2;
      Vector3 coneDir = flipped ? -up : up;
      float   coneAngle = flipped ? Mathf.PI - angle : angle;
      bool mirrored   = flipped ? behindConeFlipped  : behindCone;
      bool intersect  = flipped ? intersectDown       : intersectUp;
      if (mirrored) {
        Debug.Log($"[GRCATriDebugger]   ch={ch} vAngle={vAngle:F3} → SKIPPED (mirrored)");
        continue;
      }
      bool rspanOk = LidarCpu.GRCA_RSpan_Predict(allCW, flipped, intersect, ch, lidarPos, coneDir, coneAngle,
                                  tri, distCN, sd, sda,
                                  out uint fromSw, out uint swDiff, out bool cwSweep);
      if (!rspanOk) {
        Debug.Log($"[GRCATriDebugger]   ch={ch} vAngle={vAngle:F3} angle={angle:F3} flipped={flipped} → RSpan: NO INTERSECTION");
        continue;
      }
      Debug.Log($"[GRCATriDebugger]   ch={ch} vAngle={vAngle:F3} angle={angle:F3} flipped={flipped} → " +
                $"fromSweepId={fromSw}  sweepDiff={swDiff}  isClockWise={cwSweep}  ({swDiff+1} rays)");
      for (uint i = 0; i <= swDiff; ++i) {
        var rayId = cwSweep
          ? (fromSw + i) % LidarCpu.HNumSamples
          : (fromSw - i + LidarCpu.HNumSamples) % LidarCpu.HNumSamples;
        Vector3 dir  = LidarCpu.ComputeRayDirection(LidarCpu.HAngleMin + rayId * hInc, vAngle);
        float   dist = LidarCpu.M_T_RayTriangleIntersect(LidarCpu.transform.position, tri, dir, out Vector3 ipt);
        if (dist < float.MaxValue) {
          totalHits++;
          _grcaHits.Add(ipt);
          hitSb.AppendLine($"      HIT  ch={ch} ray={rayId}  dist={dist:F4}  pt={ipt}");
        } else {
          _grcaMisses.Add(LidarCpu.transform.position + dir * LidarCpu.RangeMax);
          hitSb.AppendLine($"      MISS ch={ch} ray={rayId}");
        }
      }
    }
    Debug.Log($"[GRCATriDebugger] GRCA total hits on this triangle: {totalHits}  GRCA-tested misses: {_grcaMisses.Count}\n{hitSb}");
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  HELPERS
  // ══════════════════════════════════════════════════════════════════════════

  private void LogTriSetup(GrcaLidarCpu.Triangle t) {
    Debug.Log($"[GRCATriDebugger] Triangle setup:\n" +
              $"  V0={t.Vertices[0]}  V1={t.Vertices[1]}  V2={t.Vertices[2]}\n" +
              $"  Edge1={t.Edge1}  Edge2={t.Edge2}\n" +
              $"  Center={t.Center}\n" +
              $"  Normal={t.Normal}\n" +
              $"  Area={t.Area:F6}  DdD=[{t.DdD[0]:F4},{t.DdD[1]:F4},{t.DdD[2]:F4}]");
  }

  private float HInc() => (LidarCpu.HAngleMax - LidarCpu.HAngleMin) / (LidarCpu.HNumSamples - 1);
  private float VInc() => LidarCpu.VNumSamples > 1
    ? (LidarCpu.VAngleMax - LidarCpu.VAngleMin) / (LidarCpu.VNumSamples - 1)
    : 0f;
}
