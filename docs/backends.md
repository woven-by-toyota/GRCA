# Backends

Backend aliases and per-backend details and labels.

Back to the [README](../README.md).

## Table of Contents

1. [Backend aliases](#backend-aliases)
2. [GRCA CPU](#grca-cpu)
   - [GRCA-CPU packet size](#grca-cpu-packet-size)
3. [GRCA CUDA](#grca-cuda)
4. [OptiX](#optix)
   - [Custom intersection (CRTI)](#custom-intersection-crti)
5. [Embree](#embree)
   - [BVH update mode (OptiX and Embree)](#bvh-update-mode-optix-and-embree)
   - [Embree ray packet size](#embree-ray-packet-size)
6. [TinyBVH CPU](#tinybvh-cpu)
7. [TinyBVH GPU](#tinybvh-gpu)
8. [Hybrid CPU](#hybrid-cpu)
9. [Hybrid GPU](#hybrid-gpu)

## Backend aliases

Group aliases expand to multiple dedicated backends. Only backends compiled into the binary are emitted.

| Alias | Expands to |
|-------|-----------|
| `grca` | `grca-cpu` + `grca-cuda` |
| `hybrid` | `hybrid-cpu` + `hybrid-gpu` |
| `all` | `grca-cpu` + `grca-cuda` + `optix` + `embree` |
| `all_t` | everything in `all` + `tinybvh-cpu` + `tinybvh-gpu` |
| `all_h` | everything in `all` + `hybrid-cpu` + `hybrid-gpu` |
| `cpu` | `grca-cpu` + `embree` |
| `cpu_t` | `grca-cpu` + `embree` + `tinybvh-cpu` |
| `cpu_h` | `grca-cpu` + `embree` + `hybrid-cpu` |
| `gpu` | `grca-cuda` + `optix` |
| `gpu_t` | `grca-cuda` + `optix` + `tinybvh-gpu` |
| `gpu_h` | `grca-cuda` + `optix` + `hybrid-gpu` |
| `gpu-all` | `grca-cuda` + `optix` + `optix-crti` |
| `bvh` | `embree` + `optix` |
| `bvh_t` | `embree` + `optix` + `tinybvh-cpu` + `tinybvh-gpu` |


## GRCA CPU

Triangle-out span-prediction algorithm, single-threaded. Built with `-O3 -march=native -ffast-math -funroll-loops`.

### GRCA-CPU packet size

Controls the SIMD width used in the SAT inner loop (Möller–Trumbore tests of multiple rays against one triangle simultaneously). Set with `cpu.packet_size` in the test config:

| Value | Inner loop stride | Intrinsic set | Notes |
|-------|------------------|---------------|-------|
| `16` | 16 rays / iter (2 × 8) | AVX2 `__m256` **(default)** | Both batches share the same broadcast constants |
| `8` | 8 rays / iter | AVX2 `__m256` | |
| `4` | 4 rays / iter | SSE4.1 `__m128` | |
| `1` | scalar | — | |

The backend label and output file prefix reflect the configured width: `GRCA_cpu_vec16`, `GRCA_cpu_vec8`, `GRCA_cpu_vec4`, `GRCA_cpu_vec1`. The SAT/BAT thresholds for GRCA-CPU are `cpu.sat_max_sweep_diff` and `cpu.sat_max_channel_diff` (default 64; see [CUDA options](configuration.md#cuda-options-cuda---)).

> **`16` vs `8`**: both use AVX2 (`__m256`) — no AVX-512 on Alder Lake. The 16-wide path processes two independent batches of 8 per outer iteration, letting the CPU interleave them for better ILP. The SIMD paths are used for the ray sweeps of both SAT and BAT triangles, on sequential (non-wraparound) ray rows; wraparound rows fall back to scalar regardless of the setting.

## GRCA CUDA

Triangle-out span-prediction algorithm on the GPU (no BVH); GRCA<sub>GPU</sub> in the paper. Dynamic mesh vertices are re-uploaded via `cudaMemcpy` each frame.

The early pass appends BAT (big-appearing) triangles to a fixed-capacity deferred `TriRay` buffer (default 800,000 entries × 212 bytes ≈ **170 MB**; set the capacity with the top-level `bat_capacity`). Each entry stores the triangle index, per-LiDAR-origin channel span (`vFromIdToId`), per-LiDAR-origin closest-vertex distance, and per-LiDAR-origin flags (origin mask, cone-apex hit state, CW/CCW winding). The late pass reads these directly — no early-pass work is repeated.

**VRAM output.** After backend init, and once per frame for CUDA/OptiX backends:
```
[VRAM] allocated: 2156.7 MB  persistent: 2898.0 MB / 15975.6 MB total
[VRAM] frame 0: +2156.7 MB (mesh: 747.7 MB, other: 1409.0 MB | total: 2898.0 MB)
```
After the benchmark ZIP is written:
```
=== Max VRAM per batch ===
  run0/batch0/grca-cuda                    +2156.7 MB (total: 2898.0 MB)
  run0/batch1/optix                       +3450.2 MB (total: 4185.8 MB)
```
All deltas are relative to the VRAM in use just before the backend is initialised. The "Max VRAM per batch" summary covers `grca-cuda`, `optix`, and `optix-crti`.

## OptiX

Hardware ray tracing via NVIDIA OptiX 9.1. Requires CUDA. One GAS is built over all triangles (static and dynamic). Build flags: `PREFER_FAST_TRACE | ALLOW_UPDATE` for pure refit, `PREFER_FAST_BUILD | ALLOW_UPDATE` for pure and periodic rebuild, plus `ALLOW_COMPACTION` when `bvh_compaction_on` is set.

### Custom intersection (CRTI)

`optix-crti` is a separate backend that uses software Möller–Trumbore triangle intersection on CUDA shader processors while keeping RT-core BVH traversal. Select it with `"backend": "optix-crti"` in a batch, or with the `gpu-all` alias to run it alongside GRCA-CUDA and `optix`. It reads the same `optix` section of the test file:

```json
{
    "optix": {
        "bvh_rebuild_after_frames": 0,
        "bvh_compaction_on": false
    },
    "batches": [
        { "backend": "gpu-all", "lidar_sim_rate": 100.0, ... }
    ]
}
```

**What differs from `optix`:**

| | `optix` | `optix-crti` |
|---|---|---|
| BVH build input | `OPTIX_BUILD_INPUT_TYPE_TRIANGLES` | `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES` (per-triangle AABB) |
| Triangle intersection | Hardware RT cores (built-in triangles) | CUDA shader processors (`__intersection__` — Möller–Trumbore) |
| BVH traversal | RT cores | RT cores (unchanged) |
| PTX loaded | `build/optix_device.ptx` | `build/optix_custom_device.ptx` |
| Backend label | `Optix_rb`, `Optix_cmp_rb`, … | `Optix_crti_rb`, `Optix_crti_cmp_rb`, … |
| API label | `Optix` | `Optix_crti` |

**Comparison purpose:** Comparing `Optix_rb` vs `Optix_crti_rb` isolates the cost of hardware RT-core triangle intersection from BVH traversal.

## Embree

Intel Embree 4.4.0, CPU ray tracing with SIMD packets. Runs single-threaded (`rtcNewDevice("threads=1,start_threads=0")`), like GRCA-CPU. Static and dynamic triangles are two separate geometries in one scene; when dynamic geometry exists the scene uses `RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST` and `LOW` scene build quality, and only the dynamic geometry is updated each frame.

### BVH update mode (OptiX and Embree)

Controlled by a single `bvh_rebuild_after_frames` integer in the respective backend section of the test file:

| Value | Mode | Label suffix | Behaviour |
|-------|------|--------------|-----------|
| `0` | Pure rebuild | `rb` | Full rebuild every frame **(default)** |
| `-1` | Pure refit | `rf` | Incremental in-place refit every frame — never rebuilds after init |
| `N > 0` | Periodic rebuild | `prN` | Rebuild on the first frame and every N+1 frames after it, refit on the N frames in between (`1` alternates rebuild and refit) |

**`bvh_compaction_on`** — when `true`, enables compaction for the built acceleration structure. Reflected in the backend label as `cmp`:

| Backend | Effect | Label examples |
|---------|--------|---------------|
| **OptiX** | Post-build `optixAccelCompact` step removes unused padding from the GAS output buffer, reducing VRAM. Same BVH quality, no traversal penalty. | `Optix_cmp_rb`, `Optix_cmp_rf`, `Optix_cmp_pr10`, `Optix_crti_cmp_rb` |
| **Embree** | Sets `RTC_SCENE_FLAG_COMPACT` on the scene, instructing Embree to use a more compact internal BVH node format. Reduces system RAM at a small traversal performance cost. | `Embree_vec8_cmp_rb`, `Embree_vec16_cmp_rf` |

See `configs/tests/main_suite/rebuild_compact.json` for a compaction example.

Embree `RTCBuildQuality` used per geometry type and mode:

| Geometry | Mode | Quality | Why |
|----------|------|---------|-----|
| Static | all | `HIGH` | SAH; built once at init |
| Dynamic | pure refit (init) | `HIGH` | Best quality; init is the only full build |
| Dynamic | periodic rebuild / pure rebuild (init + rebuilds) | `LOW` | Fast Morton builder; consistent with every periodic/per-frame rebuild |
| Dynamic | refit frame | `REFIT` | In-place node adjustment; fastest |

### Embree ray packet size

Set with `embree.packet_size` in config:

| Value | API call | SIMD width |
|-------|----------|-----------|
| `1` | `rtcIntersect1` | scalar |
| `4` | `rtcIntersect4` | 128-bit (SSE4) |
| `8` | `rtcIntersect8` | 256-bit (AVX2) |
| `16` | `rtcIntersect16` | 16-wide packet **(default)**; native 512-bit only on AVX-512 CPUs (PC1, an i7-12700H, is AVX2-only) |

The backend label and output file prefix reflect the chosen width, compaction flag, and BVH mode: `Embree_vec1_rb`, `Embree_vec1_rf`, `Embree_vec1_pr10`, `Embree_vec4_rb`, `Embree_vec8_cmp_rb`, etc.

## TinyBVH CPU

Software BVH ray tracer on CPU using [tinybvh](https://github.com/jbikker/tinybvh) (bundled; see [TinyBVH backends](../README.md#tinybvh-backends-cpu--gpu-no-extra-deps)).

**Pipeline per frame:**
1. Update dynamic vertex positions in the host-side `bvhvec4` buffer
2. Refit or rebuild the `tinybvh::BVH` on CPU (SAH builder)
3. Cast all lidar rays single-threaded using `BVH::Intersect()` (one call per ray)

Intended as a direct single-threaded comparison with Embree (packet_size=1). Unlike Embree, tinybvh does **not** apply backface culling — results near geometry edges may differ slightly.

**BVH update mode** — same `bvh_rebuild_after_frames` semantics as Embree and OptiX. Configure via the `tinybvh` section:

```json
"tinybvh": {
    "bvh_rebuild_after_frames": 0
}
```

Or set CPU and GPU independently:
```json
"tinybvh": {
    "cpu_rebuild_after_frames": -1,
    "gpu_rebuild_after_frames": 0
}
```

Backend label examples: `TinyBVH_cpu_rb`, `TinyBVH_cpu_rf`, `TinyBVH_cpu_pr4`.

## TinyBVH GPU

Software BVH ray tracer on GPU. BVH is built and refitted on CPU by tinybvh, then the nodes and BVH-sorted triangles are uploaded to GPU each frame. Ray traversal runs entirely on GPU via a custom CUDA kernel — **no RT cores are used**.

**Pipeline per frame:**
1. Update dynamic vertex positions in the host-side `bvhvec4` buffer
2. Refit or rebuild the `tinybvh::BVH` on CPU
3. Sort triangles in BVH leaf order (`primIdx` permutation) and upload to GPU (`float4` arrays)
4. Upload nodes to GPU (2 × `float4` per node = 32 bytes, matching `tinybvh::BVH::BVHNode`)
5. Pack ray origins and directions and upload to GPU (6 floats per ray)
6. Launch CUDA kernel: each thread traces one ray via iterative stack-based BVH traversal (Möller–Trumbore intersection, two-sided)
7. Readback hit distances and fill `GrcaHit` structs on CPU

**Timing breakdown** (tinybvh-gpu is CPU-bound on rebuild):

| Stage | Timing field | Notes |
|-------|-------------|-------|
| Vertex copy | `ms_dynamic_upload` | Host-side memcpy |
| BVH rebuild | `ms_bvh_rebuild_cpu` | tinybvh SAH on CPU — dominates on large scenes |
| BVH refit | `ms_bvh_refit_cpu` | Much cheaper; use `bvh_rebuild_after_frames: -1` to see GPU trace time clearly |
| GPU trace | `ms_trace_gpu` | CUDA kernel, measured with CUDA events |
| Readback | `ms_readback_cpu` | `cudaMemcpy` D→H |

Requires CUDA. Backend label examples: `TinyBVH_gpu_rb`, `TinyBVH_gpu_rf`, `TinyBVH_gpu_pr4`.

**Comparison purpose:** `tinybvh-gpu` vs `optix` isolates the benefit of hardware RT cores (OptiX uses them; tinybvh-gpu does not). `tinybvh-cpu` vs `tinybvh-gpu` isolates CPU-vs-GPU traversal with the identical BVH and ray set.

## Hybrid CPU

Splits the scene at `staticTriCount` and runs two sub-backends one after the other each frame (static pass, then dynamic pass):

- **Static geometry** → Embree BVH (all rays, static tris only). The BVH is built once at init (`HIGH` quality) and never updated, since the static pass has no dynamic geometry.
- **Dynamic geometry** → GRCA-CPU two-pass angular filter (all rays, dynamic tris only).
- **Merge** — per-ray: the closer of the two hit distances wins.

**Key points:**
- The `cpu` and `embree` sections of the test file are **not applied** to `hybrid-cpu` (they are only read for the `grca-cpu` and `embree` backends), so it always runs with the defaults: packet size 16 for both passes and SAT/BAT thresholds 64/64.
- Requires Embree (`HAVE_EMBREE`).

**Backend label:** always `Hybrid_cpu_Embree_vec16_rf_GRCA_vec16` (the format is `Hybrid_cpu_Embree_vec<embree packet>_rf_GRCA_vec<cpu packet>`).

**Timing fields:**

| Field | Source |
|-------|--------|
| `ms_lidar_sim_cpu` | Whole frame, both passes (wall time around the backend call) |
| `ms_trace_cpu` | GRCA-CPU pass only |
| `ms_trace_gpu` | Always 0: it is copied from Embree's `ms_trace_gpu`, but Embree reports its trace time in `ms_trace_cpu`, which is not carried over |
| `ms_bvh_rebuild_cpu`, `ms_bvh_refit_cpu` | Sum from both sub-backends (0 for the static Embree pass) |
| `ms_dynamic_upload` | Sum from both sub-backends |

## Hybrid GPU

Splits the scene at `staticTriCount` and runs two sub-backends one after the other each frame (static pass, then dynamic pass):

- **Static geometry** → OptiX RT cores (all rays, static tris only). The acceleration structure is built once at init and never updated, since the static pass has no dynamic geometry.
- **Dynamic geometry** → GRCA-CUDA early + late pass (all rays, dynamic tris only). All dynamic triangles are re-uploaded to GPU each frame (`staticTriCount = 0` from GRCA-CUDA's perspective).
- **Merge** — per-ray: the closer of the two hit distances wins. Done on CPU after both readbacks complete.

**Key points:**
- `bvh_rebuild_after_frames` in the `optix` config section has no effect on `hybrid-gpu`.
- The GRCA-CUDA sub-backend respects all `cuda` config options (`sat_max_*_diff`, `debug`).
- Requires CUDA + OptiX (`HAVE_OPTIX && HAVE_CUDA`).
- Both OptiX and GRCA-CUDA hold their own GPU allocations simultaneously; `hybrid-gpu` is not included in the "Max VRAM per batch" summary.

**Backend label:** `Hybrid_gpu_Optix_rf_GRCA_cuda`.

**Run order within group aliases:** `all_h` runs `optix → grca-cuda → hybrid-gpu`, then `embree → grca-cpu → hybrid-cpu`, so each hybrid result appears immediately after its component backends in the benchmark CSV; `gpu_h` and `cpu_h` run the GPU and CPU halves respectively.

**Timing fields:**

| Field | Source |
|-------|--------|
| `ms_trace_gpu` | Sum: OptiX trace + GRCA-CUDA trace |
| `ms_bvh_rebuild_gpu`, `ms_bvh_refit_gpu` | OptiX static pass only (0, since it is never updated) |
| `ms_dynamic_upload` | Sum from both sub-backends |
| `ms_readback_cpu` | Sum: OptiX + GRCA-CUDA readbacks |
| `ms_lidar_sim_gpu` | Sum: OptiX + GRCA-CUDA GPU sim time |
| `num_bat`, `num_early_t`, `num_rtic` | GRCA-CUDA dynamic pass only |
