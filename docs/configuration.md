# Configuration

Reference for the config files: structure, every field, and the main-suite test files.

Back to the [README](../README.md).

## Table of Contents

1. [Structure](#structure)
2. [Top-level fields](#top-level-fields)
3. [Main-suite rebuild sweep](#main-suite-rebuild-sweep)
4. [Active tests](#active-tests)
5. [`configs/scenes/<name>.json` fields](#configsscenesnamejson-fields)
6. [`configs/dynamic_meshes/<name>.json`](#configsdynamic_meshesnamejson)
7. [`configs/robots/<name>.json` fields](#configsrobotsnamejson-fields)
8. [Test file fields](#test-file-fields)
   - [CUDA options (`"cuda": { ... }`)](#cuda-options-cuda---)
9. [Periodic Rebuild Example](#periodic-rebuild-example)
10. [`batches` fields](#batches-fields)
11. [`static_meshes` fields (scene files) / `dynamic_meshes` fields (dynamic mesh files)](#static_meshes-fields-scene-files--dynamic_meshes-fields-dynamic-mesh-files)
12. [`robots` / `lidars` fields](#robots--lidars-fields)
13. [Random Dynamic Test](#random-dynamic-test)
14. [Orbit Pairs Test](#orbit-pairs-test)
15. [Candidate Reference Syntax](#candidate-reference-syntax)

## Structure

Config is split across five file types. Config file paths resolve relative to the directory of the top-level config; `.obj` paths inside scene and dynamic-mesh files resolve relative to the working directory (the project root):

```
benchmark_config.json                        → top-level: scenes, dynamic meshes, robots, active tests
configs/scenes/<name>.json                   → static mesh definitions only
configs/dynamic_meshes/<n>_dyn_mesh.json     → N dynamic mesh entries (sportsCar, 1–30)
configs/robots/<name>.json                   → robot + lidar array
configs/tests/<name>.json                    → one test per file (recording, backends, batches)
configs/tests/<subfolder>/<name>.json        → subfolders are supported (e.g. 3080ti mobile/fast_sat_find/)
```

The benchmark runs one session per **(scene × dynamic_mesh_count × robot_count)** combination, in that loop order.

If `benchmark_dynamic_meshes` is omitted, the scene's own `dynamic_meshes` array is used as-is.

The checked-in `benchmark_config.json` is a quick live-preview setup (Lumberyard Bistro, 1 dynamic mesh, 1 robot, `static_visual.json`). To reproduce a paper benchmark, use one of the [benchmark presets](../README.md#benchmark-presets).

## Top-level fields

| Field | Default | Description |
|-------|---------|-------------|
| `benchmark_scenes` | — | Scene files to run |
| `benchmark_dynamic_meshes` | — | Dynamic mesh files to run |
| `benchmark_robots` | — | Robot files to run |
| `active_tests` | — | Test files to run, in order (see [Active tests](#active-tests)) |
| `skip_validation` | `true` | Skip hit-distance and pose CSV writing and the end-of-batch comparison; the comparison columns in the results CSV are then placeholders (`*_pairs_*` = 0, `*_min_pct` = 100) |
| `skip_plot` | `true` | Skip all `plot_results.py` invocations (per-batch, per-run, and session-level) |
| `bat_capacity` | `800000` | GRCA-CUDA BAT buffer capacity in entries (see [GRCA CUDA](backends.md#grca-cuda)); the span-threshold presets raise it to `4500000` |
| `visualizer_on`, `mesh_vis_on`, `cone_vis_on` | see [Test file fields](#test-file-fields) | Visualizer defaults; the first active test can override them |
| `w_cam_position`, `w_cam_forward`, `w_cam_up` | — | Initial visualizer camera pose; the first active test can override them |

## Main-suite rebuild sweep

Each file in `configs/tests/main_suite/` is one run condition, set by the triangle-topology chaos mode `tri_chaos`: no deformation (ND), object-bound deformation (OBD), or scene-wide deformation (SWD). Only dynamic objects are randomized; static meshes remain fixed. Mesh vertices always move; chaos modes additionally alter per-triangle behavior. The `_hybrid` and `_compact` files repeat the three conditions with the hybrid backends or with BVH compaction on. Run indices in the output (`run_N/`) follow the order of `active_tests`.

| Test file | Condition | `tri_chaos` | `backend` | BVH compaction (OptiX + Embree) |
|-----------|-----------|-------------|-----------|---------------------------------|
| `rebuild.json` | ND | 0 | `all` | off |
| `rebuild_bound_chaos.json` | OBD | 2 | `all` | off |
| `rebuild_chaos.json` | SWD | 1 | `all` | off |
| `rebuild_hybrid.json` | ND | 0 | `hybrid` | off |
| `rebuild_bound_chaos_hybrid.json` | OBD | 2 | `hybrid` | off |
| `rebuild_chaos_hybrid.json` | SWD | 1 | `hybrid` | off |
| `rebuild_compact.json` | ND | 0 | `all` | on |
| `rebuild_bound_chaos_compact.json` | OBD | 2 | `all` | on |
| `rebuild_chaos_compact.json` | SWD | 1 | `all` | on |

The files write `tri_chaos` as `false` / `2` / `true`; booleans and integers are both accepted. Test files in other folders follow the same naming, with OBD sometimes written `chaos_bound`.

Each file has **1 batch** with `lidar_sim_rate: 100.0`, `record_frames: 500`, scale `[0.001, 30]`, seed `42`, and `pos_`/`rot_`/`scale_interpolate_frames: 0` (new random pose every frame, the paper's f.i; see [Interpolation convention](#random-dynamic-test)). `all` runs GRCA-CPU, GRCA-CUDA, OptiX, and Embree; `hybrid` runs Hybrid-CPU and Hybrid-GPU (see [Backend aliases](backends.md#backend-aliases)).

`rebuild_sub_div.json` in the same folder is not part of the main suite; it is the 50-frame test used by the subdivision preset.

Correctness checks between backends require a minimum **95%** hit-distance match.

## Active tests

`active_tests` in `benchmark_config.json` lists exactly which test files to run — each entry is a path to a test file. All listed tests run in order. Use `--active` on the CLI to override with an index selector:

```bash
./build/grca --active tests[0]/batches[1]
```

Selectors: `tests[N]` (one run), `tests[N]/batches[M]` (one batch within a run).

> **Tip — live preview without benchmarking**
> `configs/tests/static_visual.json` opens the interactive hit visualizer (`visualizer_on: true`, `mesh_vis_on: 2`, `cone_vis_on: true`) with `record_frames: -1`, so it runs until the window is closed and writes no CSVs. Place it at the top of `active_tests` to preview the scene before any benchmark run; close the window to continue to the remaining tests.

## `configs/scenes/<name>.json` fields

| Field | Description |
|-------|-------------|
| `static_meshes` | Fixed-position `.obj` meshes; loaded once per scene |

## `configs/dynamic_meshes/<name>.json`

A JSON array of dynamic mesh entries. Referenced from `benchmark_config.json` via `"benchmark_dynamic_meshes"`. `<n>_dyn_mesh.json` (n = 1–30) holds n copies of `mesh/sportsCar/sportsCar.obj`; the main suite uses 10, 20, and 30. The `<n>_<k>_sub_div_dyn_mesh.json` variants (`1_1`, `1_2`, `1_3`, `30_1`, `30_2`) use the sports car subdivided k times (4<sup>k</sup>× triangles).

## `configs/robots/<name>.json` fields

A JSON array of robot objects. Referenced from `benchmark_config.json` via `"benchmark_robots"`. The repository includes presets from 1 to 8 robots (plus multiple 8-robot variants), each with one LiDAR, so the robot count is the paper's Ω. The main-suite sweep uses `max_rays_2_robot.json`, `max_rays_4_robot.json`, `max_rays_6_robot.json`, and `max_rays_8_robot.json` (`range_max` = 1000 m); the `max_rays_<N>_robot_100.json` variants are identical except for `range_max` = 100 m and are used by the `_100` presets.

## Test file fields

| Field | Default | Description |
|-------|---------|-------------|
| `visualizer_on` | `false` | Opens the visualizer window when `true`. Integer values are treated as non-zero/zero enable flags only. |
| `mesh_vis_on` | `0` | Mesh display mode: `0` = no mesh, `1` = wireframe only, `2` = solid mesh + wireframe overlay (uses per-triangle MTL colors), `3` = same as `2` but forces flat JSON `color` for every mesh (ignores MTL) |
| `cone_vis_on` | `false` | Draw the LiDAR channel cones in the visualizer |
| `record_frames` | `-1` | Frames to record (perf, hit-dist, and pose); `-1` records nothing and keeps running (see frame-loop termination below) |
| `lidar_sim_rate` | `10` | Default simulation rate (Hz) for batches that don't set one; only read from the first active test |
| `auto_hit_view_record_frame` | `-1` | Frame to auto-write a hit-view CSV; `-1` disables. Only read from the first active test |
| `cuda` | — | CUDA options (see below) |
| `cpu` | — | `packet_size` (see [GRCA-CPU packet size](backends.md#grca-cpu-packet-size)) and `sat_max_sweep_diff` / `sat_max_channel_diff` (same meaning and defaults as in `cuda`) |
| `embree` | — | See [Embree options](backends.md#embree) |
| `optix` | — | See [OptiX options](backends.md#optix); also used by `optix-crti` |
| `tinybvh` | — | See [TinyBVH options](backends.md#tinybvh-cpu) |
| `batches` | — | Array of batch objects (see [`batches` fields](#batches-fields)) |

> **Frame loop termination**
> The loop runs until `record_frames` frames are recorded.
> - `record_frames: -1` + dedicated backend → runs **indefinitely** (Ctrl+C to stop).
> - `record_frames: -1` + any backend selector that expands to multiple backends (for example `all`, `cpu`, `gpu`, `all_t`, `gpu-all`, `hybrid`) → **rejected** with an error (first backend would never finish).

### CUDA options (`"cuda": { ... }`)

| Field | Default | Description |
|-------|---------|-------------|
| `debug` | `false` | Enable the early-T / RTIC counters (`num_early_t`, `num_rtic`); their min/avg/max are written to `runtime_info_<backend>.txt`. Also settable as top-level `cuda_debug` in the test file or with `--debug` |
| `sat_max_sweep_diff` | `64` | SAT classification threshold (paper: χ<sub>T</sub>) — triangles whose estimated sweep span exceeds this are deferred to the late pass as BAT |
| `sat_max_channel_diff` | `64` | SAT classification threshold (paper: γ<sub>T</sub>) — triangles whose channel span exceeds this are deferred to the late pass as BAT |

## Periodic Rebuild Example

**configs/tests/3080ti mobile/fast_hybrid_find/bvh_sweep_pr10.json** sets the OptiX and Embree BVHs to periodic rebuild (label suffix `pr10`):

```json
{
    "embree": {
        "bvh_rebuild_after_frames": 10,
        "packet_size": 16
    },
    "optix": {
        "bvh_compaction_on": false,
        "bvh_rebuild_after_frames": 10
    },
    ...
}
```

See [BVH update mode](backends.md#bvh-update-mode-optix-and-embree) for what `bvh_rebuild_after_frames` values mean and the build quality used per mode.

## `batches` fields

| Field | Description |
|-------|-------------|
| `backend` | A dedicated backend name or a group alias (see [Backend aliases](backends.md#backend-aliases)); `grca-cpu` if omitted |
| `lidar_sim_rate` | Simulation tick rate (Hz) |
| `random_dynamic_test` | See [Random Dynamic Test](#random-dynamic-test) |
| `orbit_pairs_test` | See [Orbit Pairs Test](#orbit-pairs-test) |
| `lidar_orbit_test` | See [Orbit Pairs Test](#orbit-pairs-test) |
| `active_test` | Name of the test object to use when a batch contains several; otherwise the first present of `orbit_pairs_test`, `lidar_orbit_test`, `random_dynamic_test` is used |

## `static_meshes` fields (scene files) / `dynamic_meshes` fields (dynamic mesh files)

| Field | Description |
|-------|-------------|
| `file` | Path to `.obj` (relative to working dir) |
| `w_position` | World-space offset `[x, y, z]` |
| `w_forward`, `w_up` | Initial orientation (dynamic meshes only) |
| `scale` | Scale factor |
| `color` | Display color `[r, g, b]` |
| `color_override` | `true` = always draw with `color` instead of the mesh's MTL colors |

## `robots` / `lidars` fields

| Field | Description |
|-------|-------------|
| `w_position` | Robot world-space origin |
| `w_forward`, `w_up` | Robot orientation (unit vectors) |
| `lidars` | Array of lidar configs |
| `hmin/hmax` | Horizontal angle range (radians) |
| `hnum` | Rays per channel (paper: χ<sub>n</sub>) |
| `vmin/vmax` | Vertical angle range (radians) |
| `vnum` | Vertical channel count (paper: γ<sub>n</sub>) |
| `range_min/max` | Hit distance limits (metres) |
| `position` | Lidar local-space offset |
| `forward`, `up` | Lidar local-space orientation (unit vectors) |

## Random Dynamic Test

Randomizes position, orientation, and optionally scale of every dynamic mesh and robot each frame within the static-mesh AABB. Fully reproducible via `seed`.

```json
"random_dynamic_test": {
    "seed": 42,
    "pos_interpolate_frames": 30,
    "rot_interpolate_frames": 30,
    "scale_interpolate_frames": -1,
    "scale_min": 0.01,
    "scale_max": 2.0,
    "tri_chaos": 0,
    "candidates": ["dynamic_meshes", "robots"]
}
```

| Field | Description |
|-------|-------------|
| `seed` | RNG seed — same seed always produces the same sequence |
| `pos_interpolate_frames` | Position randomization — see interpolation table below |
| `rot_interpolate_frames` | Orientation randomization — same convention |
| `scale_interpolate_frames` | Scale randomization — same convention; `-1` disables |
| `scale_min/max` | Random scale range. Scale is **log-uniform** (equal probability per decade) and **independent per axis** (x/y/z get separate values), producing non-uniform stretching rather than uniform resize. Applies to both mesh-level scale (`tri_chaos: 0`) and per-triangle scale (`tri_chaos: 1/2`). |
| `tri_chaos` | Per-triangle independent random transform. Scale distribution and placement region per mode: `0` = off, whole mesh is scaled/positioned/rotated as a rigid body with per-axis log-uniform scale; `1` = each triangle independently placed within the static scene AABB with per-axis log-uniform scale; `2` = each triangle independently placed within the mesh's own local AABB (transformed to world space as an OBB, also stretched per-axis), simulating skinned/deformed mesh. |
| `candidates` | Which objects to randomize — see [Candidate Syntax](#candidate-reference-syntax) |
| `robots_share_pose` | `true` = every robot uses robot 0's random pose (co-located ray origins); each LiDAR's local orientation is randomized once at start. Default `false` |

**Interpolation convention** (used by `pos_`, `rot_`, `scale_interpolate_frames`; omitted fields default to `0`):

| Value | Behaviour |
|-------|-----------|
| `-1` | Disabled — stays at configured pose |
| `0` | Instant snap — new random value every frame |
| `N > 0` | New target every N frames, linearly interpolated |

A dynamic mesh picks all its new targets at the same time, every *max* of its enabled interpolation values. With `tri_chaos` `1` or `2`, per-triangle transforms are redrawn every frame.

## Orbit Pairs Test

Rotates each specified robot around its paired dynamic mesh at a given radius and speed.

```json
"orbit_pairs_test": {
    "orbit_radius": 2.0,
    "orbit_speed": 1.0,
    "seed": 42,
    "dyn_mesh_rot_interpolate_frames": 24,
    "pair_candidates": [["robots[0]", "dynamic_meshes[0]"]]
}
```

`dyn_mesh_rot_interpolate_frames` uses the same [interpolation convention](#random-dynamic-test) as the Random Dynamic Test. `orbit_speed` is in rad/s, and `pair_candidates` entries take `robots[N]` and `dynamic_meshes[N]`.

`lidar_orbit_test` orbits individual LiDARs around one dynamic mesh instead: `dynamic_mesh_index`, `orbit_radius`, `orbit_speed`, `seed`, and `lidar_indices` (a list of `[robot, lidar]` index pairs).

## Candidate Reference Syntax

| Reference | Resolves to |
|-----------|-------------|
| `robots` | All robots |
| `robots[0]` | First robot (anything after the index, e.g. `.lidars[1]`, is ignored — the whole robot is selected) |
| `dynamic_meshes` | All dynamic meshes |
| `dynamic_meshes[1]` | Second dynamic mesh |
