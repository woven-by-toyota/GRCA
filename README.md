# GRCA test bed

Unified multi-backend LiDAR benchmark for Ubuntu 22.04 and 24.04, used for evaluating the ray-casting algorithm (GRCA) introduced in following paper:

[Geometrically Approximated Modeling for Emitter-Centric
Ray-Triangle Filtering in Arbitrarily Dynamic LiDAR Simulation](https://arxiv.org/abs/2605.10457)

All eight backends — **GRCA-CPU**, **GRCA-CUDA**, **OptiX**, **Embree**, **TinyBVH-CPU**, **TinyBVH-GPU**, **Hybrid-CPU**, and **Hybrid-GPU** — are built from a single CMake project and driven by a single `benchmark_config.json` file.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
   - [Vulkan (CSV Viewer)](#vulkan-csv-viewer)
   - [CUDA backend](#cuda-backend)
   - [OptiX backend](#optix-backend-requires-cuda)
   - [Embree backend](#embree-backend-cpu-no-gpu-required)
   - [TinyBVH backends](#tinybvh-backends-cpu--gpu-no-extra-deps)
   - [Python (for plots)](#python-for-plots)
2. [Build](#build)
3. [Configuration](#configuration)
    - [Main-suite rebuild sweep](#main-suite-rebuild-sweep)
   - [Active tests](#active-tests)
   - [Test file fields](#test-file-fields)
   - [Hybrid Test Example](#hybrid-test-example)
   - [Random Dynamic Test](#random-dynamic-test)
   - [Orbit Pairs Test](#orbit-pairs-test)
4. [Run](#run)
5. [Output](#output)
   - [Performance CSV](#performance-csv-perf_csv)
   - [Hit Distance CSV](#hit-distance-csv-hit_dist_csv)
   - [Pose CSV](#pose-csv-pose_csv)
   - [Plotting](#plotting)
6. [Tools](#tools)
   - [`compare_csv`](#compare_csv)
   - [`csv_viewer`](#csv_viewer)
   - [`poll_upload_results.py`](#poll_upload_resultspy)
7. [Backend aliases](#backend-aliases)
8. [Backends](#backends)
   - [GRCA CPU](#grca-cpu)
   - [GRCA CUDA](#grca-cuda)
   - [OptiX](#optix)
     - [Custom intersection (CRTI)](#custom-intersection-crti)
   - [Embree](#embree)
   - [TinyBVH CPU](#tinybvh-cpu)
   - [TinyBVH GPU](#tinybvh-gpu)
   - [Hybrid CPU](#hybrid-cpu)
   - [Hybrid GPU](#hybrid-gpu)
9. [Orientation Convention](#orientation-convention)
10. [Unity CPU Version](#unity-cpu-version)

---

## Prerequisites

### Vulkan (CSV Viewer)

**Ubuntu 22.04:**
```bash
sudo apt install libvulkan1 vulkan-validationlayers-dev libvulkan-dev glslang-tools libglfw3 libglfw3-dev
```
**Ubuntu 24.04:**
```bash
sudo apt install libvulkan1 vulkan-tools vulkan-validationlayers libvulkan-dev glslang-tools libglfw3 libglfw3-dev
```

### CUDA backend
```bash
sudo apt install nvidia-driver-590
sudo reboot
```
Install CUDA Toolkit 13.1:
- [Ubuntu 22.04](https://developer.nvidia.com/cuda-13-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
- [Ubuntu 24.04](https://developer.nvidia.com/cuda-13-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)

### OptiX backend (requires CUDA)
Download [OptiX 9.1.0](https://developer.nvidia.com/designworks/optix/download), then:
```bash
cd ~/Downloads
chmod +x NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64.sh
```
Add to `~/.bashrc`:
```bash
export OPTIX_ROOT=$HOME/Downloads/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64
export OptiX_INSTALL_DIR=$OPTIX_ROOT/SDK
export OPTIX_INSTALL_DIR=$OPTIX_ROOT/SDK
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### TinyBVH backends (CPU + GPU; no extra deps)

TinyBVH is bundled as a single header (`third_party/tiny_bvh.h`). `tinybvh-cpu` is always compiled; `tinybvh-gpu` is compiled only when CUDA is available.

### Embree backend (CPU; no GPU required)

Build Embree 4.4.0 from source (supported on both Ubuntu versions):
```bash
git clone https://github.com/embree/embree.git --branch v4.4.0 --depth 1
cd embree
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DEMBREE_TUTORIALS=OFF \
    -DEMBREE_ISPC_SUPPORT=OFF \
    -DEMBREE_TASKING_SYSTEM=INTERNAL \
    -DEMBREE_RAY_PACKETS=ON \
    -DEMBREE_BACKFACE_CULLING=ON \
    -DEMBREE_FILTER_FUNCTION=OFF \
    -DEMBREE_IGNORE_INVALID_RAYS=OFF \
    -DEMBREE_GEOMETRY_TRIANGLE=ON \
    -DEMBREE_GEOMETRY_QUAD=OFF \
    -DEMBREE_GEOMETRY_CURVE=OFF \
    -DEMBREE_GEOMETRY_SUBDIVISION=OFF \
    -DEMBREE_GEOMETRY_USER=OFF \
    -DEMBREE_GEOMETRY_INSTANCE=OFF \
    -DEMBREE_GEOMETRY_POINT=OFF \
    -DEMBREE_COMPACT_POLYS=OFF \
    -DEMBREE_STATIC_LIB=ON
cmake --build build -j$(nproc)
sudo cmake --install build
```
CMake detects it automatically at `/usr/local` — no extra flags needed.

Key build flags:

| Flag | Value | Reason |
|------|-------|--------|
| `EMBREE_TASKING_SYSTEM` | `INTERNAL` | Removes TBB dependency and scheduler overhead |
| `EMBREE_RAY_PACKETS` | `ON` | Enables 4/8/16-wide SIMD packet kernels |
| `EMBREE_BACKFACE_CULLING` | `ON` | Rejects back-facing triangles; all GRCA meshes are outward-facing |
| `EMBREE_FILTER_FUNCTION` | `OFF` | GRCA does not use intersection filters |
| `EMBREE_IGNORE_INVALID_RAYS` | `OFF` | GRCA guarantees well-formed rays |
| `EMBREE_GEOMETRY_TRIANGLE` | `ON` | Only geometry type used |
| `EMBREE_STATIC_LIB` | `ON` | Eliminates PLT/GOT indirection on every `rtcIntersect` call |

### Python (for plots)

`benchmark/plot_results.py` runs automatically after each batch and requires `matplotlib` and `numpy`.

**Ubuntu 22.04:**
```bash
pip install matplotlib numpy
```
**Ubuntu 24.04** (pip is externally managed):
```bash
sudo apt install python3-matplotlib python3-numpy
```
Or use a venv on either version:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib numpy
```

---

## Build

```bash
cd path/to/GRCA
make        # builds all backends and shaders
make clean  # removes files under build/
```

> Always run `grca` from the **project root**, not from inside `build/`.

The configure step prints which backends were detected:
```
=== GRCA build configuration ===
  CPU backend:              ON (always)
  CUDA backend:             TRUE
  OptiX backend:            TRUE
  Embree backend:           TRUE
  TinyBVH CPU+GPU backend:  TRUE
  Hybrid-CPU backend (Embree+GRCA): TRUE
  Hybrid-GPU backend (OptiX+GRCA):  ON
```

Binaries in `build/`:
- `grca`           — main benchmark
- `compare_csv`   — CSV comparison utility
- `csv_viewer`    — Vulkan hit-point visualizer

---

## Configuration

All parameters live in `benchmark_config.json` (or a file passed as the first CLI argument).

### Structure

Config is split across five file types, all paths relative to `benchmark_config.json`:

```
benchmark_config.json                        → top-level: scenes, dynamic meshes, robots, active tests
configs/scenes/<name>.json                   → static mesh definitions only
configs/dynamic_meshes/<n>_dyn_mesh.json     → N dynamic mesh entries (sportsCar, 1–30)
configs/robots/<name>.json                   → robot + lidar array
configs/tests/<name>.json                    → one test per file (recording, backends, batches)
configs/tests/<subfolder>/<name>.json        → subfolders are supported (e.g. fast_sat_find/, fast_hybrid_find/)
```

The benchmark runs one session per **(scene × dynamic_mesh_count × robot_count)** combination, in that loop order.

**Current repository default `benchmark_config.json`**
```json
{
    "skip_validation": true,
    "skip_plot": true,
    "benchmark_scenes": [
        "configs/scenes/Lumberyard_all.json"
    ],
    "benchmark_dynamic_meshes": [
        "configs/dynamic_meshes/1_dyn_mesh.json"
    ],
    "benchmark_robots": [
        "configs/robots/max_rays_1_robot.json"
    ],
    "active_tests": [
        "configs/tests/static_visual.json"
    ]
}
```

**Main-suite benchmark example**
```json
{
    "skip_validation": false,
    "benchmark_scenes": [
        "configs/scenes/VokseliaSpawn.json",
        "configs/scenes/Lumberyard_all.json",
        "configs/scenes/SanMiguel.json",
        "configs/scenes/RungHolt.json",
        "configs/scenes/EmeraldSquare.json",
        "configs/scenes/PowerPlant.json"
    ],
    "benchmark_dynamic_meshes": [
        "configs/dynamic_meshes/10_dyn_mesh.json",
        "configs/dynamic_meshes/20_dyn_mesh.json",
        "configs/dynamic_meshes/30_dyn_mesh.json"
    ],
    "benchmark_robots": [
        "configs/robots/max_rays_2_robot.json",
        "configs/robots/max_rays_4_robot.json",
        "configs/robots/max_rays_6_robot.json",
        "configs/robots/max_rays_8_robot.json"
    ],
    "active_tests": [
        "configs/tests/main_suite/rebuild.json",
        "configs/tests/main_suite/rebuild_bound_chaos.json",
        "configs/tests/main_suite/rebuild_chaos.json"
    ]
}
```

For the full 6-run rebuild sweep, add:
- `configs/tests/main_suite/rebuild_compact.json`
- `configs/tests/main_suite/rebuild_bound_chaos_compact.json`
- `configs/tests/main_suite/rebuild_chaos_compact.json`

If `benchmark_dynamic_meshes` is omitted, the scene's own `dynamic_meshes` array is used as-is.

### Main-suite rebuild sweep

Each run tests a specific combination of compaction (on/off) and triangle-topology chaos mode (`tri_chaos = false/true/2`). Only dynamic objects are randomized; static meshes remain fixed. Mesh vertices always move; chaos modes additionally alter per-triangle behavior.

| Run | Test file | OptiX compaction | `tri_chaos` |
|-----|-----------|-----------------|-------------|
| 0 | `rebuild.json` | off | 0 |
| 1 | `rebuild_bound_chaos.json` | off | 2 |
| 2 | `rebuild_chaos.json` | off | 1 |
| 3 | `rebuild_compact.json` | on | 0 |
| 4 | `rebuild_bound_chaos_compact.json` | on | 2 |
| 5 | `rebuild_chaos_compact.json` | on | 1 |

Each main-suite run file currently has **1 batch** with:
- `backend: "all"`
- `lidar_sim_rate: 100.0`
- `pos_interpolate_frames: 0`, `rot_interpolate_frames: 0`, `scale_interpolate_frames: 0`

See [Interpolation convention](#random-dynamic-test) for the meaning of `0` and `N > 0`.

For these main-suite files, `record_frames` is set to `500`. Since backend is `all`, each batch runs all compiled backends, not GPU-only.

Correctness checks between backends require a minimum **95%** hit-distance match.

### Active tests

`active_tests` in `benchmark_config.json` lists exactly which test files to run — each entry is a filename. All listed tests run in order. Use `--active` on the CLI to override with an index selector:

> **Tip — live preview without benchmarking**
> `configs/tests/static_visual.json` has `visualizer_on: true`, `mesh_vis_on: 2`, and `record_frames: -1` (runs until the window is closed, writes no CSVs). Place it at the top of `active_tests` to open the interactive hit visualizer before any benchmark run. `visualizer_on` only opens the viewer; `mesh_vis_on` controls how meshes are drawn (`1` = wireframe, `2` = solid-filled mesh with wireframe overlay). Close the window to continue to the remaining tests, or remove it from the list when you no longer need the preview.

```bash
./build/grca --active tests[0]/batches[1]
```

Selectors: `tests[N]` (one run), `tests[N]/batches[M]` (one batch within a run).

### `configs/scenes/<name>.json` fields

| Field | Description |
|-------|-------------|
| `static_meshes` | Fixed-position `.obj` meshes; loaded once per scene |

### `configs/dynamic_meshes/<name>.json`

A JSON array of dynamic mesh entries. Referenced from `benchmark_config.json` via `"benchmark_dynamic_meshes"`. The default set provides 1, 10, 20, or 30 entries of `mesh/sportsCar/sportsCar.obj` (files: `1_dyn_mesh.json`, `10_dyn_mesh.json`, `20_dyn_mesh.json`, `30_dyn_mesh.json`).

### `configs/robots/<name>.json` fields

A JSON array of robot objects. Referenced from `benchmark_config.json` via `"benchmark_robots"`. The repository includes presets from 1 to 8 robots (plus multiple 8-robot variants). The main-suite sweep uses `max_rays_2_robot.json`, `max_rays_4_robot.json`, `max_rays_6_robot.json`, and `max_rays_8_robot.json`.

### Test file fields

| Field | Default | Description |
|-------|---------|-------------|
| `visualizer_on` | `false` | Opens the visualizer window when `true`. Integer values are treated as non-zero/zero enable flags only. |
| `mesh_vis_on` | `0` | Mesh display mode: `0` = no mesh, `1` = wireframe only, `2` = solid mesh + wireframe overlay (uses per-triangle MTL colors), `3` = same as `2` but forces flat JSON `color` for every mesh (ignores MTL) |
| `record_frames` | `-1` | Frames to record (perf, hit-dist, and pose); `-1` disables |
| `skip_validation` | `true` | Skip hit-distance and pose CSV writing and end-of-batch comparison; match columns in results CSV are left at 0. Set in `benchmark_config.json` (top level only) |
| `skip_plot` | `true` | Skip all `plot_results.py` invocations (per-batch, per-run, and session-level). Set in `benchmark_config.json` (top level only) |
| `auto_hit_view_record_frame` | `-1` | Frame to auto-write a hit-view CSV; `-1` disables |
| `cuda` | — | CUDA options (see below) |
| `cpu` | — | See [GRCA-CPU options](#grca-cpu-packet-size) |
| `embree` | — | See [Embree options](#embree) |
| `optix` | — | See [OptiX options](#optix) |
| `optix-crti` | — | See [Custom intersection (CRTI)](#custom-intersection-crti) — same fields as `optix` |
| `tinybvh` | — | See [TinyBVH options](#tinybvh-cpu) |
| `batches` | — | Array of batch objects (see [`batches` fields](#batches-fields)) |

#### CUDA options (`"cuda": { ... }`)

| Field | Default | Description |
|-------|---------|-------------|
| `debug` | `false` | Print BAT/SAT dispatch stats each frame |
| `sat_max_sweep_diff` | `64` | SAT classification threshold — triangles whose estimated sweep span exceeds this are deferred to the late pass as BAT |
| `sat_max_channel_diff` | `64` | SAT classification threshold — triangles whose channel span exceeds this are deferred to the late pass as BAT |

### Hybrid Test Example

**configs/tests/3080ti mobile/rebuild_hybrid.json** demonstrates a hybrid configuration: refit for 10 frames, rebuild on the 11th, repeat:

```json
{
    "cpu": {
        "packet_size": 16
    },
    "embree": {
        "bvh_rebuild_after_frames": 10,
        "packet_size": 16
    },
    "optix": {
        "bvh_rebuild_after_frames": 10,
        "bvh_compaction_on": false
    },
    ...
}
```

This refits for 10 frames then rebuilds on frame 11, repeating with period 11. See [BVH update mode](#bvh-update-mode-optix-and-embree) for build flag and quality details.

> **Frame loop termination**
> The loop runs until `record_frames` frames are recorded.
> - `record_frames: -1` + dedicated backend → runs **indefinitely** (Ctrl+C to stop).
> - `record_frames: -1` + any backend selector that expands to multiple backends (for example `all`, `cpu`, `gpu`, `all_t`, `gpu-all`, `hybrid`) → **rejected** with an error (first backend would never finish).

### `batches` fields

| Field | Description |
|-------|-------------|
| `backend` | A dedicated backend name or a group alias (see [Backend aliases](#backend-aliases)) |
| `lidar_sim_rate` | Simulation tick rate (Hz) |
| `random_dynamic_test` | See [Random Dynamic Test](#random-dynamic-test) |
| `orbit_pairs_test` | See [Orbit Pairs Test](#orbit-pairs-test) |

### `static_meshes` fields (scene files) / `dynamic_meshes` fields (dynamic mesh files)

| Field | Description |
|-------|-------------|
| `file` | Path to `.obj` (relative to working dir) |
| `w_position` | World-space offset `[x, y, z]` |
| `scale` | Scale factor |
| `color` | Display color `[r, g, b]` |

### `robots` / `lidars` fields

| Field | Description |
|-------|-------------|
| `w_position` | Robot world-space origin |
| `w_forward`, `w_up` | Robot orientation (unit vectors) |
| `lidars` | Array of lidar configs |
| `hmin/hmax` | Horizontal angle range (radians) |
| `hnum` | Horizontal ray count |
| `vmin/vmax` | Vertical angle range (radians) |
| `vnum` | Vertical channel count |
| `range_min/max` | Hit distance limits (metres) |
| `position`, `forward`, `up` | Lidar local-space pose (unit vectors) |

### Random Dynamic Test

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

**Interpolation convention** (used by `pos_`, `rot_`, `scale_interpolate_frames`):

| Value | Behaviour |
|-------|-----------|
| `-1` | Disabled — stays at configured pose |
| `0` | Instant snap — new random value every frame |
| `N > 0` | New target every N frames, linearly interpolated |

### Orbit Pairs Test

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

`dyn_mesh_rot_interpolate_frames` uses the same [interpolation convention](#random-dynamic-test) as the Random Dynamic Test.

### Candidate Reference Syntax

| Reference | Resolves to |
|-----------|-------------|
| `robots` | All robots |
| `robots[0]` | First robot |
| `dynamic_meshes[1]` | Second dynamic mesh |
| `robots[0].lidars[1]` | Second lidar of first robot |

---

## Run

| Command | Action |
|---------|--------|
| `make run` | Run all batches (performance governor, pinned to core 0) |
| `make run-csv-viewer <file>` | Vulkan CSV visualizer |
| `make run-csv-compare ...` | Compare two or more CSV files |

Override backend for a one-off run (see [Backend aliases](#backend-aliases) for group aliases):
```bash
./build/grca --backend grca-cuda
./build/grca --backend all_t
./build/grca my_benchmark_config.json --active tests[1]/batches[0]
```

### Interactive controls (visualizer)

Both `grca` (Hit Viewer) and `csv_viewer` use an FPS free-roam camera:

| Input | Action |
|-------|--------|
| **Ctrl** | Toggle FPS mode on/off (locks/unlocks cursor) |
| **Mouse move** | Look around *(FPS mode)* |
| **W / S** | Move forward / backward *(FPS mode)* |
| **A / D** | Move left / right *(FPS mode)* |
| **E / Q** | Move up / down *(FPS mode)* |
| **Shift** | Fast movement *(FPS mode)* |
| **Scroll wheel** | Adjust movement speed |
| **T** | Snap camera to next lidar (cycles each press; lidar-follow mode) |
| **P** | Exit lidar-follow mode and return to free-roam |
| **M** | Toggle all meshes (static + dynamic) on/off |
| **B** | Toggle dynamic meshes on/off (overrides `M` for dynamic only) |
| **C** | Toggle lidar cones on/off |
| **F** | Focus camera on hit-point bounding box (keeps orientation) |
| **O** | Write hit CSV for current frame *(Hit Viewer only)* |
| **Esc** | Quit |

---

## Output

### Directory structure

Each (scene × dynamic_mesh_count × robot_count) combination produces one session folder. All session folders share the **benchmark start time** in their name so they sort together. The actual per-session start time is recorded inside `session_meta.csv`. After all sessions complete, they are all archived into a single zip.

```
benchmark/
├── YYYYMMDD_HHMMSS-<user>-<host>-[scene0]VokseliaSpawn-[dyn1]10_dyn_mesh-[robot0]max_rays_2_robot/   ← session folder
│   ├── session_meta.csv                ← hardware info, actual session start time, config files used
│   ├── benchmark_config.json           ← exact copy of the top-level config
│   ├── configs/                        ← only the files actually used, folder structure preserved
│   │   ├── scenes/VokseliaSpawn.json
│   │   ├── dynamic_meshes/10_dyn_mesh.json
│   │   ├── robots/max_rays_2_robot.json
│   │   └── tests/rebuild.json
│   ├── results_session.png             ← cross-run overview (only if ≥2 runs)
│   ├── run_0/
│   │   ├── results_run0.png            ← cross-batch overview (only if ≥2 batches)
│   │   ├── batch_0/
│   │   │   ├── results_run0_batch0.csv ← per-backend perf averages
│   │   │   ├── results_run0_batch0.png ← per-batch bar chart
│   │   │   ├── perf_GRCA_cuda_<ts>.csv
│   │   │   └── hit_view_GRCA_cuda_<ts>.csv
│   │   └── batch_1/ ...
│   └── run_1/ ...
├── YYYYMMDD_HHMMSS-<user>-<host>-[scene0]VokseliaSpawn-[dyn1]10_dyn_mesh-[robot1]max_rays_4_robot/   ← next session ...
├── YYYYMMDD_HHMMSS-<user>-<host>-[scene1]RungHolt-[dyn1]10_dyn_mesh-[robot0]max_rays_2_robot/
└── YYYYMMDD_HHMMSS-<user>-<host>.zip  ← all session folders archived after the full benchmark
```

Raw `hit_dist_*.csv` and `pose_*.csv` files are deleted after the per-batch correctness check — their summary stats are already captured in `results_runN_batchM.csv`.

The breakdown line printed to stderr identifies each session:
```
--- Breakdown [Run 1, Batch 4, [scene0]VokseliaSpawn-[dyn1]10_dyn_mesh-[robot0]max_rays_2_robot] (1048576 rays, 1875632 static + 3006030 dyn tris) ---
```

### Summary files

| File | Location | Contents |
|------|----------|----------|
| `session_meta.csv` | session root | Hardware info, which config files were used, scene/ray counts |
| `benchmark_config.json` | session root | Exact copy of the top-level config |
| `configs/` | session root | Only the referenced files (scene, robots, active tests), folder structure preserved |
| `results_runN_batchM.csv` | `run_N/batch_M/` | Per-backend performance metrics averaged over recorded frames |

### PNG files

| File | Location | When generated | Contents |
|------|----------|---------------|----------|
| `results_runN_batchM.png` | `run_N/batch_M/` | Every batch | Full per-batch bar chart — all metrics, one bar per backend |
| `results_runN.png` | `run_N/` | Run N has ≥2 batches | Cross-batch overview — averages across all batches, condensed layout |
| `results_session.png` | session root | Session has ≥2 runs | Cross-run overview — averages of per-run overviews, grouped by `api_label` |

**Overview PNG layout** (`results_runN.png` and `results_session.png`):

| Row | Charts |
|-----|--------|
| 1 | Avg Lidar Sim CPU · Avg BVH Rebuild CPU · Avg BVH Refit CPU |
| 2 | Avg Trace CPU · Avg Sync Readback · Fastest batch/run (Lidar CPU ms) · Slowest batch/run (Lidar CPU ms) |
| 3 | GPU metrics (Lidar Sim GPU · BVH Rebuild GPU · BVH Refit GPU · Trace GPU) — if any GPU backend ran |
| 4 | Avg FPS · Avg RTF · Avg Total CPU · Correctness check |

The fastest/slowest charts in row 2 show one bar per backend, each from the batch (or run) where that backend performed best/worst. The batch or run ID is annotated above each bar (e.g. **B0**, **B3** for batches; **R0**, **R2** for runs), with min/max error bars.

### `results_runN_batchM.csv` columns

Columns follow the same order as the per-frame perf CSV.

| Column | Description |
|--------|-------------|
| `run`, `batch` | Run and batch indices (`-1` in aggregated files) |
| `backend` | Detailed backend label (e.g. `GRCA_cpu_vec8`, `GRCA_cuda`, `Optix_rb`, `Optix_crti_rb`, `Optix_cmp_rf`, `Embree_vec1_hb10`, `TinyBVH_cpu_rb`, `TinyBVH_gpu_hb4`, `Hybrid_cpu_Embree_rf_GRCA_vec16`, `Hybrid_gpu_Optix_rf_GRCA_cuda`) |
| `api_label` | Simple API name used in session charts (`GRCA_cpu`, `GRCA_cuda`, `Optix`, `Optix_crti`, `Embree`, `TinyBVH_cpu`, `TinyBVH_gpu`, `Hybrid_cpu`, `Hybrid_gpu`) |
| `perf_frames` | Frames recorded (summed in aggregated files) |
| `lidar_sim_rate` | Simulation tick rate (Hz) |
| `avg_lidar_sim_cpu_ms` | Avg lidar simulation CPU time |
| `min_lidar_sim_cpu_ms` | Min lidar simulation CPU time across recorded frames |
| `max_lidar_sim_cpu_ms` | Max lidar simulation CPU time across recorded frames |
| `avg_dynamic_upload_ms` | Avg CPU time writing updated vertex positions into the backend buffer |
| `avg_bvh_rebuild_cpu_ms` | Avg BVH full rebuild time (CPU wall time) |
| `avg_bvh_refit_cpu_ms` | Avg BVH incremental refit time (CPU wall time) |
| `avg_trace_cpu_ms` | Avg ray-trace CPU time |
| `avg_sync_readback_ms` | Avg `cudaStreamSynchronize` stall before readback |
| `avg_readback_ms` | Avg device→host copy time |
| `avg_fill_hits_ms` | Avg time unpacking raw hit buffer into hit lists |
| `avg_lidar_sim_gpu_ms` | Avg lidar simulation GPU time |
| `min_lidar_sim_gpu_ms` | Min lidar simulation GPU time across recorded frames |
| `max_lidar_sim_gpu_ms` | Max lidar simulation GPU time across recorded frames |
| `avg_bvh_rebuild_gpu_ms` | Avg BVH full rebuild GPU time |
| `avg_bvh_refit_gpu_ms` | Avg BVH incremental refit GPU time |
| `avg_trace_gpu_ms` | Avg ray-trace GPU time |
| `init_cpu_ms` | CPU time for backend initialisation (one-off) |
| `avg_fps` | Avg simulated frames per second |
| `avg_rtf` | Avg real-time factor (`fps / lidar_sim_rate`; >1 means faster than real-time) |
| `avg_total_cpu_ms` | Avg full frame wall time including mesh prep |
| `cmp_tol`, `cmp_min_match` | Tolerance and minimum match % used for correctness checks |
| `hit_dist_pairs_matched`, `hit_dist_pairs_total`, `hit_dist_min_pct` | Hit-distance correctness check results |
| `pose_pairs_matched`, `pose_pairs_total`, `pose_min_pct` | Pose correctness check results |

### Batch summary (stderr)

After every batch, a summary is printed to stderr:

```
  --------------------------------------------------------
  Batch 0 Summary  (2 backends)
  --------------------------------------------------------

  CPU  (avg total, lower is better)
  Backend            Total(ms)  LidarSim(ms)  Trace(ms)
  -------            ---------  ------------  ---------
  1. GRCA_cuda            105.680        75.240     43.118
  2. Optix_rb            121.430        89.310     51.220

  GPU  (avg total, lower is better)
  ...

  Hit Distance Check  (tol=0.001  min-match=95%)
  hit_dist_GRCA_cuda   vs  hit_dist_Optix_rb          MATCH     dist=97.40%(max=9.80e+02)
  1/1 pairs match  (tol=0.001  min-match=95%)

  Pose Check  (tol=0.001  min-match=95%)
  pose_GRCA_cuda   vs  pose_Optix_rb            MATCH     all=100.00%
  1/1 pairs match  (tol=0.001  min-match=95%)
  --------------------------------------------------------
```

Hit distance mismatches near `range_max` are expected — different BVH algorithms disagree on grazing-angle rays at geometry boundaries. `MATCH` at 95% threshold means backends agree well enough. Run `compare_csv` without `--simple` for full per-column stats.

### Plotting

PNGs are generated **automatically** at three points during a run, with no redundant work.
Each PNG is saved twice: a normal version and a `*_anon.png` version with `GRCA` → `[I]LA` in all labels (for blind review).

| When | What is generated |
|------|-------------------|
| After each batch | `results_runN_batchM.png` + `results_runN_batchM_anon.png` |
| After each run (≥2 batches) | `results_runN.png` + `results_runN_anon.png` |
| After all runs (≥2 runs) | `results_session.png` + `results_session_anon.png` |

The plotter reads `results_runN_batchM.csv` files from `run_N/batch_M/` — no intermediate aggregate CSVs are written. `results_session.png` averages each run's batches first, then averages across runs grouped by `api_label`; the fastest/slowest run charts are derived from those per-run averages.

You can also run the plotter manually:

```bash
# Single batch — PNG saved alongside the CSV in run_N/batch_M/
python3 benchmark/plot_results.py benchmark/SESSION/run_0/batch_0/results_run0_batch0.csv

# Whole session — regenerates all batch, run, and session PNGs
python3 benchmark/plot_results.py benchmark/SESSION

# One run only — regenerates batch PNGs for that run + its run overview
python3 benchmark/plot_results.py benchmark/SESSION --run 0

# Session overview only (skips per-batch and per-run PNGs)
python3 benchmark/plot_results.py benchmark/SESSION --session-only
```

See [Python prerequisites](#python-for-plots) for installation.

### Performance CSV (`perf_*.csv`)

Written per backend per batch. The first line is a `#`-prefixed header with `lidar_sim_rate`; then a column-header row; then one data row per frame. After all data rows, per-column averages are appended as `#`-prefixed footer comments (these are stripped by the plotting scripts).

| Column | What it measures |
|--------|-----------------|
| `ms_dynamic_upload` | CPU time writing updated vertex positions into the backend buffer |
| `ms_bvh_rebuild_gpu` | Full BVH rebuild (GPU time for CUDA/OptiX; CPU wall time for Embree) |
| `ms_bvh_refit_gpu` | Incremental BVH refit |
| `ms_trace_gpu` | Ray traversal / intersection kernel |
| `ms_sync_readback` | `cudaStreamSynchronize` stall before readback |
| `ms_readback_cpu` | Device→host copy of hit results |
| `ms_fillhits` | Unpacking raw hit buffer into hit lists |
| `ms_lidar_sim_cpu` | Total lidar simulation wall time |
| `ms_total_cpu` | Full frame wall time including mesh prep |

### Hit Distance CSV (`hit_dist_*.csv`)

Written when `record_frames > -1` and `skip_validation` is not set. To keep file sizes manageable, only `record_frames / 10` frames are recorded — the set of frames is chosen randomly at run start with a fixed seed and is **shared across all batches**, so every backend samples the same frames. One row per ray per sampled frame:

```
frame,dist
0,14.32
0,1000.0   ← miss (range_max)
```

### Pose CSV (`pose_*.csv`)

Written when `record_frames > -1` and `skip_validation` is not set. World-space pose (position + orientation) for all robots, dynamic meshes, and static meshes at every recorded frame.

### Hit View CSV (`hit_view_*.csv`)

Written interactively (**O** key in visualizer) or automatically via `auto_hit_view_record_frame`. Contains world-space hit positions for visualization.

---

## Tools

### `compare_csv`

Compares two or more CSV files row-by-row with per-column statistics. Used automatically in the batch summary (`--simple` mode) and available standalone:

```bash
# Full per-column table
make run-csv-compare tol=0.001 match=98 a.csv b.csv

# One line per pair (same as used in batch summary)
./build/compare_csv --simple --tol 0.001 --match 95 a.csv b.csv c.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tol`, `-t` | `0.001` | Absolute tolerance per cell |
| `--match`, `-m` | `98.0` | Min Match% per column to pass |
| `--simple`, `-s` | off | One line per pair; skips 100% columns |

### `csv_viewer`

Vulkan-based 3D visualizer for hit-point CSVs. See [Interactive controls](#interactive-controls-visualizer).

```bash
make run-csv-viewer hits.csv
./build/csv_viewer hits.csv
```

### `poll_upload_results.py`

Watches `benchmark/` for new `.zip` files produced after each benchmark run, copies them into `results/<branch> - <short-commit>/`, and commits and pushes to the remote. Runs until stopped with Ctrl+C. Pinned to CPU core 4 so it does not interfere with the benchmark.

```bash
python3 poll_upload_results.py --interval 100 --repo .
```

| Option | Default | Description |
|--------|---------|-------------|
| `--interval` | `10` | Poll interval in seconds |
| `--repo` | `.` | Path to the GRCA repo root |

Start it in a separate terminal before (or during) a benchmark run. It will ignore any `.zip` files that already exist when it starts, and only commit new ones as they appear.

---

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

---

## Backends

### GRCA CPU

Triangle-out span-prediction algorithm, single-threaded. Built with `-O3 -march=native -ffast-math -funroll-loops`.

#### GRCA-CPU packet size

Controls the SIMD width used in the SAT inner loop (Möller–Trumbore tests of multiple rays against one triangle simultaneously). Set with `cpu.packet_size` in the test config:

| Value | Inner loop stride | Intrinsic set | Notes |
|-------|------------------|---------------|-------|
| `16` | 16 rays / iter (2 × 8) | AVX2 `__m256` **(default)** | Both batches share the same broadcast constants |
| `8` | 8 rays / iter | AVX2 `__m256` | |
| `4` | 4 rays / iter | SSE2 `__m128` | |
| `1` | scalar | — | |

The backend label and output file prefix reflect the configured width: `GRCA_cpu_vec16`, `GRCA_cpu_vec8`, `GRCA_cpu_vec4`, `GRCA_cpu_vec1`.

> **`16` vs `8`**: both use AVX2 (`__m256`) — no AVX-512 on Alder Lake. The 16-wide path processes two independent batches of 8 per outer iteration, letting the CPU interleave them for better ILP. The AVX2/SSE2 paths only activate on sequential (non-wraparound) ray rows in the SAT branch; wraparound rows fall back to scalar regardless of the setting.

### GRCA CUDA

Triangle-out span-prediction algorithm on the GPU (no BVH). Dynamic mesh vertices are re-uploaded via `cudaMemcpy` each frame.

The early pass appends BAT (big-appearing) triangles to a fixed-capacity deferred `TriRay` buffer (`MAX_TRI_RAY_COUNT = 5,000,000` entries × 212 bytes ≈ **1 GB**). Each entry stores the triangle index, per-LiDAR-origin sweep span range, per-LiDAR-origin closest-vertex distance, and per-LiDAR-origin flags (origin mask, cone-apex hit state, CW/CCW winding). The late pass reads these directly — no early-pass work is repeated.

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
All deltas are relative to VRAM used at process start.

### OptiX

Hardware ray tracing via NVIDIA OptiX 9.1. Requires CUDA.

#### Custom intersection (CRTI)

`optix-crti` is a separate backend that uses software Möller–Trumbore triangle intersection on CUDA shader processors while keeping RT-core BVH traversal. Use it the same way as `optix` in your test file:

```json
{
    "backends": ["optix", "optix-crti"],
    "optix": {
        "bvh_rebuild_after_frames": 0,
        "bvh_compaction_on": false
    },
    "optix-crti": {
        "bvh_rebuild_after_frames": 0,
        "bvh_compaction_on": false
    }
}
```

**What differs from `optix`:**

| | `optix` | `optix-crti` |
|---|---|---|
| BVH build input | `OPTIX_BUILD_INPUT_TYPE_TRIANGLES` | `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES` (per-triangle AABB) |
| Triangle intersection | Hardware RT cores (`__closesthit__`) | CUDA shader processors (`__intersection__` — Möller–Trumbore) |
| BVH traversal | RT cores | RT cores (unchanged) |
| PTX loaded | `build/optix_device.ptx` | `build/optix_custom_device.ptx` |
| Backend label | `Optix_rb`, `Optix_cmp_rb`, … | `Optix_crti_rb`, `Optix_cmp_crti_rb`, … |
| API label | `Optix` | `Optix_crti` |

**Comparison purpose:** Comparing `Optix_rb` vs `Optix_crti_rb` isolates the cost of hardware RT-core triangle intersection from BVH traversal.

### Embree

Intel Embree 4.4.0, CPU ray tracing with SIMD packets.

#### BVH update mode (OptiX and Embree)

Controlled by a single `bvh_rebuild_after_frames` integer in the respective backend section of the test file:

| Value | Mode | Behaviour |
|-------|------|-----------|
| `0` | Pure rebuild | Full rebuild every frame **(default)** |
| `-1` | Pure refit | Incremental in-place refit every frame — never rebuilds after init |
| `N > 0` | Hybrid | Refit for N frames, then rebuild on frame N+1; repeats with period N+1 |

Examples:
- `"bvh_rebuild_after_frames": -1` — pure refit, highest BVH quality at init
- `"bvh_rebuild_after_frames": 0` — pure rebuild every frame
- `"bvh_rebuild_after_frames": 1` — alternating: refit, rebuild, refit, rebuild, ...
- `"bvh_rebuild_after_frames": 10` — 10 refits then 1 rebuild, repeat

**`bvh_compaction_on`** — when `true`, enables compaction for the built acceleration structure. Reflected in the backend label as `cmp`:

| Backend | Effect | Label examples |
|---------|--------|---------------|
| **OptiX** | Post-build `optixAccelCompact` step removes unused padding from the GAS output buffer, reducing VRAM. Same BVH quality, no traversal penalty. | `Optix_cmp_rb`, `Optix_cmp_rf`, `Optix_cmp_hb10`, `Optix_cmp_crti_rb` |
| **Embree** | Sets `RTC_SCENE_FLAG_COMPACT` on the scene, instructing Embree to use a more compact internal BVH node format. Reduces system RAM at a small traversal performance cost. | `Embree_vec8_cmp_rb`, `Embree_vec16_cmp_rf` |

See `configs/tests/3080ti mobile/rebuild_hybrid.json` for a practical example.

Embree `RTCBuildQuality` used per geometry type and mode:

| Geometry | Mode | Quality | Why |
|----------|------|---------|-----|
| Static | all | `HIGH` | SAH; built once at init |
| Dynamic | pure refit (init) | `HIGH` | Best quality; init is the only full build |
| Dynamic | hybrid / rebuild (init + rebuilds) | `LOW` | Fast Morton builder; consistent with every periodic/per-frame rebuild |
| Dynamic | refit frame | `REFIT` | In-place node adjustment; fastest |

#### Embree ray packet size

Set with `embree.packet_size` in config:

| Value | API call | SIMD width |
|-------|----------|-----------|
| `1` | `rtcIntersect1` | scalar |
| `4` | `rtcIntersect4` | 128-bit (SSE4) |
| `8` | `rtcIntersect8` | 256-bit (AVX2) |
| `16` | `rtcIntersect16` | 512-bit (AVX-512) **(default)** |

The backend label and output file prefix reflect the chosen width, compaction flag, and BVH mode: `Embree_vec1_rb`, `Embree_vec1_rf`, `Embree_vec1_hb10`, `Embree_vec4_rb`, `Embree_vec8_cmp_rb`, etc.

### TinyBVH CPU

Software BVH ray tracer on CPU. Uses [tinybvh](https://github.com/jbikker/tinybvh) (bundled at `third_party/tiny_bvh.h`) — no external installation required.

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

Backend label examples: `TinyBVH_cpu_rb`, `TinyBVH_cpu_rf`, `TinyBVH_cpu_hb4`.

### TinyBVH GPU

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

Requires CUDA. Backend label examples: `TinyBVH_gpu_rb`, `TinyBVH_gpu_rf`, `TinyBVH_gpu_hb4`.

**Comparison purpose:** `tinybvh-gpu` vs `optix` isolates the benefit of hardware RT cores (OptiX uses them; tinybvh-gpu does not). `tinybvh-cpu` vs `tinybvh-gpu` isolates CPU-vs-GPU traversal with the identical BVH and ray set.

### Hybrid CPU

Splits the scene at `staticTriCount` and runs two sub-backends in parallel per frame:

- **Static geometry** → Embree BVH (all rays, static tris only). The BVH is built once at init with `ALLOW_UPDATE` flags (`bvh_rebuild_after_frames = -1` is always forced internally), then a near-zero-cost refit is called each frame since vertices never change.
- **Dynamic geometry** → GRCA-CPU two-pass angular filter (all rays, dynamic tris only).
- **Merge** — per-ray: the closer of the two hit distances wins.

**Key points:**
- `bvh_rebuild_after_frames` in the `embree` config section is **ignored** for `hybrid-cpu`; the static BVH always uses pure refit.
- The GRCA-CPU sub-backend respects all `cpu` config options (`packet_size`, `sat_max_*_diff`).
- Requires Embree (`HAVE_EMBREE`).

**Backend label:** `Hybrid_cpu_Embree_rf_GRCA_vec<N>` (e.g. `Hybrid_cpu_Embree_rf_GRCA_vec16`).

**Timing fields:**

| Field | Source |
|-------|--------|
| `ms_trace_cpu` | GRCA-CPU pass |
| `ms_trace_gpu` | Embree BVH traversal (CPU-side, reported in GPU slot) |
| `ms_bvh_refit_cpu` | Embree per-frame refit (near-zero for static geometry) |
| `ms_dynamic_upload` | Sum from both sub-backends |

### Hybrid GPU

Splits the scene at `staticTriCount` and runs two sub-backends in parallel per frame:

- **Static geometry** → OptiX RT cores (all rays, static tris only). The acceleration structure is built once at init with `OPTIX_BUILD_FLAG_ALLOW_UPDATE` (`optix_rebuild_after_frames = -1` is always forced internally), then a near-zero-cost `optixAccelUpdate` is called each frame since vertices never change.
- **Dynamic geometry** → GRCA-CUDA early + late pass (all rays, dynamic tris only). All dynamic triangles are re-uploaded to GPU each frame (`staticTriCount = 0` from GRCA-CUDA's perspective).
- **Merge** — per-ray: the closer of the two hit distances wins. Done on CPU after both readbacks complete.

**Key points:**
- `bvh_rebuild_after_frames` in the `optix` config section is **ignored** for `hybrid-gpu`; the static AS always uses pure refit.
- The GRCA-CUDA sub-backend respects all `cuda` config options (`sat_max_*_diff`, `debug`).
- Requires CUDA + OptiX (`HAVE_OPTIX && HAVE_CUDA`).
- Both OptiX and GRCA-CUDA hold their own GPU allocations simultaneously; VRAM usage is tracked and reported the same way as for the standalone GPU backends.

**Backend label:** `Hybrid_gpu_Optix_rf_GRCA_cuda`.

**Run order within group aliases** (`all_h`, `gpu_h`): `optix → grca-cuda → hybrid-gpu` then `embree → grca-cpu → hybrid-cpu`, so each hybrid result appears immediately after its component backends in the benchmark CSV.

**Timing fields:**

| Field | Source |
|-------|--------|
| `ms_trace_gpu` | Sum: OptiX trace + GRCA-CUDA trace |
| `ms_bvh_refit_gpu` | OptiX per-frame refit (near-zero for static geometry) |
| `ms_dynamic_upload` | Sum from both sub-backends |
| `ms_readback_cpu` | Sum: OptiX + GRCA-CUDA readbacks |
| `ms_lidar_sim_gpu` | Sum: OptiX + GRCA-CUDA GPU sim time |
| `num_bat`, `num_early_t`, `num_rtic` | GRCA-CUDA dynamic pass only |

---

## Orientation Convention

**R = R_y(yaw) × R_x(pitch) × R_z(roll)**

With all angles zero: forward = (0,0,1), up = (0,1,0), right = (1,0,0).

---

## Unity CPU Version

1. Install [Unity Hub](https://docs.unity3d.com/hub/manual/InstallHub.html#install-hub-linux)
2. Drag `GRCA_CPU.cs` into the Assets folder
3. Hierarchy → right-click → `Create Empty`
4. Select the GameObject → Inspector → `Add Component` → `GRCA_CPU`
5. Hierarchy → right-click → `3D Object` → `Cube`, place in front of the GameObject
6. Press Play:
   - `G` — run GRCA benchmark
   - `H` — show LiDAR hit points
