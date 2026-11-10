# GRCA test bed

Unified multi-backend LiDAR benchmark for Ubuntu 22.04 and 24.04, used for evaluating the ray-casting algorithm (GRCA) introduced in the following paper:

[Geometrically Approximated Modeling for Emitter-Centric
Ray-Triangle Filtering in Arbitrarily Dynamic LiDAR Simulation](https://arxiv.org/abs/2605.10457)

Qualitative results for the paper can be found at:
[[I]RCA Visualization](https://anonymous.4open.science/r/-I-RCA-Visualization/README.md)

The mesh files evaluated for the paper and required for this repository can be downloaded from [mesh.zip](https://zenodo.org/records/21127672/files/mesh.zip?download=1&preview=1). Unzip and copy paste to GRCA project root:

```bash
cd /path/to/GRCA
unzip /path/to/mesh.zip
cp -a mesh/. ./mesh/
```

All nine backends — **GRCA-CPU**, **GRCA-CUDA**, **OptiX**, **OptiX-CRTI**, **Embree**, **TinyBVH-CPU**, **TinyBVH-GPU**, **Hybrid-CPU**, and **Hybrid-GPU** — are built from a single CMake project and driven by a single `benchmark_config.json` file.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build](#build)
3. [Configuration](#configuration) — including the [benchmark presets](#benchmark-presets) used for the paper
4. [Run](#run)
5. [Orientation Convention](#orientation-convention)
6. [Unity CPU Version](#unity-cpu-version)
7. [Documentation](#documentation)

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

Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### OptiX backend (requires CUDA)
Download [OptiX 9.1.0](https://developer.nvidia.com/designworks/optix/download), then:
```bash
cd ~/Downloads
chmod +x NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64.sh
```
Add to `~/.bashrc` (`make` passes it to CMake as `OptiX_INSTALL_DIR`):
```bash
export OPTIX_ROOT=$HOME/Downloads/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64
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

`benchmark/plot_results.py` runs automatically after each batch when the top-level `skip_plot` is `false` (it defaults to `true`), and requires `matplotlib` and `numpy`.

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

The configure step prints which backends were detected (`CMakeLists.txt`; the build type defaults to `Release`):
```
-- === GRCA build configuration ===
--   Build type : Release
--   CPU backend: ON (always)
--   CUDA backend: TRUE
--   OptiX backend: TRUE
--   Embree backend: TRUE
--   TinyBVH CPU+GPU backend: TRUE
--   Hybrid-CPU backend (Embree+GRCA): TRUE
--   Hybrid-GPU backend (OptiX+GRCA): ON
--   Visualiser (GLFW): ON (always)
-- ================================
```

Binaries in `build/`:
- `grca`           — main benchmark
- `compare_csv`   — CSV comparison utility
- `csv_viewer`    — Vulkan hit-point visualizer

## Configuration

A run is driven by `benchmark_config.json` in the project root (or another top-level config passed as the first CLI argument).

The full config reference — file structure, every field, and the main-suite test files — is in [docs/configuration.md](docs/configuration.md).

### Benchmark presets

`presets/` holds the top-level configs used for the paper's benchmarks, all run on PC1 (i7-12700H + RTX 3080 Ti Mobile). Because config file paths resolve relative to the top-level config's directory, a preset cannot be run in place from `presets/`; copy its contents into `benchmark_config.json` at the project root first:

```bash
cp presets/benchmark_config_3080ti_main_suite.json benchmark_config.json
./build/grca
```

| Preset | Paper experiment |
|--------|------------------|
| `benchmark_config_3080ti_main_suite.json` | Results: full main suite, `range_max` = 1000 m |
| `benchmark_config_3080ti_main_suite_100.json` | Results: range-culled main suite, `range_max` = 100 m |
| `benchmark_config_3080ti_main_suite_hybrid.json` | Results: hybrid pipeline main suite, `range_max` = 1000 m |
| `benchmark_config_3080ti_main_suite_hybrid_100.json` | Results: Range+Hybrid main suite, `range_max` = 100 m |
| `benchmark_config_3080ti_hybrid_backend.json` | Hybrid Static/Dynamic Pipeline: hybrid vs. standalone backends (Power Plant, 30 dyn, Ω = 8) |
| `benchmark_config_3080ti_best_hybrid.json` | BVH Mode Selection: `rf` / `rb` / `pr1`–`pr15` under ND and OBD |
| `benchmark_config_3080ti_best_hybrid_chaos.json` | BVH Mode Selection: `rf` / `rb` / `pr1` / `pr2` under SWD |
| `benchmark_config_3080ti_extreme_pose.json` | Pose sensitivity: the eight motion configurations (Suppl., Preliminary Study) |
| `benchmark_config_3080ti_sub_div_test.json` | Mesh density (fixed scale): sports car subdivided up to 64× |
| `benchmark_config_3080ti_scale.json` | Mesh scale (fixed density): 8 scale ranges |
| `benchmark_config_3080ti_large_scale_sub_div.json` | Joint scale × density: 1× subdivided sports car at large scale ranges |
| `benchmark_config_3080ti_range.json` | Range Variation: `range_max` from 5 m to 1000 m |
| `benchmark_config_3080ti_static.json` | Scene Size Robustness: static-only scenes (Suppl.) |
| `benchmark_config_3080ti_combine_robot_frames.json` | Ray Origin Co-location: distributed vs. co-located origins, Ω = 1–8 (Suppl.) |
| `benchmark_config_3080ti_best_sat_chaos.json` | Span Threshold Selection: (γ<sub>T</sub>, χ<sub>T</sub>) over 8 scale ranges, SWD (Suppl.; the paper reports the 32–128 subset for Vokselia Spawn + 10 dyn and Power Plant + 30 dyn) |
| `benchmark_config_negative_case_sat_sweep.json` | Extended span-threshold sweep: all 36 (γ<sub>T</sub>, χ<sub>T</sub>) combinations on the main suite, ND/OBD/SWD (Suppl.) |

`benchmark_config_3080ti_compact_cost.json` (the `_compact` main-suite tests; Power Plant, 30 dyn, Ω = 8) is included for completeness but is not reported in the paper.

Notes on what the presets run:
- "hybrid" in `best_hybrid*` and `configs/tests/3080ti mobile/fast_hybrid_find/` refers to the periodic-rebuild (`prN`) sweep, not the hybrid backends.
- `best_hybrid_chaos` runs OptiX only, for 10 frames (`bvh_sweep_*_chaos.json`); `best_hybrid` runs OptiX and Embree for 50 frames.
- The `main_suite_hybrid` presets run only the hybrid backends (`backend: hybrid`); the Embree/OptiX baselines they are compared against come from the matching `main_suite` preset.
- The span-threshold test files encode both thresholds in their names, in opposite orders: `fast_sat_find/rebuild_chaos_sat_sweep_<sweep>x<channel>.json` versus `negative_case_sat_sweep/nd_obd_swd_sat_sweep_<channel>x<sweep>.json` (`sweep` = `sat_max_sweep_diff`, `channel` = `sat_max_channel_diff`). Both presets set `bat_capacity` to `4500000`.

## Run

| Command | Action |
|---------|--------|
| `make run` | Run `benchmark_config.json` in benchmark mode: sets the performance governor, disables turbo boost (both via `sudo`), and pins `grca` to core 0 |
| `make run-csv-viewer <file>` | Vulkan CSV visualizer |
| `make run-csv-compare ...` | Compare two or more CSV files |

Command-line options (see [Backend aliases](docs/backends.md#backend-aliases) for group aliases):
```bash
./build/grca --backend grca-cuda                          # override the backend of every batch
./build/grca --backend all_t
./build/grca my_benchmark_config.json --active tests[1]/batches[0]
./build/grca --refit                                      # force bvh_rebuild_after_frames = -1 for all BVH backends
./build/grca --rebuild                                    # force bvh_rebuild_after_frames = 0 for all BVH backends
./build/grca --debug                                      # same as cuda.debug = true
```

---

## Orientation Convention

**R = R_y(yaw) × R_x(pitch) × R_z(roll)**

With all angles zero: forward = (0,0,1), up = (0,1,0), right = (1,0,0).

---

## Unity CPU Version

`cs/GRCA_CPU.cs` is a standalone C# version of GRCA-CPU. Its component class is `GrcaLidarCpu`, and it uses the Input System package.

1. Install [Unity Hub](https://docs.unity3d.com/hub/manual/InstallHub.html#install-hub-linux) and add the Input System package to the project
2. Copy `cs/GRCA_CPU.cs` into the Assets folder and rename it to `GrcaLidarCpu.cs` (Unity requires a component's file name to match its class name)
3. Hierarchy → right-click → `Create Empty`
4. Select the GameObject → Inspector → `Add Component` → `Grca Lidar Cpu`
5. Set the LiDAR parameters in the Inspector (`HNumSamples`, the angle limits, and the range limits default to 0)
6. Hierarchy → right-click → `3D Object` → `Cube`, place in front of the GameObject
7. Press Play:
   - `G` — run GRCA benchmark
   - `H` — toggle LiDAR hit points
   - Each `G` run writes its hits to `Assets/hits_grca.csv`, which [`csv_viewer`](docs/tools.md#csv_viewer) can display

---

## Documentation

- [Configuration](docs/configuration.md) — config file structure, every field, and the main-suite test files
- [Output](docs/output.md) — session folders, CSV columns, batch summary, and plots
- [Backends](docs/backends.md) — backend aliases and per-backend details and labels
- [Tools](docs/tools.md) — `compare_csv`, `csv_viewer`, the results uploader, and visualizer controls
