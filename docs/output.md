# Output

What a benchmark run writes: session folders, CSV columns, the batch summary, and plots.

Back to the [README](../README.md).

## Table of Contents

1. [Directory structure](#directory-structure)
2. [PNG files](#png-files)
3. [`results_runN_batchM.csv` columns](#results_runn_batchmcsv-columns)
4. [Batch summary (stderr)](#batch-summary-stderr)
5. [Plotting](#plotting)
6. [Performance CSV (`perf_*.csv`)](#performance-csv-perf_csv)
7. [Hit Distance CSV (`hit_dist_*.csv`)](#hit-distance-csv-hit_dist_csv)
8. [Pose CSV (`pose_*.csv`)](#pose-csv-pose_csv)
9. [Hit View CSV (`hit_view_*.csv`)](#hit-view-csv-hit_view_csv)


## Directory structure

Each (scene × dynamic_mesh_count × robot_count) combination produces one session folder. All session folders share the **benchmark start time** in their name so they sort together. The actual per-session start time is recorded inside `session_meta.csv`. After all sessions complete, they are all archived into a single zip with the `zip` command (`sudo apt install zip` if missing).

```
benchmark/
├── YYYYMMDD_HHMMSS-<user>-<host>-[scene0]VokseliaSpawn-[dyn1]10_dyn_mesh-[robot0]max_rays_2_robot/   ← session folder
│   ├── session_meta.csv                ← hardware info, actual session start time, config files used, scene/ray counts
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
│   │   │   ├── results_run0_batch0.png ← per-batch bar chart (+ *_anon.png copy)
│   │   │   ├── runtime_info_grca-cuda.txt    ← per-backend run info (VRAM, BAT/RTIC counters, …)
│   │   │   ├── perf_GRCA_cuda_<ts>.csv       ← perf_<label>_<YYYYMMDD_HHMMSS>.csv
│   │   │   └── hit_view_grca-cuda_GRCA_cuda_<ts>.csv   ← hit_view_<backend>_<label>_<ts>.csv
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

## PNG files

Generated only when the top-level `skip_plot` is `false` (see [Plotting](#plotting)).

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

## `results_runN_batchM.csv` columns

Columns follow the same order as the per-frame perf CSV.

| Column | Description |
|--------|-------------|
| `run`, `batch` | Run and batch indices (`-1` when the plotter aggregates rows in memory) |
| `backend` | Detailed backend label (e.g. `GRCA_cpu_vec8`, `GRCA_cuda`, `Optix_rb`, `Optix_crti_rb`, `Optix_cmp_rf`, `Embree_vec1_pr10`, `TinyBVH_cpu_rb`, `TinyBVH_gpu_pr4`, `Hybrid_cpu_Embree_vec16_rf_GRCA_vec16`, `Hybrid_gpu_Optix_rf_GRCA_cuda`) |
| `api_label` | Simple API name used in session charts (`GRCA_cpu`, `GRCA_cuda`, `Optix`, `Optix_crti`, `Embree`, `TinyBVH_cpu`, `TinyBVH_gpu`, `Hybrid_cpu`, `Hybrid_gpu`) |
| `perf_frames` | Frames recorded (summed when the plotter aggregates rows) |
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

## Batch summary (stderr)

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
  hit_dist_GRCA_cuda   vs  hit_dist_Optix_rb          MATCH     col1=97.40%(max=9.80e+02)
  1/1 pairs match  (tol=0.001  min-match=95%)

  Pose Check  (tol=0.001  min-match=95%)
  pose_GRCA_cuda   vs  pose_Optix_rb            MATCH     all=100.00%
  1/1 pairs match  (tol=0.001  min-match=95%)
  --------------------------------------------------------
```

Hit distance mismatches near `range_max` are expected — different BVH algorithms disagree on grazing-angle rays at geometry boundaries. `MATCH` at 95% threshold means backends agree well enough. Columns are reported by position (`col1` is the `dist` column), and the raw CSVs are deleted after the check.

## Plotting

When the top-level `skip_plot` is `false`, the [PNG files](#png-files) are generated automatically after each batch, run, and session. Each PNG is saved twice: a normal version and a `*_anon.png` version with `GRCA` → `[I]LA` in all labels (for blind review).

The plotter reads `results_runN_batchM.csv` files from `run_N/batch_M/` — no intermediate aggregate CSVs are written.

You can also run the plotter manually, whatever `skip_plot` is set to:

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

See [Python prerequisites](../README.md#python-for-plots) for installation.

## Performance CSV (`perf_*.csv`)

Written per backend per batch. The first line is a `#`-prefixed header with `lidar_sim_rate`; then a column-header row; then one data row per frame. After all data rows, per-column averages are appended as `#`-prefixed footer comments (these are stripped by the plotting scripts).

| Column | What it measures |
|--------|-----------------|
| `frame` | Frame index |
| `ms_lidar_sim_cpu` | Total lidar simulation wall time |
| `ms_dynamic_upload` | CPU time writing updated vertex positions into the backend buffer |
| `ms_bvh_rebuild_cpu` | Full BVH rebuild (CPU wall time) |
| `ms_bvh_refit_cpu` | Incremental BVH refit (CPU wall time) |
| `ms_trace_cpu` | Ray-trace CPU time |
| `ms_sync_readback` | `cudaStreamSynchronize` stall before readback |
| `ms_readback_cpu` | Device→host copy of hit results |
| `ms_fillhits` | Unpacking raw hit buffer into hit lists |
| `ms_lidar_sim_gpu` | Lidar simulation GPU time |
| `ms_bvh_rebuild_gpu` | Full BVH rebuild GPU time (OptiX; Embree and TinyBVH use the `_cpu` fields) |
| `ms_bvh_refit_gpu` | Incremental BVH refit |
| `ms_init_cpu` | Backend initialisation time (one-off) |
| `sim_frame_rate` | Simulated frames per second |
| `real_time_factor` | `sim_frame_rate / lidar_sim_rate` (>1 means faster than real-time) |
| `ms_total_cpu` | Full frame wall time including mesh prep |
| `mb_readback` | MB transferred in readback |
| `hit_count` | Rays that hit geometry |

GPU trace time is not written per frame; its average is `avg_trace_gpu_ms` in `results_runN_batchM.csv`.

## Hit Distance CSV (`hit_dist_*.csv`)

Written when `record_frames > -1`, the top-level `skip_validation` is `false`, and the batch runs at least two backends (it exists only for the cross-backend check and is deleted afterwards). To keep file sizes manageable, only `record_frames / 10` frames are recorded — the set of frames is chosen randomly at run start with a fixed seed and is **shared across all batches**, so every backend samples the same frames. One row per ray per sampled frame:

```
frame,dist
0,14.32
0,1000.0   ← miss (range_max)
```

## Pose CSV (`pose_*.csv`)

Written under the same conditions as the hit-distance CSV. World-space pose (position + orientation) for all robots, dynamic meshes, and static meshes at every recorded frame.

## Hit View CSV (`hit_view_*.csv`)

Written interactively (**O** key in visualizer) or automatically via `auto_hit_view_record_frame`. Columns: `ro_id,li_id,dx,dy,dz,dist,hx,hy,hz` (robot and lidar index, ray direction, hit distance, world-space hit position). `csv_viewer` reads the first seven columns as `dx,dy,dz,dist,hx,hy,hz`, so it does not display these files correctly as written.
