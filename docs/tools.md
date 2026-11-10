# Tools

Helper tools and visualizer controls.

Back to the [README](../README.md).

## Table of Contents

1. [Interactive controls (visualizer)](#interactive-controls-visualizer)
2. [`compare_csv`](#compare_csv)
3. [`csv_viewer`](#csv_viewer)
4. [`poll_upload_results.py`](#poll_upload_resultspy)

## Interactive controls (visualizer)

`grca` (Hit Viewer) and `csv_viewer` use an FPS free-roam camera. Keys marked *Hit Viewer only* are not handled by `csv_viewer`:

| Input | Action |
|-------|--------|
| **Ctrl** | Toggle FPS mode on/off (locks/unlocks cursor) |
| **Mouse move** | Look around *(FPS mode)* |
| **W / S** | Move forward / backward *(FPS mode)* |
| **A / D** | Move left / right *(FPS mode)* |
| **E / Q** | Move up / down *(FPS mode)* |
| **Shift** | Fast movement *(FPS mode)* |
| **Scroll wheel** | Adjust movement speed |
| **T** | Snap camera to next lidar (cycles each press; lidar-follow mode) *(Hit Viewer only)* |
| **P** | Exit lidar-follow mode and return to free-roam *(Hit Viewer only)* |
| **M** | Toggle all meshes (static + dynamic) on/off *(Hit Viewer only)* |
| **B** | Toggle dynamic meshes on/off (overrides `M` for dynamic only) *(Hit Viewer only)* |
| **C** | Toggle lidar cones on/off *(Hit Viewer only)* |
| **F** | Focus camera on hit-point bounding box (keeps orientation) |
| **O** | Write hit CSV for current frame *(Hit Viewer only)* |
| **Esc** | Quit |


## `compare_csv`

Compares two or more CSV files row-by-row with per-column statistics. Columns are matched by position and reported as `col0`, `col1`, …; there is no header detection, so the first row is compared like any other row. Used automatically in the batch summary (`--simple` mode) and available standalone:

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

## `csv_viewer`

Vulkan-based 3D visualizer for hit-point CSVs with the columns `dx,dy,dz,dist,hx,hy,hz` (e.g. the Unity version's `Assets/hits_grca.csv`). See [Interactive controls](#interactive-controls-visualizer).

```bash
make run-csv-viewer hits.csv
./build/csv_viewer hits.csv
```

## `poll_upload_results.py`

Watches `benchmark/` for new `.zip` files produced after each benchmark run, copies them into `results/<branch> - <short-commit>/`, and commits and pushes to the remote. Runs until stopped with Ctrl+C. Pinned to CPU core 4 so it does not interfere with the benchmark.

```bash
python3 poll_upload_results.py --interval 100 --repo .
```

| Option | Default | Description |
|--------|---------|-------------|
| `--interval` | `10` | Poll interval in seconds |
| `--repo` | `.` | Path to the GRCA repo root |

Start it in a separate terminal before (or during) a benchmark run. It will ignore any `.zip` files that already exist when it starts, and only commit new ones as they appear. After each commit it runs `git stash`, `git pull --rebase`, and `git stash pop` before `git push`, so it touches the working tree of the repo it runs in.
