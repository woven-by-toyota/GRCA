#!/usr/bin/env python3
"""
Plot benchmark results from results.csv files.

Usage:
    # Single batch directory
    python3 benchmark/plot_results.py benchmark/SESSION/run_0/batch_0

    # Session directory — finds all results.csv files recursively
    python3 benchmark/plot_results.py benchmark/SESSION

PNGs are written alongside each results.csv that was found.
Requires: matplotlib, numpy  (pip install matplotlib numpy)
"""

import re
import sys
import csv
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import matplotlib.text as _mtext
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required.")
    print("       pip install matplotlib numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dual-save helper  (normal + anonymous "[I]LA" version)
# ---------------------------------------------------------------------------

def _anon(s):
    return s.replace("GRCA", "[I]LA")


def _save_dual(fig, path, **kwargs):
    """Save fig as *path* (normal) and *path_stem*_anon.png ([I]LA labels).
    Backend/api labels (GRCA_cuda, GRCA_cpu*, …) are replaced with [I]LA_*."""
    fig.savefig(path, **kwargs)
    fig.canvas.draw()   # force-populate tick Text objects

    restores = []   # list of zero-arg callables that undo each patch

    for ax in fig.get_axes():
        # title / axis labels — set_text() persists through savefig
        for obj in (ax.title, ax.xaxis.label, ax.yaxis.label):
            orig = obj.get_text()
            if "GRCA" in orig:
                obj.set_text(_anon(orig))
                restores.append(lambda o=obj, v=orig: o.set_text(v))

        # x-tick labels — MUST go through set_xticklabels; direct set_text()
        # is overridden by matplotlib's FixedFormatter on the next draw/save.
        xtl_objs = ax.get_xticklabels()
        xtl_strs = [t.get_text() for t in xtl_objs]
        if any("GRCA" in s for s in xtl_strs):
            kw = dict(rotation=15, ha="right", fontsize=9)
            ax.set_xticklabels([_anon(s) for s in xtl_strs], **kw)
            restores.append(lambda a=ax, v=xtl_strs, k=kw: a.set_xticklabels(v, **k))

        # inline texts
        for obj in ax.texts:
            orig = obj.get_text()
            if "GRCA" in orig:
                obj.set_text(_anon(orig))
                restores.append(lambda o=obj, v=orig: o.set_text(v))

        # legend
        leg = ax.get_legend()
        if leg:
            for obj in leg.get_texts():
                orig = obj.get_text()
                if "GRCA" in orig:
                    obj.set_text(_anon(orig))
                    restores.append(lambda o=obj, v=orig: o.set_text(v))

    # figure-level texts (suptitle etc.)
    for obj in fig.texts:
        orig = obj.get_text()
        if "GRCA" in orig:
            obj.set_text(_anon(orig))
            restores.append(lambda o=obj, v=orig: o.set_text(v))

    base, ext = os.path.splitext(path)
    anon_path = base + "_anon" + ext
    fig.savefig(anon_path, **kwargs)

    for fn in restores:
        fn()

    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}  +  {os.path.basename(anon_path)}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                try:
                    typed[k] = float(v)
                except (ValueError, TypeError):
                    typed[k] = v
            rows.append(typed)
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_PCT_FORMATTER = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")

# Fallback palette for backends not matched by name
_COLORS_FALLBACK = ["#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

# Canonical display order: optix_crti → optix → grca-cuda → hybrid-gpu → embree → grca-cpu → hybrid-cpu
_BACKEND_ORDER = ["optix_crti", "optix", "grca_cuda", "hybrid_gpu", "embree", "grca_cpu", "hybrid_cpu"]

def _backend_sort_key(name):
    lower = (name or "").lower()
    for i, key in enumerate(_BACKEND_ORDER):
        if key in lower:
            return i
    return len(_BACKEND_ORDER)

# Name-based color map: case-insensitive substring match, first hit wins.
# optix_crti must come before optix so it doesn't fall through to the generic green.
_BACKEND_COLOR_MAP = [
    ("optix_crti", "#2ECC71"),   # bright emerald — OptiX software IS (no RT-core triangle HW)
    ("optix",      "#55A868"),   # muted green    — OptiX hardware IS
    ("grca_cuda",   "#DD8452"),   # orangish
    ("hybrid_gpu", "#E8A838"),   # amber          — must precede "optix"/"embree" substrings
    ("embree",     "#4C72B0"),   # bluish
    ("grca_cpu",    "#C44E52"),   # reddish
    ("hybrid_cpu", "#845D91"),   # purple         — must precede "embree" substring
]
_fallback_idx = 0

def _backend_color(name):
    """Return a consistent color for a backend name."""
    global _fallback_idx
    lower = (name or "").lower()
    for key, color in _BACKEND_COLOR_MAP:
        if key in lower:
            return color
    # Assign from fallback palette and remember for this name
    color = _COLORS_FALLBACK[_fallback_idx % len(_COLORS_FALLBACK)]
    _fallback_idx += 1
    return color


def _setup_pct_ax(ax, lo, hi=120):
    """Apply standard percentage axis formatting: ylim, % tick labels, grid."""
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(_PCT_FORMATTER)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def correctness_chart(ax, checks, title):
    """Per-batch correctness bars (Hit Dist / Pose Match), one bar per check type."""
    active = [c for c in checks if c["pairs_total"] > 0]
    if not active:
        ax.set_visible(False)
        return

    labels = [c["label"] for c in active]
    values = [c["min_pct"] for c in active]
    crits  = [c["crit"]    for c in active]
    pairs  = [f"{c['pairs_matched']}/{c['pairs_total']}" for c in active]

    x      = np.arange(len(labels))
    colors = ["#2ca02c" if v >= cr else "#d62728" for v, cr in zip(values, crits)]
    bars   = ax.bar(x, values, color=colors, width=0.4, zorder=3)

    if crits:
        ax.axhline(crits[0], color="#333333", linewidth=1.5, linestyle="--", zorder=4)

    for bar, val, pair in zip(bars, values, pairs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.2f}%\n({pair} pairs)",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Match %", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    _setup_pct_ax(ax, 0)


def _annotate_bars(ax, bars, vals, v_min=None, v_max=None,
                   linear=False, suffix="",
                   labels_above=None):
    """Shared bar annotation: error bars + value labels. Used by all chart types."""
    centers = [b.get_x() + b.get_width() / 2 for b in bars]
    if v_min is not None and v_max is not None:
        # Always show at least a small whisker (4% of bar) even when min == max == avg.
        MIN_WHISKER = 0.04
        yerr_low  = [max(a * MIN_WHISKER, max(0, a - mn)) for a, mn in zip(vals, v_min)]
        yerr_high = [max(a * MIN_WHISKER, max(0, mx - a)) for a, mx in zip(vals, v_max)]
        # Clip whisker endpoints to the current y-axis limits so they never go off the graph.
        y_bot, y_top = ax.get_ylim()
        yerr_low  = [min(el, max(0.0, val - y_bot)) for el, val in zip(yerr_low,  vals)]
        yerr_high = [min(eh, max(0.0, y_top - val)) for eh, val in zip(yerr_high, vals)]
        ax.errorbar(centers, vals, yerr=[yerr_low, yerr_high],
                    fmt="none", ecolor="#333333", elinewidth=1.5, capsize=5, zorder=5)
        # Place min/max labels relative to the actual whisker endpoints so they
        # never overlap the cap regardless of MIN_WHISKER padding.
        whisker_bot = [val - el for val, el in zip(vals, yerr_low)]
        whisker_top = [val + eh for val, eh in zip(vals, yerr_high)]
        for cx, mn, mx, val, wb, wt in zip(centers, v_min, v_max, vals, whisker_bot, whisker_top):
            if mn > 0:
                ax.text(cx, wb * 0.90, f"{mn:.2f}{suffix}", ha="center", va="top",
                        fontsize=7, fontstretch="condensed", color="black", style="italic")
            if mx > 0:
                ax.text(cx, wt * 1.10, f"{mx:.2f}{suffix}", ha="center", va="bottom",
                        fontsize=7, fontstretch="condensed", color="black", style="italic")
            if val > 0:
                ax.text(cx, wt * 1.60 if wt > 0 else val * 1.60, f"{val:.2f}{suffix}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        for cx, val in zip(centers, vals):
            if val > 0 or linear:
                y = val + 0.4 if linear else val * 1.10
                ax.text(cx, y, f"{val:.2f}{suffix}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    if labels_above:
        for i, (cx, val, lbl) in enumerate(zip(centers, vals, labels_above)):
            if lbl and val > 0:
                y = v_max[i] * 1.80 if (v_max and v_max[i] > 0) else val * 1.60
                ax.text(cx, y, lbl, ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333333")


def bar_chart(ax, labels, values, title, ylabel="ms", color_list=None,
              val_min=None, val_max=None,
              labels_above=None):
    keep = [i for i, v in enumerate(values) if v > 0]
    if not keep:
        ax.set_visible(False)
        return
    la     = [labels_above[i] for i in keep] if labels_above is not None else None
    labels = [labels[i] for i in keep]
    values = [values[i] for i in keep]
    v_min  = [val_min[i] for i in keep] if val_min is not None else None
    v_max  = [val_max[i] for i in keep] if val_max is not None else None
    x      = np.arange(len(labels))
    colors = [color_list[i] for i in keep] if color_list else [_backend_color(l) for l in labels]
    bars   = ax.bar(x, values, color=colors, width=0.55, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"{ylabel}  [log scale]", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    top_vals = [v for v in list(values) + (v_max or []) if v > 0]
    bot_vals = [v for v in (v_min or []) + list(values) if v > 0]
    if top_vals:
        ax.set_ylim(bottom=min(bot_vals) * 0.3, top=max(top_vals) * 5)
    # Annotate after ylim is fixed so whisker clipping in _annotate_bars is correct.
    _annotate_bars(ax, bars, values, v_min, v_max, labels_above=la)


# ---------------------------------------------------------------------------
# Per-(run, batch) chart
# ---------------------------------------------------------------------------

def get(rows, col, default=0.0):
    return [r.get(col, default) or default for r in rows]


def _title_frags(rows):
    """Return (frames_part, simrate_str) title fragments from a row set."""
    frame_counts = sorted({int(r.get("perf_frames", 0) or 0) for r in rows} - {0})
    frames_part  = f"  —  {'/'.join(str(f) for f in frame_counts)} frames" if frame_counts else ""
    simrates = set()
    for r in rows:
        v = r.get("lidar_sim_rate")
        if v is not None:
            try:
                simrates.add(float(v))
            except Exception:
                pass
    simrate_str = f"  —  simrate: {sorted(simrates)[0]:.2f} Hz" if simrates else ""
    return frames_part, simrate_str


def plot_batch(rows, run_idx, batch_idx, out_dir, out_path=None, title=None, overview=False,
               extremes_data=None, extremes_label="batch", extremes_group_col="backend"):
    rows = sorted(rows, key=lambda r: _backend_sort_key(r.get("backend", "")))
    backends = [r["backend"] for r in rows]

    cpu_total       = get(rows, "avg_total_cpu_ms")
    cpu_lidar       = get(rows, "avg_lidar_sim_cpu_ms")
    cpu_lidar_min   = get(rows, "min_lidar_sim_cpu_ms")
    cpu_lidar_max   = get(rows, "max_lidar_sim_cpu_ms")
    gpu_lidar       = get(rows, "avg_lidar_sim_gpu_ms")
    gpu_lidar_min   = get(rows, "min_lidar_sim_gpu_ms")
    gpu_lidar_max   = get(rows, "max_lidar_sim_gpu_ms")
    trace_gpu       = get(rows, "avg_trace_gpu_ms")
    trace_cpu       = get(rows, "avg_trace_cpu_ms")
    init_cpu        = get(rows, "init_cpu_ms")
    avg_fps         = get(rows, "avg_fps")
    avg_rtf         = get(rows, "avg_rtf")
    dyn_upload      = get(rows, "avg_dynamic_upload_ms")
    bvh_rebuild_gpu = get(rows, "avg_bvh_rebuild_gpu_ms")
    bvh_refit_gpu   = get(rows, "avg_bvh_refit_gpu_ms")
    bvh_rebuild_cpu = get(rows, "avg_bvh_rebuild_cpu_ms")
    bvh_refit_cpu   = get(rows, "avg_bvh_refit_cpu_ms")
    readback        = get(rows, "avg_readback_ms")
    sync_readback   = get(rows, "avg_sync_readback_ms")
    fill_hits       = get(rows, "avg_fill_hits_ms")

    # Correctness stats (batch-level; same for all backends — read from row 0)
    r0            = rows[0]
    cmp_tol       = float(r0.get("cmp_tol")      or 0)
    cmp_min_match = float(r0.get("cmp_min_match") or 0)
    hd_total      = int(float(r0.get("hit_dist_pairs_total",   0) or 0))
    hd_matched    = int(float(r0.get("hit_dist_pairs_matched", 0) or 0))
    hd_min_pct    = float(r0.get("hit_dist_min_pct") or 0)
    pose_total    = int(float(r0.get("pose_pairs_total",   0) or 0))
    pose_matched  = int(float(r0.get("pose_pairs_matched", 0) or 0))
    pose_min_pct  = float(r0.get("pose_min_pct") or 0)

    cmp_checks = [
        {"label": "Hit Dist",   "min_pct": hd_min_pct,   "pairs_matched": hd_matched,  "pairs_total": hd_total,   "crit": cmp_min_match},
        {"label": "Pose Match", "min_pct": pose_min_pct, "pairs_matched": pose_matched, "pairs_total": pose_total, "crit": cmp_min_match},
    ]
    has_cmp     = any(c["pairs_total"] > 0 for c in cmp_checks)
    has_gpu     = any(v > 0 for v in gpu_lidar)
    has_bvh_gpu = any(v > 0 for v in bvh_rebuild_gpu + bvh_refit_gpu)

    # --- Row 1: lidar sim cpu + upload + BVH CPU ---
    _row1_candidates = [
        (cpu_lidar,       "Avg Lidar Sim CPU (ms, lower=better)", {"val_min": cpu_lidar_min, "val_max": cpu_lidar_max}),
        (bvh_rebuild_cpu, "Avg BVH Rebuild CPU (ms)"),
        (bvh_refit_cpu,   "Avg BVH Refit CPU (ms)"),
    ] if overview else [
        (cpu_lidar,       "Avg Lidar Sim CPU (ms, lower=better)", {"val_min": cpu_lidar_min, "val_max": cpu_lidar_max}),
        (dyn_upload,      "Avg Dynamic Upload (ms)"),
        (bvh_rebuild_cpu, "Avg BVH Rebuild CPU (ms)"),
        (bvh_refit_cpu,   "Avg BVH Refit CPU (ms)"),
    ]
    row1_charts = [c for c in _row1_candidates if any(v > 0 for v in c[0])]

    # --- Row 2: trace cpu + readback + fill hits ---
    _row2_candidates = [
        (trace_cpu,     "Avg Trace CPU (ms, lower=better)"),
        (sync_readback, "Avg Sync Readback (ms)"),
    ] if overview else [
        (trace_cpu,     "Avg Trace CPU (ms, lower=better)"),
        (sync_readback, "Avg Sync Readback (ms)"),
        (readback,      "Avg Readback (ms)"),
        (fill_hits,     "Avg Fill Hits (ms, lower=better)"),
    ]
    row2_charts = [c for c in _row2_candidates if any(v > 0 for v in c[0])]

    # --- Row 3: GPU metrics ---
    row3_charts = []
    if has_gpu:
        row3_charts.append((gpu_lidar, "Avg Lidar Sim GPU (ms, lower=better)", {"val_min": gpu_lidar_min, "val_max": gpu_lidar_max}))
    if has_bvh_gpu:
        for vals, lbl in [
            (bvh_rebuild_gpu, "Avg BVH Rebuild GPU (ms)"),
            (bvh_refit_gpu,   "Avg BVH Refit GPU (ms)"),
        ]:
            if any(v > 0 for v in vals):
                row3_charts.append((vals, lbl))
    if any(v > 0 for v in trace_gpu):
        row3_charts.append((trace_gpu, "Avg Trace GPU (ms, lower=better)"))

    # --- Row 4: init + fps + rtf + total + correctness ---
    tol_str = f"tol={cmp_tol:.4g}  min={cmp_min_match:.4g}%" if cmp_tol > 0 else f"min={cmp_min_match:.4g}%"
    row4_charts = [
        (avg_fps,   "Avg FPS (higher=better)"),
        (avg_rtf,   "Avg RTF (higher=better)"),
        (cpu_total, "Avg Total CPU (ms, lower=better)"),
    ] if overview else [
        (init_cpu,  "Init CPU (ms, lower=better)"),
        (avg_rtf,   "Avg RTF (higher=better)"),
        (cpu_total, "Avg Total CPU (ms, lower=better)"),
    ]
    if has_cmp:
        row4_charts.append(("correctness", cmp_checks, f"Correctness Check  ({tol_str})"))

    if overview and extremes_data and len(extremes_data) >= 2:
        # Per-backend: find fastest and slowest key (batch or run)
        # extremes_data: {key: [rows]}  — each row has backend / api_label
        be_key_map = {}   # backend_name → {key: row}
        for key, erows in extremes_data.items():
            for r in erows:
                be = r.get(extremes_group_col, "")
                if be:
                    be_key_map.setdefault(be, {})[key] = r

        pfx = extremes_label[0].upper()   # "B" for batch, "R" for run
        for extreme, lbl_suffix in [("fast", "fastest"), ("slow", "slowest")]:
            e_vals, e_min, e_max, e_colors, e_id_labels = [], [], [], [], []
            for be in backends:
                br = be_key_map.get(be, {})
                valid = {k: r for k, r in br.items()
                         if float(r.get("avg_lidar_sim_cpu_ms") or 0) > 0}
                if not valid:
                    e_vals.append(0); e_min.append(0); e_max.append(0)
                    e_colors.append(_backend_color(be)); e_id_labels.append("")
                    continue
                sorted_br = sorted(valid.items(),
                                   key=lambda kv: float(kv[1].get("avg_lidar_sim_cpu_ms") or 0))
                key, r = sorted_br[0] if extreme == "fast" else sorted_br[-1]
                e_vals.append(float(r.get("avg_lidar_sim_cpu_ms") or 0))
                e_min.append(float(r.get("min_lidar_sim_cpu_ms") or 0))
                e_max.append(float(r.get("max_lidar_sim_cpu_ms") or 0))
                e_colors.append(_backend_color(be))
                e_id_labels.append(f"{pfx}{key}")
            row2_charts.append((
                e_vals,
                f"{lbl_suffix.capitalize()} {extremes_label}  —  Avg Lidar Sim CPU (ms)",
                {"val_min": e_min, "val_max": e_max, "color_list": e_colors,
                 "labels_above": e_id_labels},
            ))

    rows_of_charts = [r for r in [row1_charts, row2_charts, row3_charts, row4_charts] if r]
    ncols = max(len(r) for r in rows_of_charts)
    nrows = len(rows_of_charts)

    fig, axes_grid = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4.5 * nrows),
                                  squeeze=False)
    frames_part, simrate_str = _title_frags(rows)
    full_title = title or f"Run {run_idx}  /  Batch {batch_idx}{frames_part}{simrate_str}  —  Performance Summary"
    fig.suptitle(full_title, fontsize=12, fontweight="bold", y=1.01)

    for ri, charts in enumerate(rows_of_charts):
        for ci, chart in enumerate(charts):
            if chart[0] == "correctness":
                correctness_chart(axes_grid[ri][ci], chart[1], chart[2])
            else:
                vals, lbl = chart[0], chart[1]
                kwargs = chart[2] if len(chart) > 2 else {}
                bar_chart(axes_grid[ri][ci], backends, vals, lbl, **kwargs)
        for ci in range(len(charts), ncols):
            axes_grid[ri][ci].set_visible(False)

    plt.tight_layout()
    fname = out_path or os.path.join(out_dir, f"results_run{run_idx}_batch{batch_idx}.png")
    _save_dual(plt.gcf(), fname, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Cross-batch CPU overview chart (one bar group per batch, stacked backends)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _compute_agg_rows(by_batch, group_col="backend"):
    """Average per-batch rows into one aggregate row per group_col value."""
    backends, seen = [], set()
    for brows in by_batch.values():
        for r in brows:
            be = r.get(group_col)
            if be and be not in seen:
                backends.append(be)
                seen.add(be)
    _SUM_COLS = {"perf_frames", "hit_dist_pairs_matched", "hit_dist_pairs_total",
                 "pose_pairs_matched", "pose_pairs_total"}
    agg = []
    for be in backends:
        all_be = [r for brows in by_batch.values() for r in brows if r.get(group_col) == be]
        if not all_be:
            continue
        row = dict(all_be[0])
        row["batch"] = -1
        for col in list(row.keys()):
            if col in ("backend", "api_label", "run", "batch"):
                continue
            vals = [float(r.get(col) or 0) for r in all_be]
            nz   = [v for v in vals if v > 0]
            if not nz:
                row[col] = 0.0
            elif col.startswith("min_") or col.endswith("_min_pct"):
                row[col] = min(nz)
            elif col.startswith("max_"):
                row[col] = max(nz)
            elif col in _SUM_COLS:
                row[col] = sum(vals)
            else:
                row[col] = sum(nz) / len(nz)
        agg.append(row)
    return agg


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <session_dir|results_runN_batchM.csv> [--run N]")
        sys.exit(1)

    root = sys.argv[1]

    # Optional --run N: only process that run index (generates results_runN.png, skips session)
    only_run = None
    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        if idx + 1 < len(sys.argv):
            only_run = int(sys.argv[idx + 1])

    # Optional --session-only: skip per-batch and per-run PNGs, only generate session overview
    session_only = "--session-only" in sys.argv

    # ---- Single batch CSV ----
    if os.path.isfile(root):
        m = re.match(r"results_run(\d+)_batch(\d+)\.csv$", os.path.basename(root))
        if not m:
            print(f"ERROR: unrecognised file {os.path.basename(root)}")
            sys.exit(1)
        rows = load_results(root)
        if rows:
            ri, bi = int(m.group(1)), int(m.group(2))
            plot_batch(rows, ri, bi, os.path.dirname(root))
        print("Done.")
        return

    # ---- Session directory ----
    if not os.path.isdir(root):
        print(f"ERROR: {root} is not a file or directory")
        sys.exit(1)

    # Recursively collect all results_runN_batchM.csv files
    # CSVs live at session/run_N/batch_M/ — dirpath is already the right output dir
    runs = defaultdict(dict)   # run_idx → {batch_idx: [rows]}
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            m = re.match(r"results_run(\d+)_batch(\d+)\.csv$", fname)
            if not m:
                continue
            ri, bi   = int(m.group(1)), int(m.group(2))
            csv_path = os.path.join(dirpath, fname)
            rows     = load_results(csv_path)
            if not rows:
                continue
            if not session_only and (only_run is None or ri == only_run):
                print(f"Plotting {csv_path}")
                plot_batch(rows, ri, bi, dirpath)
            runs[ri][bi] = rows

    if not runs:
        print(f"ERROR: no results_runN_batchM.csv files found in {root}")
        sys.exit(1)

    # Per-run chart: average across all batches in that run (only if 2+ batches)
    for ri in sorted(runs):
        if session_only:
            continue
        if only_run is not None and ri != only_run:
            continue
        if len(runs[ri]) < 2:
            continue
        agg = _compute_agg_rows(runs[ri])
        if agg:
            sample_rows = next(iter(runs[ri].values()))
            frames_part, _ = _title_frags(sample_rows)
            run_dir = os.path.join(root, f"run_{ri}")
            os.makedirs(run_dir, exist_ok=True)
            plot_batch(agg, ri, -1, run_dir,
                       out_path=os.path.join(run_dir, f"results_run{ri}.png"),
                       title=f"Run {ri}{frames_part}  —  Cross-Batch Overview",
                       overview=True,
                       extremes_data=runs[ri],
                       extremes_label="batch")

    # Session chart: average of per-run overviews, grouped by api_label
    if only_run is not None:
        print("Done.")
        return
    if len(runs) >= 2:
        run_overviews = {ri: _compute_agg_rows(by_batch, group_col="api_label")
                         for ri, by_batch in runs.items()}
        run_overviews = {ri: rows for ri, rows in run_overviews.items() if rows}
        if len(run_overviews) >= 2:
            session_agg = _compute_agg_rows(run_overviews, group_col="api_label")
            for r in session_agg:
                r["backend"] = r.get("api_label", r["backend"])
            if session_agg:
                plot_batch(session_agg, -1, -1, root,
                           out_path=os.path.join(root, "results_session.png"),
                           title="Cross-Run Overview",
                           overview=True,
                           extremes_data=run_overviews,
                           extremes_label="run",
                           extremes_group_col="api_label")

    print("Done.")


if __name__ == "__main__":
    main()
