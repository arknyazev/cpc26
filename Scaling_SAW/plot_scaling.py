#!/usr/bin/env python
"""
Figures for the SAW-perturbed tracing scaling study.

Reads Scaling_SAW/runs/memo.json for summary data, and loads CPU/GPU results
from .npz files for loss-fraction-vs-time curves.

Figures produced (saved as PDF + PNG in Scaling_SAW/figures/):
  1. walltime_vs_nparticles.pdf   – walltime vs N_particles, CPU vs GPU per SAW file
  2. lossfrac_vs_nparticles.pdf   – loss fraction vs N_particles
  3. walltime_vs_tol.pdf          – walltime vs tolerance
  4. lossfrac_vs_tol.pdf          – loss fraction vs tolerance
  5. walltime_vs_resolution.pdf   – walltime vs grid resolution (when multiple resolutions)
  5b. lossfrac_vs_resolution.pdf  – loss fraction vs grid resolution
  6. cpu_gpu_crossval.pdf         – CPU vs GPU loss fraction scatter (when both available)
  7. loss_vs_time_<saw>.pdf       – loss fraction vs time curves
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
try:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["CMU Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })
    fig_test, ax_test = plt.subplots()
    ax_test.set_title(r"$\alpha$")
    plt.close(fig_test)
except Exception:
    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCALING_SAW_DIR = Path(__file__).parent
MEMO_FILE       = SCALING_SAW_DIR / "runs" / "memo.json"
FIG_DIR         = SCALING_SAW_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(MEMO_FILE) as f:
    memo = json.load(f)

runs = {k: v for k, v in memo.items() if v["status"] == "finished"}
print(f"Plotting {len(runs)} finished runs.")

# ---------------------------------------------------------------------------
# SAW label and color helpers
# ---------------------------------------------------------------------------
def saw_label(stem: str) -> str:
    """Extract a short amplitude label from the SAW file stem."""
    if stem == "saw_zero":
        return r"$\delta B = 0$"
    # stem like: QH_10harmonics_scale0_000464159
    # parse the scale value: everything after "scale"
    try:
        scale_str = stem.split("scale")[1]          # "0_000464159"
        scale_val = float(scale_str.replace("_", "."))
        return rf"$\delta B/B_0 = {scale_val:.2e}$"
    except (IndexError, ValueError):
        return stem

# Assign colors to SAW files in sorted order
def _make_saw_colors(all_saws):
    cmap = plt.cm.plasma
    if len(all_saws) <= 1:
        return {s: "#1f77b4" for s in all_saws}
    return {s: cmap(i / (len(all_saws) - 1)) for i, s in enumerate(sorted(all_saws))}

all_saws = sorted({v["saw_file"] for v in runs.values()})
SAW_COLORS = _make_saw_colors(all_saws)

MODE_STYLE = {
    "cpu": dict(linestyle="-",  marker="o"),
    "gpu": dict(linestyle="--", marker="s"),
}
MODE_LABEL = {"cpu": "CPU", "gpu": "GPU"}

def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Data collector
# ---------------------------------------------------------------------------
def collect(*, vary_key, fixed: dict) -> dict:
    """
    Group finished runs by (saw_file, mode), returning sorted arrays of
    (vary_key values, walltime, loss_fraction).
    """
    groups: dict[tuple, list] = {}
    for entry in runs.values():
        if not all(entry.get(k) == v for k, v in fixed.items()):
            continue
        if entry["runtime"] is None or entry["loss_fraction"] is None:
            continue
        key = (entry["saw_file"], entry.get("mode", "cpu"))
        groups.setdefault(key, []).append(
            (entry[vary_key], entry["runtime"], entry["loss_fraction"])
        )
    result = {}
    for key, pts in groups.items():
        pts.sort()
        xs, rts, lfs = zip(*pts)
        result[key] = (np.array(xs), np.array(rts), np.array(lfs))
    return result

# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------
all_tols = sorted({v["tol"] for v in runs.values()})
all_res  = sorted({v["resolution"] for v in runs.values()})
all_nP   = sorted({v["nParticles"] for v in runs.values()})
ref_tol  = all_tols[0]      # finest tol
ref_res  = all_res[-1]      # largest resolution
ref_nP   = all_nP[-1]       # largest nParticles

# ---------------------------------------------------------------------------
# Figure 1: walltime vs nParticles
# ---------------------------------------------------------------------------
data_nP = collect(vary_key="nParticles", fixed={"resolution": ref_res, "tol": ref_tol})

if data_nP:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (saw, mode), (xs, rts, lfs) in sorted(data_nP.items()):
        ax.plot(xs, rts, color=SAW_COLORS[saw],
                label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                **MODE_STYLE[mode])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_\mathrm{particles}$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(rf"Wall time vs $N_\mathrm{{particles}}$ (res={ref_res}, tol={ref_tol:.0e})")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "walltime_vs_nparticles")

# ---------------------------------------------------------------------------
# Figure 2: loss fraction vs nParticles
# ---------------------------------------------------------------------------
if data_nP:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (saw, mode), (xs, rts, lfs) in sorted(data_nP.items()):
        ax.plot(xs, lfs, color=SAW_COLORS[saw],
                label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                **MODE_STYLE[mode])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(r"$N_\mathrm{particles}$")
    ax.set_ylabel(r"Loss fraction at $t_\mathrm{max}$")
    ax.set_title(r"Loss fraction convergence with $N_\mathrm{particles}$")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "lossfrac_vs_nparticles")

# ---------------------------------------------------------------------------
# Figure 3: walltime vs tolerance
# ---------------------------------------------------------------------------
data_tol = collect(vary_key="tol", fixed={"nParticles": ref_nP, "resolution": ref_res})

if data_tol:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (saw, mode), (xs, rts, lfs) in sorted(data_tol.items()):
        ax.plot(xs, rts, color=SAW_COLORS[saw],
                label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                **MODE_STYLE[mode])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(rf"Wall time vs tolerance ($N={ref_nP}$, res={ref_res})")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "walltime_vs_tol")

# ---------------------------------------------------------------------------
# Figure 4: loss fraction vs tolerance
# ---------------------------------------------------------------------------
if data_tol:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (saw, mode), (xs, rts, lfs) in sorted(data_tol.items()):
        ax.plot(xs, lfs, color=SAW_COLORS[saw],
                label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                **MODE_STYLE[mode])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Tolerance")
    ax.set_ylabel(r"Loss fraction at $t_\mathrm{max}$")
    ax.set_title(rf"Loss fraction vs tolerance ($N={ref_nP}$, res={ref_res})")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "lossfrac_vs_tol")

# ---------------------------------------------------------------------------
# Figure 5: walltime + loss fraction vs resolution (if multiple resolutions)
# ---------------------------------------------------------------------------
if len(all_res) > 1:
    data_res = collect(vary_key="resolution", fixed={"nParticles": ref_nP, "tol": ref_tol})
    if data_res:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for (saw, mode), (xs, rts, lfs) in sorted(data_res.items()):
            ax.plot(xs, rts, color=SAW_COLORS[saw],
                    label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                    **MODE_STYLE[mode])
        ax.set_yscale("log")
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(rf"Wall time vs grid resolution ($N={ref_nP}$, tol={ref_tol:.0e})")
        ax.legend(fontsize=8, framealpha=0.85)
        fig.tight_layout()
        _save(fig, "walltime_vs_resolution")

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for (saw, mode), (xs, rts, lfs) in sorted(data_res.items()):
            ax.plot(xs, lfs, color=SAW_COLORS[saw],
                    label=f"{saw_label(saw)} {MODE_LABEL[mode]}",
                    **MODE_STYLE[mode])
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel(r"Loss fraction at $t_\mathrm{max}$")
        ax.set_title(rf"Loss fraction vs grid resolution ($N={ref_nP}$, tol={ref_tol:.0e})")
        ax.legend(fontsize=8, framealpha=0.85)
        fig.tight_layout()
        _save(fig, "lossfrac_vs_resolution")

# ---------------------------------------------------------------------------
# Figure 6: CPU vs GPU cross-validation scatter (skipped if no GPU data)
# ---------------------------------------------------------------------------
cpu_runs = {k: v for k, v in runs.items() if v.get("mode", "cpu") == "cpu"}
gpu_runs = {k: v for k, v in runs.items() if v.get("mode", "cpu") == "gpu"}

def run_params(v):
    return (v["saw_file"], v["nParticles"], v["resolution"], v["tol"])

cpu_by_params = {run_params(v): v for v in cpu_runs.values()}
gpu_by_params = {run_params(v): v for v in gpu_runs.values()}
common_params = set(cpu_by_params) & set(gpu_by_params)

if common_params:
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    for params in sorted(common_params):
        saw, nP, res, tol = params
        cpu_lf = cpu_by_params[params]["loss_fraction"]
        gpu_lf = gpu_by_params[params]["loss_fraction"]
        if cpu_lf is None or gpu_lf is None:
            continue
        ax.scatter(gpu_lf, cpu_lf, color=SAW_COLORS[saw], s=20, zorder=3)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GPU loss fraction")
    ax.set_ylabel("CPU loss fraction")
    ax.set_title("CPU vs GPU cross-validation")

    handles = [matplotlib.patches.Patch(color=SAW_COLORS[s], label=saw_label(s))
               for s in all_saws if any(p[0] == s for p in common_params)]
    handles.append(matplotlib.lines.Line2D([], [], color="k", linestyle="--", label="Exact agreement"))
    ax.legend(handles=handles, fontsize=8, framealpha=0.85)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "cpu_gpu_crossval")
else:
    print("No CPU+GPU pairs yet — skipping cross-validation plot.")

# ---------------------------------------------------------------------------
# Figure 6: loss fraction vs time (one fig per SAW file, curves per tol/mode)
# ---------------------------------------------------------------------------
for saw in all_saws:
    saw_cpu_runs = [(k, v) for k, v in cpu_runs.items()
                    if v["saw_file"] == saw and v["nParticles"] == ref_nP
                    and v["resolution"] == ref_res]
    saw_gpu_runs = [(k, v) for k, v in gpu_runs.items()
                    if v["saw_file"] == saw and v["nParticles"] == ref_nP
                    and v["resolution"] == ref_res]

    all_mode_runs = [("cpu", saw_cpu_runs), ("gpu", saw_gpu_runs)]
    has_data = any(runs_list for _, runs_list in all_mode_runs)
    if not has_data:
        continue

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    tmax_val = None
    for mode, mode_runs in all_mode_runs:
        for run_key, entry in sorted(mode_runs, key=lambda x: x[1]["tol"]):
            results_file = SCALING_SAW_DIR / run_key / "inputs_results.npz"
            if not results_file.exists():
                continue
            d = np.load(results_file)
            if "times" not in d or "loss_frac" not in d:
                continue
            times = d["times"]
            loss_frac_t = d["loss_frac"]
            tol = entry["tol"]
            tmax_val = entry["tmax"]
            ax.loglog(times, loss_frac_t,
                      label=rf"tol={tol:.0e} {MODE_LABEL[mode]}",
                      **MODE_STYLE[mode])

    if tmax_val is not None:
        ax.set_xlim([1e-5, tmax_val])
    ax.set_ylim([1e-3, 1])
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Loss fraction")
    ax.set_title(rf"{saw_label(saw)} — loss fraction vs time ($N={ref_nP}$, res={ref_res})")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, f"loss_vs_time_{saw}")

print(f"Figures saved to {FIG_DIR}/")
