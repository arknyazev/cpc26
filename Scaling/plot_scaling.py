#!/usr/bin/env python
"""
Figures for the CPU/GPU scaling and cross-validation study.

Reads Scaling/runs/memo.json for summary data, and plot CPU results from .npz files
directly for the loss-fraction-vs-time curves.

Figures produced (saved as PDF + PNG in Scaling/figures/):
  1. walltime_vs_nparticles.pdf   – walltime vs N_particles per device, CPU vs GPU
  2. lossfrac_vs_nparticles.pdf   – loss fraction vs N_particles per device, CPU vs GPU
  3. walltime_vs_tol.pdf          – walltime vs tolerance per device, CPU vs GPU
  4. lossfrac_vs_tol.pdf          – loss fraction vs tolerance per device, CPU vs GPU
  5. walltime_vs_resolution.pdf   – walltime vs grid resolution (when data available)
  6. cpu_gpu_crossval.pdf         – CPU vs GPU loss fraction scatter (cross-validation)
  7. loss_vs_time_<device>.pdf    – loss fraction vs time from CPU results files
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
    # Verify LaTeX is available
    fig_test, ax_test = plt.subplots()
    ax_test.set_title(r"$\alpha$")
    plt.close(fig_test)
except Exception:
    # Fallback: no LaTeX
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
# Paths and constants
# ---------------------------------------------------------------------------
SCALING_DIR = Path(__file__).parent
MEMO_FILE   = SCALING_DIR / "runs" / "memo.json"
FIG_DIR     = SCALING_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(MEMO_FILE) as f:
    memo = json.load(f)

# Keep only finished runs
runs = {k: v for k, v in memo.items() if v["status"] == "finished"}
print(f"Plotting {len(runs)} finished runs.")

# ---------------------------------------------------------------------------
# Style maps
# ---------------------------------------------------------------------------
# Short device labels derived from the boozmn stem
_DEVICE_LABELS = {
    "boozmn_HSX_QHS_vacuum_ns201_aScaling": "HSX",
    "boozmn_ncsx_c09r00_fixed_aScaling":    "NCSX",
    "boozmn_new_QA_aScaling":               "QA",
    "boozmn_new_QH_aScaling":               "QH",
}
def device_label(stem: str) -> str:
    return _DEVICE_LABELS.get(stem, stem)

DEVICE_COLORS = {
    "boozmn_HSX_QHS_vacuum_ns201_aScaling": "#1f77b4",   # blue
    "boozmn_ncsx_c09r00_fixed_aScaling":    "#ff7f0e",   # orange
    "boozmn_new_QA_aScaling":               "#2ca02c",   # green
    "boozmn_new_QH_aScaling":               "#d62728",   # red
}
MODE_STYLE = {
    "cpu": dict(linestyle="-",  marker="o"),
    "gpu": dict(linestyle="--", marker="s"),
}
MODE_LABEL = {"cpu": "CPU", "gpu": "GPU"}

def color(device: str) -> str:
    return DEVICE_COLORS.get(device, "grey")

def _save(fig: plt.Figure, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Data collecter
# ---------------------------------------------------------------------------
def collect(*, vary_key, fixed: dict, runs=runs) -> dict:
    """
    Group finished runs by (device, mode), returning sorted arrays of
    (vary_key values, walltime, loss_fraction).
    """
    groups: dict[tuple, list] = {}
    for entry in runs.values():
        if not all(entry.get(k) == v for k, v in fixed.items()):
            continue
        if entry["runtime"] is None or entry["loss_fraction"] is None:
            continue
        key = (entry["device"], entry["mode"])
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
# Twin-axis figure (walltime left, loss fraction right)
# ---------------------------------------------------------------------------
def twin_figure(data: dict, xlabel: str, xscale="log"):
    """
    Returns (fig, ax_rt, ax_lf). Each key in data is (device, mode).
    """
    fig, ax_rt = plt.subplots(figsize=(4.5, 3.5))
    ax_lf = ax_rt.twinx()
    handles = []
    for (device, mode), (xs, rts, lfs) in sorted(data.items()):
        c   = color(device)
        sty = MODE_STYLE[mode]
        l1, = ax_rt.plot(xs, rts, color=c, **sty)
        ax_lf.plot(xs, lfs, color=c, alpha=0.5,
                   linestyle=sty["linestyle"], marker=sty["marker"],
                   markerfacecolor="none")
        handles.append((l1, f"{device_label(device)} {MODE_LABEL[mode]}"))

    ax_rt.set_xlabel(xlabel)
    ax_rt.set_ylabel("Wall time (s)")
    ax_lf.set_ylabel(r"Loss fraction at $t_\mathrm{max}$", rotation=270, labelpad=14)
    if xscale == "log":
        ax_rt.set_xscale("log", base=2)
        ax_rt.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_rt.set_yscale("log")
    fig.legend([h for h, _ in handles], [l for _, l in handles],
               loc="upper left", bbox_to_anchor=(0.12, 0.88),
               framealpha=0.85, edgecolor="grey")
    # Proxy legend entries for linestyle meaning
    cpu_line = matplotlib.lines.Line2D([], [], color="grey", linestyle="-",  marker="o", label="CPU")
    gpu_line = matplotlib.lines.Line2D([], [], color="grey", linestyle="--", marker="s", label="GPU")
    fig.legend(handles=[cpu_line, gpu_line], loc="lower right",
               bbox_to_anchor=(0.92, 0.12), framealpha=0.85, edgecolor="grey")
    fig.tight_layout()
    return fig, ax_rt, ax_lf

# ---------------------------------------------------------------------------
# Figure 1 & 2: walltime + loss fraction vs nParticles
# ---------------------------------------------------------------------------
# Use the most common resolution and tol as the "reference" fixed values
all_tols = sorted({v["tol"] for v in runs.values()})
all_res  = sorted({v["resolution"] for v in runs.values()})
ref_tol  = all_tols[0]     # finest tol as reference
ref_res  = all_res[-1]     # largest resolution as reference

data_nP = collect(vary_key="nParticles", fixed={"resolution": ref_res, "tol": ref_tol})

if data_nP:
    fig, ax_rt, ax_lf = twin_figure(data_nP, xlabel=r"$N_\mathrm{particles}$")
    ax_rt.set_title(rf"Scaling with $N_\mathrm{{particles}}$ "
                    rf"(res={ref_res}, tol={ref_tol:.0e})")
    _save(fig, "walltime_and_lossfrac_vs_nparticles")

    # Separate walltime figure
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (device, mode), (xs, rts, lfs) in sorted(data_nP.items()):
        ax.plot(xs, rts, color=color(device), label=f"{device_label(device)} {MODE_LABEL[mode]}",
                **MODE_STYLE[mode])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_\mathrm{particles}$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(rf"Wall time vs $N_\mathrm{{particles}}$")
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "walltime_vs_nparticles")

    # Separate loss fraction figure
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (device, mode), (xs, rts, lfs) in sorted(data_nP.items()):
        ax.plot(xs, lfs, color=color(device), label=f"{device_label(device)} {MODE_LABEL[mode]}",
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
# Figure 3 & 4: walltime + loss fraction vs tolerance
# ---------------------------------------------------------------------------
# Use the largest nParticles as reference for tolerance scan
all_nP  = sorted({v["nParticles"] for v in runs.values()})
ref_nP  = all_nP[-1]

data_tol = collect(vary_key="tol", fixed={"nParticles": ref_nP, "resolution": ref_res})

if data_tol:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (device, mode), (xs, rts, lfs) in sorted(data_tol.items()):
        ax.plot(xs, rts, color=color(device), label=f"{device_label(device)} {MODE_LABEL[mode]}",
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

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (device, mode), (xs, rts, lfs) in sorted(data_tol.items()):
        ax.plot(xs, lfs, color=color(device), label=f"{device_label(device)} {MODE_LABEL[mode]}",
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
# Figure 5: walltime vs resolution for multiple resolutions
# ---------------------------------------------------------------------------
if len(all_res) > 1:
    data_res = collect(vary_key="resolution", fixed={"nParticles": ref_nP, "tol": ref_tol})
    if data_res:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for (device, mode), (xs, rts, lfs) in sorted(data_res.items()):
            ax.plot(xs, rts, color=color(device), label=f"{device_label(device)} {MODE_LABEL[mode]}",
                    **MODE_STYLE[mode])
        ax.set_yscale("log")
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(rf"Wall time vs grid resolution ($N={ref_nP}$, tol={ref_tol:.0e})")
        ax.legend(fontsize=8, framealpha=0.85)
        fig.tight_layout()
        _save(fig, "walltime_vs_resolution")

# ---------------------------------------------------------------------------
# Figure 6: CPU vs GPU loss fraction cross-validation scatter
# ---------------------------------------------------------------------------
cpu_runs = {k: v for k, v in runs.items() if v["mode"] == "cpu"}
gpu_runs = {k: v for k, v in runs.items() if v["mode"] == "gpu"}

# Match by (device, nParticles, resolution, tol)
def run_params(v):
    return (v["device"], v["nParticles"], v["resolution"], v["tol"])

cpu_by_params = {run_params(v): v for v in cpu_runs.values()}
gpu_by_params = {run_params(v): v for v in gpu_runs.values()}
common_params = set(cpu_by_params) & set(gpu_by_params)

if common_params:
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    for params in sorted(common_params):
        device, nP, res, tol = params
        cpu_lf = cpu_by_params[params]["loss_fraction"]
        gpu_lf = gpu_by_params[params]["loss_fraction"]
        if cpu_lf is None or gpu_lf is None:
            continue
        ax.scatter(gpu_lf, cpu_lf, color=color(device), s=20, zorder=3)

    # Identity line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Exact agreement")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"GPU loss fraction")
    ax.set_ylabel(r"CPU loss fraction")
    ax.set_title("CPU vs GPU cross-validation")

    # Device legend
    handles = [matplotlib.patches.Patch(color=color(d), label=device_label(d))
               for d in DEVICE_COLORS if any(p[0] == d for p in common_params)]
    handles.append(matplotlib.lines.Line2D([], [], color="k", linestyle="--", label="Exact agreement"))
    ax.legend(handles=handles, fontsize=8, framealpha=0.85)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "cpu_gpu_crossval")

# ---------------------------------------------------------------------------
# Figure 7: loss fraction vs time from CPU results files (one fig per device)
# ---------------------------------------------------------------------------
import matplotlib.patches as mpatches

for device in DEVICE_COLORS:
    # Collect CPU runs for this device with the reference nP and resolution
    device_cpu_runs = [
        (k, v) for k, v in cpu_runs.items()
        if v["device"] == device
        and v["nParticles"] == ref_nP
        and v["resolution"] == ref_res
    ]
    if not device_cpu_runs:
        continue

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for run_key, entry in sorted(device_cpu_runs, key=lambda x: x[1]["tol"]):
        results_file = SCALING_DIR / run_key / "inputs_results.npz"
        if not results_file.exists():
            continue
        d = np.load(results_file)
        times = d["times"]
        loss_frac_t = d["loss_frac"]
        tol = entry["tol"]
        ax.loglog(times, loss_frac_t, label=rf"tol={tol:.0e}")

    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Loss fraction")
    ax.set_title(rf"{device_label(device)} — CPU loss fraction vs time "
                 rf"($N={ref_nP}$, res={ref_res})")
    ax.set_xlim([1e-5, entry["tmax"]])
    ax.set_ylim([1e-3, 1])
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, f"loss_vs_time_{device_label(device)}")

print(f"Figures saved to {FIG_DIR}/")
