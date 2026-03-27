#!/usr/bin/env python
"""
Deploy a parameter scan for CPU and/or GPU particle tracing runs.

For each combination of (device, nParticles, resolution, tol), create a run folder
containing inputs.npz, a symlink to the tracing script, and a slurm run.sh.
A memo.json is written/updated at Scaling/runs/memo.json tracking all runs.

Usage:
  python deploy_scan.py                 # deploy new runs
  python deploy_scan.py --reset-failed  # clear failed runs from memo and redeploy them
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SCALING_DIR = Path(__file__).parent
sys.path.insert(0, str(SCALING_DIR))
from inputs import Inputs

reset_failed = "--reset-failed" in sys.argv

# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------

# Paths relative to SCALING_DIR; stored as absolute paths in inputs.npz
BOOZMN_FILES = [
    "devices/boozmn/boozmn_HSX_QHS_vacuum_ns201_aScaling.nc",
    "devices/boozmn/boozmn_ncsx_c09r00_fixed_aScaling.nc",
    "devices/boozmn/boozmn_new_QA_aScaling.nc",
    "devices/boozmn/boozmn_new_QH_aScaling.nc",
]

# keep >= 512 (CPU comm_size):
NPARTICLES_VALUES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]  
RESOLUTION_VALUES = [16, 32, 64]
TOL_VALUES        = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
TMAX              = 1e-3

MODES = ["cpu", "gpu"]

# ---------------------------------------------------------------------------
# Slurm template builders
# ---------------------------------------------------------------------------

def cpu_slurm(job_name: str, script_name: str, inputs_name: str) -> str:
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4505
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=begin,end,fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d
srun -n 512 -c 1 python -u {script_name} {inputs_name}
"""

def gpu_slurm(job_name: str, script_name: str, inputs_name: str) -> str:
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=begin,end,fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate mc
python -u {script_name} {inputs_name}
"""

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

RUNS_DIR  = SCALING_DIR / "runs"
MEMO_FILE = RUNS_DIR / "memo.json"

CPU_SCRIPT = SCALING_DIR / "cpu" / "tracing_template.py"
GPU_SCRIPT = SCALING_DIR / "gpu" / "tracing_template.py"

memo: dict = {}
if MEMO_FILE.exists():
    with open(MEMO_FILE) as f:
        memo = json.load(f)

# Reset failed runs: remove from memo so they get redeployed
if reset_failed:
    failed_keys = [k for k, v in memo.items() if v["status"] == "failed"]
    for key in failed_keys:
        run_dir = SCALING_DIR / key
        # Remove any partial results file
        for f in run_dir.glob("*_results.npz"):
            f.unlink()
        del memo[key]
    print(f"Reset {len(failed_keys)} failed runs for redeployment.")

new_count = 0

for boozmn in BOOZMN_FILES:
    boozmn_abs = str(SCALING_DIR / boozmn)
    device_stem = Path(boozmn).stem
    # ICs were generated from wout files, so folder uses wout_ prefix
    ic_stem = device_stem.replace("boozmn_", "wout_", 1)
    ic_file = str(SCALING_DIR / "ICs" / ic_stem / "initial_conditions.txt")

    for mode in MODES:
        for nP in NPARTICLES_VALUES:
            for res in RESOLUTION_VALUES:
                for tol in TOL_VALUES:
                    tol_str = f"{tol:.0e}"
                    run_name = f"nP{nP}_res{res}_tol{tol_str}"
                    run_dir = RUNS_DIR / device_stem / mode / run_name
                    run_key = str(run_dir.relative_to(SCALING_DIR))

                    if run_key in memo:
                        print(f"Skipping existing run: {run_key}")
                        continue

                    run_dir.mkdir(parents=True, exist_ok=True)

                    # Write inputs.npz with absolute boozmn path
                    inputs_path = run_dir / "inputs.npz"
                    inp = Inputs(
                        boozmn_filename=boozmn_abs,
                        ic_file=ic_file,
                        nParticles=nP,
                        resolution=res,
                        tol=tol,
                        tmax=TMAX,
                    )
                    inp.to_npz(inputs_path)

                    # Symlink tracing script
                    src_script = CPU_SCRIPT if mode == "cpu" else GPU_SCRIPT
                    script_link = run_dir / src_script.name
                    if not script_link.exists():
                        script_link.symlink_to(os.path.relpath(src_script, run_dir))

                    # Write slurm script
                    slurm_fn = cpu_slurm if mode == "cpu" else gpu_slurm
                    job_name = f"{device_stem[:8]}_{mode}_{run_name}"
                    (run_dir / "run.sh").write_text(
                        slurm_fn(job_name, src_script.name, inputs_path.name)
                    )

                    memo[run_key] = {
                        "status": "deployed",
                        "mode": mode,
                        "device": device_stem,
                        "nParticles": nP,
                        "resolution": res,
                        "tol": tol,
                        "tmax": TMAX,
                        "runtime": None,
                        "loss_fraction": None,
                    }
                    new_count += 1
                    print(f"Deployed: {run_key}")

MEMO_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(MEMO_FILE, "w") as f:
    json.dump(memo, f, indent=2)

print(f"\nDeployed {new_count} new runs. Memo updated at {MEMO_FILE}")
