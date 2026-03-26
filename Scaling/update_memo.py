#!/usr/bin/env python
"""
Scan all deployed runs and update memo.json with current status.

Status transitions:
  deployed -> running   (slurm job in queue/running)
  deployed -> finished  (results file found)
  deployed -> failed    (no results, no active job)
  running  -> finished  (results file found)
  running  -> failed    (job no longer active, no results)

Reads runtime and loss_fraction from <run_dir>/inputs_results.npz when available.
"""
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

SCALING_DIR = Path(__file__).parent
MEMO_FILE   = SCALING_DIR / "runs" / "memo.json"

if not MEMO_FILE.exists():
    print("No memo.json found. Run deploy_scan.py first.")
    raise SystemExit(1)

with open(MEMO_FILE) as f:
    memo = json.load(f)

# Collect active slurm job names to detect running jobs.
# Job names are set to the run folder's last two components (mode/run_name).
active_jobs: set[str] = set()
try:
    out = subprocess.check_output(
        ["squeue", "--me", "--format=%j", "--noheader"], text=True
    )
    active_jobs = {line.strip() for line in out.splitlines() if line.strip()}
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # squeue not available (e.g., running locally)

changed = 0
for run_key, entry in memo.items():
    if entry["status"] == "finished":
        continue

    run_dir = SCALING_DIR / run_key

    # Check for results file (named inputs_results.npz by the tracing scripts)
    results_file = run_dir / "inputs_results.npz"
    if results_file.exists():
        data = dict(np.load(results_file))  # type: ignore[name-defined]
        entry["status"] = "finished"
        entry["runtime"] = float(data["runtime"])
        # CPU saves times+loss_frac array; GPU saves scalar loss_frac
        if "loss_frac" in data:
            lf = data["loss_frac"]
            entry["loss_fraction"] = float(lf[-1] if lf.ndim > 0 else lf)
        changed += 1
        continue

    # Check slurm queue — must match job_name format in deploy_scan.py
    job_name = f"{entry['device'][:8]}_{entry['mode']}_{Path(run_key).name}"
    if job_name in active_jobs:
        if entry["status"] != "running":
            entry["status"] = "running"
            changed += 1
    elif entry["status"] == "running":
        # Was running, no longer in queue, no results -> failed
        entry["status"] = "failed"
        changed += 1

with open(MEMO_FILE, "w") as f:
    json.dump(memo, f, indent=2)

print(f"Updated {changed} entries. Memo saved to {MEMO_FILE}")

# Summary
counts = Counter(e["status"] for e in memo.values())
for status, n in sorted(counts.items()):
    print(f"  {status}: {n}")
