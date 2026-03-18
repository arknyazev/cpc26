#!/usr/bin/env python
"""
Scan all deployed SAW runs and update memo.json with current status.

Status transitions:
  deployed -> running   (a cpu_driver_* job is active in squeue)
  deployed -> finished  (results file found)
  deployed -> failed    (no results, no active driver)
  running  -> finished  (results file found)
  running  -> failed    (no active driver, no results)

Reads runtime and loss_fraction from <run_dir>/inputs_results.npz when available.
"""
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

SCALING_SAW_DIR = Path(__file__).parent
MEMO_FILE       = SCALING_SAW_DIR / "runs" / "memo.json"

if not MEMO_FILE.exists():
    print("No memo.json found. Run deploy_scan.py first.")
    raise SystemExit(1)

with open(MEMO_FILE) as f:
    memo = json.load(f)

# Check whether any driver jobs are still active
drivers_active = False
try:
    out = subprocess.check_output(
        ["squeue", "--me", "--format=%j", "--noheader"], text=True
    )
    active_jobs = {line.strip() for line in out.splitlines() if line.strip()}
    drivers_active = any(j.startswith("cpu_driver_") or j.startswith("gpu_driver_") for j in active_jobs)
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # squeue not available (e.g., running locally)

changed = 0
for run_key, entry in memo.items():
    if entry["status"] == "finished":
        continue

    run_dir = SCALING_SAW_DIR / run_key

    results_file = run_dir / "inputs_results.npz"
    if results_file.exists():
        data = dict(np.load(results_file))
        entry["status"] = "finished"
        entry["runtime"] = float(data["runtime"])
        if "loss_frac" in data:
            lf = data["loss_frac"]
            entry["loss_fraction"] = float(lf[-1] if lf.ndim > 0 else lf)
        changed += 1
        continue

    if drivers_active:
        if entry["status"] != "running":
            entry["status"] = "running"
            changed += 1
    elif entry["status"] == "running":
        entry["status"] = "failed"
        changed += 1

with open(MEMO_FILE, "w") as f:
    json.dump(memo, f, indent=2)

print(f"Updated {changed} entries. Memo saved to {MEMO_FILE}")

counts = Counter(e["status"] for e in memo.values())
for status, n in sorted(counts.items()):
    print(f"  {status}: {n}")