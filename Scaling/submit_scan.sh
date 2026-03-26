#!/bin/bash
# Submit the parameter scan respecting the debug queue limit of 2 submitted jobs total.
#
# Strategy: self-resubmitting driver pattern.
#   - One CPU driver + one GPU driver are submitted (2 jobs total in queue at any time)
#   - Each driver pops the next run from a queue file, executes it via `source run.sh`,
#     then resubmits itself if the queue is not empty
#   - SBATCH directives in sourced run.sh are treated as comments — only the
#     module/conda/srun lines execute, which is correct since we are already allocated
#
# Usage:
#   bash submit_scan.sh                    # submit all cpu + gpu runs
#   bash submit_scan.sh --mode cpu         # cpu driver only
#   bash submit_scan.sh --mode gpu         # gpu driver only
#   bash submit_scan.sh --device boozmn_new_QH_aScaling   # filter by device
#   bash submit_scan.sh --dry-run          # print queue contents, no submission

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"

if [[ ! -d "${RUNS_DIR}" ]]; then
    echo "Error: ${RUNS_DIR} does not exist. Run deploy_scan.py first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE_FILTER=""
DEVICE_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE_FILTER="$2";   shift 2 ;;
        --device)  DEVICE_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true;       shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Collect and sort run.sh paths, split into cpu / gpu lists
# ---------------------------------------------------------------------------
mapfile -t ALL_SCRIPTS < <(find "${RUNS_DIR}" -name "run.sh" | sort)

CPU_SCRIPTS=()
GPU_SCRIPTS=()

for script in "${ALL_SCRIPTS[@]}"; do
    [[ -f "$(dirname "${script}")/inputs_results.npz" ]] && continue
    if [[ -n "${DEVICE_FILTER}" && "${script}" != *"${DEVICE_FILTER}"* ]]; then
        continue
    fi
    if [[ "${script}" == */cpu/* ]]; then
        CPU_SCRIPTS+=("${script}")
    elif [[ "${script}" == */gpu/* ]]; then
        GPU_SCRIPTS+=("${script}")
    fi
done

echo "Pending: ${#CPU_SCRIPTS[@]} CPU runs, ${#GPU_SCRIPTS[@]} GPU runs (finished skipped)."

if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    echo "--- CPU queue ---"
    printf '%s\n' "${CPU_SCRIPTS[@]}"
    echo ""
    echo "--- GPU queue ---"
    printf '%s\n' "${GPU_SCRIPTS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Write queue files and generate driver scripts, then submit the first driver
# ---------------------------------------------------------------------------
write_and_submit() {
    local mode="$1"      # "cpu" or "gpu"
    local -n scripts="$2"

    if [[ ${#scripts[@]} -eq 0 ]]; then
        return
    fi

    local queue_file="${RUNS_DIR}/${mode}_queue.txt"
    local driver_script="${RUNS_DIR}/${mode}_driver.sh"

    # Write queue file (one absolute run.sh path per line)
    printf '%s\n' "${scripts[@]}" > "${queue_file}"
    echo "Wrote ${#scripts[@]} entries to ${queue_file}"

    # Generate the driver script
    if [[ "${mode}" == "cpu" ]]; then
        cat > "${driver_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${mode}_driver
#SBATCH --output=${RUNS_DIR}/driver_${mode}_%j.out
#SBATCH --error=${RUNS_DIR}/driver_${mode}_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4505
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d

QUEUE="${queue_file}"

# Pop the first run from the queue
RUN_SH=\$(head -1 "\${QUEUE}")
tail -n +2 "\${QUEUE}" > "\${QUEUE}.tmp" && mv "\${QUEUE}.tmp" "\${QUEUE}"

echo "Running: \${RUN_SH}"
cd "\$(dirname "\${RUN_SH}")"
# Source run.sh — #SBATCH lines are comments, execution lines run inside this allocation
source run.sh

# Resubmit driver if queue still has entries
if [[ -s "\${QUEUE}" ]]; then
    sbatch "${driver_script}"
fi
EOF
    else
        cat > "${driver_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${mode}_driver
#SBATCH --output=${RUNS_DIR}/driver_${mode}_%j.out
#SBATCH --error=${RUNS_DIR}/driver_${mode}_%j.err
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate mc

QUEUE="${queue_file}"

# Pop the first run from the queue
RUN_SH=\$(head -1 "\${QUEUE}")
tail -n +2 "\${QUEUE}" > "\${QUEUE}.tmp" && mv "\${QUEUE}.tmp" "\${QUEUE}"

echo "Running: \${RUN_SH}"
cd "\$(dirname "\${RUN_SH}")"
source run.sh

# Resubmit driver if queue still has entries
if [[ -s "\${QUEUE}" ]]; then
    sbatch "${driver_script}"
fi
EOF
    fi

    local job_id
    job_id=$(sbatch "${driver_script}" | awk '{print $NF}')
    echo "Submitted ${mode} driver: job ${job_id} (${#scripts[@]} runs queued)"
}

if [[ -z "${MODE_FILTER}" || "${MODE_FILTER}" == "cpu" ]]; then
    write_and_submit "cpu" CPU_SCRIPTS
fi

if [[ -z "${MODE_FILTER}" || "${MODE_FILTER}" == "gpu" ]]; then
    write_and_submit "gpu" GPU_SCRIPTS
fi

echo ""
echo "Done. Monitor with: squeue --me"
echo "Update memo with:   python ${SCRIPT_DIR}/update_memo.py"
