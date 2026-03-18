#!/bin/bash
# Submit the SAW parameter scan using self-resubmitting streams.
#
# Strategy: CPU and GPU drivers are submitted simultaneously. Each driver pops
# the next run from its own queue file, executes it via `source run.sh`
# (SBATCH directives are treated as comments), then resubmits itself if its
# queue is not yet empty.
#
# CPU runs: detected by absence of /gpu/ in path — 4 parallel streams (regular)
#           or 1 stream (debug, 4-node limit)
# GPU runs: detected by /gpu/ in path — 4 parallel streams (regular)
#           or 1 stream (debug)
#
# Usage:
#   bash submit_scan.sh              # submit all pending CPU + GPU runs
#   bash submit_scan.sh --debug      # use debug queue (30 min) instead of regular (2 h)
#   bash submit_scan.sh --mode cpu   # cpu only
#   bash submit_scan.sh --mode gpu   # gpu only
#   bash submit_scan.sh --dry-run    # print queue contents, no submission

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
NSTREAMS=4

if [[ ! -d "${RUNS_DIR}" ]]; then
    echo "Error: ${RUNS_DIR} does not exist. Run deploy_scan.py first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
DEBUG=false
MODE_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true;         shift ;;
        --debug)    DEBUG=true;           shift ;;
        --mode)     MODE_FILTER="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "${DEBUG}" == true ]]; then
    CPU_QOS="debug"; CPU_WALLTIME="0:30:00"; CPU_NSTREAMS=1
    GPU_QOS="debug"; GPU_WALLTIME="0:30:00"; GPU_NSTREAMS=1
else
    CPU_QOS="regular"; CPU_WALLTIME="2:00:00"; CPU_NSTREAMS="${NSTREAMS}"
    GPU_QOS="regular"; GPU_WALLTIME="2:00:00"; GPU_NSTREAMS="${NSTREAMS}"
fi

# ---------------------------------------------------------------------------
# Collect pending run.sh paths, split into cpu / gpu
# ---------------------------------------------------------------------------
CPU_SCRIPTS=()
GPU_SCRIPTS=()
while IFS= read -r run_sh; do
    [[ -f "$(dirname "${run_sh}")/inputs_results.npz" ]] && continue
    if [[ "${run_sh}" == */gpu/* ]]; then
        GPU_SCRIPTS+=("${run_sh}")
    else
        CPU_SCRIPTS+=("${run_sh}")
    fi
done < <(find "${RUNS_DIR}" -name "run.sh" | sort)

echo "Pending: ${#CPU_SCRIPTS[@]} CPU runs, ${#GPU_SCRIPTS[@]} GPU runs (finished skipped)."

if [[ "${DRY_RUN}" == true ]]; then
    for mode in cpu gpu; do
        [[ -n "${MODE_FILTER}" && "${MODE_FILTER}" != "${mode}" ]] && continue
        if [[ "${mode}" == "cpu" ]]; then scripts=("${CPU_SCRIPTS[@]+"${CPU_SCRIPTS[@]}"}"); ns="${CPU_NSTREAMS}"
        else                             scripts=("${GPU_SCRIPTS[@]+"${GPU_SCRIPTS[@]}"}"); ns="${GPU_NSTREAMS}"; fi
        echo ""
        echo "=== ${mode^^} streams (${ns}) ==="
        idx=0
        for s in "${scripts[@]+"${scripts[@]}"}"; do
            stream=$(( idx % ns ))
            echo "  [stream ${stream}] ${s}"
            (( idx++ )) || true
        done
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# submit_stream <mode> <stream_idx> <scripts_array_name> <nstreams> <qos> <walltime>
# ---------------------------------------------------------------------------
submit_stream() {
    local mode="$1"
    local i="$2"
    local -n _scripts="$3"
    local ns="$4"
    local qos="$5"
    local walltime="$6"

    local queue_file="${RUNS_DIR}/${mode}_queue_${i}.txt"
    local driver_script="${RUNS_DIR}/${mode}_driver_${i}.sh"

    > "${queue_file}"
    local idx=0
    for script in "${_scripts[@]+"${_scripts[@]}"}"; do
        if (( idx % ns == i )); then
            echo "${script}" >> "${queue_file}"
        fi
        (( idx++ )) || true
    done

    local count
    count=$(wc -l < "${queue_file}")
    if [[ "${count}" -eq 0 ]]; then
        echo "${mode} stream ${i}: empty, skipping."
        return
    fi

    if [[ "${mode}" == "cpu" ]]; then
        cat > "${driver_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${mode}_driver_${i}
#SBATCH --output=${RUNS_DIR}/${mode}_driver_${i}_%j.out
#SBATCH --error=${RUNS_DIR}/${mode}_driver_${i}_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=${walltime}
#SBATCH --constraint=cpu
#SBATCH --qos=${qos}
#SBATCH --account=m4505
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d

QUEUE="${queue_file}"
RUN_SH=\$(head -1 "\${QUEUE}")
tail -n +2 "\${QUEUE}" > "\${QUEUE}.tmp" && mv "\${QUEUE}.tmp" "\${QUEUE}"
echo "${mode} stream ${i} running: \${RUN_SH}"
cd "\$(dirname "\${RUN_SH}")"
source run.sh
if [[ -s "\${QUEUE}" ]]; then sbatch "${driver_script}"; fi
EOF
    else
        cat > "${driver_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${mode}_driver_${i}
#SBATCH --output=${RUNS_DIR}/${mode}_driver_${i}_%j.out
#SBATCH --error=${RUNS_DIR}/${mode}_driver_${i}_%j.err
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q ${qos}
#SBATCH -t ${walltime}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate mc

QUEUE="${queue_file}"
RUN_SH=\$(head -1 "\${QUEUE}")
tail -n +2 "\${QUEUE}" > "\${QUEUE}.tmp" && mv "\${QUEUE}.tmp" "\${QUEUE}"
echo "${mode} stream ${i} running: \${RUN_SH}"
cd "\$(dirname "\${RUN_SH}")"
source run.sh
if [[ -s "\${QUEUE}" ]]; then sbatch "${driver_script}"; fi
EOF
    fi

    local job_id
    job_id=$(sbatch "${driver_script}" | awk '{print $NF}')
    echo "Submitted ${mode} stream ${i}: job ${job_id} (${count} runs queued)"
}

# ---------------------------------------------------------------------------
# Submit CPU and/or GPU streams
# ---------------------------------------------------------------------------
if [[ -z "${MODE_FILTER}" || "${MODE_FILTER}" == "cpu" ]]; then
    for i in $(seq 0 $(( CPU_NSTREAMS - 1 ))); do
        submit_stream "cpu" "${i}" CPU_SCRIPTS "${CPU_NSTREAMS}" "${CPU_QOS}" "${CPU_WALLTIME}"
    done
fi

if [[ -z "${MODE_FILTER}" || "${MODE_FILTER}" == "gpu" ]]; then
    for i in $(seq 0 $(( GPU_NSTREAMS - 1 ))); do
        submit_stream "gpu" "${i}" GPU_SCRIPTS "${GPU_NSTREAMS}" "${GPU_QOS}" "${GPU_WALLTIME}"
    done
fi

echo ""
echo "Done. Monitor with:   squeue --me"
echo "Update memo with:     python ${SCRIPT_DIR}/update_memo.py"
