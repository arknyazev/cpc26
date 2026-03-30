#!/bin/bash
#SBATCH --job-name=gpu_driver
#SBATCH --output=/global/homes/m/mczek/catapult/cpc26/Scaling/runs/driver_gpu_%j.out
#SBATCH --error=/global/homes/m/mczek/catapult/cpc26/Scaling/runs/driver_gpu_%j.err
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=mc2589@cornell.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d-dev

QUEUE="/global/homes/m/mczek/catapult/cpc26/Scaling/runs/gpu_queue.txt"

# Pop the first run from the queue
RUN_SH=$(head -1 "${QUEUE}")
tail -n +2 "${QUEUE}" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "${QUEUE}"

echo "Running: ${RUN_SH}"
cd "$(dirname "${RUN_SH}")"
source run.sh

# Resubmit driver if queue still has entries
if [[ -s "${QUEUE}" ]]; then
    sbatch "/global/homes/m/mczek/catapult/cpc26/Scaling/runs/gpu_driver.sh"
fi
