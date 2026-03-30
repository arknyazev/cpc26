#!/bin/bash
#SBATCH --job-name=cpu_driver
#SBATCH --output=/global/homes/m/mczek/catapult/cpc26/Scaling/runs/driver_cpu_%j.out
#SBATCH --error=/global/homes/m/mczek/catapult/cpc26/Scaling/runs/driver_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4505
#SBATCH --mail-user=mc2589@cornell.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d-dev

QUEUE="/global/homes/m/mczek/catapult/cpc26/Scaling/runs/cpu_queue.txt"

# Pop the first run from the queue
RUN_SH=$(head -1 "${QUEUE}")
tail -n +2 "${QUEUE}" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "${QUEUE}"

echo "Running: ${RUN_SH}"
cd "$(dirname "${RUN_SH}")"
# Source run.sh — #SBATCH lines are comments, execution lines run inside this allocation
source run.sh

# Resubmit driver if queue still has entries
if [[ -s "${QUEUE}" ]]; then
    sbatch "/global/homes/m/mczek/catapult/cpc26/Scaling/runs/cpu_driver.sh"
fi
