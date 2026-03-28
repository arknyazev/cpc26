#!/bin/bash
#SBATCH --job-name=cpu_driver
#SBATCH --output=/pscratch/sd/a/aknyazev/march26/cpc26/Scaling/runs/driver_cpu_%j.out
#SBATCH --error=/pscratch/sd/a/aknyazev/march26/cpc26/Scaling/runs/driver_cpu_%j.err
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

QUEUE="/pscratch/sd/a/aknyazev/march26/cpc26/Scaling/runs/cpu_queue.txt"

# Pop the first run from the queue
RUN_SH=$(head -1 "${QUEUE}")
tail -n +2 "${QUEUE}" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "${QUEUE}"

echo "Running: ${RUN_SH}"
cd "$(dirname "${RUN_SH}")"
# Source run.sh — #SBATCH lines are comments, execution lines run inside this allocation
source run.sh

# Resubmit driver if queue still has entries
if [[ -s "${QUEUE}" ]]; then
    sbatch "/pscratch/sd/a/aknyazev/march26/cpc26/Scaling/runs/cpu_driver.sh"
fi
