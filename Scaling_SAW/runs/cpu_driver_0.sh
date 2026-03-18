#!/bin/bash
#SBATCH --job-name=cpu_driver_0
#SBATCH --output=/Users/aknyazev/Desktop/cpc26/Scaling_SAW/runs/driver_0_%j.out
#SBATCH --error=/Users/aknyazev/Desktop/cpc26/Scaling_SAW/runs/driver_0_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m4505
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d

QUEUE="/Users/aknyazev/Desktop/cpc26/Scaling_SAW/runs/cpu_queue_0.txt"

# Pop the first run from the queue
RUN_SH=$(head -1 "${QUEUE}")
tail -n +2 "${QUEUE}" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "${QUEUE}"

echo "Stream 0 running: ${RUN_SH}"
cd "$(dirname "${RUN_SH}")"
# Source run.sh — #SBATCH lines are comments, execution lines run in this allocation
source run.sh

# Resubmit this driver if queue still has entries
if [[ -s "${QUEUE}" ]]; then
    sbatch "/Users/aknyazev/Desktop/cpc26/Scaling_SAW/runs/cpu_driver_0.sh"
fi
