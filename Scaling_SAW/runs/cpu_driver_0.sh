#!/bin/bash
#SBATCH --job-name=cpu_driver_0
#SBATCH --output=/pscratch/sd/a/aknyazev/march26/cpc26/Scaling_SAW/runs/cpu_driver_0_%j.out
#SBATCH --error=/pscratch/sd/a/aknyazev/march26/cpc26/Scaling_SAW/runs/cpu_driver_0_%j.err
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

QUEUE="/pscratch/sd/a/aknyazev/march26/cpc26/Scaling_SAW/runs/cpu_queue_0.txt"
RUN_SH=$(head -1 "${QUEUE}")
tail -n +2 "${QUEUE}" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "${QUEUE}"
echo "cpu stream 0 running: ${RUN_SH}"
cd "$(dirname "${RUN_SH}")"
source run.sh
if [[ -s "${QUEUE}" ]]; then sbatch "/pscratch/sd/a/aknyazev/march26/cpc26/Scaling_SAW/runs/cpu_driver_0.sh"; fi
