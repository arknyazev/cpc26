#!/bin/bash
#SBATCH --job-name=QH_10harmoni_nP1024_res48_tol1e-08
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m4505
#SBATCH --mail-user=aknyazev@ucsd.edu
#SBATCH --mail-type=begin,end,fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d
srun -n 512 -c 1 python -u tracing_template.py inputs.npz
