#!/bin/bash
#SBATCH --job-name=boozmn_n_cpu_nP65536_res8_tol1e-05
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m4505
#SBATCH --mail-user=mc2589@cornell.edu
#SBATCH --mail-type=begin,end,fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d-dev
srun -n 128 -c 1 python -u tracing_template.py inputs.npz
