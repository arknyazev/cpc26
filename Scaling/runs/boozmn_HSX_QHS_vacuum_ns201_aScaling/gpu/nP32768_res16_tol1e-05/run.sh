#!/bin/bash
#SBATCH --job-name=boozmn_H_gpu_nP32768_res16_tol1e-05
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -A m4505
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=mc2589@cornell.edu
#SBATCH --mail-type=begin,end,fail

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d-dev
python -u tracing_template.py inputs.npz
