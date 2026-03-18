#!/bin/zsh

ln -sf ../device/boozmn_beta2.5_QH.nc .
conda run --no-capture-output -n thea python sample_fusion_distribution.py boozmn_beta2.5_QH.nc
echo "Done.