#!/bin/zsh

echo "Running the Passing Map script on Winstell-A"
ln -s scripts/1_compute_passing_Poincare.py .
ln -s inputs/boozmn_aten_rescaled.nc boozmn.nc
conda run --no-capture-output -n thea python 1_compute_passing_Poincare.py | tee outputs/1_compute_passing_Poincare.log
echo "Done; plotting Poincare cross-section"
ln -s scripts/2_plot_passing_Poincare.py .
conda run --no-capture-output -n thea python 2_plot_passing_Poincare.py | tee outputs/2_plot_passing_Poincare.log
echo "Done; moving outputs to outputs/"
mv poincare_passing_cpu.png outputs/poincare_passing_cpu.png
mv poincare_all_points.txt outputs/poincare_all_points_passing.txt
mv initial_positions.txt outputs/initial_positions_passing.txt
mv passing_poincare.pdf outputs/passing_poincare.pdf
rm 1_compute_passing_Poincare.py
rm 2_plot_passing_Poincare.py
rm boozmn.nc
echo "All done;"