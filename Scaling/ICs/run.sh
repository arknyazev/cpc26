#!/bin/zsh

WOUT_DIR="../devices/wout"
SCRIPT="sample_fusion_distribution.py"

for f in "${WOUT_DIR}"/wout_*.nc; do
    echo "Processing: ${f}"
    conda run --no-capture-output -n thea python "${SCRIPT}" "${f}"
done

echo "Done."