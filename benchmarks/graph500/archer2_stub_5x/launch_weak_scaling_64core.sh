#!/bin/bash

CONF_DIR="./benchmarks/graph500/experiments/stub_5x_configs"
SUITE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export RESULTS_BASE_DIR="./benchmarks/graph500/archer2_stub_5x/results_64core/suite_${SUITE_TIMESTAMP}"

CONFIGS=(
    "1node.conf"
    "2node.conf"
    "4node.conf"
    "8node.conf"
    "16node.conf"
    "32node.conf"
)

echo "==========================================================="
echo " INITATING SAMPI WEAK SCALING BATTERY (64-CORE)"
echo "==========================================================="

for conf_file in "${CONFIGS[@]}"; do
    full_path="${CONF_DIR}/${conf_file}"
    
    if [ -f "$full_path" ]; then
        echo "--> Submitting: $conf_file"
        
        bash ./benchmarks/graph500/archer2_stub_5x/submit_experiment.sh "$full_path"
        
    else
        echo "--> WARNING: $conf_file not found! Skipping..."
    fi
done

echo "==========================================================="
echo " Battery submitted successfully!"
echo " Run 'squeue -u \$USER' to monitor your queue status."
echo "==========================================================="
