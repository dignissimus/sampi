#!/bin/bash

CONF_DIR="./benchmarks/graph500/experiments/stub-comparison"
export RESULTS_BASE_DIR="./benchmarks/graph500/experiments/results_archer2_64core_stub"

CONFIGS=(
    "1node_scale20.conf"
    "2node_scale21.conf"
    "4node_scale22.conf"
    "8node_scale23.conf"
    "16node_scale24.conf"
    "32node_scale25.conf"
)

echo "==========================================================="
echo " INITATING SAMPI WEAK SCALING BATTERY"
echo "==========================================================="

for conf_file in "${CONFIGS[@]}"; do
    full_path="${CONF_DIR}/${conf_file}"
    
    if [ -f "$full_path" ]; then
        echo "--> Submitting: $conf_file"
        
        bash ./benchmarks/graph500/archer2_stub_comparison/submit_experiment.sh "$full_path"
        
    else
        echo "--> WARNING: $conf_file not found! Skipping..."
    fi
done

echo "==========================================================="
echo " Battery submitted successfully!"
echo " Run 'squeue -u \$USER' to monitor your queue status."
echo "==========================================================="
