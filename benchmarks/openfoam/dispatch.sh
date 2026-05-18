#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./dispatch.sh <path_to_config.cfg>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(readlink -f "$1")"
OUT_DIR="$SCRIPT_DIR/output"

source "$CONFIG_FILE"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/result.*.txt "$OUT_DIR"/log.*

echo "=== Dispatching OpenFOAM Benchmark Jobs ==="

echo "1. Submitting Profile Job..."
sbatch $SLURM_RUN_ARGS --job-name=OF_Prof --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$SCRIPT_DIR/run_job.sh" profile 0

echo "2. Submitting $NUM_TRIALS Normal & $NUM_TRIALS Boosted Trials..."
for i in $(seq 1 $NUM_TRIALS); do
    sbatch $SLURM_RUN_ARGS --job-name=OF_Norm --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$SCRIPT_DIR/run_job.sh" normal $i
    sbatch $SLURM_RUN_ARGS --job-name=OF_Bost --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$SCRIPT_DIR/run_job.sh" boost $i
done

echo "All jobs submitted! Run ./monitor.sh $CONFIG_FILE to watch results."
