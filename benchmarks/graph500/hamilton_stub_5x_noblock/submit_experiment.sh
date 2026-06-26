#!/bin/bash

if [ -z "$1" ]; then 
    echo "Usage: $0 <path_to_experiment_config.conf>"
    exit 1
fi

CONFIG_FILE=$(realpath "$1")
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

TOTAL_TASKS=$(( NUM_NODES * TASKS_PER_NODE ))
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="${NUM_NODES}nodes_np${TOTAL_TASKS}"
if [ -z "$RESULTS_BASE_DIR" ]; then
    RESULTS_BASE_DIR="./benchmarks/graph500/experiments/results"
fi
RELATIVE_DIR="${RESULTS_BASE_DIR}/${EXP_NAME}"

echo "=> Initializing Experiment: ${EXP_NAME}"
mkdir -p "${RELATIVE_DIR}"
EXP_DIR=$(realpath "${RELATIVE_DIR}")


METADATA_FILE="${EXP_DIR}/experiment_metadata.log"
cat <<EOF > "${METADATA_FILE}"
EXPERIMENT ID: ${EXP_NAME}
TIMESTAMP:     $(date)
CONFIG USED:   ${CONFIG_FILE}
--------------------------------------------------
SCALE:         ${SCALE}
EDGEFACTOR:    ${EDGEFACTOR}
TOTAL RANKS:   ${TOTAL_TASKS} (${NUM_NODES} nodes @ ${TASKS_PER_NODE} tasks/node)
--------------------------------------------------
EOF

SLURM_ARGS=(
    "--job-name=${EXP_NAME}"
    "--nodes=${NUM_NODES}"
    "--ntasks-per-node=${TASKS_PER_NODE}"
    "--time=${TIME_LIMIT}"
    "--output=${EXP_DIR}/slurm_job_%j.log"
    "--export=ALL,EXP_DIR=${EXP_DIR},CONFIG_FILE=${CONFIG_FILE}"
)

if [ -n "${SLURM_PARTITION}" ]; then
    SLURM_ARGS+=("--partition=${SLURM_PARTITION}")
fi

if [ -n "${SLURM_ACCOUNT}" ]; then
    SLURM_ARGS+=("--account=${SLURM_ACCOUNT}")
fi

sbatch "${SLURM_ARGS[@]}" ./benchmarks/graph500/hamilton_stub_5x_noblock/worker.sh

echo "=> Job submitted! Results will populate in: ${EXP_DIR}"
