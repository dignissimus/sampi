#!/bin/bash
# ==============================================================================
# Executed by Slurm on compute nodes. 
# ==============================================================================

if [ -z "$EXP_DIR" ] || [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: EXP_DIR or CONFIG_FILE environment variables are missing."
    exit 1
fi

source "$CONFIG_FILE"
TOTAL_TASKS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# Global MPI options required for Sampi's hardware awareness
MPI_GLOBAL_OPTS="--bind-to core --map-by core"

echo "SLURM JOB ID:  ${SLURM_JOB_ID}" >> "${EXP_DIR}/experiment_metadata.log"
echo "ALLOC NODES:   ${SLURM_JOB_NODELIST}" >> "${EXP_DIR}/experiment_metadata.log"

cd "${EXP_DIR}" || exit 1

# ==============================================================================
# Phase 1: The Profiling Run
# ==============================================================================
echo "=> [$(date)] Starting Phase 1: Sampi Profiling Run on ${TOTAL_TASKS} ranks..."

LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiprofile.so" mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} \
    "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "01_profile_run.out" 2>&1

if [ ! -f "sampi_communication_profile.txt" ]; then
    echo "ERROR: sampi_communication_profile.txt was not generated! Aborting."
    exit 1
fi

# ==============================================================================
# Phase 2: The Boosted Run
# ==============================================================================
echo "=> [$(date)] Starting Phase 2: Sampi Boosted Run on ${TOTAL_TASKS} ranks..."

LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiboost.so" mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} \
    "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "02_boosted_run.out" 2>&1

echo "=> [$(date)] Experiment Complete!"

# ==============================================================================
# Phase 3: Normal run
# ==============================================================================
echo "=> [$(date)] Starting Phase 0: Vanilla Baseline Run on ${TOTAL_TASKS} ranks..."

mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} \
    "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "00_vanilla_run.out" 2>&1
