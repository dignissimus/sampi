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

# Global options required for ARCHER2 and Sampi's hardware awareness
SRUN_OPTS="--hint=nomultithread --distribution=block:block"

echo "SLURM JOB ID:  ${SLURM_JOB_ID}" >> "${EXP_DIR}/experiment_metadata.log"
echo "ALLOC NODES:   ${SLURM_JOB_NODELIST}" >> "${EXP_DIR}/experiment_metadata.log"

cd "${EXP_DIR}" || exit 1

if [ "$SCALE" -lt 24 ] && [ "$TASKS_PER_NODE" -ne 128 ]; then
    # ==============================================================================
    # Phase 1: The Profiling Run
    # ==============================================================================
    echo "=> [$(date)] Starting Phase 1: Sampi Profiling Run on ${TOTAL_TASKS} ranks..."

    srun --export=ALL,LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiprofile.so" ${SRUN_OPTS} --ntasks=${TOTAL_TASKS} \
        "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "01_profile_run.out" 2>&1

    if [ ! -f "sampi_communication_profile.txt" ]; then
        echo "ERROR: sampi_communication_profile.txt was not generated! Aborting."
        exit 1
    fi
else
    echo "=> [$(date)] Skipping Phase 1 (Profiling) and Phase 2 (Boosted) due to Two-Opt overhead or NUMA conflicts on 128-core nodes."
fi

# ==============================================================================
# Phase 2, 3, 4: Shuffled Iterations
# ==============================================================================
for iter in {1..5}; do
    echo "=============================================================================="
    echo "=> [$(date)] Starting Battery Iteration $iter"
    echo "=============================================================================="

    RUNS=("vanilla" "stub")
    if [ "$SCALE" -lt 24 ] && [ "$TASKS_PER_NODE" -ne 128 ]; then
        RUNS+=("boost")
    fi

    # Shuffle the runs array
    SHUFFLED_RUNS=($(shuf -e "${RUNS[@]}"))
    echo "=> [$(date)] Shuffled execution order for Iteration $iter: ${SHUFFLED_RUNS[*]}"

    for run in "${SHUFFLED_RUNS[@]}"; do
        if [ "$run" == "vanilla" ]; then
            echo "=> [$(date)] Starting Phase 0: Vanilla Baseline Run (Iteration $iter) on ${TOTAL_TASKS} ranks..."
            srun ${SRUN_OPTS} --ntasks=${TOTAL_TASKS} \
                "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "00_vanilla_run_${iter}.out" 2>&1
        
        elif [ "$run" == "stub" ]; then
            echo "=> [$(date)] Starting Phase 4: Sampi Stub Baseline Run (Iteration $iter) on ${TOTAL_TASKS} ranks..."
            srun --export=ALL,LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampistub.so" ${SRUN_OPTS} --ntasks=${TOTAL_TASKS} \
                "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "03_stub_run_${iter}.out" 2>&1
        
        elif [ "$run" == "boost" ]; then
            echo "=> [$(date)] Starting Phase 2: Sampi Boosted Run (Iteration $iter) on ${TOTAL_TASKS} ranks..."
            srun --export=ALL,LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiboost.so" ${SRUN_OPTS} --ntasks=${TOTAL_TASKS} \
                "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "02_boosted_run_${iter}.out" 2>&1
        fi
    done
done

echo "=> [$(date)] All 5 iterations complete!"
