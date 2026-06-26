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
MPI_GLOBAL_OPTS=""

echo "SLURM JOB ID:  ${SLURM_JOB_ID}" >> "${EXP_DIR}/experiment_metadata.log"
echo "ALLOC NODES:   ${SLURM_JOB_NODELIST}" >> "${EXP_DIR}/experiment_metadata.log"

cd "${EXP_DIR}" || exit 1

for i in {1..5}; do
    echo "=========================================================="
    echo " ITERATION $i / 5"
    echo "=========================================================="

    # Shuffle execution order to mitigate node variance
    RUNS=("vanilla" "stub")
    
    if [ "$SCALE" -lt 24 ]; then
        RUNS+=("profile" "boost")
    fi

    RUNS=( $(shuf -e "${RUNS[@]}") )

    for run_type in "${RUNS[@]}"; do
        if [ "$run_type" == "vanilla" ]; then
            echo "=> [$(date)] Starting Vanilla Baseline Run on ${TOTAL_TASKS} ranks..."
            mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "00_vanilla_run_${i}.out" 2>&1

        elif [ "$run_type" == "stub" ]; then
            echo "=> [$(date)] Starting Sampi Stub Run on ${TOTAL_TASKS} ranks..."
            LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampistub.so" mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "03_stub_run_${i}.out" 2>&1

        elif [ "$run_type" == "profile" ]; then
            echo "=> [$(date)] Starting Sampi Profiling Run on ${TOTAL_TASKS} ranks..."
            LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiprofile.so" mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "01_profile_run_${i}.out" 2>&1

            if [ ! -f "sampi_communication_profile.txt" ]; then
                echo "ERROR: sampi_communication_profile.txt was not generated! Aborting."
                exit 1
            fi

        elif [ "$run_type" == "boost" ]; then
            echo "=> [$(date)] Starting Sampi Boosted Run on ${TOTAL_TASKS} ranks..."
            LD_PRELOAD="${SAMPI_BUILD_DIR}/libsampiboost.so" mpirun ${MPI_GLOBAL_OPTS} -np ${TOTAL_TASKS} "${GRAPH500_EXEC}" ${SCALE} ${EDGEFACTOR} > "02_boosted_run_${i}.out" 2>&1
        fi
    done
done
