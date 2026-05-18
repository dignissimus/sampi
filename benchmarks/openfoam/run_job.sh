#!/bin/bash

source "$CONFIG_FILE"

module purge
module load $FOAM_MODULE
eval "$FOAM_SOURCE_CMD"

RUN_TYPE=$1
TRIAL_NUM=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/output"

BOOST_LIB="$SCRIPT_DIR/../../build/libsampiboost.so"
PROFILE_LIB="$SCRIPT_DIR/../../build/libsampiprofile.so"

cd "$SCRIPT_DIR/motorbike_base"

# Clean up step data but keep the decomposed mesh chunks
rm -rf [1-9]* processor*/*/[1-9]* postProcessing

if [ "$RUN_TYPE" == "profile" ]; then
    mpirun -x LD_PRELOAD="$PROFILE_LIB" simpleFoam -parallel > "$OUT_DIR/log.profile"
    exec_time=$(grep "ClockTime" "$OUT_DIR/log.profile" | tail -n 1 | awk '{print $3}')
    echo "$exec_time" > "$OUT_DIR/result.profile.txt"

elif [ "$RUN_TYPE" == "boost" ]; then
    mpirun -x LD_PRELOAD="$BOOST_LIB" simpleFoam -parallel > "$OUT_DIR/log.boost.${TRIAL_NUM}"
    exec_time=$(grep "ClockTime" "$OUT_DIR/log.boost.${TRIAL_NUM}" | tail -n 1 | awk '{print $3}')
    echo "$exec_time" > "$OUT_DIR/result.boost.${TRIAL_NUM}.txt"

else
    mpirun simpleFoam -parallel > "$OUT_DIR/log.normal.${TRIAL_NUM}"
    exec_time=$(grep "ClockTime" "$OUT_DIR/log.normal.${TRIAL_NUM}" | tail -n 1 | awk '{print $3}')
    echo "$exec_time" > "$OUT_DIR/result.normal.${TRIAL_NUM}.txt"
fi
