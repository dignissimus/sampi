#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./setup.sh <path_to_config.cfg>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(readlink -f "$1")"
BASE_DIR="$SCRIPT_DIR/motorbike_base"

source "$CONFIG_FILE"

echo "=== Loading OpenFOAM environment ==="
module restore
module load $FOAM_MODULE
eval "$FOAM_SOURCE_CMD"

echo "=== Copying Motorbike tutorial to base directory ==="
rm -rf "$BASE_DIR"
cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/motorBike" "$BASE_DIR"

echo "=== Injecting Mesh & Decomposition Configs ==="
sed -i "s/(20 8 8)/$MESH_DIMS/g" "$BASE_DIR/system/blockMeshDict"
sed -i "s/numberOfSubdomains [0-9]\+;/numberOfSubdomains $SUBDOMAINS;/g" "$BASE_DIR/system/decomposeParDict"
sed -i "s/n[[:space:]]\+([0-9]\+[[:space:]]\+[0-9]\+[[:space:]]\+[0-9]\+);/$HIERARCHICAL_COEFFS/g" "$BASE_DIR/system/decomposeParDict"

echo "=== Submitting Meshing Job to Slurm ==="
sbatch $SLURM_SETUP_ARGS --job-name=OF_Setup --export=ALL,CONFIG_FILE="$CONFIG_FILE" "$SCRIPT_DIR/setup_job.sh"

echo "===================================================="
echo " Setup job submitted using $1"
echo " Use 'squeue -u \$USER' to monitor its status."
echo " DO NOT run dispatch.sh until the setup job completes."
echo "===================================================="
