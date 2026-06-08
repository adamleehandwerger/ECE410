#!/bin/bash
#SBATCH --job-name=dv_setup
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

set -e

SCRATCH=$(ws_find openlane_svm)
CARAVEL=$SCRATCH/caravel_svm_project
MPW_TAG=2024.09.12-1

echo "=== dv_setup on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

module load apptainer/1.4.1-gcc-13.4.0

# --- Pull DV container ---
DV_SIF=$SCRATCH/dv.sif
if [ ! -f "$DV_SIF" ]; then
    echo "--- Pulling efabless/dv Docker image ---"
    apptainer pull $DV_SIF docker://efabless/dv:latest
else
    echo "DV SIF already present: $DV_SIF"
fi

# --- Clone caravel-lite ---
CARAVEL_LITE=$SCRATCH/caravel-lite
if [ ! -d "$CARAVEL_LITE" ]; then
    echo "--- Cloning caravel-lite @ $MPW_TAG ---"
    git clone --depth=1 --branch $MPW_TAG \
        https://github.com/efabless/caravel-lite.git \
        $CARAVEL_LITE
else
    echo "caravel-lite already at $CARAVEL_LITE"
fi

# --- Clone caravel_mgmt_soc_litex (MCW) ---
MCW_ROOT=$SCRATCH/caravel_mgmt_soc_litex
if [ ! -d "$MCW_ROOT" ]; then
    echo "--- Cloning caravel_mgmt_soc_litex @ $MPW_TAG ---"
    git clone --depth=1 --branch $MPW_TAG \
        https://github.com/efabless/caravel_mgmt_soc_litex.git \
        $MCW_ROOT
else
    echo "MCW already at $MCW_ROOT"
fi

echo ""
echo "=== dv_setup complete at $(date) ==="
echo "Next: sbatch dv_run.sh"
echo "  DV_SIF=$DV_SIF"
echo "  CARAVEL_LITE=$CARAVEL_LITE"
echo "  MCW_ROOT=$MCW_ROOT"
