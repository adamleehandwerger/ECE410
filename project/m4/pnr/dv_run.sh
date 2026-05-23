#!/bin/bash
#SBATCH --job-name=dv_svm_wb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

set -e

SCRATCH=$(ws_find openlane_svm)
CARAVEL=$SCRATCH/caravel_svm_project
CARAVEL_LITE=$SCRATCH/caravel-lite
MCW_ROOT=$SCRATCH/caravel_mgmt_soc_litex
DV_SIF=$SCRATCH/dv.sif
PDK_ROOT=$SCRATCH/pdk

TEST=svm_wb_test

echo "=== dv_run: $TEST on $(hostname) at $(date) ==="

if [ ! -f "$DV_SIF" ]; then
    echo "ERROR: DV SIF not found at $DV_SIF — run dv_setup.sh first"
    exit 1
fi

module load apptainer/1.4.1-gcc-13.4.0

echo "--- git pull caravel ---"
git -C $CARAVEL pull --ff-only || echo "WARNING: git pull failed, using local state"

# Run the DV make inside the efabless/dv container
apptainer exec \
    --bind /scratch,/tmp,/proc \
    --env CARAVEL_ROOT=$CARAVEL_LITE \
    --env MCW_ROOT=$MCW_ROOT \
    --env PDK_ROOT=$PDK_ROOT \
    --env PDK=sky130A \
    --env UPRJ_ROOT=$CARAVEL \
    $DV_SIF \
    bash -c "
        set -e
        cd $CARAVEL/verilog/dv/$TEST
        echo '--- Running: make SIM=RTL TOOLCHAIN=GCC ---'
        make SIM=RTL TOOLCHAIN=GCC 2>&1
        echo '--- RTL sim done ---'
    "

echo "=== dv_run complete at $(date) ==="

# --- Collect log output ---
SIMOUT=$(find $CARAVEL/verilog/dv/$TEST -name "*.log" -newer $CARAVEL/verilog/dv/$TEST/Makefile 2>/dev/null | head -3)
if [ -n "$SIMOUT" ]; then
    echo "=== Simulation logs ==="
    for f in $SIMOUT; do echo "--- $f ---"; cat $f; done
fi
