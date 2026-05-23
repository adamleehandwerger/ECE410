#!/bin/bash
#SBATCH --job-name=wrapper_harden
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/funphin/wrapper_harden_%j.log

# ============================================================
# user_project_wrapper hardening — OpenLane 2 (Caravel sky130A)
# Requires:
#   /scratch/funphin/openlane2.sif
#   /scratch/funphin/pdk/sky130A   (PDK_ROOT=/scratch/funphin/pdk)
#   caravel repo at /scratch/funphin/caravel_svm_project
# ============================================================

set -euo pipefail

SCRATCH=/scratch/funphin
CARAVEL=$SCRATCH/caravel_svm_project
OL2_SIF=$SCRATCH/openlane2.sif
PDK_ROOT=$SCRATCH/pdk
DESIGN_DIR=$CARAVEL/openlane/user_project_wrapper

echo "[wrapper_harden] Starting at $(date)"
echo "[wrapper_harden] Design dir : $DESIGN_DIR"
echo "[wrapper_harden] PDK root   : $PDK_ROOT"

# Pull latest caravel changes before running
cd $CARAVEL
git pull --ff-only || echo "[wrapper_harden] WARNING: git pull failed, continuing with local state"

# Run OpenLane 2 inside the Singularity container
singularity exec \
    --bind $CARAVEL:$CARAVEL \
    --bind $PDK_ROOT:$PDK_ROOT \
    --env PDK_ROOT=$PDK_ROOT \
    --env PDK=sky130A \
    $OL2_SIF \
    python3 -m openlane \
        --pdk-root $PDK_ROOT \
        --pdk sky130A \
        $DESIGN_DIR/config.json

echo "[wrapper_harden] Finished at $(date)"

# Copy outputs to a timestamped results directory
RESULTS=$SCRATCH/wrapper_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS

# Find the latest run directory OpenLane 2 created
LATEST_RUN=$(ls -td $DESIGN_DIR/runs/* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo "[wrapper_harden] OpenLane run dir: $LATEST_RUN"

    # Copy GDS/LEF/GL to caravel artifact directories
    GDS=$(find $LATEST_RUN -name "*.gds" | head -1)
    LEF=$(find $LATEST_RUN -name "*.lef" | grep -v "pdn" | head -1)
    GL=$(find $LATEST_RUN -name "*.nl.v" -o -name "*.v" | grep -i "final\|gl" | head -1)

    [ -n "$GDS" ] && cp $GDS $CARAVEL/gds/user_project_wrapper.gds  && echo "GDS -> $CARAVEL/gds/"
    [ -n "$LEF" ] && cp $LEF $CARAVEL/lef/user_project_wrapper.lef  && echo "LEF -> $CARAVEL/lef/"
    [ -n "$GL"  ] && cp $GL  $CARAVEL/verilog/gl/user_project_wrapper.v && echo "GL  -> $CARAVEL/verilog/gl/"

    cp -r $LATEST_RUN $RESULTS/openlane_run
fi

echo "[wrapper_harden] Results saved to $RESULTS"
echo "[wrapper_harden] Done."
