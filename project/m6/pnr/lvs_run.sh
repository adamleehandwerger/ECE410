#!/bin/bash
#SBATCH --job-name=m6_lvs
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

# lvs_run.sh — LVS verification for svm_compute_core and svm_top_ihp
#
# Uses SPICE netlists already extracted by LibreLane's magic-spiceextraction
# step — avoids re-running Magic (which takes 2+ hours flat, or produces
# multi-GB .ext files even hierarchically for 157k cells).
#
# Usage:
#   sbatch lvs_run.sh
#   sbatch --dependency=afterok:<top_harden_jobid> lvs_run.sh
#
# Outputs (in $ARTIFACTS/lvs/):
#   <design>_lvs.txt    — Netgen comparison report
#   <design>_netgen.log — full Netgen stdout

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
ARTIFACTS=$SCRATCH/svm_m6_artifacts
LVS_DIR=$ARTIFACTS/lvs
mkdir -p $LVS_DIR

CORE_RUN=$SCRATCH/svm_m6/project/m6/synth/runs/core_harden
TOP_RUN=$SCRATCH/svm_m6/project/m6/synth/runs/top_harden

module load apptainer/1.4.1-gcc-13.4.0
LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif

# Netgen setup from ihp-open-pdk (confirmed path)
NETGEN_SETUP=$(find $IHP_PDK_ROOT/ihp-sg13g2/libs.tech/netgen -name "*.tcl" 2>/dev/null | head -1)
if [ -z "$NETGEN_SETUP" ]; then
    NETGEN_SETUP=$(find ~/.ciel -name "ihp-sg13g2_setup.tcl" -path "*/netgen/*" 2>/dev/null | head -1)
fi

echo "=== m6 LVS at $(date) ==="
echo "NETGEN_SETUP= $NETGEN_SETUP"
echo "ARTIFACTS   = $ARTIFACTS"

if [ -z "$NETGEN_SETUP" ]; then
    echo "ERROR: Could not find netgen setup — check PDK install"; exit 1
fi

# LibreLane-extracted SPICE (8.4 MB and 12 MB — proper hierarchical extraction)
CORE_SPICE=$(ls $CORE_RUN/*/svm_compute_core.spice 2>/dev/null | tail -1)
TOP_SPICE=$(ls $TOP_RUN/*/svm_top_ihp.spice 2>/dev/null | tail -1)

echo "CORE_SPICE  = $CORE_SPICE"
echo "TOP_SPICE   = $TOP_SPICE"

if [ -z "$CORE_SPICE" ] || [ -z "$TOP_SPICE" ]; then
    echo "ERROR: LibreLane SPICE not found — run core_harden and top_harden first"; exit 1
fi

# ---------------------------------------------------------------------------
# run_lvs <design_name> <top_cell> <spice_file>
# ---------------------------------------------------------------------------
run_lvs() {
    local DESIGN=$1
    local TOP=$2
    local SPICE=$3
    local SCH=$ARTIFACTS/${DESIGN}.v
    local REPORT=$LVS_DIR/${DESIGN}_lvs.txt
    local NETGEN_LOG=$LVS_DIR/${DESIGN}_netgen.log

    echo ""
    echo "=== LVS: $DESIGN (top cell: $TOP) ==="
    echo "  SPICE: $SPICE ($(wc -l < $SPICE) lines)"
    echo "  SCH  : $SCH"

    if [ ! -f "$SCH" ]; then
        echo "  ERROR: $SCH not found — run harden first"; exit 1
    fi

    echo "  [1/1] Netgen: layout SPICE vs GL Verilog"
    apptainer exec --bind /scratch,/home,/tmp $LIBRELANE_SIF \
        netgen -batch lvs \
            "$SPICE $TOP" \
            "$SCH $TOP" \
            $NETGEN_SETUP \
            $REPORT \
        2>&1 | tee $NETGEN_LOG

    echo ""
    echo "  --- LVS Result ($DESIGN) ---"
    if grep -q "Circuits match uniquely" $REPORT 2>/dev/null; then
        echo "  [PASS] $DESIGN LVS: CLEAN"
    elif grep -q "Netlists do not match" $REPORT 2>/dev/null; then
        echo "  [FAIL] $DESIGN LVS: MISMATCH"
        grep -A 10 "Netlists do not match" $REPORT | head -15
    else
        echo "  [UNKNOWN] No match/mismatch keyword — inspect $REPORT"
        tail -20 $REPORT
    fi
}

# ---------------------------------------------------------------------------
# Run LVS for both designs
# ---------------------------------------------------------------------------
run_lvs svm_compute_core svm_compute_core "$CORE_SPICE"
run_lvs svm_top_ihp      svm_top_ihp      "$TOP_SPICE"

echo ""
echo "=== LVS complete at $(date) ==="
echo "Reports in $LVS_DIR/:"
ls -lh $LVS_DIR/
