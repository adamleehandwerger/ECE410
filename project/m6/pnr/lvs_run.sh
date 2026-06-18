#!/bin/bash
#SBATCH --job-name=m6_lvs
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

# lvs_run.sh — LVS verification for svm_compute_core and svm_top_ihp
#
# Runs Magic (GDS→SPICE) + Netgen (SPICE vs GL Verilog) inside the
# LibreLane SIF.  Requires completed harden artifacts in $ARTIFACTS/.
#
# Usage:
#   sbatch lvs_run.sh
#   sbatch --dependency=afterok:<top_harden_jobid> lvs_run.sh
#
# Outputs (in $ARTIFACTS/lvs/):
#   <design>_extracted.spice  — Magic-extracted SPICE netlist
#   <design>_lvs.txt          — Netgen comparison report
#   <design>_netgen.log       — full Netgen stdout

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
ARTIFACTS=$SCRATCH/svm_m6_artifacts
LVS_DIR=$ARTIFACTS/lvs
mkdir -p $LVS_DIR

module load apptainer/1.4.1-gcc-13.4.0
LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif

# IHP magic / netgen support files (resolved at runtime to handle PDK layout variations)
MAGIC_TECH=$(find $IHP_PDK_ROOT/ihp-sg13g2/libs.tech/magic -name "sg13g2.tech" | head -1)
NETGEN_SETUP=$(find $IHP_PDK_ROOT/ihp-sg13g2/libs.tech/netgen -name "*.tcl" | head -1)

echo "=== m6 LVS at $(date) ==="
echo "MAGIC_TECH  = $MAGIC_TECH"
echo "NETGEN_SETUP= $NETGEN_SETUP"
echo "ARTIFACTS   = $ARTIFACTS"

# ---------------------------------------------------------------------------
# run_lvs <design_name> <top_cell>
# ---------------------------------------------------------------------------
run_lvs() {
    local DESIGN=$1
    local TOP=$2
    local GDS=$ARTIFACTS/${DESIGN}.gds
    local SCH=$ARTIFACTS/${DESIGN}.v
    local SPICE=$LVS_DIR/${DESIGN}_extracted.spice
    local REPORT=$LVS_DIR/${DESIGN}_lvs.txt
    local NETGEN_LOG=$LVS_DIR/${DESIGN}_netgen.log

    echo ""
    echo "=== LVS: $DESIGN (top cell: $TOP) ==="

    for F in "$GDS" "$SCH"; do
        if [ ! -f "$F" ]; then
            echo "  ERROR: $F not found — run harden first"; exit 1
        fi
    done

    # ------------------------------------------------------------------
    # Stage 1: Magic GDS → SPICE extraction
    # ------------------------------------------------------------------
    cat > /tmp/magic_extract_${DESIGN}.tcl << MAGIC_EOF
gds read ${GDS}
load ${TOP}
flatten ${TOP}
extract all
ext2spice hierarchy on
ext2spice format ngspice
ext2spice -o ${SPICE}
quit
MAGIC_EOF

    echo "  [1/2] Magic: $GDS → $SPICE"
    apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF \
        magic -T $MAGIC_TECH -noconsole -dnull \
        < /tmp/magic_extract_${DESIGN}.tcl 2>&1 \
        | tee $LVS_DIR/${DESIGN}_magic.log

    if [ ! -s "$SPICE" ]; then
        echo "  ERROR: Magic produced empty or missing SPICE — check magic.log"; exit 1
    fi
    echo "  Extracted: $(wc -l < $SPICE) lines → $SPICE"

    # ------------------------------------------------------------------
    # Stage 2: Netgen — SPICE (layout) vs Verilog GL (schematic)
    # ------------------------------------------------------------------
    echo "  [2/2] Netgen: $SPICE vs $SCH"
    apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF \
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
        echo "  First mismatch:"
        grep -A 10 "Netlists do not match" $REPORT | head -15
    else
        echo "  [UNKNOWN] No match/mismatch keyword — inspect $REPORT"
        tail -20 $REPORT
    fi
}

# ---------------------------------------------------------------------------
# Run LVS for both designs
# ---------------------------------------------------------------------------
run_lvs svm_compute_core svm_compute_core
run_lvs svm_top_ihp      svm_top_ihp

echo ""
echo "=== LVS complete at $(date) ==="
echo "Reports in $LVS_DIR/:"
ls -lh $LVS_DIR/
