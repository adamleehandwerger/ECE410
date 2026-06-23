#!/bin/bash
#SBATCH --job-name=m6_top_harden
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

# m6_top_harden.sh — Harden svm_top_ihp with IHP SG13G2 PDK on Orca
#
# Run AFTER core_harden.sh completes successfully (or via SLURM dependency):
#   sbatch --dependency=afterok:<core_harden_jobid> top_harden.sh
#
# Requires $ARTIFACTS/svm_compute_core.{gds,lef,v} from core_harden.
# Treats svm_compute_core as a hardened macro (black-box).
#
# Prerequisites (same as core_harden.sh):
#   $SCRATCH/librelane_3.0.4.sif   — LibreLane 3.0.4 Apptainer SIF
#   $SCRATCH/ihp-open-pdk          — IHP-Open-PDK cloned
#   $SCRATCH/svm_m6                — ECE410 repo (latest pull)

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
SVM_M6=$SCRATCH/svm_m6
DESIGN_DIR=$SVM_M6/project/m6/synth
ARTIFACTS=$SCRATCH/svm_m6_artifacts
GDS_STAGE=$SVM_M6/project/m6/pnr/gds
mkdir -p $ARTIFACTS $GDS_STAGE

echo "=== m6_top_harden: svm_top_ihp (IHP SG13G2) on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"
echo "IHP_PDK_ROOT=$IHP_PDK_ROOT"

# --- Verify core macro artifacts ---
echo "--- Verifying core macro artifacts ---"
for F in svm_compute_core.gds svm_compute_core.lef svm_compute_core.v; do
    if [ ! -f "$ARTIFACTS/$F" ]; then
        echo "ERROR: $ARTIFACTS/$F not found — run core_harden.sh first"
        exit 1
    fi
    ls -lh "$ARTIFACTS/$F"
done

# --- Stage macro files where top_config.json expects them ---
cp $ARTIFACTS/svm_compute_core.gds $GDS_STAGE/
cp $ARTIFACTS/svm_compute_core.lef $GDS_STAGE/
cp $ARTIFACTS/svm_compute_core.v   $GDS_STAGE/

# --- Black-box stub is committed in rt1/ — just verify it exists ---
if [ ! -f "$SVM_M6/project/m6/rt1/compute_core_bb.v" ]; then
    echo "ERROR: compute_core_bb.v not found in rt1/ — check repo"
    exit 1
fi
echo "Black-box stub: $(ls -lh $SVM_M6/project/m6/rt1/compute_core_bb.v)"

# --- Pull latest m6 RTL ---
echo "--- git pull svm_m6 ---"
git -C $SVM_M6 pull --ff-only || echo "WARNING: git pull failed, using local state"

# --- LibreLane SIF ---
module load apptainer/1.4.1-gcc-13.4.0

LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif
if [ ! -f "$LIBRELANE_SIF" ]; then
    echo "ERROR: $LIBRELANE_SIF not found."
    exit 1
fi
echo "LibreLane SIF: $(ls -lh $LIBRELANE_SIF)"
apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF librelane --version 2>/dev/null

# --- Verify inputs ---
echo "--- Top-level inputs ---"
ls -lh $DESIGN_DIR/top_config.json
ls -lh $DESIGN_DIR/svm_top_ihp.sdc
ls -lh $SVM_M6/project/m6/rt1/top.sv
ls -lh $SVM_M6/project/m6/rt1/compute_core_bb.v
ls -lh $GDS_STAGE/svm_compute_core.lef
ls -lh $GDS_STAGE/svm_compute_core.gds

# --- DRT checkpoint resume (preserve routing if it exists) ---
RUN_DIR=$DESIGN_DIR/runs/top_harden
DRT_ODB=$(find $RUN_DIR/44-openroad-detailedrouting -name "*.odb" 2>/dev/null | head -1)
if [ -n "$DRT_ODB" ]; then
    echo "--- DRT checkpoint found: $DRT_ODB — resuming from OpenROAD.FillInsertion ---"
    RESUME_FROM="--from OpenROAD.FillInsertion"
else
    echo "--- No DRT checkpoint — running from scratch ---"
    rm -rf $RUN_DIR
    RESUME_FROM=""
fi

# --- Run LibreLane ---
echo "--- Running librelane (svm_top_ihp, IHP SG13G2, macro=svm_compute_core) ---"
apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF \
    librelane \
    --pdk ihp-sg13g2 \
    --run-tag top_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    $RESUME_FROM \
    $DESIGN_DIR/top_config.json 2>&1

echo "=== Top harden done at $(date) ==="

# --- Collect outputs ---
echo "=== Output artifacts ==="
FINAL_GDS=$(find $RUN_DIR -name "*.gds" 2>/dev/null | grep -i final | head -1)
FINAL_LEF=$(find $RUN_DIR -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_GL=$(find $RUN_DIR -name "*.nl.v" -o -name "*.v" 2>/dev/null | grep -i final | head -1)

[ -n "$FINAL_GDS" ] && cp $FINAL_GDS $ARTIFACTS/svm_top_ihp.gds && echo "GDS -> $ARTIFACTS/"
[ -n "$FINAL_LEF" ] && cp $FINAL_LEF $ARTIFACTS/svm_top_ihp.lef && echo "LEF -> $ARTIFACTS/"
[ -n "$FINAL_GL"  ] && cp $FINAL_GL  $ARTIFACTS/svm_top_ihp.v   && echo "GL  -> $ARTIFACTS/"

echo ""
echo "Timing summary:"
find $RUN_DIR -name "*.rpt" 2>/dev/null | xargs grep -l "wns\|slack" 2>/dev/null | head -3 | \
    xargs -I{} sh -c 'echo "--- {} ---" && grep -E "wns|tns|slack" {} | head -5'

echo ""
echo "DRC summary:"
find $RUN_DIR -name "*drc*" -o -name "*klayout*" 2>/dev/null | grep -i report | head -3 | \
    xargs -I{} sh -c 'echo "--- {} ---" && tail -10 {}'

echo ""
echo "=== m6 top harden complete — artifacts in $ARTIFACTS/ ==="
echo "=== Run KLayout DRC to confirm 0 violations ==="
echo "  klayout -b -r \$IHP_PDK_ROOT/ihp-sg13g2/libs.tech/klayout/tech/drc/sg13g2.lydrc \\"
echo "          -rd input=$ARTIFACTS/svm_top_ihp.gds"
