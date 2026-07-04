#!/bin/bash
#SBATCH --job-name=m6_chip_harden
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

# m6_chip_harden.sh — Full-chip harden of chip_top with sg13g2 pad ring.
# Wraps svm_top_ihp in sg13g2_IOPad ring + bondpads + sealring.
# Requires svm_compute_core macro artifacts from core_harden (jobs 94594).
#
# Run:
#   sbatch chip_harden.sh

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
SVM_M6=$SCRATCH/svm_m6
DESIGN_DIR=$SVM_M6/project/m6/synth
ARTIFACTS=$SCRATCH/svm_m6_artifacts
GDS_STAGE=$SVM_M6/project/m6/pnr/gds
mkdir -p $ARTIFACTS $GDS_STAGE

echo "=== m6_chip_harden: chip_top (IHP SG13G2, Chip flow) on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

# ── Verify core macro artifacts ────────────────────────────────────────
echo "--- Verifying svm_compute_core macro artifacts ---"
for F in svm_compute_core.gds svm_compute_core.lef; do
    if [ ! -f "$ARTIFACTS/$F" ]; then
        echo "ERROR: $ARTIFACTS/$F not found — run core_harden.sh first"
        exit 1
    fi
    ls -lh "$ARTIFACTS/$F"
done

# Stage macro files where chip_config.yaml expects them
cp $ARTIFACTS/svm_compute_core.gds $GDS_STAGE/
cp $ARTIFACTS/svm_compute_core.lef $GDS_STAGE/

# ── Pull latest repo ───────────────────────────────────────────────────
echo "--- git pull ---"
git -C $SVM_M6 pull --ff-only || echo "WARNING: git pull failed, using local state"

# Restore bondpad GDS from artifacts (git-lfs not available on Orca;
# git pull replaces the real GDS with the LFS pointer stub)
BONDPAD_GDS=$SVM_M6/project/m6/ip/sg13g2_ip__bondpad_70x70/final/gds/sg13g2_ip__bondpad_70x70.gds
BONDPAD_CACHE=$ARTIFACTS/sg13g2_ip__bondpad_70x70.gds
if [ -f "$BONDPAD_CACHE" ]; then
    cp $BONDPAD_CACHE $BONDPAD_GDS
    echo "Restored bondpad GDS from artifacts cache ($(ls -lh $BONDPAD_GDS | awk '{print $5}'))"
else
    echo "ERROR: bondpad GDS not in artifacts cache — run: scp <local>/sg13g2_ip__bondpad_70x70.gds orca:$BONDPAD_CACHE"
    exit 1
fi

# ── Verify chip_config.yaml and chip_top.sv exist ─────────────────────
ls -lh $DESIGN_DIR/chip_config.yaml
ls -lh $DESIGN_DIR/chip_top.sdc
ls -lh $SVM_M6/project/m6/rt1/chip_top.sv
ls -lh $SVM_M6/project/m6/rt1/top.sv
ls -lh $SVM_M6/project/m6/rt1/compute_core_bb.v
ls -lh $SVM_M6/project/m6/ip/sg13g2_ip__bondpad_70x70/final/gds/sg13g2_ip__bondpad_70x70.gds
ls -lh $SVM_M6/project/m6/ip/sg13g2_ip__bondpad_70x70/final/lef/sg13g2_ip__bondpad_70x70.lef

# ── LibreLane SIF ──────────────────────────────────────────────────────
module load apptainer/1.4.1-gcc-13.4.0
LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif
if [ ! -f "$LIBRELANE_SIF" ]; then
    echo "ERROR: $LIBRELANE_SIF not found."
    exit 1
fi
apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF librelane --version 2>/dev/null

# ── Run LibreLane Chip flow ────────────────────────────────────────────
RUN_DIR=$DESIGN_DIR/runs/chip_harden
echo "--- Running chip_top Chip flow ---"
RESUME_FLAG=""
if [ -d "$RUN_DIR" ]; then
    echo "Existing run found — resuming from last successful step"
    RESUME_FLAG="--from-last-checkpoint"
fi

apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF \
    librelane \
    --pdk ihp-sg13g2 \
    --run-tag chip_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    $RESUME_FLAG \
    $DESIGN_DIR/chip_config.yaml 2>&1

echo "=== Chip harden done at $(date) ==="

# ── Collect outputs ───────────────────────────────────────────────────
FINAL_GDS=$(find $RUN_DIR -name "*.gds" 2>/dev/null | grep -i "final\|stream" | head -1)
FINAL_LEF=$(find $RUN_DIR -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_GL=$(find $RUN_DIR -name "*.nl.v" -o -name "*.v" 2>/dev/null | grep -i final | head -1)

[ -n "$FINAL_GDS" ] && cp $FINAL_GDS $ARTIFACTS/chip_top.gds     && echo "GDS -> $ARTIFACTS/chip_top.gds"
[ -n "$FINAL_LEF" ] && cp $FINAL_LEF $ARTIFACTS/chip_top.lef     && echo "LEF -> $ARTIFACTS/chip_top.lef"
[ -n "$FINAL_GL"  ] && cp $FINAL_GL  $ARTIFACTS/chip_top.v       && echo "GL  -> $ARTIFACTS/chip_top.v"

echo ""
echo "Timing summary:"
find $RUN_DIR -name "*.rpt" 2>/dev/null | xargs grep -l "wns\|slack" 2>/dev/null | head -3 | \
    xargs -I{} sh -c 'echo "--- {} ---" && grep -E "wns|tns|slack" {} | head -5'

echo ""
echo "DRC summary:"
find $RUN_DIR -name "*drc*" -o -name "*klayout*" 2>/dev/null | grep -i report | head -3 | \
    xargs -I{} sh -c 'echo "--- {} ---" && tail -5 {}'

echo "=== chip_top full-chip harden complete ==="
echo "=== Submit GDS to IHP shuttle after DRC/LVS signoff ==="
