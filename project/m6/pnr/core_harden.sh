#!/bin/bash
#SBATCH --job-name=m6_core_harden
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

# m6_core_harden.sh — Harden svm_compute_core with IHP SG13G2 PDK on Orca
#
# Prerequisites (run setup_ihp_pdk.sh first):
#   $SCRATCH/ihp-open-pdk   — IHP-Open-PDK cloned
#   $SCRATCH/svm_m6         — ECE410 repo cloned (or latest pull)
#   $SCRATCH/ol2_venv_mf    — OpenLane 2 venv (reused from m5)
#   $SCRATCH/openlane2.sif  — OpenLane 2 Apptainer SIF (reused from m5)
#   $SCRATCH/.nix           — nix-portable with yosys-with-plugins (reused from m5)
#
# Output: GDS/LEF/GL written to $SCRATCH/svm_m6_artifacts/

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
VOLARE_PDK_ROOT=$SCRATCH/pdk
SVM_M6=$SCRATCH/svm_m6
DESIGN_DIR=$SVM_M6/project/m6/synth
ARTIFACTS=$SCRATCH/svm_m6_artifacts
mkdir -p $ARTIFACTS

echo "=== m6_core_harden: svm_compute_core (IHP SG13G2) on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"
echo "IHP_PDK_ROOT=$IHP_PDK_ROOT"

# --- Verify PDK ---
if [ ! -d "$IHP_PDK_ROOT" ]; then
    echo "ERROR: IHP PDK not found at $IHP_PDK_ROOT"
    echo "  Run setup_ihp_pdk.sh first."
    exit 1
fi

# --- Pull latest m6 RTL ---
echo "--- git pull svm_m6 ---"
git -C $SVM_M6 pull --ff-only || echo "WARNING: git pull failed, using local state"

# --- LibreLane SIF (ships its own Yosys/OpenROAD/KLayout) ---
module load apptainer/1.4.1-gcc-13.4.0

LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif
if [ ! -f "$LIBRELANE_SIF" ]; then
    echo "ERROR: $LIBRELANE_SIF not found. Run pull_librelane job first."
    exit 1
fi
echo "LibreLane SIF: $(ls -lh $LIBRELANE_SIF)"
apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF librelane --version 2>/dev/null

# --- Verify inputs ---
echo "--- Design inputs ---"
ls -lh $DESIGN_DIR/core_config.json
ls -lh $DESIGN_DIR/svm_compute_core.sdc
ls -lh $SVM_M6/project/m6/rt1/compute_core.sv
ls -lh $SVM_M6/project/m6/rt1/interface.sv

# --- Clean previous run ---
RUN_DIR=$DESIGN_DIR/runs/core_harden
echo "--- Removing old run dir ---"
rm -rf $RUN_DIR

# --- Run OpenLane 2 ---
echo "--- Running librelane inside SIF (IHP SG13G2, NUM_SV=600, RAM_LATENCY=3) ---"
apptainer exec --bind /scratch,/tmp $LIBRELANE_SIF \
    librelane \
    --pdk ihp-sg13g2 \
    --run-tag core_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    $DESIGN_DIR/core_config.json 2>&1

echo "=== Core harden done at $(date) ==="

# --- Collect outputs ---
echo "=== Output artifacts ==="
FINAL_GDS=$(find $RUN_DIR -name "*.gds" 2>/dev/null | grep -i final | head -1)
FINAL_LEF=$(find $RUN_DIR -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_GL=$(find $RUN_DIR -name "*.nl.v" -o -name "*.v" 2>/dev/null | grep -i final | head -1)

[ -n "$FINAL_GDS" ] && cp $FINAL_GDS $ARTIFACTS/svm_compute_core.gds && echo "GDS -> $ARTIFACTS/"
[ -n "$FINAL_LEF" ] && cp $FINAL_LEF $ARTIFACTS/svm_compute_core.lef && echo "LEF -> $ARTIFACTS/"
[ -n "$FINAL_GL"  ] && cp $FINAL_GL  $ARTIFACTS/svm_compute_core.v   && echo "GL  -> $ARTIFACTS/"

echo ""
echo "Timing summary:"
find $RUN_DIR -name "*.rpt" 2>/dev/null | xargs grep -l "wns\|slack" 2>/dev/null | head -3 | \
    xargs -I{} sh -c 'echo "--- {} ---" && grep -E "wns|tns|slack" {} | head -5'

echo ""
echo "=== Next step: sbatch $SVM_M6/project/m6/pnr/top_harden.sh ==="
