#!/bin/bash
# recover_workspace.sh — Restore expired openlane_svm workspace and re-submit jobs
#
# Run this interactively on Orca (not via sbatch):
#   bash project/m6/pnr/recover_workspace.sh
#
# What it does:
#   1. Allocates a fresh openlane_svm workspace (60 days)
#   2. Copies librelane SIF, PDK, and all artifacts from .removed backup
#   3. Pulls latest git (bug-fixed scripts)
#   4. Submits: top_harden → (afterok) lvs_run, fast_hold_1p65V

set -e

REMOVED=/scratch/.removed/funphin-openlane_svm-1781770202

echo "=== Recovering openlane_svm workspace at $(date) ==="

# 1. Allocate fresh workspace
ws_allocate openlane_svm 60
SCRATCH=$(ws_find openlane_svm)
echo "SCRATCH = $SCRATCH"

# 2. Copy data from .removed backup
echo ""
echo "--- Copying SIF (1.4 GB) ---"
cp $REMOVED/librelane_3.0.4.sif $SCRATCH/

echo "--- Copying ihp-open-pdk ---"
cp -r $REMOVED/ihp-open-pdk $SCRATCH/

echo "--- Copying svm_m6 repo ---"
cp -r $REMOVED/svm_m6 $SCRATCH/

echo "--- Copying svm_m6_artifacts ---"
cp -r $REMOVED/svm_m6_artifacts $SCRATCH/

echo "--- Copy complete ---"
du -sh $SCRATCH/*/

# 3. Pull latest git (contains bug-fixed scripts)
echo ""
echo "--- Pulling latest git ---"
cd $SCRATCH/svm_m6
git pull origin main
echo "HEAD = $(git log --oneline -1)"

# 4. Submit jobs
echo ""
echo "--- Submitting jobs ---"

# top_harden (Verilator BB stub fix applied in top_config.json)
TOP_JID=$(sbatch --parsable project/m6/pnr/top_harden.sh)
echo "top_harden       job $TOP_JID"

# lvs_run depends on top_harden (needs top GDS) AND core harden (already done)
LVS_JID=$(sbatch --parsable --dependency=afterok:$TOP_JID project/m6/pnr/lvs_run.sh)
echo "lvs_run          job $LVS_JID (afterok:$TOP_JID)"

# fast_hold is independent — uses existing core_harden ODB
HOLD_JID=$(sbatch --parsable project/m6/pnr/fast_hold_1p65V.sh)
echo "fast_hold_1p65V  job $HOLD_JID"

echo ""
echo "=== Submitted at $(date) ==="
echo "Monitor with: squeue -u funphin"
echo "Logs: \$SCRATCH/svm_m6/<job>.out or \$ARTIFACTS/<design>/"
