#!/bin/bash
# resubmit_chain.sh — Resubmit full m6 job chain after a failed core_harden
#
# Run interactively on Orca after job 94289 exits with DRT violations:
#   bash project/m6/pnr/resubmit_chain.sh
#
# Pulls latest git (picks up 58% density + DRT_OPT_ITERS=128 in core_config.json)
# then chains: core_harden → top_harden (afterok) → lvs_run (afterok top)
#                          └→ fast_hold_1p65V (afterok)

set -e

SCRATCH=$(ws_find openlane_svm)
SVM_M6=$SCRATCH/svm_m6

echo "=== m6 resubmit at $(date) ==="
echo "SCRATCH=$SCRATCH"

# Pull latest git (contains 58% density + DRT_OPT_ITERS=128)
echo "--- git pull ---"
git -C $SVM_M6 pull --ff-only
echo "HEAD = $(git -C $SVM_M6 log --oneline -1)"

# Cancel any pending dependents from the previous failed run
echo "--- Cancelling stale pending jobs ---"
scancel --user=funphin --state=PENDING 2>/dev/null || true

# Submit chain
CORE_JID=$(sbatch --parsable $SVM_M6/project/m6/pnr/core_harden.sh)
echo "core_harden      job $CORE_JID"

TOP_JID=$(sbatch --parsable --dependency=afterok:$CORE_JID $SVM_M6/project/m6/pnr/top_harden.sh)
echo "top_harden       job $TOP_JID (afterok:$CORE_JID)"

LVS_JID=$(sbatch --parsable --dependency=afterok:$TOP_JID $SVM_M6/project/m6/pnr/lvs_run.sh)
echo "lvs_run          job $LVS_JID (afterok:$TOP_JID)"

HOLD_JID=$(sbatch --parsable --dependency=afterok:$CORE_JID $SVM_M6/project/m6/pnr/fast_hold_1p65V.sh)
echo "fast_hold_1p65V  job $HOLD_JID (afterok:$CORE_JID)"

echo ""
echo "=== Submitted at $(date) ==="
echo "Monitor with: squeue -u funphin"
