#!/bin/bash
#SBATCH --job-name=mpw_precheck
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

set -e

SCRATCH=$(ws_find openlane_svm)
CARAVEL=$SCRATCH/caravel_svm_project
PDK_ROOT=$SCRATCH/pdk
PRECHECK_SIF=$SCRATCH/mpw-precheck.sif

echo "=== mpw_precheck on $(hostname) at $(date) ==="

# --- Pull latest repo state (reset to origin/main; skip LFS smudge) ---
# GDS files tracked by git-lfs — skip smudge to avoid downloading stale pointers.
# Always restore from canonical OL2 run-directory outputs after the reset.
GDS_CORE=$CARAVEL/gds/svm_compute_core.gds
GDS_WRAP=$CARAVEL/gds/user_project_wrapper.gds

# Canonical GDS sources in the OL2 run dirs (authoritative, always full-size)
OL_GDS_CORE=$(find $CARAVEL/openlane/svm_compute_core/runs/core_harden -name 'svm_compute_core.gds' 2>/dev/null | xargs ls -S 2>/dev/null | head -1)
OL_GDS_WRAP=$(find $CARAVEL/openlane/user_project_wrapper/runs/wrapper_harden -name 'user_project_wrapper.gds' 2>/dev/null | xargs ls -S 2>/dev/null | head -1)

echo "GDS sources: core=$OL_GDS_CORE wrap=$OL_GDS_WRAP"

rm -f $CARAVEL/.git/refs/remotes/origin/main.lock $CARAVEL/.git/index.lock
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL fetch origin
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL reset --hard origin/main

# Restore real GDS from OL2 run dirs unconditionally (avoids stale/partial backup)
[ -n "$OL_GDS_CORE" ] && [ -f "$OL_GDS_CORE" ] && cp "$OL_GDS_CORE" "$GDS_CORE" && echo "Restored core GDS ($(du -sh $GDS_CORE | cut -f1))"
[ -n "$OL_GDS_WRAP" ] && [ -f "$OL_GDS_WRAP" ] && cp "$OL_GDS_WRAP" "$GDS_WRAP" && echo "Restored wrapper GDS ($(du -sh $GDS_WRAP | cut -f1))"

# --- Verify required artifacts exist ---
echo "--- Checking required files ---"
PASS=1
for f in \
    $CARAVEL/gds/svm_compute_core.gds \
    $CARAVEL/gds/user_project_wrapper.gds \
    $CARAVEL/lef/svm_compute_core.lef \
    $CARAVEL/lef/user_project_wrapper.lef \
    $CARAVEL/verilog/gl/svm_compute_core.v \
    $CARAVEL/verilog/gl/user_project_wrapper.v \
    $CARAVEL/info.yaml; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename $f) ($(du -sh $f | cut -f1))"
    else
        echo "  MISSING: $f"
        PASS=0
    fi
done
[ $PASS -eq 0 ] && { echo "FAIL: required artifacts missing"; exit 1; }

module load apptainer/1.4.1-gcc-13.4.0

if [ ! -f $PRECHECK_SIF ]; then
    echo "--- Pulling mpw-precheck container ---"
    apptainer pull $PRECHECK_SIF docker://efabless/mpw_precheck:latest
fi

mkdir -p $CARAVEL/precheck_results
RESULTS=$CARAVEL/precheck_results/precheck.log
> $RESULTS

# --- Magic DRC on svm_compute_core ---
# Write TCL into a location that is bind-mounted into the container (/project)
echo "--- Running Magic DRC on svm_compute_core ---" | tee -a $RESULTS
cat > $CARAVEL/drc_core.tcl << 'MAGICEOF'
drc off
gds read /project/gds/svm_compute_core.gds
load svm_compute_core
drc on
drc check
set drc_count [drc list count total]
puts "DRC error count: $drc_count"
if {$drc_count == 0} { puts "PASS: svm_compute_core DRC clean" } \
else { puts "FAIL: svm_compute_core DRC $drc_count errors" }
quit -noprompt
MAGICEOF

apptainer exec \
    --bind $CARAVEL:/project \
    --bind $PDK_ROOT:/pdk \
    $PRECHECK_SIF \
    bash -c "cd /project && magic -dnull -noconsole -rcfile /pdk/sky130A/libs.tech/magic/sky130A.magicrc /project/drc_core.tcl" \
    2>&1 | grep -E "DRC|PASS|FAIL|error count" | tee -a $RESULTS
rm -f $CARAVEL/drc_core.tcl

# --- SPDX license check ---
echo "--- Checking SPDX headers ---" | tee -a $RESULTS
MISSING_SPDX=0
for f in $CARAVEL/verilog/rtl/svm_compute_core.sv $CARAVEL/verilog/rtl/user_project_wrapper.sv; do
    if grep -q "SPDX" "$f" 2>/dev/null; then
        echo "  OK SPDX: $(basename $f)" | tee -a $RESULTS
    else
        echo "  WARN no SPDX: $(basename $f)" | tee -a $RESULTS
    fi
done

echo "=== precheck done at $(date) ===" | tee -a $RESULTS
echo "=== Summary ==="
grep -E "PASS|FAIL|OK|MISSING|WARN" $RESULTS
