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
# Preserve real GDS files — git reset --hard replaces them with LFS pointer stubs
GDS_CORE=$CARAVEL/gds/svm_compute_core.gds
GDS_WRAP=$CARAVEL/gds/user_project_wrapper.gds
GDS_BACKUP=/tmp/svm_gds_backup
OL_GDS=$CARAVEL/openlane/svm_compute_core/runs/core_harden/51-magic-streamout/svm_compute_core.gds

mkdir -p $GDS_BACKUP
[ -f $GDS_CORE ] && cp $GDS_CORE $GDS_BACKUP/
[ -f $GDS_WRAP ] && cp $GDS_WRAP $GDS_BACKUP/

rm -f $CARAVEL/.git/refs/remotes/origin/main.lock $CARAVEL/.git/index.lock
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL fetch origin
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL reset --hard origin/main

# Restore real GDS if reset left LFS pointer stubs (< 1 MB)
if [ $(stat -c%s $GDS_CORE 2>/dev/null || echo 0) -lt 1048576 ]; then
    if [ -f $GDS_BACKUP/svm_compute_core.gds ]; then
        cp $GDS_BACKUP/svm_compute_core.gds $GDS_CORE
    elif [ -f $OL_GDS ]; then
        cp $OL_GDS $GDS_CORE
    fi
fi
if [ $(stat -c%s $GDS_WRAP 2>/dev/null || echo 0) -lt 1048576 ]; then
    [ -f $GDS_BACKUP/user_project_wrapper.gds ] && cp $GDS_BACKUP/user_project_wrapper.gds $GDS_WRAP
fi

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
