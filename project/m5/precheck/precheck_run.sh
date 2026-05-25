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

echo "=== mpw_precheck on $(hostname) at $(date) ==="

# --- Pull latest repo state ---
git -C $CARAVEL pull --ff-only || echo "WARNING: git pull failed"

# --- Verify required artifacts exist ---
echo "--- Checking required files ---"
for f in \
    $CARAVEL/gds/svm_compute_core.gds \
    $CARAVEL/gds/user_project_wrapper.gds \
    $CARAVEL/lef/svm_compute_core.lef \
    $CARAVEL/lef/user_project_wrapper.lef \
    $CARAVEL/verilog/gl/svm_compute_core.v \
    $CARAVEL/verilog/gl/user_project_wrapper.v; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename $f) ($(du -sh $f | cut -f1))"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

module load apptainer/1.4.1-gcc-13.4.0

# Pull precheck SIF if not present
PRECHECK_SIF=$SCRATCH/mpw-precheck.sif
if [ ! -f $PRECHECK_SIF ]; then
    echo "--- Pulling mpw-precheck container ---"
    apptainer pull $PRECHECK_SIF docker://efabless/mpw_precheck:latest
fi

echo "--- Running mpw-precheck ---"
apptainer exec \
    --bind $CARAVEL:/project \
    --bind $PDK_ROOT:/pdk \
    $PRECHECK_SIF \
    python3 /usr/local/bin/precheck.py \
        --input-directory /project \
        --pdk-path /pdk \
        --output-directory /project/precheck_results \
        --manifest /project/info.yaml \
        2>&1 | tee $CARAVEL/precheck_results/precheck.log

echo "=== precheck done at $(date) ==="
echo "=== Results ==="
cat $CARAVEL/precheck_results/precheck.log | grep -E "PASS|FAIL|ERROR|check"
