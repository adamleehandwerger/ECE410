#!/bin/bash
#SBATCH --job-name=dv_svm_wb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

set -e

SCRATCH=$(ws_find openlane_svm)
CARAVEL=$SCRATCH/caravel_svm_project
CARAVEL_LITE=$SCRATCH/caravel-lite
MCW_ROOT=$SCRATCH/caravel_mgmt_soc_litex
DV_SIF=$SCRATCH/dv.sif
PDK_ROOT=$SCRATCH/pdk

TEST=svm_wb_test

echo "=== dv_run: $TEST on $(hostname) at $(date) ==="

if [ ! -f "$DV_SIF" ]; then
    echo "ERROR: DV SIF not found at $DV_SIF — run dv_setup.sh first"
    exit 1
fi

# --- Pull latest caravel repo state (reset to origin/main; skip LFS smudge) ---
echo "--- git sync caravel ---"
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL fetch origin
GIT_LFS_SKIP_SMUDGE=1 git -C $CARAVEL reset --hard origin/main

module load apptainer/1.4.1-gcc-13.4.0

# Run the DV make inside the efabless/dv container.
# Pass all make-infrastructure paths explicitly so env.makefile's $(DESIGNS)
# default assignments are never needed.
apptainer exec \
    --bind /scratch,/tmp,/proc \
    --env CARAVEL_ROOT=$CARAVEL_LITE \
    --env CARAVEL_VERILOG_PATH=$CARAVEL_LITE/verilog \
    --env MCW_ROOT=$MCW_ROOT \
    --env CORE_VERILOG_PATH=$MCW_ROOT/verilog \
    --env FIRMWARE_PATH=$MCW_ROOT/verilog/dv/firmware \
    --env VERILOG_PATH=$MCW_ROOT/verilog \
    --env USER_PROJECT_VERILOG=$CARAVEL/verilog \
    --env PDK_ROOT=$PDK_ROOT \
    --env PDK=sky130A \
    --env UPRJ_ROOT=$CARAVEL \
    --env GCC_PATH=/foss/tools/riscv-gnu-toolchain-rv32i/217e7f3debe424d61374d31e33a091a630535937/bin \
    --env GCC_PREFIX=riscv32-unknown-linux-gnu \
    --env PATH=/foss/tools/iverilog/cc0a8c8dd2fef69c4f7fb8219542b1c03a71a3b4/bin:/foss/tools/riscv-gnu-toolchain-rv32i/217e7f3debe424d61374d31e33a091a630535937/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    $DV_SIF \
    bash -c "
        set -e
        cd $CARAVEL/verilog/dv/$TEST
        echo '--- Running: make SIM=RTL TOOLCHAIN=GCC ---'
        make SIM=RTL TOOLCHAIN=GCC 2>&1
        echo '--- RTL sim done ---'
    "

echo "=== dv_run complete at $(date) ==="

# --- Print any simulation log output ---
SIMOUT=\$(find $CARAVEL/verilog/dv/$TEST -name "*.log" 2>/dev/null | head -5)
if [ -n "\$SIMOUT" ]; then
    echo "=== Simulation logs ==="
    for f in \$SIMOUT; do echo "--- \$f ---"; cat "\$f"; done
fi
