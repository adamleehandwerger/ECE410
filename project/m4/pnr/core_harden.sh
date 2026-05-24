#!/bin/bash
#SBATCH --job-name=core_harden
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

set -e

SCRATCH=$(ws_find openlane_svm)
PDK_ROOT=$SCRATCH/pdk
CARAVEL=$SCRATCH/caravel_svm_project
DESIGN_DIR=$CARAVEL/openlane/svm_compute_core

echo "=== core_harden: svm_compute_core on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

# --- Pull latest changes ---
echo "--- git pull caravel ---"
git -C $CARAVEL pull --ff-only || echo "WARNING: git pull failed, using local state"

# --- Activate the existing ol2 venv ---
OL2_VENV=$SCRATCH/ol2_venv_mf
source $OL2_VENV/bin/activate
echo "OpenLane2: $(openlane --version 2>/dev/null || echo 'version check failed')"

# --- Load apptainer and set up EDA tool wrappers ---
module load apptainer/1.4.1-gcc-13.4.0

OL2_SIF=$SCRATCH/openlane2.sif
echo "SIF: $(ls -lh $OL2_SIF 2>/dev/null || echo 'SIF NOT FOUND')"

TOOL_WRAPPERS=$SCRATCH/eda-wrappers
mkdir -p $TOOL_WRAPPERS

for TOOL in openroad magic klayout netgen verilator iverilog opensta sta; do
    cat > $TOOL_WRAPPERS/$TOOL << WRAP
#!/bin/bash
exec apptainer exec --bind /scratch,/tmp $OL2_SIF $TOOL "\$@"
WRAP
    chmod +x $TOOL_WRAPPERS/$TOOL
done

PROOT=$SCRATCH/.nix/.nix-portable/bin/proot
NIX_STORE=$SCRATCH/.nix/.nix-portable/nix
NIX_YOSYS=$(ls -d $SCRATCH/.nix/.nix-portable/nix/store/*-yosys-with-plugins/bin/yosys 2>/dev/null | head -1)
cat > $TOOL_WRAPPERS/yosys << WRAP
#!/bin/bash
export PYTHONPATH=$OL2_VENV/lib/python3.13/site-packages\${PYTHONPATH:+:\$PYTHONPATH}
exec $PROOT -b $NIX_STORE:/nix $NIX_YOSYS "\$@"
WRAP
chmod +x $TOOL_WRAPPERS/yosys

export PATH=$TOOL_WRAPPERS:$PATH
echo "yosys:    $(yosys --version 2>&1 | grep 'Yosys' | head -1)"
echo "openroad: $(openroad --version 2>&1 | head -1)"
echo "magic:    $(magic --version 2>&1 | head -1 || echo n/a)"

# --- Confirm key files exist ---
echo "--- Design inputs ---"
ls -lh $DESIGN_DIR/config.json
ls -lh $CARAVEL/verilog/rtl/svm_compute_core.sv

# --- Clean previous run ---
echo "--- Removing old run dir ---"
rm -rf $DESIGN_DIR/runs/core_harden

# --- Run OpenLane 2 ---
echo "--- Running openlane ---"
openlane \
    --pdk sky130A \
    --pdk-root $PDK_ROOT \
    --run-tag core_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    $DESIGN_DIR/config.json 2>&1

echo "=== Done at $(date) ==="

# --- Collect outputs ---
OUT_BASE=$DESIGN_DIR/runs/core_harden
echo "=== Output files ==="
find $OUT_BASE -name "*.gds" -o -name "*.lef" -o -name "*.def" 2>/dev/null | grep -i final | head -10
ls -lh $OUT_BASE 2>/dev/null || echo "No output dir found at $OUT_BASE"

# --- Copy GDS/LEF/GL to caravel artifact dirs ---
FINAL_GDS=$(find $OUT_BASE -name "*.gds" 2>/dev/null | grep -i final | head -1)
FINAL_LEF=$(find $OUT_BASE -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_GL=$(find $OUT_BASE -name "*.nl.v" -o -name "*.v" 2>/dev/null | grep -i final | head -1)

[ -n "$FINAL_GDS" ] && cp $FINAL_GDS $CARAVEL/gds/svm_compute_core.gds  && echo "GDS -> gds/"
[ -n "$FINAL_LEF" ] && cp $FINAL_LEF $CARAVEL/lef/svm_compute_core.lef  && echo "LEF -> lef/"
[ -n "$FINAL_GL"  ] && cp $FINAL_GL  $CARAVEL/verilog/gl/svm_compute_core.v && echo "GL  -> verilog/gl/"

echo "=== core_harden complete ==="
