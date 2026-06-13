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

# m6_top_harden.sh — Harden svm_top_ihp (standalone top) with IHP SG13G2 PDK
#
# Run AFTER core_harden.sh has completed successfully.
# Requires $ARTIFACTS/svm_compute_core.{gds,lef,v} from core_harden.
#
# Flow: svm_compute_core is treated as a hardened macro (black box).
# svm_top_ihp (SPI slave + register file + clock gate) is synthesised
# around it and routed on Metal1-Metal5.

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
SVM_M6=$SCRATCH/svm_m6
DESIGN_DIR=$SVM_M6/project/m6/synth
ARTIFACTS=$SCRATCH/svm_m6_artifacts

echo "=== m6_top_harden: svm_top_ihp (IHP SG13G2) on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

# --- Verify core macro artifacts exist ---
echo "--- Verifying core macro artifacts ---"
for F in svm_compute_core.gds svm_compute_core.lef svm_compute_core.v; do
    if [ ! -f "$ARTIFACTS/$F" ]; then
        echo "ERROR: $ARTIFACTS/$F not found — run core_harden.sh first"
        exit 1
    fi
    ls -lh $ARTIFACTS/$F
done

# --- Copy artifacts where top_config.json expects them ---
mkdir -p $SVM_M6/project/m6/pnr/gds
cp $ARTIFACTS/svm_compute_core.gds $SVM_M6/project/m6/pnr/gds/
cp $ARTIFACTS/svm_compute_core.lef $SVM_M6/project/m6/pnr/gds/
cp $ARTIFACTS/svm_compute_core.v   $SVM_M6/project/m6/pnr/gds/

# Generate black-box stub for synthesis (no logic, just port declarations)
python3 - << 'PYEOF'
import re, sys
with open(f"{sys.argv[1]}/svm_compute_core.v") as f:
    src = f.read()
# Extract module header up to first semicolon after port list
m = re.search(r'(module\s+svm_compute_core\s*\(.*?\)\s*;)', src, re.DOTALL)
if m:
    with open(f"{sys.argv[1]}/svm_compute_core_bb.v", "w") as f:
        f.write("// Black-box stub for top-level synthesis\n")
        f.write(m.group(1) + "\nendmodule\n")
    print("Black-box stub written")
else:
    print("WARNING: could not extract module header — using full GL netlist as blackbox")
    import shutil
    shutil.copy(f"{sys.argv[1]}/svm_compute_core.v",
                f"{sys.argv[1]}/svm_compute_core_bb.v")
PYEOF "$SVM_M6/project/m6/pnr/gds"

# Also copy to rt1/ where top_config.json references it
cp $SVM_M6/project/m6/pnr/gds/svm_compute_core_bb.v \
   $SVM_M6/project/m6/rt1/compute_core_bb.v

# --- git pull latest m6 ---
echo "--- git pull svm_m6 ---"
git -C $SVM_M6 pull --ff-only || echo "WARNING: git pull failed"

# --- Activate OL2 venv ---
OL2_VENV=$SCRATCH/ol2_venv_mf
source $OL2_VENV/bin/activate

# --- EDA tool wrappers ---
module load apptainer/1.4.1-gcc-13.4.0

OL2_SIF=$SCRATCH/openlane2.sif
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

# --- Verify inputs ---
echo "--- Top-level inputs ---"
ls -lh $DESIGN_DIR/top_config.json
ls -lh $DESIGN_DIR/svm_top_ihp.sdc
ls -lh $SVM_M6/project/m6/rt1/top.sv
ls -lh $SVM_M6/project/m6/rt1/compute_core_bb.v

# --- Clean previous run ---
RUN_DIR=$DESIGN_DIR/runs/top_harden
rm -rf $RUN_DIR

# --- Run OpenLane 2 ---
echo "--- Running openlane (svm_top_ihp, IHP SG13G2) ---"
openlane \
    --pdk sg13g2 \
    --pdk-root $IHP_PDK_ROOT \
    --run-tag top_harden \
    --jobs $SLURM_CPUS_PER_TASK \
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
echo "=== m6 harden complete — artifacts in $ARTIFACTS/ ==="
echo "=== Run KLayout DRC manually to confirm 0 violations ==="
echo "  klayout -b -r \$IHP_PDK_ROOT/ihp-sg13g2/libs.tech/klayout/tech/drc/sg13g2.lydrc \\"
echo "          -rd input=$ARTIFACTS/svm_top_ihp.gds"
