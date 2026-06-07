#!/bin/bash
#SBATCH --job-name=wrapper_harden
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
# Orca: use "long" partition (up to 7 days); "gpu" partition not available

set -e

SCRATCH=$(ws_find openlane_svm)
PDK_ROOT=$SCRATCH/pdk
CARAVEL=$SCRATCH/caravel_svm_project
DESIGN_DIR=$CARAVEL/openlane/user_project_wrapper

echo "=== wrapper_harden: user_project_wrapper on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

# --- Pull latest caravel changes ---
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

# All tools except yosys come from the SIF (openroad, magic, etc.)
for TOOL in openroad magic klayout netgen verilator iverilog opensta sta; do
    cat > $TOOL_WRAPPERS/$TOOL << WRAP
#!/bin/bash
exec apptainer exec --bind /scratch,/tmp $OL2_SIF $TOOL "\$@"
WRAP
    chmod +x $TOOL_WRAPPERS/$TOOL
done

# yosys: use nix yosys-with-plugins (0.46) via proot — supports PyOSYS (-y flag)
# SIF yosys 0.38 does not support -y, which OpenLane v2.3.10 requires for JsonHeader
PROOT=$SCRATCH/.nix/.nix-portable/bin/proot
NIX_STORE=$SCRATCH/.nix/.nix-portable/nix
NIX_YOSYS=$(ls -d $SCRATCH/.nix/.nix-portable/nix/store/*-yosys-with-plugins/bin/yosys 2>/dev/null | head -1)
cat > $TOOL_WRAPPERS/yosys << WRAP
#!/bin/bash
# Expose venv site-packages so yosys embedded Python can find click and other
# openlane deps (nix yosys uses its own Python; PYTHONPATH bridges the gap)
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
ls -lh $DESIGN_DIR/macro.cfg
ls -lh $CARAVEL/lef/svm_compute_core.lef
ls -lh $CARAVEL/gds/svm_compute_core.gds
ls -lh $CARAVEL/verilog/gl/svm_compute_core.v
ls -lh $CARAVEL/verilog/rtl/user_project_wrapper.sv

# --- Clean previous run ---
echo "--- Removing old run dir ---"
rm -rf $DESIGN_DIR/runs/wrapper_harden

OUT_BASE=$DESIGN_DIR/runs/wrapper_harden

# =============================================================================
# Phase 1: Full synthesis through GlobalRouting.
# SYNTH_ELABORATE_ONLY=0 (full synthesis) produces a proper gate-level netlist
# that OpenSTA can parse. FP_TEMPLATE_MATCH_MODE=permissive handles the
# applydeftemplate mismatch for the 6 unused power BTERMs
# (vccd2, vdda1, vdda2, vssa1, vssa2, vssd2).
# Phase 1 stops before DetailedRouting because those BTERMs acquire multiple
# physical PIN geometries from the Caravel power ring template, which causes
# TritonRoute DRT-0302 ("Unsupported multiple pins on bterm vccd2").
# =============================================================================
echo "--- Phase 1: Synthesis through GlobalRouting ---"
openlane \
    --pdk sky130A \
    --pdk-root $PDK_ROOT \
    --run-tag wrapper_harden \
    --to OpenROAD.GlobalRouting \
    --jobs $SLURM_CPUS_PER_TASK \
    $DESIGN_DIR/config.json 2>&1

# =============================================================================
# ODB patch: delete the 6 unused power BTERMs from the OpenROAD database.
# After applydeftemplate, these BTERMs have multiple physical PIN shapes (one
# per power ring segment in the template). TritonRoute rejects multi-pin BTERMs
# with DRT-0302. Deleting them is safe: they are not connected to anything in
# this design (svm_compute_core only uses vccd1/vssd1).
# The ODB is modified in-place; Phase 2's --from DetailedRouting reads it via
# the GlobalRouting step's state_out.json which still points to the same path.
# =============================================================================
echo "--- Patching ODB: deleting unused power BTERMs ---"
GRT_DIR=$(ls -d $OUT_BASE/*-openroad-globalrouting 2>/dev/null | tail -1)
if [ -z "$GRT_DIR" ]; then
    echo "ERROR: GlobalRouting step directory not found under $OUT_BASE"
    ls $OUT_BASE 2>/dev/null
    exit 1
fi

GRT_ODB=$(python3 -c "
import json, sys
with open('$GRT_DIR/state_out.json') as f:
    state = json.load(f)
odb = state.get('odb') or ''
if odb:
    print(odb)
else:
    print('NOT_FOUND', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

echo "GlobalRouting ODB: $GRT_ODB"
if [ -z "$GRT_ODB" ] || [ ! -f "$GRT_ODB" ]; then
    echo "ERROR: ODB not found. Dumping state keys:"
    python3 -c "import json; d=json.load(open('$GRT_DIR/state_out.json')); print(dict((k,v) for k,v in d.items() if v and k not in ('metrics',)))"
    exit 1
fi

cat > /tmp/delete_power_bterms.tcl << EOF
read_db {$GRT_ODB}
set block [ord::get_db_block]
foreach net_name {vccd2 vdda1 vdda2 vssa1 vssa2 vssd2} {
    set bterm [\$block findBTerm \$net_name]
    if {\$bterm != "NULL"} {
        puts "  Deleting power BTERM: \$net_name"
        odb::dbBTerm_destroy \$bterm
    } else {
        puts "  BTERM \$net_name not found (already absent)"
    }
}
write_db {$GRT_ODB}
puts "ODB patch complete."
exit
EOF

openroad -no_init -exit /tmp/delete_power_bterms.tcl
echo "--- ODB patch done ---"

# =============================================================================
# Phase 2: DetailedRouting through end.
# GlobalRouting step has state_out.json pointing to the patched ODB.
# OpenLane will not re-run prior steps (they all have state_out.json).
# =============================================================================
echo "--- Phase 2: DetailedRouting onward ---"
# --skip OpenROAD.IRDropReport: PSM-0069 is expected for sub-chips without
# VSRC_LOC_FILES (not a top-level package integration). Advisory only.
openlane \
    --pdk sky130A \
    --pdk-root $PDK_ROOT \
    --run-tag wrapper_harden \
    --from OpenROAD.DetailedRouting \
    --skip OpenROAD.IRDropReport \
    --skip KLayout.XOR \
    --skip KLayout.DRC \
    --jobs $SLURM_CPUS_PER_TASK \
    $DESIGN_DIR/config.json 2>&1

echo "=== Done at $(date) ==="

# --- Collect outputs ---
echo "=== Output files ==="
find $OUT_BASE -name "*.gds" -o -name "*.lef" -o -name "*.def" 2>/dev/null | grep -i final | head -10
ls -lh $OUT_BASE 2>/dev/null || echo "No output dir found at $OUT_BASE"

# --- Copy GDS/LEF/GL to caravel artifact dirs ---
FINAL_GDS=$(find $OUT_BASE -name "*.gds" 2>/dev/null | grep -i final | head -1)
FINAL_LEF=$(find $OUT_BASE -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_GL=$(find $OUT_BASE -name "*.nl.v" -o -name "*.v" 2>/dev/null | grep -i final | head -1)

[ -n "$FINAL_GDS" ] && cp $FINAL_GDS $CARAVEL/gds/user_project_wrapper.gds  && echo "GDS -> gds/"
[ -n "$FINAL_LEF" ] && cp $FINAL_LEF $CARAVEL/lef/user_project_wrapper.lef  && echo "LEF -> lef/"
[ -n "$FINAL_GL"  ] && cp $FINAL_GL  $CARAVEL/verilog/gl/user_project_wrapper.v && echo "GL  -> verilog/gl/"

echo "=== wrapper_harden complete ==="
