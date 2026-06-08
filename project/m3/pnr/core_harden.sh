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
SKY130A=$PDK_ROOT/sky130A
CARAVEL=$SCRATCH/caravel_svm_project
DESIGN_DIR=$CARAVEL/openlane/svm_compute_core
OUT_BASE=$DESIGN_DIR/runs/core_harden

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
rm -rf $OUT_BASE

# =========================================================
# Phase 1: Synthesis → Global Routing (checkpoint before DRT)
# =========================================================
echo "--- Phase 1: openlane --to OpenROAD.GlobalRouting ---"
openlane \
    --pdk sky130A \
    --pdk-root $PDK_ROOT \
    --run-tag core_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    --to OpenROAD.GlobalRouting \
    $DESIGN_DIR/config.json 2>&1

echo "Phase 1 complete at $(date)"

# --- Patch GRT DEF: replace probe cells (belt-and-suspenders), strip met5 TRACKS ---
# SYNTH_DONT_USE_LIST should prevent probe cells from synthesis, but patch anyway.
# RT_MAX_LAYER=met4 should prevent met5 tracks, but strip any that appear.
GRT_DEF=$(find "$OUT_BASE" -name "*.def" -path "*GlobalRouting*" 2>/dev/null | sort | tail -1)
[ -z "$GRT_DEF" ] && GRT_DEF=$(find "$OUT_BASE" -name "*.def" 2>/dev/null | sort | tail -1)

if [ -n "$GRT_DEF" ]; then
    echo "--- Patching GRT DEF: $GRT_DEF ---"
    PROBE_COUNT=$(grep -c 'probe_p_8\|probec_p_8' "$GRT_DEF" 2>/dev/null || true)
    MET5_COUNT=$(grep -c 'TRACKS.*LAYER met5' "$GRT_DEF" 2>/dev/null || true)
    echo "  probe cells: ${PROBE_COUNT:-0}   met5 TRACKS: ${MET5_COUNT:-0}"
    sed -i \
        -e 's/sky130_fd_sc_hd__probe_p_8/sky130_fd_sc_hd__buf_8/g' \
        -e 's/sky130_fd_sc_hd__probec_p_8/sky130_fd_sc_hd__buf_8/g' \
        -e '/TRACKS.*LAYER met5/d' \
        "$GRT_DEF"
    echo "  Patch applied"
else
    echo "WARNING: No GRT DEF found — skipping probe cell patch"
fi

# =========================================================
# Phase 2: Detailed Routing → sign-off
# =========================================================
echo "--- Phase 2: openlane --from OpenROAD.DetailedRouting ---"
openlane \
    --pdk sky130A \
    --pdk-root $PDK_ROOT \
    --run-tag core_harden \
    --jobs $SLURM_CPUS_PER_TASK \
    --from OpenROAD.DetailedRouting \
    $DESIGN_DIR/config.json 2>&1

echo "=== OpenLane complete at $(date) ==="

# --- Locate final outputs ---
echo "=== Output files ==="
find $OUT_BASE -name "*.gds" -o -name "*.lef" -o -name "*.def" 2>/dev/null | grep -i final | head -10
ls -lh $OUT_BASE 2>/dev/null || echo "No output dir found at $OUT_BASE"

FINAL_GDS=$(find $OUT_BASE -name "*.gds" 2>/dev/null | grep -i final | head -1)
FINAL_LEF=$(find $OUT_BASE -name "*.lef" 2>/dev/null | grep -i "final\|abstract" | grep -v pdn | head -1)
FINAL_DEF=$(find $OUT_BASE -name "*.def" 2>/dev/null | grep -i "detail\|drt\|final" | sort | tail -1)
FINAL_GL=$(find $OUT_BASE \( -name "*.nl.v" -o -name "*.v" \) 2>/dev/null | grep -i final | head -1)

# =========================================================
# Magic GDS fallback — if OpenLane exports a stub (<1 MB)
# load full standard cell GDS before reading DEF
# =========================================================
GDS_SIZE=0
[ -n "$FINAL_GDS" ] && GDS_SIZE=$(stat -c%s "$FINAL_GDS" 2>/dev/null || echo 0)
echo "OpenLane GDS: ${FINAL_GDS:-none} (${GDS_SIZE} bytes)"

if [ "${GDS_SIZE:-0}" -lt 1048576 ]; then
    echo "--- GDS stub or missing (${GDS_SIZE}B) — running Magic export ---"
    if [ -n "$FINAL_DEF" ]; then
        MAGIC_GDS=/tmp/svm_compute_core_magic.gds
        MAGIC_TCL=/tmp/magic_export_core.tcl
        cat > "$MAGIC_TCL" << MEOF
drc off
crashbackups stop
gds read $SKY130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds
lef read $SKY130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef
lef read $SKY130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef
def read $FINAL_DEF
gds write $MAGIC_GDS
quit
MEOF
        echo "  DEF: $FINAL_DEF"
        echo "  TCL: $MAGIC_TCL"
        magic -noconsole -dnull < "$MAGIC_TCL" 2>&1 || echo "WARNING: magic exited non-zero"
        MAGIC_SIZE=$(stat -c%s "$MAGIC_GDS" 2>/dev/null || echo 0)
        if [ "$MAGIC_SIZE" -gt 1048576 ]; then
            echo "Magic GDS valid: $(ls -lh $MAGIC_GDS)"
            FINAL_GDS=$MAGIC_GDS
        else
            echo "WARNING: Magic GDS still invalid (${MAGIC_SIZE}B) — no valid GDS available"
        fi
    else
        echo "WARNING: No DRT DEF found — cannot run Magic fallback"
    fi
else
    echo "OpenLane GDS valid: $(ls -lh $FINAL_GDS)"
fi

# --- Copy GDS/LEF/GL to caravel artifact dirs ---
[ -n "$FINAL_GDS" ] && cp "$FINAL_GDS" $CARAVEL/gds/svm_compute_core.gds  && echo "GDS -> gds/"
[ -n "$FINAL_LEF" ] && cp "$FINAL_LEF" $CARAVEL/lef/svm_compute_core.lef  && echo "LEF -> lef/"
[ -n "$FINAL_GL"  ] && cp "$FINAL_GL"  $CARAVEL/verilog/gl/svm_compute_core.v && echo "GL  -> verilog/gl/"

echo "=== core_harden complete at $(date) ==="
