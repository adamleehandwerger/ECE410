#!/bin/bash
# setup_ihp_pdk.sh — One-time IHP SG13G2 PDK setup on Orca
# Run interactively (not via SLURM) before submitting core_harden.sh.
#
# Usage:
#   bash setup_ihp_pdk.sh
#
# What it does:
#   1. Clones IHP-Open-PDK into $SCRATCH/ihp-open-pdk
#   2. Verifies the sg13g2_stdcell Liberty file is present
#   3. Clones this project's m6 RTL into $SCRATCH/svm_m6
#   4. Prints environment variables to add to ~/.bashrc

set -e

SCRATCH=$(ws_find openlane_svm)
IHP_PDK_ROOT=$SCRATCH/ihp-open-pdk
SVM_M6=$SCRATCH/svm_m6

echo "=== IHP SG13G2 PDK setup on $(hostname) at $(date) ==="
echo "SCRATCH=$SCRATCH"

# --- Clone IHP Open PDK ---
if [ -d "$IHP_PDK_ROOT/.git" ]; then
    echo "IHP PDK already present — pulling latest..."
    git -C $IHP_PDK_ROOT pull --ff-only || echo "WARNING: pull failed, using local state"
else
    echo "Cloning IHP-Open-PDK..."
    git clone --depth 1 https://github.com/IHP-GmbH/IHP-Open-PDK.git $IHP_PDK_ROOT
fi

# --- Verify key PDK files ---
LIB_FILE=$IHP_PDK_ROOT/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_typ_1p20V_25C.lib
LEF_FILE=$IHP_PDK_ROOT/ihp-sg13g2/libs.ref/sg13g2_stdcell/lef/sg13g2_stdcell.lef
ICG_FILE=$IHP_PDK_ROOT/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/sg13g2_dlclkp_1.lib 2>/dev/null || true

echo ""
echo "--- Verifying PDK files ---"
if [ -f "$LIB_FILE" ]; then
    echo "Liberty file  : OK  ($LIB_FILE)"
else
    echo "ERROR: Liberty file not found at $LIB_FILE"
    echo "  IHP PDK directory structure may have changed — check ihp-open-pdk repo."
    exit 1
fi

if [ -f "$LEF_FILE" ]; then
    echo "LEF file      : OK  ($LEF_FILE)"
else
    echo "ERROR: LEF not found at $LEF_FILE"
    exit 1
fi

# Check for sg13g2_dlclkp_1 (ICG cell used by top.sv)
if grep -r "sg13g2_dlclkp_1" $IHP_PDK_ROOT/ihp-sg13g2/libs.ref/ --include="*.lib" -l 2>/dev/null | head -1; then
    echo "ICG cell      : OK  (sg13g2_dlclkp_1 found in Liberty)"
else
    echo "WARNING: sg13g2_dlclkp_1 not found in Liberty files"
    echo "  The ICG cell may be in a different file — check PDK contents."
fi

# --- Clone svm_m6 RTL ---
if [ -d "$SVM_M6/.git" ]; then
    echo ""
    echo "svm_m6 repo already present — pulling..."
    git -C $SVM_M6 pull --ff-only || echo "WARNING: pull failed"
else
    echo ""
    echo "Cloning ECE410 m6 RTL..."
    git clone https://github.com/adamleehandwerger/ECE410.git $SVM_M6
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Add these to your ~/.bashrc or source before running SLURM jobs:"
echo "  export IHP_PDK_ROOT=$IHP_PDK_ROOT"
echo "  export PDK=sg13g2"
echo "  export SVM_M6=$SVM_M6"
echo ""
echo "Next step: sbatch $SVM_M6/project/m6/pnr/core_harden.sh"
