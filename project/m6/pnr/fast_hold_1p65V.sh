#!/bin/bash
#SBATCH --job-name=m6_hold_1p65V
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=handwerg@pdx.edu

# fast_hold_1p65V.sh — Hold timing at fast_1p65V_m40C corner
#
# Reads the post-PnR ODB from core_harden and runs OpenROAD STA with
# sg13g2_stdcell_fast_1p65V_m40C.lib — the corner not included in the
# default LibreLane STA_CORNERS for IHP SG13G2.
#
# At 1.65 V / -40 °C cells are ~15-20% faster than at 1.32 V, making
# hold timing harder.  The existing post-PnR WNS at 1p32V is +0.089 ns
# with 12 806 dlygate hold buffers already inserted.
#
# Usage: sbatch fast_hold_1p65V.sh

set -e

SCRATCH=$(ws_find openlane_svm)
SVM_M6=$SCRATCH/svm_m6
RUN=$SVM_M6/project/m6/synth/runs/core_harden
ARTIFACTS=$SCRATCH/svm_m6_artifacts
OUTDIR=$ARTIFACTS/fast_hold_1p65V
mkdir -p $OUTDIR

module load apptainer/1.4.1-gcc-13.4.0
LIBRELANE_SIF=$SCRATCH/librelane_3.0.4.sif

# Derive ciel PDK base from the existing signoff config (version-hash stable)
CIEL_BASE=$(python3 -c "
import json, sys
d = json.load(open('$RUN/55-openroad-stapostpnr/config.json'))
lib = d['LIB']['nom_fast_1p32V_m40C'][0]
print(lib.split('/libs.ref')[0])
")

echo "=== fast_1p65V_m40C hold check at $(date) ==="
echo "CIEL_BASE = $CIEL_BASE"
echo "ODB       = $RUN/53-odb-cellfrequencytables/svm_compute_core.odb"

export LIB_FAST_1P65=$CIEL_BASE/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_fast_1p65V_m40C.lib
export TECH_LEF=$CIEL_BASE/libs.ref/sg13g2_stdcell/lef/sg13g2_tech.lef
export CELL_LEF=$CIEL_BASE/libs.ref/sg13g2_stdcell/lef/sg13g2_stdcell.lef
export ODB_FILE=$RUN/53-odb-cellfrequencytables/svm_compute_core.odb
export SDC_FILE=$SVM_M6/project/m6/synth/svm_compute_core.sdc
export OUTDIR=$OUTDIR

# Verify files
for F in "$LIB_FAST_1P65" "$TECH_LEF" "$CELL_LEF" "$ODB_FILE" "$SDC_FILE"; do
    [ -f "$F" ] || { echo "ERROR: missing $F"; exit 1; }
done

apptainer exec --bind /scratch,/home,/tmp $LIBRELANE_SIF \
    openroad -no_splash -exit $SVM_M6/project/m6/pnr/fast_hold_1p65V.tcl \
    2>&1 | tee $OUTDIR/openroad.log

echo ""
echo "=== Results (see openroad.log for full timing paths) ==="
grep -A 2 "Hold Summary" $OUTDIR/openroad.log 2>/dev/null || true
echo ""
echo "=== Done at $(date) — full log: $OUTDIR/openroad.log ==="
