# fast_hold_1p65V.tcl — Hold analysis at fast_1p65V_m40C corner
# Invoked by fast_hold_1p65V.sh inside the LibreLane SIF via:
#   openroad -no_splash -exit fast_hold_1p65V.tcl
#
# Required env vars (set by wrapper):
#   LIB_FAST_1P65  — path to sg13g2_stdcell_fast_1p65V_m40C.lib
#   TECH_LEF       — sg13g2_tech.lef
#   CELL_LEF       — sg13g2_stdcell.lef
#   ODB_FILE       — post-PnR .odb from core_harden run
#   SDC_FILE       — svm_compute_core.sdc
#   OUTDIR         — output directory for reports

set lib_1p65 $::env(LIB_FAST_1P65)
set tech_lef $::env(TECH_LEF)
set cell_lef $::env(CELL_LEF)
set odb_file $::env(ODB_FILE)
set sdc_file $::env(SDC_FILE)
set outdir   $::env(OUTDIR)

puts "=== fast_1p65V_m40C Hold Check ==="
puts "LIB : $lib_1p65"
puts "ODB : $odb_file"
puts "SDC : $sdc_file"
puts "OUT : $outdir"

read_lef $tech_lef
read_lef $cell_lef

# Define corner then read liberty — create_timing_corner does not exist in OpenROAD
define_corners fast_1p65V_m40C
read_liberty -corner fast_1p65V_m40C $lib_1p65

read_db  $odb_file
read_sdc $sdc_file

file mkdir $outdir

# -file and -append are not supported in this OpenROAD build; all output
# goes to stdout and is captured by the shell's tee into openroad.log
puts "=== Hold Paths (fast_1p65V_m40C) ==="
report_checks -path_delay min -sort_by_slack -corner fast_1p65V_m40C

puts ""
puts "=== Hold Summary (fast_1p65V_m40C) ==="
report_worst_slack -min
report_tns
