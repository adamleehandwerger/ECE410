# Full-dataset RTL cosim of svm_compute_core (600-SV pipelined, behavioral SRAM).
SIM      = icarus
TOPLEVEL = svm_compute_core
MODULE   = test_full_dataset_cosim
RTL_DIR  = ../rt1

VERILOG_SOURCES  = /tmp/m7_timescale.v
VERILOG_SOURCES += $(RTL_DIR)/compute_core.sv
VERILOG_SOURCES += $(RTL_DIR)/alpha_sram.sv

COMPILE_ARGS += -g2012 -DSIMULATION -DALPHA_SRAM_BEHAV_CELL -Wno-timescale
export COCOTB_HDL_TIMEUNIT      = 1ns
export COCOTB_HDL_TIMEPRECISION = 1ps

include $(shell cocotb-config --makefiles)/Makefile.sim
