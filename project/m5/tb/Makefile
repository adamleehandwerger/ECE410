# ECE410 m5 — cocotb Wishbone interface simulation
# Exercises user_project_wrapper via its WB register interface.
# Outputs: asic_preds.csv  (consumed by confusion_comparison_m5.py)
#
# Usage:
#   make cosim          # run WB interface simulation → asic_preds.csv
#   python3 confusion_comparison_m5.py   # generate 2-column confusion matrix plot

SIM      = icarus
TOPLEVEL = user_project_wrapper
MODULE   = tb_wb_cosim

RTL_DIR  = ../../m5/rt1

VERILOG_SOURCES  = $(RTL_DIR)/user_project_wrapper.sv
VERILOG_SOURCES += $(RTL_DIR)/svm_compute_core.sv
VERILOG_SOURCES += sky130_stubs.v

# SystemVerilog + macro definitions
COMPILE_ARGS += -g2012
COMPILE_ARGS += -DMPRJ_IO_PADS=38
COMPILE_ARGS += -Wno-timescale


# cocotb Python dependencies (must be installed in the active virtualenv)
# pip install cocotb scikit-learn numba wfdb matplotlib numpy

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: cosim
cosim:
	$(MAKE) sim

.PHONY: plot
plot: asic_preds.csv
	python3 confusion_comparison_m5.py

.PHONY: all
all: cosim plot
