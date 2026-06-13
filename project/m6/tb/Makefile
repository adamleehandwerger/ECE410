# ECE410 m6 — cocotb SPI cosimulation (svm_top_ihp, IHP SG13G2 target)
# Drives svm_top_ihp via SPI slave interface; outputs ../sim/asic_preds.csv
#
# Usage:
#   make cosim              # full 300-sample SPI cosim → asic_preds.csv
#   make cosim N=25         # quick subset (COSIM_N_EVAL=25)
#   make latency            # RAM_LATENCY unit test (iverilog standalone, < 1 s)
#   make plot               # regenerate confusion_comparison_m6.png
#   make all                # cosim + plot

SIM      = icarus
TOPLEVEL = svm_top_ihp
MODULE   = tb_spi_cosim

RTL_DIR  = ../rt1

VERILOG_SOURCES  = $(RTL_DIR)/top.sv
VERILOG_SOURCES += $(RTL_DIR)/compute_core.sv
VERILOG_SOURCES += $(RTL_DIR)/interface.sv

# SystemVerilog + simulation macro (skips sg13g2_dlclkp_1 ICG instantiation)
COMPILE_ARGS += -g2012
COMPILE_ARGS += -DSIMULATION
COMPILE_ARGS += -Wno-timescale

# Pass N_EVAL override: make cosim N=25
ifdef N
    export COSIM_N_EVAL=$(N)
endif

# pip install cocotb scikit-learn wfdb matplotlib numpy
include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: cosim
cosim:
	$(MAKE) sim

.PHONY: latency
latency:
	iverilog -g2012 -DSIMULATION \
	    -o /tmp/svm_lat_tb_m6.out \
	    $(RTL_DIR)/compute_core.sv svm_ram_latency_tb.sv && \
	/tmp/svm_lat_tb_m6.out

.PHONY: plot
plot:
	python3 ../sim/confusion_comparison_m6.py

.PHONY: all
all: cosim plot
