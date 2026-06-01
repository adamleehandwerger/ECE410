SIM                 = icarus
TOPLEVEL_LANG       = verilog

VERILOG_SOURCES     = $(PWD)/svm_compute_core.sv
TOPLEVEL            = svm_compute_core
COCOTB_TEST_MODULES = test_svm_compute_core

include $(shell cocotb-config --makefiles)/Makefile.sim
