# ECE410 — Milestone 2: RTL Verification & Synthesis

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** sky130A (SkyWater 130 nm open-PDK)  
**Accuracy:** 98.67% on MIT-BIH (sklearn = hardware, zero gap)  
**Status:** 19/19 tests passing (10 iverilog + 9 cocotb); synthesis complete

The m2 milestone delivers a fully verified, ASIC-ready RTL implementation of a
fixed-point RBF-SVM compute core. Feature vector: 256-dim multi-scale
(128 single-beat + 64 10-beat mean + 64 RR-interval). Fixed-point: Q6.10 (16-bit).

---

## Directory Structure

```
m2/
├── README.md                  ← this file
├── README_errorcodes.md       ← error code reference (13 codes, reset/sticky behavior)
├── README_interface.md        ← port-level interface specification
├── interface_spec.md          ← full interface spec with signal table and timing
├── design_summary.md          ← design decisions with alternatives and trade-offs
├── block_diagram.png          ← architecture block diagram (v6)
├── generate_block_diagram.py  ← Python script that renders block_diagram.png (matplotlib)
├── confusion_comparison.png   ← sklearn vs. hardware confusion matrix comparison
├── confusion_comparison.py    ← script that generates confusion_comparison.png
├── rt1/                       ← RTL source files
│   ├── top.sv                 ← top-level module (svm_compute_core) — FSM, datapath,
│   │                              FIFO, distance matrix, Horner LUT kernel engine
│   └── svm_interfaces.sv      ← SystemVerilog interface definitions (svm_data_if,
│                                  svm_ctrl_if) used by the compute core
├── tb/                        ← Testbenches and verification
│   ├── README.md              ← testbench overview and how to run
│   ├── Makefile               ← iverilog/cocotb build rules; `make all` runs all tests
│   ├── tb_top.sv              ← full-pipeline 5-heartbeat classification testbench
│   ├── tb_svm_params.svh      ← auto-generated SV include: dual coefficients,
│   │                              intercepts, SV counts for all 5 classes
│   ├── tb_error_codes.sv      ← unit test: all 13 error codes, sticky latch, reset-clear
│   ├── tb_backpressure.sv     ← unit test: kernel_valid hold; late kernel_ready release
│   ├── tb_consecutive.sv      ← unit test: back-to-back heartbeat processing
│   ├── tb_dist_boundary.sv    ← unit test: accumulator saturation boundary
│   ├── tb_dist_zero.sv        ← unit test: D=0 → kernel_out=1024 (exp(0)=1 in Q6.10)
│   ├── tb_gamma_zero.sv       ← unit test: γ=0 edge case
│   ├── tb_interface.sv        ← unit test: port signal protocol compliance
│   ├── tb_min_sv.sv           ← unit test: minimum SV count (1 SV per class)
│   ├── tb_multi_heartbeat.sv  ← unit test: num_samples=3 loop-back
│   ├── tb_param_write.sv      ← unit test: runtime parameter write via param_write_en
│   ├── tb_power.sv            ← unit test: clock-gate idle power behavior
│   ├── tb_warmup.sv           ← unit test: warmup-state exit and start-pulse timing
│   ├── tb_results.md          ← recorded pass/fail results for all testbenches
│   ├── test_svm_compute_core.py ← cocotb Python testbench (9 tests via simulation API)
│   ├── sv_ram.hex             ← 256-SV × 256-feature support vector memory image
│   ├── test_features.hex      ← 5-heartbeat feature vectors (one per class)
│   ├── test_labels.hex        ← ground-truth class labels for test_features.hex
│   ├── expected_kernels.hex   ← pre-computed expected RBF kernel outputs
│   └── expected_preds.hex     ← pre-computed expected class predictions
├── sim/                       ← Simulation outputs
│   ├── cosim_run.log          ← full cocotb simulation log (all 9 tests)
│   └── cosim_waveform.png     ← VCD-derived waveform screenshot (classification run)
└── synth/                     ← Synthesis results (OpenLane 2, sky130A)
    ├── config.json            ← OpenLane 2 config: CLOCK_PERIOD=10ns, sky130_fd_sc_hd
    ├── area_report.txt        ← cell count, total area, utilization (pre-DRT estimate)
    ├── timing_report.txt      ← setup timing: critical path, WNS, TNS
    ├── power_report.txt       ← internal/switching/leakage power breakdown
    ├── critical_path.md       ← annotated critical path explanation
    └── openlane_run.log       ← full OpenLane synthesis run log
```

---

## Key Design Parameters (m2)

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat + 64 10-beat mean + 64 RR) |
| Support vectors | up to 256 per run |
| Classes | 5 (Normal, PVC, AFib, VT, SVT) |
| Fixed-point | Q6.10, 16-bit signed |
| Clock target | 100 MHz (10 ns period) |
| SV RAM | External (host-side), read via FIFO interface |
| FIFO depth | 4096 entries |

## Quick Start

```bash
# Run all iverilog unit tests
cd tb && make all

# Run cocotb co-simulation
cd tb && make cocotb

# Regenerate block diagram
python3 generate_block_diagram.py
```

## Differences from m1

m2 targets synthesis only (no physical layout). m3 keeps the same 256-dim feature
vector, moves argmax from the wrapper into the core (class labels written to
work_ram), completes full place-and-route to GDS, and integrates into the Caravel
chipIgnite wrapper for tape-out submission.
