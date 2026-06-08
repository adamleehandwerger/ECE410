# ECE410 — Milestone 3: Place-and-Route

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd  
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys + OpenROAD + TritonRoute)  
**Architecture:** Batch v9 — host pre-loads SV + input matrix; ASIC classifies autonomously  
**Status:** `svm_compute_core` hardened (job 91966, 0 DRC, +7.83 ns WNS). `user_project_wrapper` hardening and Caravel submission → m4.

---

## Directory Structure

```
m3/
├── README.md                    ← this file — full m3 catalog
├── README_caravel.md            ← Caravel chipIgnite overview (reference; submission in m4)
├── README_errorcodes.md         ← 13 error codes, sticky latch, reset-clear reference
├── README_mcu.md                ← MCU integration guide (batch pre-load protocol)
├── design_summary.md            ← P&R results: area, power, timing, architecture
├── block_diagram.png            ← hardware block diagram (v9, batch architecture)
├── generate_block_diagram.py    ← renders block_diagram.png (matplotlib)
├── confusion_comparison_m4.py   ← Numba Q6.10 vs ASIC confusion matrix script
├── confusion_comparison_m4.png  ← sklearn vs ASIC confusion matrix plot
├── throughput_comparison.txt    ← inference time and power comparison summary (ASIC vs CPU)
│
├── rt1/                         ← RTL source (v9, final)
│   ├── top.sv                   ← top-level Caravel wrapper (user_project_wrapper)
│   ├── compute_core.sv          ← SVM core: NUM_SV=500, alpha_addr[8:0], batch FSM
│   └── interface.sv             ← SystemVerilog interface definitions (svm_data_if,
│                                    svm_ctrl_if) used by the compute core
│
├── tb/                          ← Unit testbenches (13 tests, all PASS)
│   ├── README.md                ← testbench overview and how to run
│   ├── Makefile                 ← iverilog/cocotb build rules; `make all` runs all tests
│   ├── dv_Makefile              ← Caravel DV framework Makefile (Wishbone C test)
│   ├── tb_top.sv                ← full-pipeline 5-heartbeat classification testbench
│   ├── tb_svm_params.svh        ← auto-generated SV include: dual coefficients,
│   │                                intercepts, SV counts for all 5 classes
│   ├── tb_error_codes.sv        ← unit test: all 13 error codes, sticky latch, reset-clear
│   ├── tb_backpressure.sv       ← unit test: kernel_valid hold; late kernel_ready release
│   ├── tb_consecutive.sv        ← unit test: back-to-back heartbeat processing
│   ├── tb_dist_boundary.sv      ← unit test: accumulator saturation boundary
│   ├── tb_dist_zero.sv          ← unit test: D=0 → kernel_out=1024 (exp(0)=1 in Q6.10)
│   ├── tb_gamma_zero.sv         ← unit test: γ=0 edge case
│   ├── tb_interface.sv          ← unit test: port signal protocol compliance
│   ├── tb_min_sv.sv             ← unit test: minimum SV count (1 SV per class)
│   ├── tb_multi_heartbeat.sv    ← unit test: num_samples=3 loop-back
│   ├── tb_param_write.sv        ← unit test: runtime parameter write via param_write_en
│   ├── tb_power.sv              ← unit test: clock-gate idle power behavior
│   ├── tb_warmup.sv             ← unit test: warmup-state exit and start-pulse timing
│   ├── tb_results.md            ← recorded pass/fail results for all testbenches
│   ├── test_svm_compute_core.py ← cocotb Python testbench (9 tests via simulation API)
│   ├── svm_wb_test.c            ← Wishbone C test (Caravel DV framework)
│   ├── svm_wb_test_tb.v         ← Verilog wrapper for Caravel DV test
│   ├── sv_ram.hex               ← 256-SV × 256-feature support vector memory image
│   ├── test_features.hex        ← 5-heartbeat feature vectors (one per class)
│   ├── test_labels.hex          ← ground-truth class labels for test_features.hex
│   ├── expected_kernels.hex     ← pre-computed expected RBF kernel outputs
│   └── expected_preds.hex       ← pre-computed expected class predictions
│
├── sim/                         ← Simulation outputs (Wishbone cocotb cosim)
│   ├── cosim_run.log            ← full cosim log: 300 samples, 97.67% accuracy
│   ├── cosim_waveform.png       ← VCD-derived waveform screenshot (classification run)
│   └── asic_preds.csv           ← 300 ASIC predictions (last cosim run)
│
├── synth/                       ← Place-and-route summary (OL2 job 91966, sky130A)
│   ├── config.json              ← OpenLane 2 config: CLOCK_PERIOD=25 ns, sky130_fd_sc_hd
│   ├── openlane_run.log         ← full P&R run log (synthesis → DRC, SLURM job 91966)
│   ├── timing_report.txt        ← STA: setup WNS +7.83 ns, hold WNS +0.30 ns, 0 violations
│   ├── area_report.txt          ← 2500×2500 µm, ~146K cells, ~14% utilization
│   ├── power_report.txt         ← 66 mW active; internal/switching/leakage breakdown
│   └── critical_path.md         ← annotated critical path through distance accumulator
│
├── bench/                       ← Benchmark: ASIC vs optimized Python
│   ├── benchmark.md             ← accuracy, throughput, power, energy efficiency tables
│   ├── benchmark_data.csv       ← raw measurements (ASIC measured; CPU estimated)
│   ├── roofline_final.png       ← dual-panel roofline + power-efficiency chart
│   └── roofline_final.py        ← script that generates roofline_final.png (matplotlib)
│
└── pnr/                         ← Full P&R artifacts (scripts, GDS, SDC)
    ├── config.json              ← OL2 wrapper config
    ├── core_config.json         ← OL2 core config (svm_compute_core)
    ├── core_harden.sh           ← SLURM: hardens svm_compute_core on Orca
    ├── wrapper_harden.sh        ← SLURM: hardens user_project_wrapper (→ m4)
    ├── svm_compute_core.sdc     ← timing constraints (core)
    ├── base_user_project_wrapper.sdc ← timing constraints (wrapper)
    ├── macro.cfg                ← macro placement config
    ├── wrapper_config.json      ← OL2 config for user_project_wrapper
    ├── timing_report.txt        ← STA setup report (job 91966)
    ├── hold_timing_report.txt   ← STA hold report (job 91966)
    ├── power_report.txt         ← power report (job 91966)
    ├── area_report.txt          ← area/utilization report (job 91966)
    ├── drc_report.txt           ← DRC: 0 violations
    ├── critical_path.md         ← critical path analysis
    ├── dv_run.sh                ← Caravel DV run script
    ├── dv_setup.sh              ← Caravel DV environment setup
    └── gds/
        └── svm_compute_core.gds ← 226 MB hardened GDS (job 91966)
```

---

## Key Design Parameters

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat + 64 10-beat + 64 RR history) |
| Support vectors | 500 total (100 per class, 5 classes) |
| Fixed-point | Q6.10, 16-bit signed |
| Gamma / C | 0.25 / 1.0 |
| Clock | 40 MHz (25 ns) |
| Setup WNS | +7.83 ns (TT, 0 violations) |
| Hold WNS | +0.30 ns (TT, 0 violations) |
| Active power | 66 mW → 0.284 mW avg at 80 bpm (0.431% duty cycle) |
| Die | 2500 × 2500 µm, ~14% utilization, ~146K cells |
| DRC | 0 violations |
| ASIC accuracy | 97.67% (293/300) — exact match with sklearn, zero gap |

## Quick Start

```bash
# Run all iverilog unit tests
cd tb && make all

# Run cocotb co-simulation (300 samples, ~96 min)
cd tb && make cocotb

# Regenerate block diagram
python3 generate_block_diagram.py

# Regenerate roofline plot
cd bench && python3 roofline_final.py   # or rerun benchmark script
```

## Differences from m2

m2 targeted synthesis only (no physical layout). m3 adds:
- Full place-and-route to GDS (OL2 job 91966, sky130A)
- Batch architecture v9 — off-chip SRAM pre-load via 19-bit GPIO bus
- NUM_SV increased to 500 (100/class), alpha_addr widened to 9-bit
- Caravel Wishbone register map (CONTROL, STATUS, NUM_SAMPLES, ALPHA_WR)
- Benchmark suite (bench/) comparing ASIC vs sklearn and Numba on Orca

## Next Step → m4

Wrapper hardening (`user_project_wrapper`) and Caravel submission in `project/m4/`.  
See `m4/README.md` for wrapper harden results, RAM_LATENCY feature, and precheck.
