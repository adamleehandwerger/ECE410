# ECE410 — Milestone 4: Place-and-Route, GDS Tape-Out & Caravel Integration

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**Accuracy:** 96.39% on MIT-BIH (sklearn = hardware, 0.00% gap)
**Status:** svm_compute_core P&R complete (job 91947) — GDS/LEF/GL committed; wrapper hardening running (job 91948)

The m4 milestone delivers a fully hardened GDSII layout of `svm_compute_core`
integrated into the Efabless Caravel chipIgnite `user_project_wrapper`. Feature
vector: 256-dim (128 single-beat morphology + 64 10-beat context + 64 100-beat
context). SV RAM is off-chip via GPIO/LA. Argmax inside the SVM core writes class
labels to work_ram. Timing is clean at TT corner: +7.9 ns setup slack, 0 violations.

---

## Key Results (OL2 job 91947, nom_tt_025C_1v80)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns), **TT CLEAN — 0 violations** |
| Setup WNS | +7.923 ns |
| Hold WNS | +0.297 ns |
| Active power | 66 mW → **~0.26 mW avg** at 80 bpm (0.4% duty cycle) |
| 14-day battery | 119 days headroom from SVM core alone |
| Cells | 146,311 standard cells |
| Die | 2500 × 2500 µm, 14.1% utilization |
| DRC | **0 violations** |

---

## Directory Structure

```
m4/
├── README.md                  ← this file
├── README_caravel.md          ← Caravel submission requirements and repo structure
├── README_errorcodes.md       ← SVM error code definitions
├── README_mcu.md              ← MCU firmware interface guide
├── design_summary.md          ← P&R results, OL2 flow, design decisions
├── block_diagram.png          ← architecture block diagram (v7, Caravel integration)
├── generate_block_diagram.py  ← renders block_diagram.png (matplotlib)
├── confusion_comparison.png   ← sklearn vs. hardware confusion matrix comparison
├── confusion_comparison.py    ← generates confusion_comparison.png
├── rt1/                       ← RTL source (256-feature, Caravel-integrated)
│   ├── svm_compute_core.sv    ← compute core: FSM, FIFO, distance, Horner LUT kernel
│   │                              FEATURE_DIM=256, NUM_SV=250, FIFO_DEPTH=512
│   ├── svm_fifo_sram.sv       ← FIFO SRAM macro interface
│   ├── svm_sv_ram.sv          ← SV RAM off-chip GPIO/LA arbiter
│   └── user_project_wrapper.sv ← Caravel wrapper: Wishbone, clock gate (ICG),
│                                  work_ram (class labels), GPIO/LA pin assignments
├── pnr/                       ← P&R scripts, configs, and reports
│   ├── config.json            ← OL2 config for svm_compute_core (job 91947)
│   │                              sky130A, sky130_fd_sc_hd, CLOCK_PERIOD=25ns
│   ├── wrapper_config.json    ← OL2 config for user_project_wrapper (job 91948)
│   ├── core_harden.sh         ← SLURM script: hardens svm_compute_core on Orca
│   ├── wrapper_harden.sh      ← SLURM script: hardens user_project_wrapper on Orca
│   ├── svm_compute_core.sdc   ← SDC constraints (40 MHz, propagated clock)
│   ├── macro.cfg              ← Macro placement: u_svm at (253, 554) N
│   ├── base_user_project_wrapper.sdc ← SDC for wrapper harden
│   ├── dv_setup.sh            ← SLURM: pulls efabless/dv SIF + caravel-lite
│   ├── dv_run.sh              ← SLURM: runs svm_wb_test RTL DV in container
│   ├── area_report.txt        ← 146K cells, 14.1% util, 2500×2500 µm
│   ├── timing_report.txt      ← Setup: +7.923 ns WNS (TT, 0 vios); SS/FF noted
│   ├── hold_timing_report.txt ← Hold: +0.297 ns WNS (TT), 0 violations
│   ├── power_report.txt       ← 66 mW active, ~0.26 mW avg (wearable budget)
│   ├── drc_report.txt         ← 0 DRC violations (OL2 integrated TritonRoute)
│   ├── critical_path.md       ← Critical path: TT clean, SS/FF corner analysis
│   └── gds/
│       └── svm_compute_core.gds  ← 181 MB GDS (git-lfs)
├── tb/                        ← Testbenches and Caravel chip-level DV
│   ├── README.md
│   ├── Makefile               ← iverilog/cocotb; `make all` runs all tests
│   ├── tb_top.sv              ← full-pipeline 5-heartbeat classification TB
│   ├── tb_svm_params.svh      ← SV params: 256-feature, 250-SV model
│   ├── tb_error_codes.sv      ← unit test: error codes, sticky latch, reset-clear
│   ├── tb_backpressure.sv     ← unit test: kernel_valid hold; late kernel_ready
│   ├── tb_consecutive.sv      ← unit test: back-to-back heartbeat processing
│   ├── tb_dist_boundary.sv    ← unit test: accumulator saturation boundary
│   ├── tb_dist_zero.sv        ← unit test: D=0 → kernel_out=1024
│   ├── tb_gamma_zero.sv       ← unit test: γ=0 edge case
│   ├── tb_interface.sv        ← unit test: port signal protocol compliance
│   ├── tb_min_sv.sv           ← unit test: minimum SV count (1 SV per class)
│   ├── tb_multi_heartbeat.sv  ← unit test: num_samples=3 loop-back
│   ├── tb_param_write.sv      ← unit test: runtime parameter write
│   ├── tb_power.sv            ← unit test: clock-gate idle power behavior
│   ├── tb_warmup.sv           ← unit test: warmup-state exit, start-pulse timing
│   ├── tb_results.md          ← pass/fail results for all testbenches
│   ├── svm_wb_test_tb.v       ← Caravel chip-level DV (mprj_io[31:16]=0xBB91 pass)
│   ├── svm_wb_test.c          ← RISC-V firmware: Wishbone register read/write
│   ├── dv_Makefile            ← Efabless standard DV Makefile
│   ├── test_svm_compute_core.py ← cocotb testbench (9 tests)
│   ├── sv_ram.hex             ← 250-SV × 256-feature SV memory image
│   ├── test_features.hex      ← 5-heartbeat 256-feature test vectors
│   ├── test_labels.hex        ← ground-truth class labels
│   ├── expected_kernels.hex   ← pre-computed expected RBF kernel outputs
│   └── expected_preds.hex     ← pre-computed expected class predictions
└── sim/
    ├── cosim_run.log          ← cocotb simulation log
    └── cosim_waveform.png     ← VCD-derived waveform screenshot
```

---

## Key Design Parameters

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat morph + 64 10-beat context + 64 100-beat context) |
| Support vectors | 250 (capped at design parameter) |
| Classes | 5 (Normal, PVC, AFib, VT, SVT) |
| Fixed-point | Q6.10, 16-bit signed |
| FIFO depth | 512 words |
| Clock target | 40 MHz (25 ns, sky130_fd_sc_hd TT) |
| SV RAM | Off-chip: GPIO[24:10] (addr), GPIO[25] (ren), LA[15:0] (rdata) |
| Work RAM | On-chip 2 KB register array (2048 × 16-bit), Wishbone-accessible |
| svm_compute_core die | 2500 × 2500 µm |
| Wrapper die | 2920 × 3520 µm (Caravel fixed, user_project_area) |
| DRC violations | **0** |

## Caravel Wishbone Memory Map (base `0x30000000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x00 | FIFO_DATA | WO | write 16-bit feature word |
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn [3]=kern_ready |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class |
| +0x0C | NUM_SAMPLES | RW | [9:0] heartbeats per run |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |
| +0x38 | WORK_RD | WO | [10:0] address to latch from work_ram |
| +0x3C | STATUS2 | RO | [15:0] work_ram read data |

## Running on Orca

```bash
# Harden svm_compute_core:
sbatch ~/ece410/core_harden.sh

# After core completes, harden user_project_wrapper:
sbatch ~/ece410/wrapper_harden.sh

# After dv_setup.sh completes, run register-access DV:
sbatch ~/ece410/dv_run.sh
```

## Differences from m3

| | m3 | m4 |
|--|----|----|
| P&R flow | Manual OpenROAD scripts | OpenLane 2 v2.3.10 Classic |
| Clock target | 100 MHz (incorrect) | **40 MHz** |
| Setup timing (TT) | −14.04 ns (violated) | **+7.923 ns (clean)** |
| Active power | 575 mW | **66 mW** |
| Die size | 2895 µm² (manual) | 2500×2500 µm (OL2 floorplan) |
| Utilization | 50% | 14.1% |
| SRAM macros | 4× sky130_sram_1kbyte | None (register arrays) |
| FIFO depth | 4096 | 512 |
| Caravel | Not integrated | user_project_wrapper hardened |
