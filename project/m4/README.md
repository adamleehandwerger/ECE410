# ECE410 — Milestone 4: Place-and-Route, GDS Tape-Out & Caravel Integration

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** sky130A (SkyWater 130 nm open-PDK)  
**Accuracy:** 96.39% on MIT-BIH (sklearn = hardware, 0.00% gap, 154/256 SVs active)  
**Status:** DRT complete — 0 DRC violations; GDS/LEF/GL committed to Caravel repo

The m4 milestone delivers a fully hardened GDSII layout of `svm_compute_core`
integrated into the Efabless Caravel chipIgnite `user_project_wrapper`. Feature
vector: 256-dim (128 single-beat morphology + 64 10-beat context + 64 100-beat
context). SV RAM moved off-chip via GPIO/LA to eliminate the unavailable
sky130_sram macro. Argmax moved into the SVM core; class labels written to
work_ram instead of raw kernel values.

---

## Directory Structure

```
m4/
├── README.md                  ← this file
├── README_caravel.md          ← Caravel submission requirements and repo structure
├── design_summary.md          ← P&R results, design decisions, m3→m4 comparison
├── block_diagram.png          ← architecture block diagram (v7, Caravel integration)
├── generate_block_diagram.py  ← Python script that renders block_diagram.png (matplotlib)
├── confusion_comparison.png   ← sklearn vs. hardware confusion matrix comparison
├── confusion_comparison.py    ← script that generates confusion_comparison.png
├── rt1/                       ← RTL source files (128-feature, Caravel-integrated)
│   ├── svm_compute_core.sv    ← compute core: FSM, FIFO, distance matrix, Horner LUT
│   │                              kernel; FEATURE_DIM=256 NUM_SV=250 FIFO_DEPTH=8192
│   ├── svm_fifo_sram.sv       ← feature FIFO backed by sky130 SRAM macro interface
│   ├── svm_sv_ram.sv          ← SV RAM arbiter — off-chip GPIO/LA interface logic
│   └── user_project_wrapper.sv ← Caravel wrapper: Wishbone decode, clock gate (ICG),
│                                  work_ram (class labels), GPIO/LA pin assignments
│                                  Wishbone map (base 0x30000000):
│                                    +0x04 CONTROL  +0x08 STATUS  +0x0C NUM_SAMPLES
│                                    +0x10-0x20 NUM_SV[0-4]  +0x24 PARAM_WR
│                                    +0x38 WORK_RD  +0x3C STATUS2
├── pnr/                       ← Place-and-route scripts, configs, and reports
│   ├── config.json            ← OpenLane 2 config for svm_compute_core standalone harden
│   │                              (sky130A, sky130_fd_sc_hd, CLOCK_PERIOD=25ns)
│   ├── wrapper_config.json    ← OpenLane 2 config for user_project_wrapper harden
│   │                              (SYNTH_ELABORATE_ONLY=1, fixed 2920×3520 µm die,
│   │                               FP_DEF_TEMPLATE, RT_MAX_LAYER=met4)
│   ├── wrapper_harden.sh      ← SLURM script: hardens user_project_wrapper on Orca
│   │                              (normal partition, 24h; uses nix yosys-with-plugins
│   │                               0.46 via proot for PyOSYS + SIF for OpenROAD/Magic)
│   ├── macro.cfg              ← Macro placement: u_svm at (609, 909) N orientation
│   ├── dv_setup.sh            ← SLURM script: pulls efabless/dv Apptainer SIF and
│   │                              clones caravel-lite + MCW on Orca (run once)
│   ├── dv_run.sh              ← SLURM script: runs svm_wb_test RTL DV inside container
│   ├── area_report.txt        ← DRT-complete area: 2.895 mm², 50% utilization, sky130A
│   ├── timing_report.txt      ← Setup timing: WNS −14.04 ns, max clock ~41.6 MHz
│   ├── hold_timing_report.txt ← Hold timing report post-DRT
│   ├── power_report.txt       ← Total power: 575 mW (internal + switching + leakage)
│   ├── drc_report.txt         ← DRC: 0 violations (TritonRoute drt_v12/v13, li1–met4)
│   └── critical_path.md       ← Annotated critical path (distance accumulator chain)
├── tb/                        ← Testbenches and Caravel chip-level DV
│   ├── README.md              ← testbench overview and how to run
│   ├── Makefile               ← iverilog/cocotb build rules; `make all` runs all tests
│   ├── tb_top.sv              ← full-pipeline 5-heartbeat classification testbench
│   ├── tb_svm_params.svh      ← SV params for 256-feature 250-SV model
│   ├── tb_error_codes.sv      ← unit test: all error codes, sticky latch, reset-clear
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
│   ├── tb_warmup.sv           ← unit test: warmup-state exit and start-pulse timing
│   ├── tb_results.md          ← recorded pass/fail results for all testbenches
│   ├── svm_wb_test_tb.v       ← Caravel chip-level DV testbench: instantiates full
│   │                              Caravel SoC, monitors mprj_io[31:16] for 0xBB91 (pass)
│   ├── svm_wb_test.c          ← RISC-V firmware: writes/reads NUM_SAMPLES and NUM_SV[0-4]
│   │                              via Wishbone (base 0x30000000); signals pass via GPIO
│   ├── dv_Makefile            ← Efabless standard DV Makefile (MCW_ROOT includes)
│   ├── test_svm_compute_core.py ← cocotb Python testbench (9 tests)
│   ├── sv_ram.hex             ← 250-SV × 256-feature SV memory image
│   ├── test_features.hex      ← 5-heartbeat 256-feature test vectors
│   ├── test_labels.hex        ← ground-truth class labels
│   ├── expected_kernels.hex   ← pre-computed expected RBF kernel outputs
│   └── expected_preds.hex     ← pre-computed expected class predictions
└── sim/                       ← Simulation outputs
    ├── cosim_run.log          ← cocotb simulation log
    └── cosim_waveform.png     ← VCD-derived waveform screenshot
```

---

## Key Design Parameters (m4)

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat morph + 64 10-beat context + 64 100-beat context) |
| Support vectors | 256 (154 active at 96.39% accuracy) |
| Classes | 5 (Normal, PVC, AFib, VT, SVT) |
| Fixed-point | Q6.10, 16-bit signed |
| Clock target | 40 MHz (25 ns period, sky130_fd_sc_hd) |
| SV RAM | Off-chip via GPIO[24:10] (addr) + GPIO[25] (ren) + LA[15:0] (rdata) |
| Work RAM | On-chip 2 KB register array (2048 × 16-bit), Wishbone-accessible |
| Die size | 2920 × 3520 µm (Caravel fixed, user_project_area) |
| DRC violations | **0** |

## Caravel Wishbone Memory Map (base `0x30000000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=work_ram[0] class |
| +0x0C | NUM_SAMPLES | RW | [9:0] heartbeats per classification run |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] support vectors per class |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |
| +0x38 | WORK_RD | WO | [10:0] address to latch from work_ram |
| +0x3C | STATUS2 | RO | [15:0] work_ram read data |

## Running the Wrapper Harden on Orca

```bash
# On Orca — first time only (pulls efabless/dv SIF + caravel-lite + MCW):
sbatch ~/ece410/dv_setup.sh

# Harden the user_project_wrapper (24h job, normal partition):
cd /scratch/funphin-openlane_svm/caravel_svm_project/openlane/user_project_wrapper
sbatch ~/ece410/wrapper_harden.sh

# After dv_setup completes, run the register-access DV test:
sbatch ~/ece410/dv_run.sh
```

## Differences from m3

| | m3 | m4 |
|--|----|----|
| Feature dim | 256 | 256 (unchanged) |
| Argmax | External (wrapper) | Internal (core → work_ram) |
| SV RAM | External FIFO interface | Off-chip GPIO/LA |
| Layout | Synthesis only | DRT complete, GDS/LEF/GL |
| Caravel | Not integrated | user_project_wrapper hardened |
| DV | iverilog + cocotb | + Caravel chip-level RISC-V DV |
