# ECE410 SVM ASIC — Milestone 5 (m5)

**Student:** Adam Handwerger · handwerg@pdx.edu  
**Course:** ECE410, Portland State University  
**Project:** 5-class Cardiac Arrhythmia Classifier, RBF-SVM ASIC, sky130A  
**Revision:** v9 (batch architecture, NUM_SV=500, 97.67% accuracy)

---

## Results

| Metric | Value |
|--------|-------|
| ASIC accuracy | **97.67%** (293/300) |
| sklearn accuracy | 97.67% (293/300) |
| Accuracy gap | **0.00%** — exact match |
| Support vectors | 500 total (100/class) |
| Feature dim | 256 (128 + 64 + 64 multi-scale) |
| Gamma / C | 0.25 / 1.0 (Q6.10 fixed-point) |
| Inference time | 3.23 ms / beat @ 40 MHz |
| Active power | ~66 mW |
| Avg power (80 bpm) | 0.284 mW |
| Core die area | 2500 × 2500 µm (6.25 mm²) |
| Wrapper die area | 2920 × 3520 µm (10.28 mm², Caravel fixed) |
| Core setup WNS | +7.83 ns (TT, 0 violations) |
| Core hold WNS | +0.30 ns (TT, 0 violations) |
| Core DRC | 0 violations |

---

## Directory Structure

```
project/m5/
├── README.md                   ← this file
├── README_submission.md        ← Caravel submission requirements & status
├── block_diagram.png           ← hardware block diagram (v9)
├── design_summary.md           ← full design summary: area, power, timing
├── generate_block_diagram.py   ← generates block_diagram.png
│
├── rt1/                        ← RTL source (v9, final)
│   ├── svm_compute_core.sv     ← SVM core: NUM_SV=500, alpha_addr[8:0]
│   └── user_project_wrapper.sv ← Caravel wrapper: reg_alpha_wr[24:0]
│
├── sim/                        ← Simulation / cosim
│   ├── Makefile                ← `make sim` runs Wishbone cosim
│   ├── README.md               ← sim setup & usage
│   ├── tb_wb_cosim.py          ← cocotb testbench (Wishbone batch cosim)
│   ├── sky130_stubs.v          ← sky130 cell stubs for Icarus
│   ├── asic_preds.csv          ← 300 ASIC predictions (last run)
│   ├── confusion_comparison_m5.py  ← generates comparison plot
│   ├── confusion_comparison_m5.png ← sklearn vs ASIC confusion matrices
│   ├── throughput_comparison.txt   ← inference time / power summary
│   └── sim_build/              ← Icarus compiled sim (generated)
│
├── pnr/                        ← Place-and-route reports
│   ├── timing_report.txt       ← STA results (jobs 91966/91967)
│   ├── drc_report.txt          ← DRC/LVS results
│   ├── area_report.txt         ← Die area and utilization
│   ├── power_report.txt        ← Active and average power
│   ├── base_user_project_wrapper.sdc  ← timing constraints
│   ├── macro.cfg               ← macro placement config
│   ├── wrapper_config.json     ← OpenLane wrapper config
│   └── wrapper_harden.sh       ← wrapper harden SLURM script
│
├── precheck/                   ← Efabless mpw-precheck
│   ├── precheck_run.sh         ← SLURM script to run precheck on Orca
│   └── precheck_results.txt    ← results (pending precheck run)
│
└── submission/
    └── checklist.md            ← submission checklist (all items tracked)
```

---

## Caravel Repo (`caravel_svm_project`)

Physical artifacts live in the separate caravel repo at  
`https://github.com/adamleehandwerger/caravel_svm_project`:

```
caravel_svm_project/
├── gds/
│   ├── svm_compute_core.gds        ← 226 MB  (job 91966, LFS)
│   └── user_project_wrapper.gds    ← 230 MB  (job 91967, LFS)
├── lef/
│   ├── svm_compute_core.lef        ← 94 KB   (job 91966)
│   └── user_project_wrapper.lef    ← 195 KB  (job 91967)
├── verilog/
│   ├── rtl/
│   │   ├── svm_compute_core.sv     ← v9 RTL (NUM_SV=500)
│   │   └── user_project_wrapper.sv ← v9 RTL (reg_alpha_wr[24:0])
│   └── gl/
│       ├── svm_compute_core.v      ← 13 MB GL netlist (job 91966)
│       └── user_project_wrapper.v  ← 78 KB GL netlist (job 91967)
└── openlane/
    ├── svm_compute_core/           ← OL2 config + SDC
    └── user_project_wrapper/       ← OL2 config + macro.cfg
```

---

## Running the Cosim

```bash
cd project/m5/sim

# Full 300-sample Wishbone cosim (~96 min)
PYTHONUNBUFFERED=1 make sim

# Quick subset (25 samples, any gamma)
COSIM_N_EVAL=25 COSIM_GAMMA=0.25 PYTHONUNBUFFERED=1 make sim

# Generate confusion matrix plot
python3 confusion_comparison_m5.py
```

Requires: `pip install cocotb scikit-learn wfdb matplotlib numpy`  
PhysioNet cache: `~/.physionet_cache/` (avoids 45-min re-download)  
NumPy cache: `/tmp/cosim_cache_ecg_n300_d256.npz` (cleared on reboot)

---

## Off-chip RAM Protocol

```
Address: {row[10:0], col[7:0]} = 19-bit
  Rows 0–499    → SV matrix     (500 × 256 × 2 B = 256 KB)
  Rows 500–1499 → input matrix  (1000 × 256 × 2 B = 512 KB)

GPIO[28:10] = ram_addr[18:0]   (ASIC drives)
GPIO[29]    = ram_ren           (ASIC drives, 1-cycle strobe)
LA[15:0]    = ram_rdata[15:0]  (host drives, 1-cycle latency)
```

---

## Wishbone Register Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=err_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] heartbeats per batch |
| +0x10–0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class (max 100 each) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data (γ, C, bias) |
| +0x28 | ALPHA_WR | WO | [24:16]=sv_global_idx (9-bit) [15:0]=alpha Q6.10 |

---

*Last updated: 2026-05-25*
