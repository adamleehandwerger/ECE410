# ECE410 — Milestone 4: svm_compute_core Hardened (Batch Architecture v9)

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys + OpenROAD + TritonRoute)
**Architecture:** Batch v9 — host pre-loads SV + input matrix; ASIC classifies autonomously
**Status:** svm_compute_core hardened (job 91966). Wrapper hardening in m5.

---

## Key Results (OL2 job 91966, nom_tt_025C_1v80)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns), **TT CLEAN — 0 violations** |
| Setup WNS | +7.83 ns |
| Hold WNS | +0.30 ns |
| Active power | 66 mW → **0.284 mW avg** at 80 bpm (0.431% duty cycle) |
| 14-day battery | 108 days headroom from SVM core alone |
| Cells | ~146K standard cells |
| Die | 2500 × 2500 µm, ~14% utilization |
| DRC | **0 violations** |
| GDS | 226 MB |
| ASIC accuracy | **97.67%** (293/300) = sklearn, zero gap |

---

## Batch Architecture (v9)

```
Host MCU (low-power continuous)
    │  Collect 1000 heartbeats, extract 256-dim features per beat
    │  Pre-load SV matrix    → off-chip SRAM rows 0..499
    │  Pre-load input matrix → off-chip SRAM rows 500..1499
    ▼
CONTROL[start=1] via Wishbone
    │
    ▼
svm_compute_core (burst, 40 MHz)
    │
    ├── LOAD_INPUT: reads 256 features from off-chip RAM (GPIO/LA bus)
    │       → stores in local feature_bank[256]
    ├── COMPUTE_DIST: reads 500 SVs × 256 words from off-chip RAM
    │       → accumulates Σ(xᵢ - svᵢ)² in distance engine
    ├── COMPUTE_KERNEL: Horner LUT → exp(-γ·d²)
    ├── OUTPUT_RESULT: accumulates alpha-weighted kernel score per class
    └── WRITE_CLASS: argmax → sample_rdy + class_out
            │
            ├─► sample_rdy / IRQ[0] — one pulse per beat
            └─► done / IRQ[1] — one pulse when batch finishes
```

Off-chip RAM: `{row[10:0], col[7:0]}` = 19-bit address.
ASIC drives `GPIO[28:10]` = `ram_addr`, `GPIO[29]` = `ram_ren`.
Host drives `LA[15:0]` = `ram_rdata` with 1-cycle latency.

---

## Key Design Parameters

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat + 64 10-beat + 64 RR history) |
| Support vectors | 500 total (100 per class, 5 classes) |
| Fixed-point | Q6.10, 16-bit signed |
| Gamma | 0.25 (γ=0x0100 in Q6.10, zero quantization error) |
| Clock | 40 MHz (25 ns) |
| Off-chip RAM | 19-bit GPIO bus (rows 0–499=SVs, 500–1499=inputs) |
| Die | 2500 × 2500 µm |
| DRC | 0 violations |

---

## Caravel Wishbone Memory Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in batch (1–1000) |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class (max 100 each) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data (γ, C, bias) |
| +0x28 | ALPHA_WR | WO | [24:16]=sv_global_idx (9-bit) [15:0]=alpha Q6.10 |

---

## GPIO Pin Assignments

| GPIO    | Signal           | Dir | Description                         |
|---------|------------------|-----|-------------------------------------|
| [2:0]   | `class_out`      | out | Class label (0=N, 1=PVC, 2=AFib, 3=VT, 4=SVT) |
| [3]     | `sample_rdy`     | out | Pulses once per classified beat     |
| [4]     | `done`           | out | Pulses once at end of batch         |
| [5]     | `error`          | out | Asserted on fault                   |
| [9:6]   | `error_code`     | out | 4-bit fault code                    |
| [28:10] | `ram_addr[18:0]` | out | 19-bit off-chip SRAM address        |
| [29]    | `ram_ren`        | out | Off-chip SRAM read enable           |
| LA[15:0]| `ram_rdata`      | in  | Host drives SRAM read data (1-cycle)|

---

## Directory Structure

```
m4/
├── README.md                    ← this file
├── README_caravel.md            ← Caravel submission overview
├── README_errorcodes.md         ← Error code reference
├── README_mcu.md                ← MCU integration guide
├── design_summary.md            ← P&R results and architecture
├── block_diagram.png            ← hardware block diagram
├── generate_block_diagram.py    ← renders block_diagram.png
├── confusion_comparison_m4.py   ← Numba Q6.10 vs ASIC confusion matrix
├── confusion_comparison_m4.png  ← output plot
├── throughput_comparison.txt    ← inference speed and power summary
├── rt1/                         ← RTL source (v9, final)
│   ├── svm_compute_core.sv      ← core: NUM_SV=500, alpha_addr[8:0]
│   └── user_project_wrapper.sv  ← Caravel wrapper: reg_alpha_wr[24:0]
├── pnr/                         ← P&R scripts, configs, reports
│   ├── config.json              ← OL2 config for svm_compute_core
│   ├── core_harden.sh           ← SLURM: hardens svm_compute_core on Orca
│   ├── wrapper_harden.sh        ← SLURM: hardens user_project_wrapper (→ m5)
│   ├── timing_report.txt        ← +7.83 ns setup WNS (job 91966)
│   ├── power_report.txt         ← 66 mW active, 0.284 mW avg
│   ├── drc_report.txt           ← 0 DRC violations
│   ├── area_report.txt          ← 2500×2500 µm, ~14% utilization
│   └── critical_path.md         ← critical path analysis
├── sim/
│   └── asic_preds.csv           ← 300 ASIC predictions (97.67%)
└── tb/                          ← unit testbenches (13 tests, all PASS)
    ├── tb_top.sv
    ├── tb_error_codes.sv
    └── ...
```

---

## Next Step → m5

Wrapper hardening (`user_project_wrapper`) is in `project/m5/`.
See `m5/README.md` for wrapper harden results, submission artifacts, and precheck.
