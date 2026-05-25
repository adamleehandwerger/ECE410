# ECE410 — Milestone 4: Batch RBF-SVM Cardiac Arrhythmia Classifier

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**Architecture:** Batch v8 — ASIC autonomously classifies 1000×256-dim beats
**Status:** RTL updated to batch architecture (v8); new DRT in progress

The m4 milestone delivers a fully hardened GDSII layout of `svm_compute_core`
integrated into the Efabless Caravel chipIgnite `user_project_wrapper`. The
**batch architecture** (v8) removes the streaming input FIFO: the host collects
up to 1000 heartbeats at low power, pre-loads both the SV matrix and the input
matrix into off-chip SRAM, then fires a single `start` pulse. The ASIC drives
both loads autonomously over a unified 19-bit GPIO address bus.

Feature vector: 256-dim (128 single-beat morphology + 64 10-beat context +
64 100-beat context). SV RAM and input matrix are off-chip (GPIO/LA bus).
Per-sample results fire via IRQ[0] / `sample_rdy`; batch-done via IRQ[1] /
`svm_done`. Timing is clean at TT corner: +7.9 ns setup slack, 0 violations
(prior job 91947; new DRT in progress for updated RTL).

---

## Key Results (OL2 job 91947, nom_tt_025C_1v80 — prior RTL)

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

*New DRT re-hardening in progress for v8 batch RTL.*

---

## Batch Architecture (v8)

```
Host MCU (low-power continuous)
    │  Collect 1000 heartbeats, extract 256-dim features per beat
    │  Pre-load SV matrix   → off-chip SRAM rows 0..249
    │  Pre-load input matrix → off-chip SRAM rows 250..1249
    ▼
CONTROL[start=1] via Wishbone
    │
    ▼
svm_compute_core (burst, 40 MHz)
    │
    ├── LOAD_INPUT: reads 256 features from off-chip RAM (GPIO/LA bus)
    │       → stores in local feature_bank[256]
    ├── COMPUTE_DIST: reads 250 SVs × 256 words from off-chip RAM
    │       → accumulates Σ(xᵢ - svᵢ)² in distance engine
    ├── COMPUTE_KERNEL: Horner LUT → exp(-γ·d²)
    ├── OUTPUT_RESULT: accumulates kernel score per class
    └── WRITE_CLASS: argmax → sample_rdy + class_out
            │
            ├─► sample_rdy / IRQ[0] — one pulse per beat
            └─► svm_done / IRQ[1] — one pulse when batch finishes
```

Off-chip RAM address encoding: `{row[10:0], col[7:0]}` = 19-bit.
ASIC drives `GPIO[28:10]` = `ram_addr`, `GPIO[29]` = `ram_ren`.
Host drives `LA[15:0]` = `ram_rdata` with 1-cycle latency.

---

## Key Design Parameters

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat morph + 64 10-beat + 64 100-beat context) |
| Support vectors | 250 total (up to 50 per class, 5 classes) |
| Batch size | Up to 1000 samples |
| Fixed-point | Q6.10, 16-bit signed |
| Clock target | 40 MHz (25 ns, sky130_fd_sc_hd TT) |
| Off-chip RAM | Unified 19-bit GPIO bus (SVs + input matrix) |
| svm_compute_core die | 2500 × 2500 µm |
| Wrapper die | 2920 × 3520 µm (Caravel fixed) |
| DRC violations | **0** (prior run; new DRT pending) |

---

## Caravel Wishbone Memory Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done(batch) [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in this batch (1–1000) |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |

**Removed vs. v7:** FIFO_DATA (0x00), WORK_RD (0x38), STATUS2 (0x3C).

---

## GPIO Pin Assignments

| GPIO    | Signal          | Dir | Description                           |
|---------|-----------------|-----|---------------------------------------|
| [2:0]   | `class_out`     | out | Class label, stable when sample_rdy   |
| [3]     | `sample_rdy`    | out | Pulses once per beat classified        |
| [4]     | `svm_done`      | out | Pulses once at end of batch            |
| [5]     | `svm_error`     | out | Asserted on fault                      |
| [9:6]   | `error_code`    | out | 4-bit fault code                       |
| [28:10] | `ram_addr[18:0]`| out | 19-bit off-chip SRAM address           |
| [29]    | `ram_ren`       | out | Off-chip SRAM read enable              |
| LA[15:0]| `ram_rdata`     | in  | Host drives SRAM read data (1-cycle)  |

---

## Running on Orca

```bash
# Harden svm_compute_core (new RTL — run first):
sbatch ~/ece410/core_harden.sh

# After core GDS is ready, harden user_project_wrapper:
sbatch ~/ece410/wrapper_harden.sh
```

---

## Directory Structure

```
m4/
├── README.md                   ← this file
├── README_caravel.md           ← Caravel submission requirements
├── README_errorcodes.md        ← Error code reference (v8 batch)
├── README_mcu.md               ← MCU integration guide (v8 batch protocol)
├── design_summary.md           ← P&R results, architecture, design decisions
├── block_diagram.png           ← architecture block diagram
├── generate_block_diagram.py   ← renders block_diagram.png
├── confusion_comparison.png    ← sklearn vs. hardware confusion matrix
├── confusion_comparison.py     ← generates confusion_comparison.png
├── rt1/                        ← RTL source (batch v8)
│   ├── svm_compute_core.sv     ← compute core: batch FSM, distance, Horner LUT
│   │                               FEATURE_DIM=256, NUM_SV=250, MAX_BATCH=1000
│   ├── svm_fifo_sram.sv        ← (legacy — not used in v8)
│   ├── svm_sv_ram.sv           ← (legacy — not used in v8)
│   └── user_project_wrapper.sv ← Caravel wrapper: Wishbone, batch_active clock gate,
│                                   19-bit GPIO RAM bus, per-sample IRQ outputs
├── pnr/                        ← P&R scripts, configs, and reports
│   ├── config.json             ← OL2 config for svm_compute_core
│   ├── wrapper_config.json     ← OL2 config for user_project_wrapper
│   ├── core_harden.sh          ← SLURM: hardens svm_compute_core on Orca
│   ├── wrapper_harden.sh       ← SLURM: hardens user_project_wrapper on Orca
│   ├── svm_compute_core.sdc    ← SDC constraints (40 MHz, propagated clock)
│   ├── macro.cfg               ← Macro placement: u_svm at (253, 554) N
│   ├── base_user_project_wrapper.sdc ← SDC for wrapper harden
│   ├── area_report.txt         ← 146K cells, 14.1% util (prior run)
│   ├── timing_report.txt       ← Setup: +7.923 ns WNS (prior run)
│   ├── hold_timing_report.txt  ← Hold: +0.297 ns WNS (prior run)
│   ├── power_report.txt        ← 66 mW active, ~0.26 mW avg
│   ├── drc_report.txt          ← 0 DRC violations (prior run)
│   ├── critical_path.md        ← Critical path analysis
│   └── gds/
│       └── svm_compute_core.gds ← GDS from prior run (new run in progress)
├── tb/                         ← Testbenches
│   ├── README.md
│   └── ...
└── sim/
    ├── cosim_run.log           ← cocotb simulation log (prior run)
    └── cosim_waveform.png      ← VCD-derived waveform (prior run)
```

---

## Differences from v7 (streaming) to v8 (batch)

| | v7 (m4 prior) | v8 (m4 current) |
|--|---------------|-----------------|
| Input path | Stream 256 words to FIFO per beat | Pre-load input matrix in SRAM |
| FIFO | 512-deep register array | **Removed** |
| work_ram | 64-entry result buffer | **Removed** |
| SV RAM bus | GPIO[24:10]=sv_ram_addr (15-bit) | GPIO[28:10]=ram_addr (19-bit, unified) |
| Input bus | Wishbone FIFO_DATA writes | Same GPIO/LA bus, different address rows |
| Per-sample result | Poll work_ram after `done` | `sample_rdy` / IRQ[0] per beat |
| Batch done | IRQ[0]=done | IRQ[1]=svm_done |
| WB registers | 8 (incl. FIFO_DATA, WORK_RD, STATUS2) | 5 (FIFO/WORK regs removed) |
| Clock gate | `qspi_valid` based | `batch_active` register (no gap) |
