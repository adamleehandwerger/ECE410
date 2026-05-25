# SVM Compute Core — Full-Chip Design Summary (m5: Wrapper Hardening)

**Project:** Multi-Class Cardiac Arrhythmia Detection — Caravel chipIgnite Tape-Out
**Technology:** sky130A / sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic
**Architecture:** Batch v8 — host pre-loads SV + input matrix; ASIC classifies autonomously
**RTL freeze:** m4/rt1 (batch v8 — svm_compute_core.sv, user_project_wrapper.sv)

---

## Component Summary

### svm_compute_core (m4, batch v8 — new DRT in progress)

| Metric | Value (prior job 91947) |
|--------|------------------------|
| Clock | 40 MHz (25 ns), TT corner clean |
| Setup WNS (TT) | +7.923 ns — 0 violations |
| Hold WNS (TT) | +0.297 ns — 0 violations |
| Active power | 66 mW (new: expected lower — FIFO removed) |
| Avg power (80 bpm, batch) | **~0.15 mW** (0.23% duty cycle) |
| Cells | 146,311 (new: expected fewer — FIFO + work_ram removed) |
| Die | 2500 × 2500 µm, 14.1% utilization (new: expected lower) |
| DRC | 0 violations |
| GDS | Pending new DRT |
| LEF | Pending new DRT |
| GL netlist | Pending new DRT |

### user_project_wrapper (m5, pending new DRT after core re-harden)

| Metric | Value |
|--------|-------|
| Die | 2920 × 3520 µm (Caravel fixed) |
| Macro | u_svm at (253, 554) N — 2500 × 2500 µm footprint |
| Clock | wb_clk_i (Caravel), gated to svm_gclk via ICG |
| CTS | Disabled (RUN_CTS: 0) — wrapper uses macro clock |
| STA | Post-route TBD (pending new core + wrapper DRT) |
| GDS | Pending |

---

## Batch Architecture (v8)

### What the Host Does

```
MCU (low-power, continuous)
    │
    │  1. Collect 1000 heartbeats (250 Hz ECG → feature extraction)
    │  2. Load SV matrix  (250 SVs × 256 features) → SRAM rows 0..249
    │  3. Load input matrix (1000 beats × 256 features) → SRAM rows 250..1249
    │  4. Write NUM_SAMPLES = 1000, fire CONTROL[start]
    │
    ▼  ASIC takes over:
    ├── LOAD_INPUT per beat: 256 cycles (reads from SRAM via GPIO/LA)
    ├── COMPUTE_DIST per SV: 258 cycles (reads SV from same SRAM)
    ├── COMPUTE_KERNEL: 20 cycles (Horner LUT exp approximation)
    └── WRITE_CLASS: argmax → sample_rdy (IRQ[0]) → class_out stable
                    last beat → svm_done (IRQ[1])
```

### Off-chip RAM Bus

| Signal | Pin | Direction | Description |
|--------|-----|-----------|-------------|
| `ram_addr[18:0]` | GPIO[28:10] | ASIC out | Row×256 + column |
| `ram_ren` | GPIO[29] | ASIC out | Read strobe |
| `ram_rdata[15:0]` | LA[15:0] | Host in→ASIC | 1-cycle latency response |

Address layout: `{row[10:0], col[7:0]}`. Rows 0..249 = SVs; rows 250..1249 = inputs.

---

## Full-Chip Power Estimate (Batch Architecture)

| Subsystem | Active Power | Duty Cycle | Avg Power |
|-----------|-------------|-----------|-----------|
| svm_compute_core (batch) | 66 mW | 0.23% (1.75 s / 750 s) | **0.15 mW** |
| Caravel management SoC | ~5 mW | ~5% | ~0.25 mW |
| ECG frontend (analog) | ~0.5 mW | 100% | 0.5 mW |
| BLE (optional, logging) | ~10 mW | ~0.1% | ~0.01 mW |
| **Total estimated** | — | — | **~0.9 mW** |

Battery budget: 200 mAh @ 3.7V = 740 mWh → **740 hours / 0.9 mW ≈ 34 days**.
14-day target met with 2.4× margin. SVM core alone: ~200 days.

---

## Wishbone Register Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in batch (1–1000) |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |

**Removed vs. v7:** FIFO_DATA (0x00), WORK_RD (0x38), STATUS2 (0x3C).

---

## Caravel Submission Artifacts

### Required files (caravel_svm_project repo)

| File | Status | Notes |
|------|--------|-------|
| `gds/svm_compute_core.gds` | ⏳ New DRT in progress | Prior run: 181 MB |
| `lef/svm_compute_core.lef` | ⏳ New DRT in progress | Prior run: 108 KB |
| `verilog/gl/svm_compute_core.v` | ⏳ New DRT in progress | Prior run: 13 MB |
| `gds/user_project_wrapper.gds` | ⏳ Pending core + wrapper DRT | — |
| `lef/user_project_wrapper.lef` | ⏳ Pending | — |
| `verilog/gl/user_project_wrapper.v` | ⏳ Pending | — |

### Efabless mpw-precheck gates

| Check | Status |
|-------|--------|
| Manifest | ⏳ Pending new artifacts |
| Consistency | ⏳ |
| XOR (Magic vs. KLayout) | ⏳ |
| DRC (Magic) | ⏳ |
| LVS (netgen) | ⏳ |
| Antenna check | ⏳ |

---

## Design Architecture

### Feature Pipeline (256-dim, 40 MHz burst)

```
ECG signal (250 Hz)
    │
    ▼
Feature extraction (host MCU)
    │  256 features per beat:
    │  128 single-beat morphology
    │   64 10-beat context
    │   64 100-beat context
    ▼
Host pre-loads SRAM:
    │  rows 0..249   = SV matrix   (250 × 256 × Q6.10)
    │  rows 250..N+249 = input matrix (N × 256 × Q6.10)
    ▼
CONTROL[start=1] via Wishbone
    │
    ▼
svm_compute_core (ASIC burst):
    ├── LOAD_INPUT   — fetch 256 words from SRAM → feature_bank
    ├── COMPUTE_DIST — fetch SV from SRAM, Σ(xᵢ-svᵢ)² for all 250 SVs
    ├── COMPUTE_KERNEL — Horner LUT: exp(-γ·d²)
    ├── OUTPUT_RESULT  — accumulate kernel scores per class
    └── WRITE_CLASS    — argmax → sample_rdy + class_out
                              └─► IRQ[0] per beat, IRQ[1] at batch end
```

### Fixed-Point Precision (Q6.10)

- 16-bit signed: 1 sign + 6 integer + 10 fractional bits
- γ = 0.25 → exactly 0x0100 in Q6.10 → zero quantization error
- Hardware accuracy = sklearn accuracy: **96.39% on MIT-BIH, 0.00% gap**

---

## Yosys / OpenSTA Compatibility Fixes

Carried forward from job 91947, now also applying to updated RTL:

| Fix | Status |
|-----|--------|
| Unpacked array output port | ✅ Removed |
| `return` in function case arms | ✅ Fixed |
| `$mem` inference on feature_bank | ✅ `ram_style = "registers"` |
| Non-constant async reset | ✅ `arm_interrupted` removed |
| `sim_sram_models.sv` as gate-level | ✅ `sta-blackbox` |
| OpenSTA corner.tcl `catch {}` | ✅ Applied on Orca |
| 512-depth FIFO `$mem` | ✅ **Moot — FIFO removed in v8** |
| 64-entry work_ram | ✅ **Moot — work_ram removed in v8** |

---

## Repository Structure (caravel_svm_project)

```
caravel_svm_project/
├── gds/
│   ├── svm_compute_core.gds        ← pending new DRT
│   └── user_project_wrapper.gds    ← pending
├── lef/
│   ├── svm_compute_core.lef        ← pending new DRT
│   └── user_project_wrapper.lef    ← pending
├── verilog/
│   ├── rtl/
│   │   ├── svm_compute_core.sv     ← batch v8 RTL ← m4/rt1/
│   │   ├── user_project_wrapper.sv ← batch v8 RTL ← m4/rt1/
│   │   └── sim_sram_models.sv      ← behavioral models (sta-blackbox)
│   └── gl/
│       ├── svm_compute_core.v      ← pending new DRT
│       └── user_project_wrapper.v  ← pending
└── openlane/
    ├── svm_compute_core/           ← OL2 config + SDC
    └── user_project_wrapper/       ← OL2 config + macro.cfg
```

---

*Document version: m5 · 2026-05-24 — batch v8; new DRT in progress*
