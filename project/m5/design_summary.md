# SVM Compute Core — Full-Chip Design Summary (m5: Wrapper & Submission)

**Project:** Multi-Class Cardiac Arrhythmia Detection — Caravel chipIgnite Tape-Out
**Technology:** sky130A / sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic
**RTL freeze:** m4 (svm_compute_core.sv, user_project_wrapper.sv)

---

## Component Summary

### svm_compute_core (m4, job 91947 — COMPLETE)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns), TT corner clean |
| Setup WNS (TT) | +7.923 ns — 0 violations |
| Hold WNS (TT) | +0.297 ns — 0 violations |
| Active power | 66 mW |
| Avg power (80 bpm) | ~0.26 mW |
| Cells | 146,311 |
| Die | 2500 × 2500 µm, 14.1% utilization |
| DRC | 0 violations |
| GDS | 181 MB (Magic stream-out) |
| LEF | 108 KB |
| GL netlist | 13 MB |

### user_project_wrapper (m5, job 91948 — IN PROGRESS)

| Metric | Value |
|--------|-------|
| Die | 2920 × 3520 µm (Caravel fixed) |
| Macro | u_svm at (253, 554) N — 2500×2500 µm footprint |
| Macro margin | X: 167 µm, Y: 466 µm |
| Clock | wb_clk_i (Caravel), gated to svm_gclk via ICG |
| CTS | Disabled (RUN_CTS: 0) — wrapper uses macro clock |
| STA | Post-route (TBD — pending job 91948) |
| GDS | TBD (expected ~1–2 GB) |

---

## Full-Chip Power Estimate

Wearable target: strap-on cardiac monitor, 14-day rechargeable battery.

| Subsystem | Active Power | Duty Cycle | Avg Power |
|-----------|-------------|-----------|-----------|
| svm_compute_core | 66 mW | 0.4% (3 ms / 750 ms) | 0.26 mW |
| Caravel management SoC | ~5 mW (estimate) | ~5% | ~0.25 mW |
| ECG frontend (analog) | ~0.5 mW | 100% (continuous) | 0.5 mW |
| BLE (data logging, optional) | ~10 mW | ~0.1% | ~0.01 mW |
| **Total estimated** | — | — | **~1.0 mW** |

Battery budget: 200 mAh @ 3.7V = 740 mWh → **740 hours / 1.0 mW ≈ 30 days**.
14-day target met with 2× margin. SVM core alone is 119 days.

---

## Caravel Submission Artifacts

### Required files (caravel_svm_project repo)

| File | Status | Size |
|------|--------|------|
| `gds/svm_compute_core.gds` | ✅ Local + ECE410 repo LFS | 181 MB |
| `lef/svm_compute_core.lef` | ✅ Committed | 108 KB |
| `verilog/gl/svm_compute_core.v` | ✅ Committed | 13 MB |
| `gds/user_project_wrapper.gds` | ⏳ Pending job 91948 | ~1–2 GB |
| `lef/user_project_wrapper.lef` | ⏳ Pending job 91948 | ~1 MB |
| `verilog/gl/user_project_wrapper.v` | ⏳ Pending job 91948 | ~1 MB |

### Efabless mpw-precheck gates

| Check | Expected | Status |
|-------|----------|--------|
| Manifest | All required files present | ⏳ |
| Consistency | Netlists match GDS hierarchy | ⏳ |
| XOR (Magic vs. KLayout) | No differences | ⏳ |
| DRC (Magic) | 0 violations | ⏳ |
| LVS (netgen) | No shorts/opens | ⏳ |
| Antenna check | No violations | ⏳ |

---

## Design Architecture

### Feature Pipeline (256-dim, 40 MHz)

```
ECG signal (250 Hz sample rate)
    │
    ▼
Feature extraction (host MCU)
    │  256 features per heartbeat
    │  128 single-beat morphology (RR intervals, waveform shape)
    │   64 10-beat context (short-term rhythm)
    │   64 100-beat context (long-term rhythm)
    ▼
Wishbone FIFO write (0x30000000)
    │  256 × 16-bit Q6.10 values
    ▼
svm_compute_core (ASIC, 40 MHz)
    │
    ├── FIFO (512 × 16-bit register array)
    ├── Feature bank (256 × 16-bit)
    ├── Distance engine: Σ(xᵢ - svᵢ)² for each of 250 SVs
    │       → Q6.10 fixed-point accumulation
    │       → Horner LUT: exp(-γ·d²) ≈ exp(-int) × exp(-frac)
    ├── Score accumulation: Σ αᵢ·yᵢ·K(x, svᵢ) per class + bias
    └── Argmax: class label → work_ram[sample_counter]
    │
    ▼
done pulse → IRQ → MCU reads STATUS (0x30000008)
    │  class_out[2:0] = {Normal, PVC, AFib, VT, SVT}
    ▼
Optional: read work_ram[0..N-1] for batch of N heartbeat labels
```

### Fixed-Point Precision (Q6.10)

- 16-bit signed: 1 sign + 6 integer + 10 fractional bits
- Range: −32 to +31.999 (covers all feature values after normalization)
- γ = 0.25 → exactly 0x0100 in Q6.10 → no quantization error
- Hardware accuracy = sklearn accuracy: **96.39% on MIT-BIH, 0.00% gap**

---

## Yosys / OpenSTA Compatibility Fixes Applied

All fixes are in `svm_compute_core.sv` (m4 RTL, frozen):

| Issue | Fix Applied |
|-------|-------------|
| Unpacked array output port `bias_reg[5]` | Port removed |
| `return expr` in function case arms | Replaced with `fn_name = expr` |
| `$mem` inference in FIFO and feature_bank | `(* ram_style = "registers" *)` |
| Non-constant async reset (`$_ALDFFE_PNP_`) | `arm_interrupted` signal removed |
| `sim_sram_models.sv` treated as gate-level | `/// sta-blackbox` added |
| OpenSTA `corner.tcl` unsupported properties | 6 `catch {}` patches on Orca |

---

## Repository Structure (caravel_svm_project)

```
caravel_svm_project/
├── gds/
│   ├── svm_compute_core.gds        ← 181 MB (local + ECE410 LFS)
│   └── user_project_wrapper.gds    ← pending
├── lef/
│   ├── svm_compute_core.lef        ← 108 KB ✅
│   └── user_project_wrapper.lef    ← pending
├── verilog/
│   ├── rtl/
│   │   ├── svm_compute_core.sv     ← RTL (frozen at m4)
│   │   ├── user_project_wrapper.sv ← RTL (frozen at m4)
│   │   └── sim_sram_models.sv      ← behavioral models (sta-blackbox)
│   └── gl/
│       ├── svm_compute_core.v      ← 13 MB GL netlist ✅
│       └── user_project_wrapper.v  ← pending
└── openlane/
    ├── svm_compute_core/           ← OL2 config + SDC (job 91947)
    └── user_project_wrapper/       ← OL2 config + DEF template (job 91948)
```

---

*Document version: m5 · 2026-05-24 — wrapper pending job 91948*
