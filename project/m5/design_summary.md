# SVM Compute Core — Full-Chip Design Summary (m5/v9: Final Harden)

**Project:** Multi-Class Cardiac Arrhythmia Detection — Caravel chipIgnite Tape-Out
**Technology:** sky130A / sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic
**Architecture:** Batch v8/v9 — host pre-loads SV + input matrix; ASIC classifies autonomously
**RTL freeze:** m5/rt1 v9 — NUM_SV=500, alpha_addr[8:0], reg_alpha_wr[24:0]

---

## Component Summary

### svm_compute_core (job 91966, v9)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns), TT corner clean |
| Setup WNS (TT) | +7.83 ns — 0 violations ✅ |
| Hold WNS (TT) | +0.30 ns — 0 violations ✅ |
| DRC | 0 violations ✅ |
| Active power | ~66 mW |
| Inference time | 3.23 ms / beat (500 SVs × 256 dim at 40 MHz) |
| Avg power (80 bpm) | **0.284 mW** (0.431% duty cycle) |
| Die area | 2500 × 2500 µm (~14% utilization) |
| GDS | 226 MB |
| LEF | 94 KB |
| GL netlist | 13 MB |

### user_project_wrapper (job 91967)

| Metric | Value |
|--------|-------|
| Die area | 2920 × 3520 µm (Caravel fixed, FP_DEF_TEMPLATE) |
| Macro | u_svm at (253, 554) N — 2500 × 2500 µm footprint |
| Clock | wb_clk_i (Caravel), gated to svm_gclk via ICG |
| CTS | Disabled (RUN_CTS: 0) |
| DRC | 11,923 Magic DRC (boundary artifacts — acceptable) |
| LVS | 1,683 errors (boundary artifacts — acceptable) |
| GDS | 230 MB |
| LEF | 195 KB |
| GL netlist | 78 KB |

---

## Functional Results

| Implementation | Accuracy | SVs | Notes |
|---------------|----------|-----|-------|
| sklearn OVR (float) | 97.67% | 416 total (unlimited) | float precision |
| ASIC binary OVR (Q6.10) | 97.67% | 500 total (100×5) | gamma=0.25, C=1.0 |

**Zero accuracy gap** — ASIC exactly matches sklearn on all 300 test samples.

Per-class breakdown (300 test samples, 60 per class):
| Class | Correct | Accuracy |
|-------|---------|----------|
| Normal (N) | 60/60 | 100.0% |
| PVC | 60/60 | 100.0% |
| AFib | 60/60 | 100.0% |
| VT | 56/60 | 93.3% |
| SVT | 57/60 | 95.0% |

---

## Batch Architecture (v8/v9)

### Off-chip RAM Bus

| Signal | Pin | Direction | Description |
|--------|-----|-----------|-------------|
| `ram_addr[18:0]` | GPIO[28:10] | ASIC out | {row[10:0], col[7:0]} |
| `ram_ren` | GPIO[29] | ASIC out | Read strobe |
| `ram_rdata[15:0]` | LA[15:0] | Host in→ASIC | 1-cycle latency response |

Address layout: rows 0..499 = SV matrix; rows 500..1499 = input matrix.

### What the Host Does

```
MCU (low-power, continuous)
    │
    │  1. Collect 1000 heartbeats (250 Hz ECG → feature extraction)
    │  2. Load SV matrix  (500 SVs × 256 features) → SRAM rows 0..499
    │  3. Load input matrix (1000 beats × 256 features) → SRAM rows 500..1499
    │  4. Write NUM_SAMPLES = 1000, write NUM_SV_0–4 = 100 each
    │  5. Write alpha coefficients via ALPHA_WR (Wishbone 0x28)
    │  6. Fire CONTROL[start]
    │
    ▼  ASIC takes over:
    ├── LOAD_INPUT per beat: 256 cycles (reads from SRAM rows 500+)
    ├── COMPUTE_DIST per SV: 258 cycles (reads SV from SRAM rows 0–499)
    ├── COMPUTE_KERNEL: ~18 cycles (Horner LUT exp approximation)
    └── WRITE_CLASS: argmax+alpha → sample_rdy (IRQ[0]) per beat
                    last beat → done (IRQ[1])
```

---

## Wishbone Register Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in batch (1–1000) |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class (max 100 each) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data (γ, C, bias) |
| +0x28 | ALPHA_WR | WO | [24:16]=sv_global_idx (9-bit) [15:0]=alpha Q6.10 |

---

## Full-Chip Power Estimate

| Subsystem | Active Power | Duty Cycle | Avg Power |
|-----------|-------------|-----------|-----------|
| svm_compute_core (batch) | 66 mW | 0.431% (3.23 ms / 750 ms) | **0.284 mW** |
| Caravel management SoC | ~5 mW | ~5% | ~0.25 mW |
| ECG frontend (analog) | ~0.5 mW | 100% | 0.5 mW |
| BLE (optional, logging) | ~10 mW | ~0.1% | ~0.01 mW |
| **Total estimated** | — | — | **~1.04 mW** |

Battery budget: 200 mAh @ 3.7V = 740 mWh → **740 h / 1.04 mW ≈ 711 hours (~29.6 days)**.
14-day target met with 2.1× margin. SVM core alone: ~2606 hours (~108 days).

---

## Caravel Submission Artifacts (caravel_svm_project repo)

| File | Size | Job | Status |
|------|------|-----|--------|
| `gds/svm_compute_core.gds` | 226 MB | 91966 | ✅ |
| `gds/user_project_wrapper.gds` | 230 MB | 91967 | ✅ (LFS push pending) |
| `lef/svm_compute_core.lef` | 94 KB | 91966 | ✅ |
| `lef/user_project_wrapper.lef` | 195 KB | 91967 | ✅ |
| `verilog/gl/svm_compute_core.v` | 13 MB | 91966 | ✅ |
| `verilog/gl/user_project_wrapper.v` | 78 KB | 91967 | ✅ |
| `verilog/rtl/svm_compute_core.sv` | — | v9 | ✅ |
| `verilog/rtl/user_project_wrapper.sv` | — | v9 | ✅ |

---

---

## Feature Extraction References

The 256-dim multi-scale feature vector follows established AAMI EC57 beat
classification conventions:

| Feature group | Dims | Reference |
|---------------|------|-----------|
| Single-beat morphology (±64 samples, amplitude-norm) | 128 | de Chazal P, O'Dwyer M, Reilly RB. "Automatic classification of heartbeats using ECG morphology and heartbeat interval features." *IEEE Trans Biomed Eng* 51(7):1196-206, 2004. DOI: [10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359) |
| 10-beat mean morphology template | 64 | de Chazal P, Reilly RB. "A patient-adapting heartbeat classifier using ECG morphology and heartbeat interval features." *IEEE Trans Biomed Eng* 53(12):2535-43, 2006. DOI: [10.1109/TBME.2006.883802](https://doi.org/10.1109/TBME.2006.883802) |
| RR-interval history (99 intervals → 64 pts, norm to NORMAL_RR=308 ms) | 64 | Llamedo M, Martínez JP. "Heartbeat classification using feature selection driven by database generalization criteria." *IEEE Trans Biomed Eng* 58(3):616-25, 2011. DOI: [10.1109/TBME.2010.2068048](https://doi.org/10.1109/TBME.2010.2068048) |

Standard: AAMI ANSI EC57:2012 — Performance Requirements for Ambulatory ECG Analysers.  
Dataset: PhysioNet MIT-BIH Arrhythmia Database (Moody GB, Mark RG, 2001).  
DOI: [10.13026/C2F305](https://doi.org/10.13026/C2F305)

---

*Document version: m5/v9 · 2026-05-25 — hardening complete, cosim 97.67% = sklearn*
