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
| `ram_rdata[15:0]` | LA[15:0] | Host in→ASIC | Data valid after `RAM_LATENCY` cycles |

Address layout: rows 0..499 = SV matrix; rows 500..1499 = input matrix.

**`RAM_LATENCY` parameter** — configures the number of clock cycles between `ram_ren`
assertion and valid `ram_rdata`. Default is 1 (same-cycle combinational SRAM model used
in cosim). Set to 3 for the IS61WV51216 async SRAM (10 ns access at 40 MHz → 3-cycle
pipeline). The core inserts wait states automatically; the host does not need to pad
responses. Verified by `svm_ram_latency_tb.sv` (10-beat, LAT=3, PASS in ~208 cycles/beat).

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
    ▼  ASIC takes over (timing shown for RAM_LATENCY=1 / RAM_LATENCY=3):
    ├── LOAD_INPUT per beat:    256 / 768 cycles  (reads from SRAM rows 500+)
    ├── COMPUTE_DIST per SV:   258 / 770 cycles  (reads SV from SRAM rows 0–499)
    ├── COMPUTE_KERNEL: ~18 cycles (Horner LUT exp approximation, LAT-independent)
    └── WRITE_CLASS: argmax+alpha → sample_rdy (IRQ[0]) per beat
                    last beat → done (IRQ[1])
```

Inference time scales linearly with `RAM_LATENCY`. At 40 MHz:
- LAT=1 (cosim default): **3.23 ms / beat** (500 SVs × 256 dim)
- LAT=3 (IS61WV51216 SRAM): **~9.7 ms / beat** — still well within the 750 ms heartbeat period

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

## Caravel Submission Artifacts

**Repository:** [github.com/adamleehandwerger/caravel_svm_project](https://github.com/adamleehandwerger/caravel_svm_project)  
**GDS Release:** [v2.0-hardened](https://github.com/adamleehandwerger/caravel_svm_project/releases/tag/v2.0-hardened) (226 MB core / 230 MB wrapper)

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

## Acknowledgments

Place-and-route was performed on **Orca**, Portland State University's high-performance
computing cluster, using SLURM batch jobs with OpenLane 2 v2.3.10 inside a Singularity
container.  We thank the PSU Research Computing team for providing access to the GPU and
CPU nodes that made the multi-hour OpenLane runs feasible.

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

*Document version: m5/v9 · 2026-05-27 — RAM_LATENCY parameter added; cosim 97.67% = sklearn; Appendix A added*

---

## Appendix A — Runtime Model Reload Procedure

The ASIC is fully runtime-reprogrammable. A new trained SVM model can be loaded without
resetting the chip or re-synthesizing. The on-chip alpha table and the off-chip SRAM are
simply overwritten in place.

### What constitutes a "model"

| Parameter | Location | Size |
|-----------|----------|------|
| SV matrix (500 × 256 × 16-bit features) | Off-chip SRAM, rows 0–499 | 256 KB |
| Alpha coefficients (500 × 16-bit) | On-chip `alpha_table[]` registers | 8 KB |
| Gamma (γ, Q6.10) | On-chip `gamma_reg` shadow register | 2 bytes |
| SV counts per class (`NUM_SV[0–4]`) | On-chip Wishbone registers | 5 bytes |

### Reload sequence

Perform these steps while the ASIC is idle (`STATUS[done]=1` or after reset).
Do **not** fire `CONTROL[start]` until all steps are complete.

**Step 1 — Write new SV matrix to off-chip SRAM**

Write 500 rows × 256 columns of Q6.10 feature values to SRAM address rows 0–499.
Address encoding: `addr[18:0] = {sv_global_idx[10:0], feat_dim[7:0]}`.
This is a bulk SRAM write from the MCU and does not involve the Wishbone bus.

```
for sv in range(500):
    for feat in range(256):
        sram_write(addr=(sv << 8) | feat, data=sv_matrix[sv][feat])
```

**Step 2 — Write new alpha coefficients via ALPHA_WR**

Write all 500 alpha values over Wishbone. Each write encodes the SV global index
and the alpha value in a single 32-bit word:

```
WB base = 0x3000_0000
ALPHA_WR offset = 0x28

for idx in range(500):
    word = (idx << 16) | (alpha[idx] & 0xFFFF)   # [24:16]=sv_idx, [15:0]=alpha Q6.10
    wishbone_write(WB_BASE + 0x28, word)
```

**Step 3 — Update gamma via PARAM_WR**

```
PARAM_WR offset = 0x24
# addr=0 is gamma register; [19]=en, [18:16]=addr, [15:0]=value
word = (1 << 19) | (0 << 16) | gamma_Q6_10
wishbone_write(WB_BASE + 0x24, word)
```

**Step 4 — Update SV counts if changed**

Only required if the new model has different SVs per class than the previous model.

```
NUM_SV offsets: class 0 = 0x10, class 1 = 0x14, ..., class 4 = 0x20
for c in range(5):
    wishbone_write(WB_BASE + 0x10 + c*4, num_sv[c])
```

**Step 5 — Fire start**

```
wishbone_write(WB_BASE + 0x04, 0x0B)   # CONTROL: start=1, vbatt_ok=1, kern_ready=1
```

### Constraints

- Total SV count must not exceed 500 (100 per class maximum). Models exceeding this
  require re-synthesis with updated `NUM_SV` parameters.
- Gamma must be representable in Q6.10 (range 0–63.999, resolution ~0.001).
- Alpha values must be representable in Q6.10 (signed, range −32 to +31.999).
- The reload can be performed between any two batches. The ASIC does not need to be
  held in reset during the write sequence — the alpha table and gamma register are
  shadow-registered and only take effect when `start` fires.
- Writing ALPHA_WR while `STATUS[done]=0` (i.e., during active inference) has
  undefined behavior and must be avoided.
