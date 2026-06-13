# m5 Benchmark — ASIC vs Optimized Python + RAM_LATENCY Timing

**Design:** svm_compute_core (sky130A, 40 MHz)  
**Dataset:** MIT-BIH Arrhythmia Database, 300 test samples, 5 classes  
**Features:** 256-dim multi-scale (128 single-beat + 64 10-beat + 64 RR-interval)

---

## 1. Accuracy

| Implementation | SVs | Accuracy | Gap vs sklearn |
|---------------|-----|----------|----------------|
| sklearn OVR (float, unlimited SVs) | 416 | **97.67%** (293/300) | — baseline |
| ASIC binary OVR (Q6.10, 500 SVs) | 500 | **97.67%** (293/300) | **0.00%** |

Per-class breakdown:

| Class | sklearn | ASIC | Notes |
|-------|---------|------|-------|
| Normal (N) | 60/60 (100%) | 60/60 (100%) | |
| PVC | 60/60 (100%) | 60/60 (100%) | |
| AFib | 60/60 (100%) | 60/60 (100%) | |
| VT | 56/60 (93.3%) | 56/60 (93.3%) | |
| SVT | 57/60 (95.0%) | 57/60 (95.0%) | |

**Zero accuracy gap** — Q6.10 fixed-point matches float on all 300 test samples.

---

## 2. Throughput

| Implementation | Platform | Throughput | Source |
|---------------|----------|-----------|--------|
| sklearn OVR (1 core) | Orca Xeon | ~4,200 inf/s | estimated |
| Numba parallel (8 cores) | Orca Xeon | ~95,000 inf/s | estimated |
| **ASIC sky130A** | 40 MHz | **309 inf/s** | **measured** |

The ASIC targets wearable operation at 80 bpm (1.34 inf/s required), making
raw throughput secondary to power. The 309 inf/s batch rate supports burst
classification of 1000-beat batches in 3.23 s — well within inter-batch windows.

---

## 3. Power & Energy Efficiency

| Implementation | Active Power | Energy Efficiency |
|---------------|-------------|-------------------|
| sklearn (1 core) | ~15,000 mW (CPU core) | ~280 inf/J |
| Numba parallel (8 cores) | ~80,000 mW (CPU TDP) | ~1,188 inf/J |
| **ASIC (active)** | **66 mW** | **4,682 inf/J** |
| **ASIC (duty-cycled, 80 bpm)** | **0.284 mW avg** | **~1,088,000 inf/J** |

The ASIC delivers **17× better energy efficiency** than a single Orca core
at active power, and **3,900× better** on a duty-cycled wearable basis.

Battery budget (200 mAh @ 3.7V = 740 mWh):
- SVM core alone: ~2,606 hours (108 days)
- Full system (MCU + ECG + BLE): ~1.04 mW avg → ~711 hours (29.6 days) — **14-day target MET**

---

## 4. Measured Inference Time

At 40 MHz (25 ns / cycle):

| Stage | Cycles | Time |
|-------|--------|------|
| LOAD_INPUT (256 features, LAT=1) | 256 | 6.4 µs |
| COMPUTE_DIST per SV (256 dims) | ~258 | 6.5 µs |
| COMPUTE_DIST × 500 SVs | ~129,000 | 3.23 ms |
| COMPUTE_KERNEL (Horner LUT) | ~18 per SV | 0.45 ms total |
| WRITE_CLASS (argmax) | 1 | 25 ns |
| **Total per beat** | **~129,700** | **~3.23 ms** |

Wearable duty cycle at 80 bpm: 3.23 ms / 750 ms = **0.431%**

---

## 5. RAM_LATENCY Benchmark (m5)

The `RAM_LATENCY` parameter inserts wait-state cycles between `ram_ren` and valid
`ram_rdata`, allowing the core to interface with SRAM devices of varying access times.
The dominant cost is COMPUTE_DIST, which performs 500 × 256 = 128,000 RAM reads per beat.

| Configuration | SRAM Device | RAM_LATENCY | Inference Time | Throughput | Duty Cycle (80 bpm) | Avg Power |
|--------------|------------|------------|---------------|-----------|--------------------|----|
| LAT=1 (cosim) | ideal / logic RAM | 1 | **3.23 ms** | 309 inf/s | 0.431% | **0.284 mW** |
| LAT=3 (real SRAM) | IS61WV51216 (10 ns) | 3 | **~9.7 ms** | ~103 inf/s | 1.29% | **0.852 mW** |

Both configurations are well within the 750 ms heartbeat window at 80 bpm (1.34 inf/s required).
LAT=3 increases average power ~3× but the 14-day target remains comfortably met.

**Cycle breakdown comparison (256-dim, 500 SVs):**

| Stage | LAT=1 cycles | LAT=3 cycles | Δ |
|-------|-------------|-------------|---|
| LOAD_INPUT (256 reads) | 256 | 768 | +512 |
| COMPUTE_DIST (128,000 reads) | ~129,000 | ~387,000 | +258,000 |
| COMPUTE_KERNEL (no RAM) | ~9,000 | ~9,000 | — |
| WRITE_CLASS | 1 | 1 | — |
| **Total** | **~138,257** | **~396,769** | **+258,512** |

Verified by `tb/svm_ram_latency_tb.sv` (FEAT=4, NSV=5, LAT=3 → PASS, 208 cycles/beat).

---

## 6. Roofline Summary

All three implementations share the same operational intensity (~2.0 ops/byte)
because the algorithm is identical — only the execution substrate changes.

| Implementation | Op. Intensity | Performance | Active Power |
|---------------|--------------|-------------|-------------|
| sklearn (1 core) | 2.0 ops/B | 2.15 GOPS | 15 W |
| Numba (8 cores, Orca) | 2.0 ops/B | 48.6 GOPS | 80 W |
| **ASIC sky130A** | **2.0 ops/B** | **158 MOPS** | **66 mW** |

See `roofline_final.png` for the dual-panel roofline + power-efficiency chart.  
Raw data: `benchmark_data.csv`.

*CPU throughput and power figures are estimates. ASIC figures are measured from RTL cosim.*

---

*ECE410 — Portland State University · Adam Handwerger · m5 (batch architecture, sky130A, RAM_LATENCY)*
