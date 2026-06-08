# m4 Benchmark — ASIC vs Optimized Python

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

## 5. Roofline Summary

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

*ECE410 — Portland State University · Adam Handwerger · m4 (batch architecture, sky130A)*
