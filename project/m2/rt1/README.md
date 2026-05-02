# m2 — Interface Specification Revision Notes

## Overview

This document records every change made to `Chiplet_Interface_Specifications.md`
between the original (baseline) version and the updated version that aligns with
the implemented RTL in `svm_compute_core.sv`.

---

## Parameter Changes

| Parameter | Previous | Updated | Reason |
|-----------|----------|---------|--------|
| Interface protocol | SPI (1 data lane) | QSPI (4 data lanes) | RTL ports are named `qspi_valid`, `qspi_data`, `qspi_ready`; quad-SPI gives 4× bandwidth at the same clock rate |
| Feature data format | float32 — 4 B/feature | Q6.10 fixed-point — 2 B/feature | RTL `DATA_WIDTH=16` throughout; fixed-point halves transfer volume and eliminates FPU hardware |
| Batch size | 100 heartbeats | 1000 heartbeats | RTL `MAX_BATCH_SIZE=1000`; 7-day battery target requires large burst batches |
| Support vectors | 200 SVs | 250 SVs | RTL `NUM_SV=250`; sklearn training produces 246 hardware-subset SVs across 5 classes |
| Chiplet internal clock | 100 MHz | 50 MHz | RTL `CLK_PERIOD=20 ns`; conservative first-tapeout target to close timing |
| Distance engine architecture | 16 parallel units | 1 sequential unit | RTL instantiates one `distance_matrix` module; area and power constraint for implantable device |
| Encoding artifacts | Present (Ã, â, ×) | Corrected (×, →, ≥) | UTF-8 corruption in source file |

---

## Derived Bandwidth Changes

| Metric | Previous | Updated | Driver |
|--------|----------|---------|--------|
| Effective interface data rate | 4 Mbps (0.5 MB/s) | 16 Mbps (2 MB/s) | 4-lane QSPI at same 4 MHz clock |
| Input data per batch | 102.4 KB | 512 KB | 10× more samples × ½ data format = 5× net increase |
| Output data per batch | 80 KB | 500 KB | 10× more samples × 250 SVs (was 200) × ½ format = 6.25× net increase |
| Total I/O per batch | 182.4 KB | 1,012 KB | Combined effect above |
| Input transfer time | 205 ms | 256 ms | Higher volume (+5×) partially offset by wider bus (+4×) |
| Output transfer time | 160 ms | 250 ms | Higher volume (+6.25×) partially offset by wider bus (+4×) |
| Total transfer time | 365 ms | 506 ms | Net increase despite wider bus |

---

## Compute Time Changes

| Metric | Previous | Updated | Driver |
|--------|----------|---------|--------|
| Per-SV compute | ~1.28 cycles (16 units) | 278 cycles (1 unit) | Parallelism removed; sequential pipeline |
| Horner latency | ~15 ops | 18 cycles | Correct RTL: SCALE + SCALE2 + 15 Horner states + OUTPUT |
| Per-sample compute | ~1.9 ms | ~1.4 ms | 1000× more samples; 250 SVs × 278 cycles @ 50 MHz |
| Total compute (full batch) | 30 ms | 1,400 ms | Sequential × 1000 samples × 250 SVs |

---

## Bottleneck Analysis Change

| Dimension | Previous | Updated |
|-----------|----------|---------|
| **Bound type** | Interface-bound | **Compute-bound** |
| Overhead factor | 12.2× (interface / compute) | 2.76× (compute / interface) |
| Chiplet idle fraction | 91.8% waiting for SPI | ~15% waiting on QSPI |
| Compute utilisation | 8.2% | ~85% |
| Primary optimisation target | Increase SPI clock speed | Parallelise distance_matrix engine |

The removal of the 16-parallel-unit assumption is the dominant cause of the flip:
a single sequential engine running at half the assumed clock takes ~46× longer than
the 16-unit array at 100 MHz, turning an interface-bound design into a compute-bound one.

---

## Performance Summary

```
                    PREVIOUS                        UPDATED
                    ────────────────────────────    ──────────────────────────────────
Protocol            SPI, 1 lane, 4 MHz              QSPI, 4 lanes, 4 MHz
Data format         float32 (4 B)                   Q6.10 (2 B)
Batch size          100 heartbeats                  1000 heartbeats
SVs                 200                             250
Clock               100 MHz                         50 MHz
─────────────────────────────────────────────────────────────────────────────
Input transfer      205 ms                          256 ms  (hidden by FIFO pipelining)
Compute             30 ms                           1,400 ms
Output transfer     160 ms                          250 ms
─────────────────────────────────────────────────────────────────────────────
Total latency       395 ms / 100 hb                1,650 ms / 1000 hb
Throughput          ~253 hb/sec                    ~606 hb/sec
Bound type          Interface (12.2×)               Compute (2.76×)
```

---

## Optimisation Path (updated)

Because the bottleneck has moved to compute, the roadmap changes:

| Step | Action | Compute time | Total latency | Bound |
|------|--------|-------------|---------------|-------|
| Baseline (this RTL) | 1 distance engine | 1,400 ms | 1,650 ms | Compute (2.76×) |
| N=4 parallel engines | 4 distance engines | 350 ms | 600 ms | Near-balanced |
| N=16 parallel engines | 16 distance engines | 87.5 ms | 337 ms | Interface (5.8×) |
| N=16 + 25 MHz QSPI | 16 engines + faster bus | 87.5 ms | 128 ms | Balanced |
