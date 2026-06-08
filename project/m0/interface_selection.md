# Chiplet Interface Specifications

## Chiplet Design Overview

Custom hardware accelerator chiplet for cardiac SVM inference using RBF kernel computation with dedicated distance matrix and kernel evaluation units.

---

## Interface Specifications

### Communication Interface
- **Protocol:** SPI (Serial Peripheral Interface)
- **Mode:** SPI Mode 0 (CPOL=0, CPHA=0)
- **Configuration:** Full-duplex, master-slave
- **Data width:** 8-bit (1 byte per transfer)

### Interface Clock Speed
- **SPI Clock Frequency:** 4 MHz
- **Transfer Rate:** 4 Mbps (500 KB/s)

---

## Bandwidth Requirements

### Data Transfer Requirements

**Input Data (per batch of 100 heartbeats):**
- 100 samples × 256 features × 4 bytes (float32) = **102.4 KB**
- Transfer time @ 4 MHz SPI: **102.4 KB ÷ 0.5 MB/s = 204.8 ms ≈ 205 ms**

**Output Data (per batch):**
- Kernel matrix: 100 samples × 200 support vectors × 4 bytes = **80 KB**
- Transfer time @ 4 MHz SPI: **80 KB ÷ 0.5 MB/s = 160 ms**

**Total SPI Transfer Time per Batch:**
- Input transfer: 205 ms
- Output transfer: 160 ms
- **Total: 365 ms**

### Chiplet Internal Bandwidth

**Distance Matrix Computation:**
- 100 samples × 200 support vectors = 20,000 distance calculations
- Each distance: 256 multiply-add operations
- With 16 parallel distance units @ 100 MHz:
  - Operations: 20,000 × 256 = 5.12M operations
  - Time: 5.12M ÷ (16 × 100M) = **3.2 ms**
- **Effective compute bandwidth: ~1.6 GOPS**

**Kernel Computation (Horner's method for exp approximation):**
- 20,000 RBF kernel evaluations
- ~15 operations per kernel (polynomial approximation)
- Time: 300K operations ÷ 100M = **3 ms**

**Chiplet Computation Bandwidth:**
- **Peak compute: ~100 MOPS** (100 MHz clock)
- **Effective sustained: ~1.6 GOPS** (with 16 parallel units)

---

## Interface Bandwidth Analysis

### Rated Interface Bandwidth
| Metric | Specification |
|--------|--------------|
| **SPI Clock** | 4 MHz |
| **Data Rate** | 4 Mbps (0.5 MB/s) |
| **Theoretical Bandwidth** | 0.5 MB/s = **0.0005 GB/s** |

### Required Bandwidth (per batch)
| Direction | Data Size | Required Time | Required Bandwidth |
|-----------|-----------|--------------|-------------------|
| **Input (MCU → Chiplet)** | 102.4 KB | 205 ms | 0.5 MB/s |
| **Output (Chiplet → MCU)** | 80 KB | 160 ms | 0.5 MB/s |
| **Total per batch** | 182.4 KB | 365 ms | 0.5 MB/s |

### Is the Design Interface-Bound?

**Yes, the design is significantly interface-bound.**

**Analysis:**
- **Chiplet compute time:** 20-35 ms (distance + kernel computation)
- **SPI transfer time:** 365 ms (input + output)
- **Interface overhead:** **365 ms ÷ 35 ms = 10.4× slower than compute**

**Breakdown:**
```
Total latency per batch (100 heartbeats):
├─ Input SPI transfer:     205 ms  (56.2%)
├─ Chiplet computation:     30 ms  (8.2%)  ← Actual work
└─ Output SPI transfer:    160 ms  (43.8%)
   TOTAL:                  395 ms

Interface-bound factor: 365 ms ÷ 30 ms = 12.2×
```

The chiplet spends **91.8% of time waiting for SPI transfers** and only **8.2% doing actual computation**.

---

## Performance Bottleneck Analysis

### Current SPI @ 4 MHz
- **Throughput:** 253 heartbeats/sec
- **Latency:** 395 ms per 100 heartbeats
- **Utilization:** Chiplet idle 91.8% of the time

### If SPI Upgraded to 20 MHz (5× faster)
- **Input transfer:** 41 ms
- **Computation:** 30 ms
- **Output transfer:** 32 ms
- **Total:** 103 ms
- **Interface overhead:** Only 2.4× vs compute
- **Throughput:** 970 heartbeats/sec (3.8× improvement)

### If SPI Upgraded to 50 MHz (12.5× faster)
- **Input transfer:** 16.4 ms
- **Computation:** 30 ms
- **Output transfer:** 12.8 ms
- **Total:** 59.2 ms
- **Now compute-bound!** Chiplet utilization: 50.7%
- **Throughput:** 1,690 heartbeats/sec (6.7× improvement)

---

## Host Platform Specifications

### MCU (Microcontroller Unit)
- **Platform:** ARM Cortex-M7 or equivalent
- **Clock Speed:** 400-600 MHz
- **RAM:** 512 KB - 1 MB SRAM
- **Flash:** 2-4 MB program memory
- **Peripherals:**
  - SPI master controller (4-50 MHz capable)
  - ADC for ECG signal acquisition (12-16 bit, 1 kSPS)
  - DMA for efficient data transfers
  - Low-power modes for battery operation

### Power Budget
- **MCU active:** 50-100 mW
- **Chiplet active:** 185 mW (peak during computation)
- **Chiplet average:** 0.62 mW (with 0.33% duty cycle)
- **Total system:** ~51-101 mW average

### Communication Architecture
```
ECG Sensor → ADC → MCU (Feature Extraction) → SPI → Chiplet (SVM Inference) → SPI → MCU (Decision)
```

**Data Flow:**
1. MCU samples ECG at 360 Hz
2. MCU extracts 256 features per heartbeat (~50-100 ms)
3. MCU buffers 100 heartbeats
4. MCU sends batch to chiplet via SPI (205 ms)
5. Chiplet computes kernel matrix (30 ms)
6. Chiplet returns results via SPI (160 ms)
7. MCU performs final SVM decision function (2 ms)

---

## Interface Optimization Recommendations

### Option 1: Increase SPI Clock Speed
- **Upgrade to 20 MHz:** Reduces transfer time from 365 ms → 73 ms
- **Cost:** Minimal (parameter change)
- **Benefit:** 5× faster transfers, 3.8× overall throughput improvement

### Option 2: Use Parallel Interface (QSPI/Octal SPI)
- **Quad SPI @ 50 MHz:** 4 lanes × 50 MHz = 25 MB/s
- **Transfer time:** 182.4 KB ÷ 25 MB/s = **7.3 ms**
- **Total latency:** 7.3 ms (in) + 30 ms (compute) + 7.3 ms (out) = **44.6 ms**
- **Now compute-bound!**

### Option 3: On-Chip Feature Extraction
- **Move feature extraction to chiplet**
- **Transfer raw ECG samples instead:** 100 beats × 256 samples × 2 bytes = 51.2 KB
- **Reduces input bandwidth by 50%**
- **Adds computation time but eliminates MCU bottleneck**

### Recommended Solution
**Upgrade to 20 MHz SPI** (easiest, 3.8× improvement) or **QSPI @ 50 MHz** (if compute-bound performance needed).

---

## Summary Table

| Specification | Value |
|--------------|-------|
| **Interface Type** | SPI (Serial Peripheral Interface) |
| **Interface Clock** | 4 MHz |
| **Data Rate** | 4 Mbps (0.5 MB/s) |
| **Bandwidth (GB/s)** | 0.0005 GB/s |
| **Required Bandwidth** | 0.5 MB/s (matches rated) |
| **Interface-Bound?** | **Yes** (12.2× overhead) |
| **Transfer Time** | 365 ms per batch |
| **Compute Time** | 30 ms per batch |
| **Interface Utilization** | 100% (saturated) |
| **Chiplet Utilization** | 8.2% (idle 91.8% waiting for transfers) |
| **Host Platform** | ARM Cortex-M7 @ 400-600 MHz |
| **Host RAM** | 512 KB - 1 MB SRAM |

---

## Conclusion

The current chiplet design is **significantly interface-bound** due to the 4 MHz SPI bottleneck. The chiplet can compute results in 30 ms but spends 365 ms transferring data, resulting in only **8.2% compute utilization**.

**Key Findings:**
- ✅ Chiplet computation is efficient (30 ms for 100 samples)
- ❌ SPI interface is the bottleneck (365 ms for transfers)
- ⚠️ Chiplet sits idle 91.8% of the time waiting for data
- 🎯 Upgrading to 20 MHz SPI would provide 3.8× improvement
- 🎯 QSPI @ 50 MHz would make the design compute-bound (optimal)

**Generated:** April 19, 2026
