# CLLM — Accelerator Benchmarking

*SV = support vector*

---

## 6. Baseline vs. Final Design

### 6.1 Implementations

| | Baseline | Final Design |
|---|---|---|
| Platform | Python sklearn, CPU (float64) | sky130A ASIC, $\mathbb{Q}_{6.10}$, 40 MHz |
| Mode | Batch — 1,000 beats loaded at once | Streaming — one beat at a time |
| Precision | 64-bit float | 16-bit fixed-point |
| Clock | ~3 GHz (laptop CPU) | 40 MHz |

---

### 6.2 Execution Time

The ASIC processes one beat in $N_{cyc}$ cycles at 40 MHz. With $N_{sv} = 500$ SVs and 256 features per SV, one pipeline pass requires:

$$N_{cyc} = N_{sv} \times N_{feat} = 500 \times 256 = 128{,}000 \text{ cycles/beat}$$

$$t_{beat} = \frac{128{,}000}{40 \times 10^6} \approx 3.2 \text{ ms/beat}$$

The Python baseline computes 500,000 kernel evaluations (1,000 beats $\times$ 500 SVs), each requiring $\sim$256 multiply-accumulate pairs plus $\exp()$:

$$N_{ops}^{baseline} = 1{,}000 \times 500 \times (2 \times 256 + 1) = 256.5 \times 10^6 \text{ ops}$$

At an effective CPU throughput of $\sim$2 GFLOPS (memory-bound, float64):

$$t_{baseline} \approx \frac{256.5 \times 10^6}{2 \times 10^9} \approx 128 \text{ ms for 1,000 beats}$$

| | Baseline (Python) | Final Design (ASIC) |
|---|---|---|
| Time per beat | ~0.13 ms | 3.2 ms |
| Time for 1,000 beats | **~128 ms** | **3,200 ms** |
| Speedup (baseline/ASIC) | 1× (reference) | 0.04× (25× slower) |

---

### 6.3 Throughput

$$\text{Throughput} = \frac{\text{beats}}{\text{time (s)}}$$

| | Baseline (Python) | Final Design (ASIC) |
|---|---|---|
| Throughput (beats/sec) | ~7,800 | ~312 |
| Relative throughput | 25× | 1× |

The ASIC is 25× lower throughput than sklearn on a laptop CPU. This trade-off is intentional: the ASIC targets **power efficiency**, not peak throughput.

---

### 6.4 Memory Usage

| Region | Baseline (Python, batch) | Final Design (ASIC, streaming) |
|---|---|---|
| Feature data | $1{,}000 \times 256 \times 8 = 2.0$ MB | $256 \times 2 = 512$ B (one beat, on-chip) |
| SV model | $500 \times 256 \times 8 = 1.0$ MB | $500 \times 256 \times 2 = 256$ KB (off-chip SRAM) |
| Kernel matrix | $1{,}000 \times 500 \times 8 = 4.0$ MB | Not materialised — computed and discarded per SV |
| $\alpha$ weights | $500 \times 8 = 4$ KB | $500 \times 2 = 1$ KB (off-chip SRAM) |
| Accumulators | Implicit in kernel matrix | $5 \times 4 = 20$ B (on-chip registers) |
| **Total** | **~7.0 MB** | **~257 KB** |

The ASIC reduces working memory by **~27×** by never materialising the kernel matrix, streaming one SV at a time and accumulating directly into 5 class registers.

---

### 6.5 Summary

| Metric | Baseline (Python) | Final Design (ASIC) | Ratio |
|---|---|---|---|
| Time / 1,000 beats | 128 ms | 3,200 ms | ASIC 25× slower |
| Throughput | ~7,800 beats/sec | ~312 beats/sec | ASIC 25× lower |
| Memory | ~7.0 MB | ~257 KB | ASIC 27× smaller |
| Power | ~5–15 W (CPU) | **0.284 mW** (avg) | ASIC ~30,000× lower |

The ASIC sacrifices throughput and raw speed to achieve orders-of-magnitude lower power. At 80 bpm the device classifies one beat every 750 ms — the 3.2 ms classification window represents only **0.4% active duty cycle**, which is what drives the average power down to 0.284 mW and enables a 29.6-day battery life.

For high-throughput applications (hospital batch classification) a systolic-array design on a deep-submicron node is the correct architecture, as detailed in Appendix B of the design summary.
