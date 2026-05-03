# SVM Compute Core — Precision Rationale

## Table 1: Bit-Width Justification

| Signal / Internal Node | Width | Justification |
|---|---|---|
| All data words (features, SVs, γ, C, kernel) | 16-bit Q6.10 | Range ±31.999 covers ECG features normalized to [−1, 1] and kernel output ∈ (0, 1]; 32-bit float would triple area and power |
| `diff` / `diff_squared` | 32-bit (2×DATA_WIDTH) | `\|diff\|` ≤ 2048 raw; squaring → up to 2^22; 16-bit overflows on the first squared term for any large input |
| `accumulator` | 40-bit (2×DATA_WIDTH + 8) | 256 terms × max squared diff ≈ 2^30; extra 8 bits guard inputs up to ±4 per dim; 32-bit saturates above ±1 |
| `dist_out` | 20-bit (DIST_WIDTH) | 16-bit saturates at ~64 Q6.10 for 256 dims, collapsing inter-class kernels; 20-bit covers worst-case 1,040,384 raw < 2^20; 21 bits adds no benefit |
| Horner `temp` | 36-bit (DATA_WIDTH + DIST_WIDTH) | Holds full γ(16-bit) × dist(20-bit) product before the FRAC_BITS shift; truncating to 32 bits first loses 4 bits, biasing kernel by up to 0.016 |
| `sv_ram_addr` | 18-bit | Natural `{sv_base[9:0], feat_rd_addr[7:0]}` concat; FEATURE_DIM=256=2^8 makes the 8-bit lower field exact; 2 spare MSBs add no cost |
| `work_ram_addr` | 18-bit | MAX_BATCH_SIZE × NUM_SV = 250,000; 2^18 = 262,144 > 250,000; 17-bit (131,072) is insufficient |
| `param_addr` | 2-bit | 2 programmable registers (γ = 2'b00, C = 2'b01); 1-bit too narrow; 2-bit allows future extension to 4 |
| `num_sv_per_class` | 8-bit per class | Max per-class SVs ≤ NUM_SV = 250 ≤ 255; exact uint8 fit; 9 bits would waste a register bit per class |
| `num_samples` | 10-bit | MAX_BATCH_SIZE = 1000; 2^10 = 1024 > 1000; 9-bit only reaches 512 |
| FIFO count | 14-bit (ADDR_WIDTH + 1) | FIFO_DEPTH = 8192 = 2^13; needs 14-bit counter to distinguish full (count = 8192) from empty (count = 0); 13-bit aliases the two states |

---

## Table 2: Quantization Error vs. FP32

| Node | Format | ½-LSB or MAE | FP32 Reference (½-ulp) | Ratio vs FP32 |
|---|---|---|---|---|
| Features, SVs, γ, C | Q6.10 | 4.88 × 10⁻⁴ | 5.96 × 10⁻⁸ @ v = 1.0 | ~8192× |
| `diff` (x[k] − sv[k]) | Q6.10 † | 9.77 × 10⁻⁴ ‡ | 1.19 × 10⁻⁷ @ v = 1.0 | ~8192× |
| `diff_squared` | Q12.20 † | 0 (exact multiply) | ~6 × 10⁻⁸ @ v = 0.01 | exact |
| `accumulator` | Q12.20 † | 0 (exact sum) | — | exact |
| `dist_out` | Q6.10 | 4.88 × 10⁻⁴ | 5.96 × 10⁻⁷ @ v = 10 | ~820× |
| Horner x = γ · dist | Q6.10 | 4.88 × 10⁻⁴ | 2.98 × 10⁻⁸ @ v = 0.5 | ~16384× |
| `kernel_out` (measured end-to-end) | Q6.10 | MAE = 0.00744 § | 0.00449 ¶ | 1.66× |

**Notes**

† Stored in a wider exact container with no truncation at this node; error is inherited from Q6.10 inputs, not introduced here.

‡ Worst case: sum of two independent Q6.10 input ½-LSBs (one from the feature, one from the SV).

§ End-to-end measured MAE vs. sklearn FP64 over 300 test samples × 246 hardware-subset SVs.

¶ FP32 Horner with the same 15th-order polynomial and zero quantization — intrinsic approximation error only. The 1.66× ratio shows how much overhead the fixed-point datapath adds above the common polynomial floor.

> **Address and control signals** (`sv_ram_addr`, `work_ram_addr`, `param_addr`, `num_sv_per_class`, `num_samples`, FIFO count) are integer counters with no fractional component; quantization error vs. FP32 is not applicable.
