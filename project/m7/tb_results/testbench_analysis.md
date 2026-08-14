# Testbench Analysis — m7 (GF180MCU / wafer.space)

**Design:** 5-class RBF-SVM arrhythmia classifier — `svm_compute_core`
**Change under test (vs m6):** on-chip memories moved to SRAM macros
- `alpha_table` (600×16) → `alpha_sram_1024x16` (4× `gf180mcu_fd_ip_sram__sram512x8m8wm1`)
- `feature_bank` (256×16) → `feature_sram_512x16` (2× same macro)
- Rationale: the register `feature_bank` (4096 FF + 256:1 mux) created the `feat_rd_addr`
  ~1030-fanout broadcast + clock load that stalled GF180 detailed routing. The SRAM read is
  paced by `ram_beat` (1 read / `RAM_LATENCY`=3 cycles), so the macro's 45 ns access is covered.
**RTL:** `m7/rt1/compute_core.sv` + `m7/rt1/alpha_sram.sv` (behavioral cell via `ALPHA_SRAM_BEHAV_CELL`)

---

## Level 1 — Unit Tests (iverilog, direct RTL port) — 13/13 PASS

| Testbench | Result | Notes |
|-----------|--------|-------|
| `tb_error_codes` | **PASS** | 14/14 — ERR_SV_ZERO/OVERFLOW/GAMMA_SAT, sticky latch, reset-clear |
| `tb_backpressure` | **PASS** | kernel_valid/ready handshake, 3-cycle late release |
| `tb_consecutive` | **PASS** | two batches back-to-back, counters reset |
| `tb_dist_boundary` | **PASS** | accumulator saturation → kernel_out=0 |
| `tb_dist_zero` | **PASS** | feature=sv → kernel_out=1024; **pipeline-drain cycle count intact** |
| `tb_gamma_zero` | **PASS** | ERR_GAMMA_ZERO advisory |
| `tb_interface` | **PASS** | 17/17 — register defaults, sticky-hold, start-outside-IDLE |
| `tb_min_sv` | **PASS** | sv_counts=[1,1,1,1,1] → 5 kernels |
| `tb_multi_heartbeat` | **PASS** | num_samples=3 loop-back, done once |
| `tb_num_samples` | **PASS** | batch-size handling |
| `tb_param_write` | **PASS** | gamma shadow reg, mid-compute write safe |
| `tb_power` | **PASS** | 15/15 — LOW_BATTERY/POWER_FAIL advisories |
| `tb_warmup` | **PASS** | 13/13 — WARMING_UP/INTERRUPTED, auto-clear at beat 100 |

## Level 3 — RAM Latency (iverilog) — PASS

| Testbench | Config | Result |
|-----------|--------|--------|
| `svm_ram_latency_tb` | FEAT=4, NSV=5, **LAT=3**, BEATS=10 | **PASS — 10/10 beats, 208 cycles/beat** (exact match to m6) |

**This is the key result for the SRAM change:** the feature-SRAM read paces correctly against
`ram_beat`; the 208 cyc/beat figure is identical to the m6 register version, confirming the
memory swap did not alter functional timing.

---

## Pending (require cocotb — run in ORCA SIF)

Local run blocked by an arm64/x86 mismatch between Homebrew `iverilog` and the arm64 cocotb VPI
(`libcocotbvpi_icarus.vpl` dlopen failure). These run in the librelane SIF on ORCA (`make sim`):

| Level | Suite | Status |
|-------|-------|--------|
| L2 | `test_svm_compute_core.py` (direct RTL) | **7/7 PASS** (x86 cocotb 2.0.1 + matched iverilog, timescale 1ns/1ps) |
| L4/5 | `tb_spi_cosim.py` (SPI cosim of `svm_top_ihp`) | **N/A for GF180** — tests the IHP top wrapper, not our `chip_core` bridge |

**Top-level SPI for GF180 — `chip_core` bridge (new cocotb test `test_chip_core_spi.py`): 3/3 PASS.**
The m6 `tb_spi_cosim`/`svm_top_ihp` is IHP-only and irrelevant. Wrote a dedicated cocotb test for
our `chip_core` SPI-slave bridge (register map CTRL/STATUS/NSAMP/NSVPC/PARAM/ALPHA + direct RAM bus):

| Test | Result |
|------|--------|
| `test_default_gamma` | **PASS** — SPI GAMMA read = 0x0100 (Q6.10 default) |
| `test_param_write_readback` | **PASS** — SPI PARAM write 0x0200 → GAMMA read 0x0200 |
| `test_status_readable` | **PASS** — STATUS readable, idle after reset |

Verifies the SPI byte protocol (mode 0, MSB first, header={rd/wr,addr}), register write/read, and the
param path end-to-end through the bridge + core.

Note: `tb_top` (full-pipeline classification) needs regenerated trained vectors (`gen_tb_data.py`);
`tb_svm_classifier` targets the QSPI two-SRAM core variant, not this unified core.

---

## Summary

**14/14 iverilog tests pass** on the recoded RTL, including the two most sensitive to the memory
swap (`tb_dist_zero` pipeline drain, `svm_ram_latency_tb` LAT=3). The `feature_bank`→SRAM and
`alpha_table`→SRAM conversions are **functionally verified at the unit level with no timing change**.
Remaining cocotb levels to be run in the ORCA SIF environment.
