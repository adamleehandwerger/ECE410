---
geometry: "margin=2.2cm"
fontsize: 10pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{enumitem}
  - \setlist[itemize]{itemsep=3pt, topsep=3pt}
  - \pagestyle{plain}
title: "Testbench Analysis — m3"
subtitle: "5-Class Cardiac Arrhythmia Classifier — RBF-SVM ASIC (Pre-Netlist)"
author: "Adam Handwerger · ECE410, Portland State University"
date: "2026-05-01"
---

The m3 verification strategy targets `svm_compute_core` at the pre-netlist level using
a QSPI streaming interface with an on-chip FIFO. The interface drives feature vectors
word-by-word into the core via `qspi_valid` / `qspi_data` / `qspi_ready` handshake.
All 22 tests pass at this milestone.

**Note:** The QSPI/FIFO streaming interface was replaced in m5 by a Wishbone register
map and off-chip SRAM accessed via GPIO/LA pins. See `m5/tb/testbench_analysis.md`
for the updated verification strategy.

---

# Level 1 — Unit Tests (iverilog, Direct RTL Port)

**Location:** `m3/tb/` · **Simulator:** Icarus Verilog 13.0 · **Run:** `cd m3/tb && make all`

These 13 testbenches exercise individual FSM paths, error conditions, and datapath corner
cases in isolation. They drive `svm_compute_core` RTL ports directly — no wrapper — using
a small parameterization (FEATURE\_DIM=16, NUM\_SV=5–10) so simulation completes in
milliseconds.

## tb\_svm\_classifier.sv — Full 5-Class Pipeline

Classifies one representative heartbeat per class (Normal, PVC, AFib, VT, SVT) using the
trained SVM parameters from `tb_svm_params.svh`. Programs gamma (0.25 -> 0x0100 in Q6.10),
sets SV counts for all five classes, streams a 256-dimensional feature vector through the
QSPI FIFO interface (`qspi_valid` / `qspi_data`), services the SV RAM read requests from a
precomputed hex file, and collects kernel outputs. After all SVs are processed, applies the
OVR decision function (weighted kernel sum + bias) and compares predicted class against the
expected label. Pass criterion: 4 of 5 correct classifications, no error flag asserted.

## tb\_error\_codes.sv — Fault Detection and Sticky Latch

Exercises all diagnostic error codes using two DUT instances. The main instance
(FEATURE\_DIM=16, NUM\_SV=10, FIFO\_DEPTH=256) tests ERR\_SV\_ZERO (0x1) by setting all
SV counts to zero, ERR\_SV\_OVERFLOW (0x2) by loading more SVs than NUM\_SV, and
ERR\_GAMMA\_SAT (0x4) by writing gamma above saturation. A second instance (FIFO\_DEPTH=4)
tests ERR\_FIFO\_OVERFLOW (0x5) by streaming more words than the FIFO can hold. After each
fault the testbench verifies the sticky latch (error\_code holds for 50 idle cycles),
pulses `rst_n`, and confirms both `error` and `error\_code` return to zero.

**Note:** ERR\_FIFO\_OVERFLOW (0x5) and the FIFO\_DEPTH parameter are removed in m5.
The batch architecture eliminates the on-chip FIFO; input data lives in off-chip SRAM.

## tb\_backpressure.sv — QSPI Backpressure Handshake

Tests the `kernel_valid` / `kernel_ready` flow-control handshake across three scenarios.
Sub-test A: `kernel_ready` permanently asserted (baseline). Sub-test B: `kernel_ready`
released on the same clock as `kernel_valid` rises (same-cycle acknowledgment). Sub-test C:
3-cycle delay before `kernel_ready` — exposed a bug where `kernel_valid` pulsed for only
one cycle and the FSM stalled permanently. The fix (set/clear register holding `kernel_valid`
until acknowledged) is verified here.

## tb\_consecutive.sv — Back-to-Back Batch Resets

Runs two complete classification batches without an intervening `rst_n`. After the first
`done`, immediately pulses `start` with a new `num_samples` and fresh feature stream.
Verifies that `sample_counter`, `class_counter`, and `sv_counter` reset at the batch
boundary and no error flags carry over.

## tb\_dist\_boundary.sv — Accumulator Saturation

Worst-case feature pair: input = 0x7FFF (+31.999 Q6.10), SV = 0x8000 (−32.0 Q6.10).
For FEATURE\_DIM=16, the squared differences saturate the 20-bit accumulator at 0xFFFFF.
Verifies the Horner LUT maps saturated distance to `kernel_out = 0` (exp(−∞) = 0).

## tb\_dist\_zero.sv — Zero-Distance Kernel Identity

All input features equal corresponding SV features (both 0x0400 = 1.0 Q6.10). Verifies
kernel output is exactly 1024 (= 1.0 Q6.10). Also the canonical check for the 2-cycle
pipeline drain: reading the accumulator one cycle early produces a nonzero distance and
fails the kernel identity check.

## tb\_gamma\_zero.sv — Gamma Zero Advisory

Writes gamma = 0 and classifies one beat. All kernels evaluate to 1.0; all class scores
are identical. Verifies ERR\_GAMMA\_ZERO (0x6) asserts as advisory while the batch
completes without hanging.

## tb\_interface.sv — Port Signal Protocol

25 assertions covering reset state, register defaults, and protocol edge cases. Checks:
all outputs (done, error, kernel\_valid, qspi\_ready) deasserted after `rst_n`; `gamma_reg`
reads back 0x0100; `start` mid-batch is ignored; a 2-sample batch produces exactly two
`sample_rdy` pulses and one `done`.

## tb\_min\_sv.sv — Minimum SV Configuration

One SV per class (5 total). Verifies 5 kernel outputs, one `done`, all kernels = 1024
(features and SVs both zero, exp(0) = 1.0).

## tb\_multi\_heartbeat.sv — Three-Beat Loop-Back

`num_samples = 3`. Verifies the FSM returns to LOAD\_INPUT after each beat, `done` fires
once after the third beat, and `sample_rdy` pulses three times.

## tb\_param\_write.sv — Gamma Shadow Register

Writes a new gamma mid-classification (during COMPUTE\_DIST) and verifies it does not
take effect until the next batch. An earlier revision lacked the shadow register and
mid-compute writes corrupted the kernel sum.

## tb\_power.sv — Battery Fault Behavior

ERR\_LOW\_BATTERY (0xA): advisory, fires on `vbatt_warn`, allows classification to
complete, clears when `vbatt_warn` deasserts. ERR\_POWER\_FAIL (0xB): blocking, fires on
`vbatt_ok` deassert, prevents new batches. 16 checks validate latch, clear, and IDLE
recovery behavior.

## tb\_warmup.sv — Warmup Advisory Sequence

ERR\_WARMING\_UP (0x8) fires from clean start, persists through beats 1–99.
ERR\_INTERRUPTED (0x9) fires if `rst_n` during warmup. Five sub-tests cover normal
sequence, mid-warmup reset, sticky fault override, advisory clearance at beat 100, and
re-warming after reset.

---

# Level 2 — Integration Tests (cocotb, Direct RTL Port)

**Location:** `m3/tb/` · **Simulator:** Icarus Verilog + cocotb 2.0.1 · **Run:** `cd m3/tb && make cocotb`

9 tests using Python coroutines to drive RTL ports with programmatic stimulus generation
and floating-point reference comparison.

## test\_reset\_outputs

All outputs (done, error, kernel\_valid, qspi\_ready) must be deasserted within 120 ns
of `rst_n` release. Gate test for the remaining 8.

## test\_param\_programming

Writes gamma (0x0100) and C (0x0400) via `param_write_en` and reads back via `gamma_reg`
and `c_reg`. Verifies Q6.10 encoding round-trips through the register path.

## test\_sv\_counts\_set

Programs [60, 45, 55, 50, 40] = 250 SVs total. Confirms independent storage and no
ERR\_SV\_OVERFLOW.

## test\_sv\_counts\_unequal\_stress

Extreme distribution [100, 10, 80, 40, 20]. Confirms the SV counter loop handles
non-uniform class boundaries correctly.

## test\_qspi\_fifo\_load

Streams a full 256-word feature vector through the QSPI interface (`qspi_valid` /
`qspi_data` / `qspi_ready`) and confirms the FIFO absorbs all 256 words without asserting
backpressure. Simulation time ~1042 clock cycles at one word per cycle.

**Note:** Replaced in m5 by `test_wb_sram_load` which uses the GPIO/LA SRAM interface.

## test\_qspi\_backpressure

Overflows the FIFO by continuing `qspi_valid` after full. Verifies `qspi_ready` deasserts,
no extra words are written, and ERR\_FIFO\_OVERFLOW does not fire when the host respected
backpressure.

**Note:** Replaced in m5 by `test_wb_ram_latency` which tests the LAT=2 SRAM wait-state
logic.

## test\_default\_gamma\_fixed\_point

Reads `gamma_reg` after reset. Confirms 0x0100 (= 0.25 Q6.10) — default correctly
encoded as integer, not floating-point literal.

## test\_full\_pipeline\_small\_batch

Complete single-sample pipeline: programs parameters, 2 SVs per class (10 total), streams
feature vector, responds to SV RAM reads, waits for `done`. Liveness check only — no
accuracy check.

## test\_kernel\_output\_range

Same as above but checks every `kernel_out` lies in [0, 1024] Q6.10. Written after an
early revision produced kernel values above 1024 due to a Horner coefficient sign flip.

---

# Summary

| Level | Testbench(es) | Interface | Framework | Tests | Result |
|-------|---------------|-----------|-----------|-------|--------|
| 1 — Unit | tb\_svm\_classifier, tb\_error\_codes, tb\_backpressure, tb\_consecutive, tb\_dist\_boundary, tb\_dist\_zero, tb\_gamma\_zero, tb\_interface, tb\_min\_sv, tb\_multi\_heartbeat, tb\_param\_write, tb\_power, tb\_warmup | Direct RTL (QSPI/FIFO) | iverilog | 13 | **13/13 PASS** |
| 2 — Integration | test\_reset\_outputs, test\_param\_programming, test\_sv\_counts\_set, test\_sv\_counts\_unequal\_stress, test\_qspi\_fifo\_load, test\_qspi\_backpressure, test\_default\_gamma\_fixed\_point, test\_full\_pipeline\_small\_batch, test\_kernel\_output\_range | Direct RTL (QSPI/FIFO) | cocotb | 9 | **9/9 PASS** |
| **Total** | | | | **22** | **22/22 PASS** |

**Interface evolution:** m3 used QSPI streaming with an on-chip FIFO (FIFO\_DEPTH=256).
m4 moved to Wishbone registers. m5 added the off-chip SRAM interface (GPIO address bus,
LA data bus) with configurable RAM\_LATENCY for the IS61WV51216 async SRAM. See
`m5/tb/testbench_analysis.md` for the final m5 verification strategy.

---
*ECE410 · Portland State University · Adam Handwerger · sky130A · MIT-BIH Arrhythmia Database*
