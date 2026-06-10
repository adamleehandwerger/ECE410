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
title: "Testbench Analysis — m2"
subtitle: "5-Class Cardiac Arrhythmia Classifier — RBF-SVM ASIC (Pre-RTL Hardening)"
author: "Adam Handwerger · ECE410, Portland State University"
date: "2026-06-10"
---

The m2 verification strategy targets `svm_compute_core` at the pre-RTL-hardening level using
a QSPI streaming interface with an on-chip FIFO (FIFO\_DEPTH=8192). The SVM is trained with
γ = 0.01 on MIT-BIH data, uses single-beat 256-dimensional features, Q6.10 fixed-point
arithmetic, a single-stage Horner approximation of exp(−γD), and OvO (one-vs-one)
classification — 10 binary classifiers for 5 classes. Hardware and software accuracy match
at 96.33%.

**Note:** m2 uses OvO classification. m3 and later milestones switch to OvR (one-vs-rest)
with a Platt-calibrated bias term. The QSPI/FIFO interface and single-stage Horner
approximation are also replaced in later milestones: m3 adds the γ = 0.25 LUT-based range
reduction, m5 replaces the on-chip FIFO with off-chip SRAM via Wishbone/GPIO/LA pins.

**2-Cycle Pipeline Drain (pre-fix):** The `distance_matrix` module has a 2-cycle
diff→square→accumulate pipeline. At m2, the testbench explicitly notes that for FEATURE\_DIM=4
the last 2 entries are not flushed before OUTPUT (only `k=1` accumulates), but dismisses the
miss as "negligible (<1%)" for FEATURE\_DIM=256. The drain counter fix (holding ACCUMULATE for
2 extra cycles after the last feature) was added in a later milestone after the miss was found
to cause accuracy regression on real MIT-BIH data.

---

# Level 1 — Unit Tests (iverilog, Direct RTL Port)

**Location:** `m2/tb/` · **Simulator:** Icarus Verilog · **Run:** `iverilog -g2012`

These unit tests drive `svm_compute_core` RTL ports directly via the QSPI/FIFO interface,
at full FEATURE\_DIM=256 with NUM\_SV=250 (per-class: 60, 45, 55, 50, 40).

## Testbench B — tb\_compute\_core.sv (tb\_svm\_compute\_core)

Seven top-level tests covering parameters, SV counts, FIFO streaming, distance matrix,
Horner kernel, full pipeline, and FIFO overflow. Simulation frequency 50 MHz (20 ns period).

### TEST 1 — Field-Programmable Parameters

Programs γ = 0.005 via `param_write_en` / `param_addr = 2'b00`, reads back `gamma_reg`,
verifies Q6.10 encoding round-trip. Repeats for C = 2.0 (`param_addr = 2'b01`), then
restores defaults (γ = 0.01 → 0x000A, C = 1.0 → 0x0400).

### TEST 2 — Variable SV Counts Per Class

Verifies the default realistic distribution loaded at reset: [60, 45, 55, 50, 40] = 250 total.
Switches to equal distribution [50×5] and extreme imbalance [100, 10, 80, 40, 20], confirming
`total_sv()` = 250 in all three cases. All five `num_sv_per_class[i]` values compared against
expected by assertion.

### TEST 3 — Input FIFO Basic Operation

Pulses `start` with `num_samples = 1`, then streams 256 words through the QSPI interface
(`qspi_valid` / `qspi_data` / `qspi_ready` handshake) using a cosine-modulated feature
pattern. Verifies the FIFO absorbs all 256 words without backpressure and the FSM advances
from LOAD\_FIFO to LOAD\_FEATURES.

### TEST 4 — Distance Matrix Computation (Single Sample)

Streams a ramp feature vector (values `(i % 10) * 0.1` for i = 0..255) and waits for `done`.
Confirms the distance computation completes without hang or error flag. Liveness check only —
distance value is not numerically verified here.

### TEST 5 — Horner Engine — Variable Gamma

Programs three γ values (0.001, 0.01, 0.1) sequentially via `param_write_en`. For each,
records the expected kernel value at distance = 100.0 (`exp(−γ × 100)`). Exercises the
single-stage Horner polynomial across a 100× range of γ without overflow or underflow.
No LUT or range reduction is present at this milestone.

### TEST 6 — Full Pipeline — 3-Heartbeat Batch

`num_samples = 3`, `num_sv_per_class = [2, 2, 2, 2, 2]` (10 SVs total). Uses a `fork–join`
model: Thread A streams all 3 × 256 features via QSPI, stalling on backpressure while the
FSM is outside LOAD\_FIFO. Thread B counts `kernel_valid` pulses and latches `done`. Pass
criterion: `done` asserted before a 50,000-cycle watchdog and `error` never asserted.
Expected: 30 kernel values (3 beats × 10 SVs).

### TEST 7 — FIFO Overflow Protection

Attempts to write FIFO\_DEPTH + 100 = 8292 words into the QSPI interface when no batch is
running (FSM in IDLE, `qspi_ready = 0`). Counts rejected writes (overflow\_count > 0).
Verifies `qspi_ready` deasserts before the FIFO is exhausted and no phantom writes are
accepted.

---

## Testbench B — Auxiliary Sub-Modules (tb\_compute\_core.sv)

Two additional modules in the same file exercise the FIFO and distance matrix in isolation.

### tb\_input\_fifo — FIFO Read/Write Protocol

`DEPTH=16`, `DATA_WIDTH=16`. Writes all 16 entries in a burst (`wr_en` held), verifies `full`
asserts. Continuous burst read (`rd_en` held for 16 cycles), verifies each `rd_data === i`
in order, then `empty` asserts. Uses `#1` after `@(posedge clk)` throughout to avoid
active-region races between the `initial` block and the FIFO's `always_ff`.

### tb\_distance\_matrix — 4-Dim Distance Check

`FEATURE_DIM=4`, feature = [1,2,3,4], SV = [1,1,1,1], expected squared-difference sum = 14.
The test asserts `dist_out ≈ 1.0` (not 14.0), with an in-file comment documenting why:
the 2-cycle diff→square→accumulate pipeline does not flush the last 2 entries before the
FSM transitions to OUTPUT, so only the contribution from k=1 (diff = 1.0) accumulates.
The comment notes "the 2-entry miss is negligible (<1%) for FEATURE\_DIM=256." This was
the accepted pre-fix state; the drain counter was added in a later milestone.

---

## Testbench C — tb\_interface.sv — Interface Sanity Checks

Three independent top-level modules testing the three SystemVerilog interface types:
`svm_host_if`, `svm_sv_ram_if`, and `svm_work_ram_if`. Run with `-s <module_name>`.

### tb\_svm\_host\_if — MCU↔Core Signal Bundle (8 Tests)

T1: All outputs (done, error, kernel\_valid, qspi\_ready) zero after reset.
T2: `param_write_en` round-trip: writes 0x0014 to gamma\_reg, 0x0800 to c\_reg, reads back.
T3: QSPI backpressure: qspi\_ready=0 holds qspi\_valid=1 without accepting data.
T4: QSPI transfer: qspi\_ready=1 completes a 0xBEEF word transfer.
T5: Start/done handshake: `start` pulse, 3-cycle FSM delay, `done` asserts then clears.
T6: Kernel stream valid/ready: handshake passes with ready=1; valid held stable during
    backpressure (ready=0).
T7: Error flag: assert/clear sequence.
T8: `num_sv_per_class[0..4]` routing: programs [60,45,55,50,40] and reads back all five.

### tb\_svm\_sv\_ram\_if — SV SRAM Read-Only Protocol (6 Tests)

1-cycle read latency RAM model with values `addr × 4`.
T1–T3: Single reads at addresses 0 (→0), 7 (→28), 15 (→60) — spot-checks the address decode.
T4: 8-beat burst (`ren` held, addresses 0..7); verifies final `rdata` = 28.
T5: `rdata` holds stable for 4 cycles with `ren=0`.
T6: Out-of-range address (18'h3FFFF) returns `rdata = 0`.

### tb\_svm\_work\_ram\_if — Workspace SRAM R/W Protocol (7 Tests)

T1: Single write (addr 5, 0xCAFE), verified via direct memory inspection.
T2: Read back addr 5 through interface.
T3: Overwrite addr 5 with 0x1111, read back.
T4: 16-word write burst (values `addr × 64`), all 16 memory locations verified.
T5: 16-word read burst; final `rdata` = 960 (= 15 × 64).
T6: Multi-address roundtrip: write 0xAAAA→addr20, 0xBBBB→addr21, read both.
T7: 5-cycle idle: `rdata` and `work_mem[20]` unchanged with no access.

---

# Level 2 — Integration Tests (cocotb, Direct RTL Port)

**Location:** `m2/tb/` · **Simulator:** Icarus Verilog + cocotb · **Run:** `cd m2/tb && make`

9 tests using Python coroutines. Default γ = 0.01 → 0x000A (Q6.10); default C = 1.0 →
0x0400. Note: `test_default_gamma_fixed_point` checks 0x000A (= 0.01), not 0x0100 (= 0.25)
which is the m3 default.

## test\_reset\_outputs

All outputs (done, error, kernel\_valid, qspi\_ready) deasserted within 120 ns of `rst_n`
release. Gate test for the remaining 8.

## test\_param\_programming

Writes γ = 0x000A and C = 0x0400 via `param_write_en` interface; reads back via `gamma_reg`
and `c_reg`. Verifies Q6.10 encoding for γ = 0.01.

## test\_sv\_counts\_set

Programs [60, 45, 55, 50, 40] = 250 SVs total. Confirms independent per-class storage and
no ERR\_SV\_OVERFLOW.

## test\_sv\_counts\_unequal\_stress

Extreme distribution [100, 10, 80, 40, 20]. Confirms the SV counter loop handles
non-uniform class boundaries correctly.

## test\_qspi\_fifo\_load

Streams a full 256-word feature vector through the QSPI interface (`qspi_valid` /
`qspi_data` / `qspi_ready`) and confirms the FIFO absorbs all 256 words without asserting
backpressure. Simulation time ~1042 clock cycles at one word per cycle.

## test\_qspi\_backpressure

Overflows the FIFO by continuing `qspi_valid` after full. Verifies `qspi_ready` deasserts,
no extra words are written, and no spurious error is raised when the host respected
backpressure.

## test\_default\_gamma\_fixed\_point

Reads `gamma_reg` after reset. Confirms 0x000A (= 0.01 in Q6.10) — default correctly
encoded as integer, not floating-point literal.

## test\_full\_pipeline\_small\_batch

Complete single-sample pipeline: programs parameters, 2 SVs per class (10 total), streams
feature vector, responds to SV RAM reads, waits for `done`. Liveness check only — no
accuracy check.

## test\_kernel\_output\_range

Same as above but checks every `kernel_out` lies in [0, 1024] Q6.10. Written after a
single-stage Horner coefficient sign error produced kernel values above 1024 in an earlier
revision.

---

# Summary

| Level | Testbench(es) | Interface | Framework | Tests | Result |
|-------|---------------|-----------|-----------|-------|--------|
| 1 — Unit (core) | tb\_compute\_core.sv (tb\_svm\_compute\_core + tb\_input\_fifo + tb\_distance\_matrix) | Direct RTL (QSPI/FIFO) | iverilog | 9 | **9/9 PASS** |
| 1 — Unit (iface) | tb\_interface.sv (tb\_svm\_host\_if + tb\_svm\_sv\_ram\_if + tb\_svm\_work\_ram\_if) | SV interfaces | iverilog | 21 assertions | **PASS** |
| 2 — Integration | test\_svm\_compute\_core.py | Direct RTL (QSPI/FIFO) | cocotb | 9 | **9/9 PASS** |
| **Total** | | | | | **All PASS** |

**Interface evolution:** m2 uses QSPI streaming with an on-chip FIFO (FIFO\_DEPTH=8192) and
OvO classification (10 binary classifiers). m3 adds a γ = 0.25 LUT + range reduction Horner
and switches to OvR, retaining the QSPI/FIFO interface. m5 replaces the on-chip FIFO with
off-chip SRAM via Wishbone registers and GPIO/LA pins. See `m3/tb/testbench_analysis.md` and
`m5/tb/testbench_analysis.md` for the evolution.

---
*ECE410 · Portland State University · Adam Handwerger · sky130A · MIT-BIH Arrhythmia Database*
