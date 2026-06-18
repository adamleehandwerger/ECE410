---
geometry: "margin=2.2cm"
fontsize: 10pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{enumitem}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \setlist[itemize]{itemsep=3pt, topsep=3pt}
  - \pagestyle{plain}
title: "Testbench Analysis"
subtitle: "5-Class Cardiac Arrhythmia Classifier --- RBF-SVM ASIC (m6, IHP SG13G2)"
author: "Adam Handwerger $\\cdot$ ECE410, Portland State University"
date: "2026-06-17"
---

The design implements a one-versus-rest (OVR) radial basis function support vector machine
(RBF-SVM) over five arrhythmia classes
$\mathcal{C} = \{\text{Normal},\,\text{PVC},\,\text{AFib},\,\text{VT},\,\text{SVT}\}$.
For an input feature vector
$\mathbf{x} \in \mathbb{R}^{256}$
and the $k$-th class support-vector set
$\mathcal{S}_k = \{\mathbf{sv}^{(k)}_i\}_{i=1}^{N_k}$,
the OVR decision function is

$$f_k(\mathbf{x}) = \sum_{i=1}^{N_k} \alpha^{(k)}_i\, K\!\left(\mathbf{x},\, \mathbf{sv}^{(k)}_i\right) - \rho_k, \qquad
  K(\mathbf{x},\mathbf{sv}) = \exp\!\left(-\gamma\,\|\mathbf{x} - \mathbf{sv}\|^2\right),$$

and the predicted label is $\hat{y} = \arg\max_k f_k(\mathbf{x})$.
All internal arithmetic uses $Q_{6.10}$ fixed-point representation:
a signed 16-bit integer $\hat{v}$ encodes the real value $v = \hat{v}\cdot 2^{-10}$.
The parameter $\gamma = 0.25$ encodes to $\hat{\gamma} = \lfloor 0.25 \cdot 2^{10}\rceil = \texttt{0x0100}$,
and the identity $K(\mathbf{x},\mathbf{x}) = 1.0$ encodes to $1024 = \texttt{0x0400}$.

The verification strategy spans five levels. Levels 1 and 2 target `svm_compute_core`
through its RTL ports directly. Level 3 characterises the `RAM_LATENCY` wait-state logic
for the physical IS62WV51216 asynchronous SRAM. Levels 4 and 5 drive the complete
`svm_top_ihp` IHP SG13G2 wrapper through its SPI slave register interface.
The Caravel Wishbone interface and management SoC (Levels 4--6 in milestone m5) are
replaced by a standalone SPI slave compatible with the nRF52840 MCU
(CPOL$=0$, CPHA$=0$, 40-bit frames: 8-bit address $\|$ 32-bit data, MSB first).
Of 30 tests, 23 pass and 7 (Level 4) are pending.

---

# Level 1 --- Unit Tests (iverilog, Direct RTL Port)

**Location:** `m6/tb/` $\cdot$ **Simulator:** Icarus Verilog 13.0 $\cdot$
**Invocation:** `cd m6/tb && make all`

Fourteen testbenches exercise individual FSM paths, error conditions, and datapath corner
cases in isolation. Each test drives `svm_compute_core` RTL ports directly — no SPI bus,
no top-level wrapper — so a failure localises immediately to the logic under test.
All tests use reduced parameterisations ($\text{FEATURE\_DIM}=16$, $N_k \in [1,2]$)
so that simulation completes in under $10\,\text{ms}$ and waveform inspection is practical.

## tb\_top.sv --- Full 5-Class Pipeline

Classifies one representative heartbeat per class using trained SVM parameters from
`tb_svm_params.svh`. The testbench programs
$\hat{\gamma} = \texttt{0x0100}$,
sets $N_k$ for all five classes, streams a 256-dimensional feature vector
$\mathbf{x} \in \mathbb{Z}_{Q_{6.10}}^{256}$
through the GPIO/SRAM interface, services SV RAM read requests from a precomputed
hex file, and collects all $\sum_k N_k$ kernel outputs.
After all support vectors are processed, the OVR decision function is applied and
the predicted class $\hat{y}$ is compared to the expected label.
Pass criterion: at least 4 of 5 beats correctly classified with no error flag asserted.

## tb\_error\_codes.sv --- Fault Detection and Sticky Latch

Exercises all diagnostic error codes in sequence:
$\texttt{ERR\_SV\_ZERO}=\texttt{0x1}$,
$\texttt{ERR\_SV\_OVERFLOW}=\texttt{0x2}$,
$\texttt{ERR\_NUM\_SAMPLES\_ZERO}=\texttt{0x7}$,
$\texttt{ERR\_GAMMA\_SAT}=\texttt{0x4}$.
After triggering each fault, the testbench verifies that `error_code` holds its
value for 50 idle cycles (sticky latch property), then pulses `rst_n` and confirms
that both `error` and `error_code` return to zero.

## tb\_backpressure.sv --- Kernel-Ready Handshake

Tests the `kernel_valid` / `kernel_ready` flow-control handshake across three
scenarios: (A) `kernel_ready` permanently asserted; (B) `kernel_ready` released on
the same cycle that `kernel_valid` rises; (C) a 3-cycle delay before `kernel_ready`
asserts. Scenario C exposed a defect in a prior revision where `kernel_valid`
pulsed for exactly one cycle, causing the FSM to stall indefinitely.

## tb\_consecutive.sv --- Back-to-Back Batch Resets

Executes two complete classification batches without an intervening `rst_n`, confirming
that the FSM and all internal counters reset cleanly between batches.
Key invariants: `sample_counter`, `class_counter`, and `sv_counter` all return to zero
at the batch boundary; `done` asserts exactly once per batch; no error flags carry over.
This reflects normal wearable operation with continuous 1000-beat batches.

## tb\_dist\_boundary.sv --- Accumulator Saturation

Presents a worst-case input pair: $\hat{x}_j = \texttt{0x7FFF}$
(corresponding to $+31.999$ in $Q_{6.10}$) and
$\widehat{sv}_j = \texttt{0x8000}$ ($-32.0$ in $Q_{6.10}$),
maximising the squared component difference
$(\hat{x}_j - \widehat{sv}_j)^2 = (2^{15}-1+2^{15})^2$.
For $\text{FEATURE\_DIM}=16$ this saturates the 20-bit squared-distance
accumulator at $\texttt{0xFFFFF}$.
Verifies that the Horner LUT correctly maps the saturated distance to
$\hat{K} = 0$ (i.e., $e^{-\infty} = 0$).

## tb\_dist\_zero.sv --- Zero-Distance Kernel Identity

Sets every input feature equal to its corresponding SV feature,
$\hat{x}_j = \widehat{sv}_j = \texttt{0x0400}$ ($= 1.0$ in $Q_{6.10}$) for all $j$,
so $\|\mathbf{x} - \mathbf{sv}\|^2 = 0$ and $K(\mathbf{x},\mathbf{sv}) = 1.0$.
Verifies that the kernel output is exactly $1024 = \texttt{0x0400}$.
This is the canonical check for the 2-cycle pipeline drain: if the FSM reads the
accumulator one cycle early, the final dimension is excluded and the test fails.

## tb\_gamma\_zero.sv --- Gamma Zero Advisory

Writes $\hat{\gamma} = 0$ and classifies a single heartbeat.
Verifies that $\texttt{ERR\_GAMMA\_ZERO} = \texttt{0x6}$ asserts as an advisory
flag while the core still completes the batch without hanging.

## tb\_interface.sv --- Port Signal Protocol

Checks 25 individual assertions covering reset state, register defaults, and
protocol edge cases, including: all outputs deasserted after `rst_n`;
$\hat{\gamma} = \texttt{0x0100}$ at reset; a `start` pulse received outside IDLE
is ignored; a 2-sample batch terminates with exactly two `sample_rdy` pulses
followed by exactly one `done`.

## tb\_min\_sv.sv --- Minimum SV Configuration

Configures one support vector per class ($N_k = 1$ for all $k$, $\sum_k N_k = 5$)
and classifies one beat. Verifies that exactly 5 kernel outputs are produced,
`done` fires once, and all kernel outputs equal $1024$. This exercises the loop
termination logic at its minimal boundary condition.

## tb\_multi\_heartbeat.sv --- Three-Beat Loop-Back

Sets `num_samples`$= 3$ and streams three identical feature vectors. Verifies that
`done` fires exactly once after the third beat (not once per beat) and that
`sample_rdy` pulses three times with the correct `class_out` value each time.

## tb\_param\_write.sv --- Gamma Shadow Register

Attempts to write a new $\hat{\gamma}$ mid-classification (during COMPUTE\_DIST) and
verifies that the new value does not take effect until the next batch.
Confirms the shadow-register fix: $\hat{\gamma}$ is latched at the `start` edge
and the live `param_data` path is disconnected during computation.

## tb\_power.sv --- Battery Fault Behaviour

Tests the two-tier battery monitoring interface.
$\texttt{ERR\_LOW\_BATTERY} = \texttt{0xA}$ is advisory: it asserts when `vbatt_warn`
rises but allows the current classification to complete.
$\texttt{ERR\_POWER\_FAIL} = \texttt{0xB}$ is blocking: it asserts when `vbatt_ok`
deasserts and prevents `start` from launching a new batch.
In m6 the SPI register map exposes `vbatt_ok` at CONTROL[1] and `vbatt_warn` at
CONTROL[2]; the core fault logic is unchanged from m5.

## tb\_warmup.sv --- Warmup Advisory Sequence

Verifies $\texttt{ERR\_WARMING\_UP} = \texttt{0x8}$ and
$\texttt{ERR\_INTERRUPTED} = \texttt{0x9}$ across five sub-tests:
the normal 100-beat warmup sequence; mid-warmup reset; real-fault override
($\texttt{ERR\_GAMMA\_SAT}$ latching sticky over an active advisory); advisory
clearance at beat 100; and the re-warming condition after a reset that fires
at count $= 100$.

## tb\_num\_samples.sv --- num\_samples Parameter Coverage

Dedicated coverage for the `num\_samples` input across five sub-tests.
(T1) `num\_samples`$=1$: baseline --- `done` fires once, `sample\_rdy` fires once,
kernel count $= N_{\mathrm{SV}} = 5$.
(T2) `num\_samples`$=4$: `sample\_rdy` fires four times; `class\_out` is in $[0,4]$ on
every pulse.
(T3) `num\_samples`$= \text{MAX\_BATCH}=8$: boundary case --- kernel count $= 8 \times N_{\mathrm{SV}} = 40$.
(T4) `num\_samples`$=0$ triggers $\texttt{ERR\_NUM\_SAMPLES\_ZERO}=\texttt{0x7}$;
the testbench verifies the sticky-latch property by holding 50 idle cycles and
confirming that both `error` and `error\_code` are unchanged, then pulses `rst\_n`
and asserts both return to zero.
(T5) Changes `num\_samples` from 2 to 3 between consecutive batches without an
intervening reset; each batch produces the correct per-batch `sample\_rdy` count
and kernel total.

---

# Level 2 --- Integration Tests (cocotb, Direct RTL Port)

**Location:** `m6/tb/` $\cdot$ **Simulator:** Icarus Verilog + cocotb 2.0.1 $\cdot$
**Invocation:** `cd m6/tb && make cocotb`

Seven tests use Python coroutines to drive `svm_compute_core` RTL ports directly,
enabling programmatic stimulus generation, $Q_{6.10}$ reference computation, and
numerical result comparison. The DUT interface is identical to m5; no SPI overhead
is introduced at this level.

## test\_reset\_outputs

Confirms that every output port (`done`, `error`, `kernel_valid`, `sample_rdy`)
deasserts within $120\,\text{ns}$ of `rst_n` release.

## test\_param\_programming

Writes $\hat{\gamma} = \texttt{0x0100}$ and $\hat{C} = \texttt{0x0400}$ via the
`param_write_en` / `param_addr` / `param_data` interface and reads back via
`gamma_reg` and `c_reg` output ports. Confirms that the $Q_{6.10}$ encoding
round-trips correctly: $\lfloor 0.25 \cdot 2^{10}\rceil = 256 = \texttt{0x0100}$.

## test\_sv\_counts\_flat

Two-part test of the `num\_sv\_per\_class\_flat` encoding.
First, $\mathbf{N} = [2, 2, 2, 2, 2]$ (sum $= 10 \ll N_{\mathrm{SV}}=500$) is applied
and the core is pulsed; confirms no $\texttt{ERR\_SV\_ZERO}$ or
$\texttt{ERR\_SV\_OVERFLOW}$ fires.
Second, $\mathbf{N} = [200, 200, 200, 200, 200]$ (sum $= 1000 > 500$) is applied
and confirms $\texttt{ERR\_SV\_OVERFLOW}=\texttt{0x2}$ asserts on the next start pulse.

## test\_default\_gamma

Reads `gamma_reg` immediately after reset and confirms it holds
$\texttt{0x0100}$ ($= 0.25$ in $Q_{6.10}$), verifying that the RTL reset value
is correctly encoded.

## test\_full\_pipeline

Executes a complete single-sample pipeline with a constant SRAM model
(all words $= \texttt{0x0400}$, so $\mathbf{x} = \mathbf{sv} = 1.0$,
$\|\mathbf{x}-\mathbf{sv}\|^2 = 0$, $K = 1.0$).
Sets $N_k = 2$ for all $k$ (10 SVs total) and waits for `done`.
Pass criteria: `done` fires exactly once; no real fault asserted; all 10 kernel
outputs equal $1024$.

## test\_kernel\_range

Drives a sine-wave SRAM ($\widehat{sv}_{j} = 0.3\sin(r \cdot 0.3 + j \cdot 0.05)$
for SV rows; input rows $= 0$), producing non-trivial distances.
Asserts that every kernel output satisfies $0 \leq \hat{K} \leq 1024$ in $Q_{6.10}$.
Values outside this interval indicate fixed-point overflow, a sign error in the
Horner polynomial, or an incorrect LUT entry.

## test\_multi\_sample

Sets `num_samples`$= 3$ and classifies three identical beats with the constant SRAM
model. Verifies that `done` fires exactly once (not once per beat) and that the
total kernel count equals $30 = 3 \times 10$.

---

# Level 3 --- Feature Test: RAM\_LATENCY (iverilog, Direct RTL Port)

**Location:** `m6/tb/` $\cdot$ **Simulator:** Icarus Verilog 13.0

**Invocation:**
```
iverilog -g2012 -DSIMULATION -o /tmp/svm_lat_tb.out \
    ../rt1/compute_core.sv svm_ram_latency_tb.sv && /tmp/svm_lat_tb.out
```

## svm\_ram\_latency\_tb.sv --- Wait-State Logic for Physical SRAM

Verifies the `RAM_LATENCY` parameter introduced in m5 to support the IS62WV51216
asynchronous SRAM (10 ns address-to-data access time; LAT$=3$ required at 40 MHz
with PCB parasitics, corresponding to a 25 ns clock period with $3 \times 8.3\,\text{ns}$
of available setup margin).

The testbench instantiates `svm_compute_core` with
FEATURE\_DIM$=4$, $N_k = 1$ for all $k$, MAX\_BATCH\_SIZE$=10$, RAM\_LATENCY$=3$.
The SRAM model is a 3-stage shift register on the address bus: `ram_rdata` presents
the word at `addr_pipe[2]`, faithfully modelling three-cycle address-to-data latency.

In m6 the physical SRAM data path is sourced from `ram_rdata_in[15:0]` --- dedicated
input pads wired directly to SRAM DQ[15:0] without relaying through the management
SoC Logic Analyzer interface. This eliminates the 2-cycle `la_data_in`
re-synchronisation latency present in the m5 Caravel path, so
RAM\_LATENCY$=3$ now represents device latency only, not device$+$SoC relay.

Ten beats are classified with all features and SV values set to
$\hat{x}_j = \widehat{sv}_j = \texttt{0x0100}$
($= 1.0$ in $Q_{6.10}$), giving $K = \exp(0) = 1.0 \Rightarrow \hat{K} = 1024$
and expected class $= 0$ (Normal). Confirmed: `sample_rdy` fires $10\times$,
`done` fires once, no sticky error. Throughput: $208\,\text{cycles/beat}$.

---

# Level 4 --- SPI Unit Tests (cocotb, SPI Slave, svm\_top\_ihp)

**Location:** `m6/tb/` $\cdot$ **Simulator:** Icarus Verilog + cocotb 2.0.1 $\cdot$
**Invocation:** `cd m6/tb && make spi_unit` ($\approx 8\,\text{s}$)

## tb\_spi\_unit.py --- SPI Register Interface Unit Tests

Seven targeted tests exercise the SPI register map through `svm_top_ihp`. All tests
use a minimal 5-SV-per-class configuration ($\sum_k N_k = 25$) so each test
completes within $400{,}000\,\text{ns}$. The SPI helper drives 40-bit frames:
an 8-bit address byte (bit[7]$=0$ for write, bit[7]$=1$ for read) followed by
32 data bits, MSB first, CPOL$=0$, CPHA$=0$.
The signal `spi_csn` is held low for the full frame and released to latch writes.

The register map under test is defined by the SPI FSM in `rt1/top.sv`:

| Address | Name | Access | Width | Change from m5 |
|---------|------|--------|-------|----------------|
| \texttt{0x01} | CONTROL | RW | 32 | Replaces WB \texttt{0x04} |
| \texttt{0x02} | STATUS | RO | 32 | Replaces WB \texttt{0x08} |
| \texttt{0x03} | NUM\_SAMPLES | RW | 10 | Replaces WB \texttt{0x0C} |
| \texttt{0x04}--\texttt{0x08} | NUM\_SV[0--4] | RW | 8 | Replaces WB \texttt{0x10}--\texttt{0x20} |
| \texttt{0x09} | PARAM\_WR | WO | 20 | Replaces WB \texttt{0x24} |
| \texttt{0x0A} | ALPHA\_WR | WO | 26 | sv\_idx now 10-bit in [25:16] (was 9-bit [24:16]) |

## test\_spi\_reset\_outputs

Reads STATUS (\texttt{0x02}) via SPI immediately after `rst_n`. Confirms:
STATUS[0]$=\text{done}=0$; STATUS[1]$=\text{error}=0$;
STATUS[8:6]$=\text{class}=0$; STATUS[9]$=\text{sample\_rdy}=0$.
Also confirms CONTROL (\texttt{0x01}) reads back \texttt{0x00000008}
(reset default: `vbatt_warn`$=1$).

## test\_spi\_gamma\_register

Writes $\gamma = 0.25$ (encoded as $\hat{\gamma} = \texttt{0x00000100}$) to
PARAM\_WR (\texttt{0x09}) using the frame encoding
\[\texttt{data} = (1 \ll 19) \mid (0 \ll 16) \mid \texttt{0x0100},\]
then reads back the internal `gamma_reg` signal via the cocotb DUT hierarchy.
Confirms that the $Q_{6.10}$ value $\texttt{0x0100}$ round-trips correctly
through the SPI write path and the core parameter latch.

## test\_spi\_num\_sv\_registers

Writes five distinct SV counts
$\mathbf{N} = [10, 20, 15, 25, 30]$ to NUM\_SV[0..4] (addresses
\texttt{0x04}--\texttt{0x08}) and reads each back via SPI read transactions.
Confirms $\sum_k N_k = 100$ and that each register holds its value independently.
Note that ALPHA\_WR is now a 26-bit write: the `sv_global_idx` field occupies
bits [25:16] (10 bits, supporting the range $0$--$499$) versus the m5 9-bit
field [24:16] (range $0$--$499$ with one unused bit).

## test\_spi\_alpha\_load

Loads 25 alpha coefficients linearly varying from $+0.50$ to $+0.26$ in $Q_{6.10}$
through ALPHA\_WR (\texttt{0x0A}) using the encoding
$\texttt{data} = (\texttt{sv\_idx} \ll 16) \mid \hat{\alpha}$.
Verifies that the STATUS error flag remains clear after all 25 writes complete
and that the clock gate (`svm_clk_en`) correctly deasserts between writes when
the drain counter expires.

## test\_spi\_sram\_load

Configures the full 5-SV-per-class SVM, connects a LAT$=3$ SRAM model to
`ram_rdata_in[15:0]` and `ram_addr_out[18:0]`, then fires CONTROL[0]$=\text{start}$
via SPI. Monitors `sample_rdy` on the output pad directly (not via `la_data_in`
as in m5). Confirms that the ASIC classifies the single-sample batch without a
sticky error ($\texttt{error\_code} < \texttt{0x8}$). The advisory
$\texttt{ERR\_WARMING\_UP} = \texttt{0x8}$ is expected and excluded from the
failure criterion.

## test\_spi\_ram\_latency

Identical to `test_spi_sram_load` but uses a LAT$=1$ ideal SRAM model to verify
that the wait-state logic correctly handles the minimum-latency case and that
the same classification result is produced independently of the latency setting.

## test\_spi\_start\_clear

Fires CONTROL[0]$=\text{start}$ via SPI, waits for the `done` output pad to assert,
then reads STATUS (\texttt{0x02}) to confirm the done bit is set. Writes
CONTROL$=\texttt{0x00000000}$ to clear start, then monitors `sample_rdy` for
100 cycles to confirm the FSM returned to IDLE without re-triggering. Verifies
the auto-clear behaviour: CONTROL[0] deasserts on the clock edge following the
SPI CS\# rising edge.

---

# Level 5 --- SPI System Cosimulation (cocotb, SPI Slave, svm\_top\_ihp)

**Location:** `m6/tb/` $\cdot$ **Simulator:** Icarus Verilog + cocotb

**Invocation:** `PYTHONUNBUFFERED=1 make sim` (300 samples, $\approx16\,\text{h}$)
or `COSIM_N_EVAL=25 make sim` (25-sample subset for quick validation)

## tb\_spi\_cosim.py --- 300-Sample SPI Cosimulation

This is the primary system-level verification. The testbench acts as the nRF52840
host MCU and drives the full `svm_top_ihp` wrapper exclusively through the SPI
slave register interface --- the same path used by production firmware.

**Changes from m5 `tb_wb_cosim.py`** are summarised in the following table:

| Aspect | m5 (Wishbone / Caravel) | m6 (SPI / IHP SG13G2) |
|--------|-------------------------|----------------------|
| Configuration bus | Wishbone B4 at \texttt{0x3000\_0000} | SPI slave, CPOL$=0$, CPHA$=0$ |
| SRAM data path | `la_data_in[15:0]` (via management SoC) | `ram_rdata_in[15:0]` (dedicated pads) |
| Top module | `user_project_wrapper` | `svm_top_ihp` |
| $\alpha$ address width | 9-bit [24:16] | 10-bit [25:16] --- supports full $0$--$499$ |
| SV allocation $\mathbf{N}$ | $[95,95,95,120,95]$ | $[95,95,95,120,95]$ (unchanged) |
| NUM\_SAMPLES reset default | $300$ (written at startup) | $1000$ (sticky hardware default) |
| Sample-ready signal | GPIO[3] on `io_out` | `sample_rdy` output pad |
| SRAM latency modelled | LAT$=1$ (la relay adds $\sim2\,\text{cycles}$) | LAT$=3$ (device latency only) |
| Event-driven sampling | Polling loop ($158\times10^6$ callbacks) | `await First(RisingEdge(sample_rdy), Timer(...))` |

**Setup phase.** Real ECG records are loaded from MIT-BIH, SVDB, and INCART via the
`wfdb` PhysioNet interface. An 80/20 stratified split with
`random_state`$=42$ yields 300 test samples (60 per class). An sklearn OVR-SVM
with $\gamma = 0.25$, $C = 1.0$ is trained in floating-point. The 500 coefficients
$\{\hat{\alpha}^{(k)}_i\}$ and the SV matrix are quantised to $Q_{6.10}$ and written
to a Python dictionary that models the off-chip IS62WV51216 SRAM.

**Configuration sequence.** The following SPI transactions are issued at startup:

1. Write NUM\_SAMPLES (\texttt{0x03}) $= 300$
2. Write NUM\_SV[0..4] (\texttt{0x04}--\texttt{0x08}) $= [95, 95, 95, 120, 95]$
3. Write PARAM\_WR (\texttt{0x09}) 500 times to load all $\hat{\alpha}^{(k)}_i$
   (10-bit `sv_idx` in bits [25:16])
4. Write CONTROL (\texttt{0x01}) $= \texttt{0x00000007}$
   ($\text{start}=1$, $\text{vbatt\_ok}=1$, $\text{vbatt\_warn}=1$)

**SRAM model.** A cocotb coroutine monitors `ram_ren_out` and `ram_addr_out[18:0]`
on the output pads. When `ram_ren_out` asserts, the coroutine drives the
corresponding 16-bit word onto `ram_rdata_in[15:0]` after LAT$=3$ clock cycles,
directly modelling the IS62WV51216 and eliminating the management SoC relay latency
of the m5 path.

**Classification loop.** After `start` fires, the ASIC drives the SRAM bus
autonomously. The cocotb coroutine uses
\[\texttt{await First(RisingEdge(dut.sample\_rdy), Timer(timeout\_ns, "ns"))}\]
to wake Python exactly once per sample rather than once per clock cycle, reducing
callback overhead from $158 \times 10^6$ to 300 per 300-sample batch.
Results are accumulated and written to `../sim/asic_preds.csv`.

**Result.** 295 of 300 samples are classified correctly:

$$\text{Accuracy}_{\text{ASIC}} = \frac{295}{300} = 98.33\% \;>\; \frac{293}{300} = 97.67\% = \text{Accuracy}_{\text{float}}.$$

The 2-sample advantage of the ASIC over the sklearn float reference arises from
quantisation-favourable boundary effects: the $Q_{6.10}$ rounding of the RBF kernel
shifts the decision margin $f_k(\mathbf{x}) - f_{k'}(\mathbf{x})$ across zero in the
correct direction for 2 samples that float misclassifies.
The 5 remaining errors occur on beats at the VT/SVT morphological boundary where
both float and $Q_{6.10}$ agree the samples are ambiguous.

---

# Summary

| Level | Testbench(es) | Interface | Framework | $n$ | Result |
|-------|---------------|-----------|-----------|-----|--------|
| 1 --- Unit | tb\_top, tb\_error\_codes, tb\_backpressure, tb\_consecutive, tb\_dist\_boundary, tb\_dist\_zero, tb\_gamma\_zero, tb\_interface, tb\_min\_sv, tb\_multi\_heartbeat, tb\_num\_samples, tb\_param\_write, tb\_power, tb\_warmup | Direct RTL | iverilog | 14 | 14/14 PASS |
| 2 --- Integration | test\_reset\_outputs, test\_param\_programming, test\_sv\_counts\_flat, test\_default\_gamma, test\_full\_pipeline, test\_kernel\_range, test\_multi\_sample | Direct RTL | cocotb | 7 | 7/7 PASS |
| 3 --- Feature | svm\_ram\_latency\_tb | Direct RTL | iverilog | 1 | 1/1 PASS |
| 4 --- SPI Unit | tb\_spi\_unit (7 tests) | SPI + svm\_top\_ihp | cocotb | 7 | pending |
| 5 --- SPI System | tb\_spi\_cosim | SPI + svm\_top\_ihp | cocotb | 1 | PASS --- 98.33\% |
| **Total** | | | | **30** | **23/23 PASS + 7 pending** |

Level 6 (Caravel management SoC DV) from m5 is not applicable to m6: the IHP SG13G2
target is a standalone design with no management SoC. The nRF52840 MCU role is
fulfilled by the SPI master firmware, which is validated at Level 5 through the
`tb_spi_cosim.py` host emulation.

---
*ECE410 $\cdot$ Portland State University $\cdot$ Adam Handwerger $\cdot$ IHP SG13G2 $\cdot$ MIT-BIH + SVDB + INCART*
