# SVM Compute Core — Design Summary (m4: Batch Architecture v8)

**Project:** Multi-Class Cardiac Arrhythmia Detection
**RTL:** `svm_compute_core.sv` (batch v8 — 256-feature, 250 SVs, Q6.10 fixed-point)
**Architecture:** Batch — host pre-loads SV matrix + input matrix; ASIC classifies autonomously
**Accuracy:** 96.39% on MIT-BIH (sklearn = hardware, 0.00% gap)
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**Status:** RTL v8 complete; new DRT in progress (prior job 91947 results below)

---

## P&R Results (OL2 job 91947, nom_tt_025C_1v80 — prior RTL)

| Metric | Value |
|--------|-------|
| Clock target | 40 MHz (25 ns period) |
| Setup WNS (TT) | **+7.923 ns — CLEAN, 0 violations** |
| Hold WNS (TT) | **+0.297 ns — CLEAN, 0 violations** |
| Active power | **66 mW** (42.8 mW internal + 23.2 mW switching) |
| Avg power (80 bpm) | **~0.26 mW** (~0.4% duty cycle) |
| Standard cells | 146,311 |
| Die area | 2500 × 2500 µm (6.25 mm²) |
| Utilization | 14.1% |
| DRC violations | **0** |
| Wire length | 1,565,010 µm |

*These metrics are from the prior streaming architecture (v7). The batch
architecture (v8) removes the 512-deep FIFO and 64-entry work_ram, replacing
them with the LOAD_INPUT state machine. Utilization and power will decrease;
all timing constraints remain the same (40 MHz, 25 ns).*

---

## 1. Batch Architecture (v8)

### What Changed from v7

| Component | v7 (streaming) | v8 (batch) |
|-----------|----------------|------------|
| Input path | FIFO_DATA WB writes (256 words/beat) | Off-chip SRAM via GPIO/LA bus |
| Input FIFO | 512 × 16-bit register array | **Removed** |
| work_ram | 64 × 16-bit result buffer | **Removed** |
| SV RAM bus | GPIO[24:10] = 15-bit sv_ram_addr | GPIO[28:10] = 19-bit unified ram_addr |
| Input RAM | Wishbone (host pushes) | Same GPIO/LA bus (ASIC pulls) |
| Per-beat output | Poll work_ram after batch done | `sample_rdy` IRQ[0] per beat |
| Batch done signal | IRQ[0] | IRQ[1] |
| Clock gate | `qspi_valid` based (could open/close between beats) | `batch_active` register (stays open entire batch) |
| WB registers | FIFO_DATA (0x00), WORK_RD (0x38), STATUS2 (0x3C) | **All three removed** |
| FSM states | IDLE → LOAD_FIFO → LOAD_FEATURES → COMPUTE_DIST → … | IDLE → LOAD_INPUT → COMPUTE_DIST → … |

### Off-chip RAM Address Map

```
Address = {row[10:0], col[7:0]}  (19-bit)

Rows   0 .. 249   SV matrix      (250 SVs × 256 features × 2 B = 128 KB)
Rows 250 .. 1249  Input matrix   (1000 beats × 256 features × 2 B = 512 KB max)

Maximum address: 1250 × 256 − 1 = 319 999  →  19 bits
```

### LOAD_INPUT State

The LOAD_INPUT state replaces LOAD_FIFO + LOAD_FEATURES:

```
cycle  0:     ram_addr = {NUM_SV + sample_counter, 0},  ram_ren = 1
cycle  1:     ram_rdata valid → feature_bank[0] latched
cycle  1..256: advance feat_wr_addr, ram_ren high while < FEATURE_DIM
cycle  258:   feat_wr_count == 256 → transition to COMPUTE_DIST
```

One-cycle RAM latency is absorbed by registering `feat_wr_en` and `feat_wr_addr`.
9-bit counters prevent 8-bit wrap at FEATURE_DIM = 256 (critical fix).

### batch_active Clock Gate

Without `batch_active`, the ICG would close between the `start` pulse and the
first clock cycle the FSM turns active, stalling indefinitely. The `batch_active`
register solves this:

```systemverilog
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)              batch_active <= 0;
    else if (reg_control[0]) batch_active <= 1;   // set on start
    else if (svm_done)       batch_active <= 0;   // clear on batch done
end
wire svm_clk_en = batch_active | reg_control[0] | core_warming | (drain_cnt > 0);
```

---

## 2. FSM

```
IDLE
 │  start && vbatt_ok
 ▼
LOAD_INPUT  ──────────────────────────────────┐
 │  feat_wr_count == FEATURE_DIM              │ (loop: next sample)
 ▼                                            │
COMPUTE_DIST                                  │
 │  dist_done                                 │
 ▼                                            │
COMPUTE_KERNEL                                │
 │  horner_done                               │
 ▼                                            │
OUTPUT_RESULT                                 │
 │  kernel_valid (advance sv/class counter)   │
 │  last_sv && last_class                     │
 ▼                                            │
WRITE_CLASS ─── last_heartbeat ─────► IDLE   │
 │                                            │
 └──── !last_heartbeat ──────────────────────┘
```

Cycle budget per sample (approx.):
- LOAD_INPUT: 256 + 2 = 258 cycles
- Per SV: COMPUTE_DIST (258) + COMPUTE_KERNEL (20) + OUTPUT_RESULT (1) = 279 cycles
- 250 SVs: 69,750 cycles
- WRITE_CLASS: 1 cycle
- **Total per sample: ≈ 70,009 cycles**
- **1000-sample batch: ≈ 70 M cycles at 40 MHz ≈ 1.75 s**

---

## 3. Fixed-Point Format — Q6.10

Unchanged from m3/v7. 16-bit signed, 10 fractional bits.

| Value | Q6.10 | Hex |
|-------|-------|-----|
| γ = 0.25 | 256 | `0x0100` |
| γ = 1.0 | 1024 | `0x0400` |
| Feature range | ±32 | `0x8000..0x7FFF` |

Quantization accuracy verified: sklearn = hardware, **96.39%, 0.00% gap**.

---

## 4. Yosys / OpenSTA Compatibility Fixes

All fixes carried forward from v7 (job 91947):

| Issue | Fix |
|-------|-----|
| `output logic [W-1:0] bias_reg [5]` — unpacked array port | Port removed |
| `return expr` in case arms | Changed to `fn_name = expr` |
| `feature_bank [FEATURE_DIM]` → `$mem` inference | `(* ram_style = "registers" *)` |
| `$_ALDFFE_PNP_` non-constant async reset | `arm_interrupted` / `interrupted` removed |
| `sim_sram_models.sv` treated as gate-level by OpenSTA | `/// sta-blackbox` directive |
| OpenSTA corner.tcl unsupported properties | `catch {}` wrappers on Orca |

**v8 specific:** `input_fifo` (FIFO_DEPTH=512 register array) removed entirely —
no `ram_style` annotation needed. `work_ram` (64 entries) also removed.

---

## 5. Power Analysis

### Active Power (prior job 91947, 40 MHz TT)

| Component | Power |
|-----------|-------|
| Internal logic | 42.8 mW |
| Switching | 23.2 mW |
| Leakage | ~0.4 µW |
| **Total active** | **66.0 mW** |

### Wearable Budget (batch architecture)

The batch model changes the duty cycle calculation:

| Parameter | Value |
|-----------|-------|
| Batch size | 1000 beats |
| Time to collect 1000 beats at 80 bpm | 750 s |
| Time to classify 1000 beats (ASIC, 40 MHz) | ~1.75 s |
| Duty cycle | 1.75 / 750 = **0.23%** |
| **Avg SVM core power** | 66 mW × 0.0023 = **~0.15 mW** |
| 200 mAh @ 3.7V (740 mWh) | **~200 days** from SVM core alone |

The batch architecture improves average power by ~2× compared to per-beat
streaming, since the ASIC is idle for longer between classification bursts.

---

## 6. Timing — 40 MHz, TT Corner

Setup WNS +7.923 ns (prior run). Critical path: FIFO read-pointer decode →
feature-bank mux → distance accumulator. With FIFO removed in v8, the critical
path is expected to shift to the distance accumulator or Horner engine, which
have similar register-to-register depths. New DRT result will confirm.

SS/FF corner violations expected (sky130_fd_sc_hd inherent at extreme corners).
TT is the target corner for ECE410 submission.

---

## 7. Caravel Integration

### GPIO / LA Assignment

| Signal | Pins | Direction |
|--------|------|-----------|
| `class_out[2:0]` | GPIO[2:0] | output |
| `sample_rdy` | GPIO[3] / IRQ[0] | output |
| `svm_done` | GPIO[4] / IRQ[1] | output |
| `svm_error` | GPIO[5] | output |
| `error_code[3:0]` | GPIO[9:6] | output |
| `ram_addr[18:0]` | GPIO[28:10] | output |
| `ram_ren` | GPIO[29] | output |
| `ram_rdata[15:0]` | LA[15:0] | input (host drives) |

### Clock Gate

```
wb_clk_i → sky130_fd_sc_hd__dlclkp_1 (ICG) → svm_gclk → svm_compute_core
                    ↑
              svm_clk_en = batch_active | reg_control[0] | core_warming | drain_cnt>0
```

In SIMULATION (`define SIMULATION`), the ICG is replaced with a simple AND gate.

---

## 8. Submission Artifacts

| File | Location | Status |
|------|----------|--------|
| `svm_compute_core.gds` | `pnr/gds/` + caravel repo | Prior run (new DRT pending) |
| `svm_compute_core.lef` | caravel repo | Prior run |
| `svm_compute_core.v` (GL) | caravel repo | Prior run |
| `user_project_wrapper.gds` | caravel repo | Pending new wrapper DRT |
| `user_project_wrapper.lef` | caravel repo | Pending |
| `user_project_wrapper.v` (GL) | caravel repo | Pending |

---

*Document version: m4 v8 batch · 2026-05-24*
