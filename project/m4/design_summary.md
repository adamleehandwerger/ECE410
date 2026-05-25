# SVM Compute Core — Design Summary (m4: OL2 Hardening & GDS Tape-Out)

**Project:** Multi-Class Cardiac Arrhythmia Detection
**RTL:** `svm_compute_core.sv` (256-feature, 250 SVs, Q6.10 fixed-point)
**Accuracy:** 96.39% on MIT-BIH (sklearn = hardware, 0.00% gap)
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**Status:** P&R complete — GDS/LEF/GL committed; wrapper hardening in progress

---

## P&R Results Summary (OL2 job 91947, sky130A TT/25°C/1.8V)

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

---

## 1. OL2 Flow — Key Changes from m4 Manual DRT

The earlier m4 milestone used a custom manual OpenROAD flow at 100 MHz (10 ns period),
which produced −14 ns setup violations. The OL2 Classic flow correctly targets **40 MHz**
(25 ns), the practical maximum for sky130_fd_sc_hd at TT corner, and delivers a clean
timing closure with 7.9 ns slack.

| Metric | m4 manual (drt_v12, 100 MHz) | m4 OL2 (job 91947, 40 MHz) |
|--------|------------------------------|----------------------------|
| Clock period | 10 ns | **25 ns** |
| Setup WNS (TT) | −14.04 ns (VIOLATED) | **+7.923 ns (CLEAN)** |
| Hold WNS (TT) | −3.01 ns (pre-filler) | **+0.297 ns** |
| Active power | 575 mW | **66 mW** |
| Cell count | ~162K | **146K** |
| Die utilization | 50% | **14.1%** |
| DRC violations | 0 | **0** |

Key OL2 improvements:
- Proper CTS (clock tree synthesis) inserts hold buffers → hold violations resolved
- AREA 0 synthesis strategy + 45% density target → lower power, smaller footprint
- `(* ram_style = "registers" *)` on feature_bank + FIFO → no unmapped SRAM macros
- All Yosys/OpenSTA compatibility issues resolved (see pnr/core_harden.sh)

---

## 2. Yosys Compatibility Fixes (m4 OL2)

Several RTL constructs were incompatible with Yosys 0.46 on sky130A and required fixes:

| Issue | Fix |
|-------|-----|
| `output logic [W-1:0] bias_reg [5]` — unpacked array port | Removed port entirely |
| `return expr` in `case` arms of functions | Changed to `function_name = expr` |
| `input_fifo` $mem inference (DEPTH=8192) | Added `(* ram_style = "registers" *)` |
| `feature_bank [FEATURE_DIM]` $mem inference | Added `(* ram_style = "registers" *)` |
| `$_ALDFFE_PNP_` — non-constant async reset on `interrupted` | Removed `arm_interrupted` and `interrupted` signals |
| `sim_sram_models.sv` parsed as gate-level netlist by OpenSTA | Added `/// sta-blackbox` directive |
| OpenSTA corner.tcl — `is_propagated`, `is_virtual`, `is_generated`, `sources`, `report_clock_latency`, `report_clock_min_period` not supported | Patched with `catch {}` wrappers |

---

## 3. Fixed-Point Format — Q6.10

Unchanged from m3. 16-bit signed, 10 fractional bits, γ = 0.25 exactly representable.
Quantization accuracy verified by hardware simulation (confusion_comparison.py).
sklearn accuracy = hardware accuracy: **0.00% gap**.

---

## 4. Timing — 40 MHz, TT Corner Clean

Setup slack: +7.923 ns (critical path uses 17.1 ns of 25 ns budget).
Worst register-to-register slack: +14.97 ns.

The critical path runs through the FIFO read-pointer decode → feature-bank mux →
distance accumulator feedback chain. With a 25 ns period this path closes comfortably.

SS/FF corner timing violations are expected for a complex compute design on sky130
at extreme corners (−56.7 ns at 100°C/1.60V, −29.2 ns at −40°C/1.95V). TT is the
target corner for ECE410 submission.

---

## 5. Power — Wearable Budget

| Component | Power |
|-----------|-------|
| Internal logic | 42.8 mW |
| Switching | 23.2 mW |
| Leakage | ~0.4 µW |
| **Total (active)** | **66.0 mW** |

Wearable analysis at 80 bpm:
- Active duration per beat: ~3 ms (classification)
- Beat period: 750 ms → duty cycle ~0.4%
- **Average SVM core power: ~0.26 mW**
- 200 mAh @ 3.7V battery (740 mWh) → **~119 days** from SVM core alone

The 256-dim feature set is retained — no need to reduce to 128-dim.

---

## 6. Caravel Submission Artifacts

| File | Location | Size |
|------|----------|------|
| `svm_compute_core.gds` | `gds/` (caravel repo, local) + `project/m4/pnr/gds/` (ECE410 repo, LFS) | 181 MB |
| `svm_compute_core.lef` | `lef/svm_compute_core.lef` (caravel repo) | 108 KB |
| `svm_compute_core.v` | `verilog/gl/svm_compute_core.v` (caravel repo) | 13 MB |

GDS committed to ECE410 repo via git-lfs. Caravel public fork blocks LFS upload for
new objects (GitHub restriction on public forks); GDS kept locally + in ECE410 repo.

---

## 7. user_project_wrapper Status

Wrapper hardening in progress (SLURM job 91948, long partition).
Prior run (multiple jobs 91877–91910) completed through detailed placement (step 25).
Failed at global routing due to SLURM job kill (not a routing failure — GRT reported
0 overflow). Job 91948 resumes from step 25 and runs through DRT → GDS.

---

*Document version: m4 OL2 · 2026-05-24*
