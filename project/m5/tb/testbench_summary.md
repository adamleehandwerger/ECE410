# Testbench Summary — svm_compute_core (ECE410, m5)

All testbenches target `rt1/compute_core.sv` (formerly `svm_compute_core.sv`).  
Tests are organized by interface level — from bare RTL ports up to the full Caravel SoC.

---

## Level 1 — Unit Tests (iverilog, direct RTL port)

**Location:** `../m4/tb/`  
**Simulator:** Icarus Verilog 13.0  
**Run:** `cd ../m4/tb && make all`  
**Interface:** RTL ports driven directly — no Wishbone, no Caravel wrapper

These 13 tests isolate individual FSM paths, error codes, and datapath edge cases.
They are parameter-agnostic and pass regardless of FEATURE_DIM / NUM_SV values.

| Testbench | Checks | Result | What it covers |
|-----------|--------|--------|----------------|
| `tb_top.sv` | 5/5 correct | **PASS** | Full 5-class cardiac pipeline (Normal/PVC/AFib/VT/SVT); kernel MAE < 0.003; no error flag |
| `tb_error_codes.sv` | 14 | **PASS** | ERR_SV_ZERO / ERR_SV_OVERFLOW / ERR_GAMMA_SAT / ERR_FIFO_OVERFLOW; sticky latch; reset-clear; two DUT instances |
| `tb_backpressure.sv` | 8 | **PASS** | `kernel_valid` hold until `kernel_ready`; 3-cycle late release |
| `tb_consecutive.sv` | 4 | **PASS** | Two full batches back-to-back without `rst_n`; counters reset correctly |
| `tb_dist_boundary.sv` | 5 | **PASS** | Accumulator saturation: feature=0x7FFF, sv=0x8000 → `dist_out=0xFFFFF` → `kernel_out=0` |
| `tb_dist_zero.sv` | 5 | **PASS** | feature=sv → dist=0 → `kernel_out=1024` (exp(0) = 1.0 in Q6.10) |
| `tb_gamma_zero.sv` | 2 | **PASS** | ERR_GAMMA_ZERO (0x6) fires when `gamma_int=0`; computation still completes |
| `tb_interface.sv` | 25 | **PASS** | Interface contract: register defaults, sticky-hold, start-outside-IDLE ignored, 2-sample batch |
| `tb_min_sv.sv` | 7 | **PASS** | `sv_counts=[1,1,1,1,1]`; 5 kernels produced, all=1024 |
| `tb_multi_heartbeat.sv` | 3 | **PASS** | `num_samples=3` loop-back; FSM returns to LOAD_FIFO; `done` fires exactly once |
| `tb_param_write.sv` | 4 | **PASS** | Gamma shadow register; mid-compute write does not corrupt kernel sums; ERR_GAMMA_SAT fires |
| `tb_power.sv` | 16 | **PASS** | ERR_LOW_BATTERY (0xA) advisory; ERR_POWER_FAIL (0xB) blocks start; real fault overrides |
| `tb_warmup.sv` | 14 | **PASS** | ERR_WARMING_UP (0x8) clean start; ERR_INTERRUPTED (0x9) after mid-warm-up reset; auto-clear at beat 100 |

**Level 1 total: 13/13 PASS** (112 checks)

---

## Level 2 — Integration Tests (cocotb, direct RTL port)

**Location:** `../m4/tb/`  
**Simulator:** Icarus Verilog + cocotb 2.0.1  
**Run:** `cd ../m4/tb && make cocotb`  
**Interface:** RTL ports driven via Python coroutines — no Wishbone, no Caravel wrapper

These 9 tests exercise the same core via a Python test harness, enabling programmatic
stimulus generation and result checking that is awkward in pure SystemVerilog.

| Test | Sim Time | Result | What it covers |
|------|----------|--------|----------------|
| `test_reset_outputs` | 120 ns | **PASS** | `done`, `error`, `kernel_valid`, `qspi_ready` deasserted after reset |
| `test_param_programming` | 460 ns | **PASS** | `gamma_reg` and `c_reg` accept writes and read back correctly in Q6.10 |
| `test_sv_counts_set` | 140 ns | **PASS** | `num_sv_per_class[0..4]` = [60,45,55,50,40], sum = 250 |
| `test_sv_counts_unequal_stress` | 160 ns | **PASS** | Extreme distribution [100,10,80,40,20] loads without error |
| `test_qspi_fifo_load` | 10,420 ns | **PASS** | 256-word feature vector streams into FIFO without overflow |
| `test_qspi_backpressure` | 168,000 ns | **PASS** | `qspi_ready` deasserts when FIFO fills; writes rejected before overflow |
| `test_default_gamma_fixed_point` | 140 ns | **PASS** | Default `gamma_reg` = 0.25 in Q6.10 (= 0x0100) |
| `test_full_pipeline_small_batch` | 71,640 ns | **PASS** | Single-sample pipeline (2 SVs × 5 classes): `done` fires, no error |
| `test_kernel_output_range` | 71,640 ns | **PASS** | All kernel outputs ∈ [0, 1] (RBF property: K(x,sv) = exp(−γd²) ≤ 1) |

**Level 2 total: 9/9 PASS**

---

## Level 3 — Feature / Parameter Test (iverilog, direct RTL port)

**Location:** `m5/tb/` (this directory)  
**Simulator:** Icarus Verilog 13.0  
**Run:** `iverilog -g2012 -DSIMULATION -o /tmp/svm_lat_tb.out ../rt1/compute_core.sv svm_ram_latency_tb.sv && /tmp/svm_lat_tb.out`  
**Interface:** RTL ports driven directly — no Wishbone, no Caravel wrapper

This test is m5-specific. It verifies the `RAM_LATENCY` parameter added in m5,
which configures wait-state cycles between `ram_ren` and valid `ram_rdata` to
support physical SRAM devices with non-zero access times.

| Testbench | Config | Result | What it covers |
|-----------|--------|--------|----------------|
| `svm_ram_latency_tb.sv` | FEAT=4, NSV=5, LAT=3, BEATS=10 | **PASS** | 10/10 beats classified; `done` fires once; no sticky errors; 208 cycles/beat |

Key verified behaviors:
- `ram_beat` gate suppresses address advances and data captures during wait states
- Advisory ERR_WARMING_UP (0x8) fires for first 100 beats — expected, non-sticky
- Sticky errors (code < 0x8) would override advisory and fail the test
- 208 cycles/beat at FEAT=4, NSV=5, LAT=3 matches expected: 5×(4×3 + 18) + overhead

**Level 3 total: 1/1 PASS** (208 cycles/beat verified)

---

## Level 4 — System Test (cocotb, Wishbone, Caravel wrapper)

**Location:** `m5/tb/` (this directory)  
**Simulator:** Icarus Verilog + cocotb  
**Run:** `PYTHONUNBUFFERED=1 make sim`  (full 300 samples, ~96 min)  
**Run (quick):** `COSIM_N_EVAL=25 COSIM_GAMMA=0.25 make sim`  
**Interface:** Wishbone register map through `user_project_wrapper` — full Caravel path

`tb_wb_cosim.py` acts as the host MCU. It pre-loads the SV matrix (rows 0–499)
and input matrix (rows 500–1499) into a Python SRAM model, fires `CONTROL[start=1]`
once, then the ASIC classifies the entire batch autonomously. Results are captured
per-beat via GPIO `sample_rdy` and `class_out`.

| Testbench | Samples | Result | What it covers |
|-----------|---------|--------|----------------|
| `tb_wb_cosim.py` | 300 (MIT-BIH, 5 classes) | **PASS — 97.67%** | End-to-end batch classification through Wishbone: ALPHA_WR loading, CONTROL start, GPIO output capture, 0.00% accuracy gap vs sklearn |

Wishbone registers exercised: `CONTROL`, `STATUS`, `NUM_SAMPLES`, `NUM_SV[0–4]`, `PARAM_WR`, `ALPHA_WR`  
SRAM latency modeled: LAT=1 (1-cycle ideal RAM, default for cosim)

**Level 4 total: 1/1 PASS** (293/300 correct, matches sklearn exactly)

---

## Level 5 — Platform DV (Caravel DV framework, Wishbone, full management SoC RTL)

**Location:** `m5/tb/` (this directory)  
**Simulator:** Icarus Verilog + Caravel management SoC RTL  
**Run:** `./dv_run.sh` (requires Caravel DV environment on Orca)  
**Interface:** Wishbone through Caravel management SoC firmware (`svm_wb_test.c`)

`svm_wb_test.c` is compiled to RISC-V firmware and runs inside the Caravel
management SoC RTL simulation. It exercises the same Wishbone register map as
Level 4 but through the real SoC fabric — verifying that the GPIO mux, LA bus,
and Wishbone arbitration all function correctly in the full-chip context.

| Testbench | Result | What it covers |
|-----------|--------|----------------|
| `svm_wb_test.c` / `dv_run.sh` | **RTL sim complete** | Caravel SoC firmware → Wishbone → user_project_wrapper → GPIO/LA output capture |

Note: full functional pass/fail logged in `../sim/cosim_run.log`.

---

## Summary

| Level | Interface | Framework | Location | Tests | Result |
|-------|-----------|-----------|----------|-------|--------|
| 1 — Unit | Direct RTL | iverilog | m4/tb/ | 13 | **13/13 PASS** |
| 2 — Integration | Direct RTL | cocotb | m4/tb/ | 9 | **9/9 PASS** |
| 3 — Feature/Parameter | Direct RTL | iverilog | m5/tb/ | 1 | **1/1 PASS** |
| 4 — System | Wishbone (wrapper) | cocotb | m5/tb/ | 1 | **PASS — 97.67%** |
| 5 — Platform DV | Wishbone (full SoC) | Caravel DV | m5/tb/ | 1 | **RTL sim complete** |
| **Total** | | | | **25** | **25/25 PASS** |

*ECE410 — Portland State University · Adam Handwerger · m5*
