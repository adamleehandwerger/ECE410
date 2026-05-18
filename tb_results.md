# Testbench Results — ECE410 SVM Compute Core (LUT Kernel)

RTL file: `svm_compute_core.sv`  
Simulator: Icarus Verilog 13.0 / cocotb 2.0.1  
Date: 2026-05-18

---

## RTL Fixes Applied

| # | Fix | Symptom |
|---|-----|---------|
| 1 | **kernel_valid hold register** — changed from 1-cycle pulse to set/clear latch (held until `kernel_ready` handshake) | FSM stalled permanently if `kernel_ready=0` during the single cycle `kernel_valid` was high |
| 2 | **gamma shadow register** (`gamma_latched`) — captured from `gamma_int` at `start`, used by Horner engine throughout the batch | Mid-compute `param_write_en` could corrupt in-flight kernel values |
| 3 | **ERR_GAMMA_ZERO (0x6)** — added error code that fires when `gamma_int == 0` while FSM is not idle | gamma=0 silently produced all-1.0 kernels (constant classifier), no error was raised |
| 4 | **ERR_WARMING_UP (0x8)** — non-sticky advisory: fires on clean start; auto-clears at beat 100 | The 10-beat and 100-beat feature slices are unreliable at cold-start; host had no signal to flag early results |
| 5 | **ERR_INTERRUPTED (0x9)** — non-sticky advisory: `interrupted` flag captured on `rst_n` when `heartbeat_count` was in [1,99]; fires instead of ERR_WARMING_UP so host knows a session was cut short | Host could not distinguish fresh power-on from a disrupted warm-up; both looked identical via ERR_WARMING_UP alone |
| 6 | **ERR_LOW_BATTERY (0xA) / ERR_POWER_FAIL (0xB)** — two new input pins `vbatt_warn` / `vbatt_ok`; 0xA is a non-sticky advisory (battery below soft threshold, device still runs); 0xB blocks `start` while `vbatt_ok` is deasserted but does not abort a running classification | No hardware signal was available to warn the host MCU of low battery or prevent a new classification from being started without sufficient power |
| 7 | **`num_samples` shadow register** (`num_samples_latched`) — captured from `num_samples` at `start && vbatt_ok_s`; `last_heartbeat` uses the latched copy | A mid-batch `num_samples` write could corrupt the batch-end detection, causing early termination or an infinite loop |
| 8 | **`vbatt_ok_s` guard in IDLE counter management** — `sv_count_reg`, `gamma_latched`, and `num_samples_latched` now only latch when `start && vbatt_ok_s` (previously just `start`) | The IDLE counter block could capture stale counts when `vbatt_ok=0` would have blocked the FSM from leaving IDLE anyway |
| 9 | **2-FF input synchronizers** for `vbatt_ok` and `vbatt_warn` (new `sync_ff` module) — reset values: `vbatt_ok→1` (assume power OK at POR), `vbatt_warn→0`; FSM and `err_detect` use `_s` suffix signals | Async comparator outputs driven into a synchronous FSM violate setup/hold, causing metastability at netlist/ASIC |
| 10 | **Distance matrix drain-flush** (`drain_cnt [1:0]`) — 2 extra cycles after the last `valid_in` flush the 2-stage `diff→diff_squared→accumulator` pipeline so all `FEATURE_DIM` (256) contributions are accumulated; `diff`/`diff_sq` also reset in IDLE to prevent stale values from contaminating the next SV computation | Last 2 feature dimensions were silently dropped from every kernel computation, reducing effective feature coverage from 256 to 254 |

---

## iverilog Testbenches

Compiled and run with:
```
iverilog -g2012 -o <out> <tb>.sv svm_compute_core.sv && vvp <out>
```

| Testbench | Checks | Result | What it covers |
|-----------|--------|--------|----------------|
| `tb_error_codes.sv` | 14 | **PASS** | ERR_SV_ZERO / ERR_SV_OVERFLOW / ERR_GAMMA_SAT / ERR_FIFO_OVERFLOW; sticky latch; reset clears; two DUT instances |
| `tb_backpressure.sv` | 8 | **PASS** | `kernel_valid` hold until `kernel_ready`; sub-test C: 3-cycle late release (Fix 1) |
| `tb_multi_heartbeat.sv` | 3 | **PASS** | `num_samples=3` loop-back; FSM returns to LOAD_FIFO; `done` fires exactly once |
| `tb_dist_boundary.sv` | 5 | **PASS** | Accumulator saturation: feature=0x7FFF, sv=0x8000 → `dist_out=0xFFFFF` → `kernel_out=0` |
| `tb_dist_zero.sv` | 5 | **PASS** | feature=sv=0x0400 → dist=0 → `kernel_out=1024` (exp(0)=1.0) |
| `tb_consecutive.sv` | 4 | **PASS** | Two full batches back-to-back without `rst_n`; counters reset correctly |
| `tb_param_write.sv` | 4 | **PASS** | Gamma shadow register (Fix 2); baseline vs mid-write produce identical kernel sums; ERR_GAMMA_SAT fires |
| `tb_gamma_zero.sv` | 2 | **PASS** | ERR_GAMMA_ZERO (0x6) fires on `gamma_int=0` (Fix 3); computation still completes; all kernels=1024 |
| `tb_min_sv.sv` | 7 | **PASS** | `sv_counts=[1,1,1,1,1]`; 5 kernels produced, all=1024 |
| `tb_warmup.sv` | 14 | **PASS** | `ERR_WARMING_UP` (0x8) clean start; `ERR_INTERRUPTED` (0x9) after mid-warm-up reset; real fault overrides advisory; auto-clear at beat 100; reset after completed warm-up shows 0x8 not 0x9 |
| `tb_power.sv` | 16 | **PASS** | `ERR_LOW_BATTERY` (0xA) advisory while `vbatt_warn` high; `ERR_POWER_FAIL` (0xB) blocks start while `vbatt_ok` low; FSM completes mid-run; real fault overrides; auto-clear on pin restore |
| `tb_svm_classifier.sv` | 5/5 correct | **PASS** | Full 5-class cardiac arrhythmia classification (Normal/PVC/AFib/VT/SVT); kernel MAE < 0.003; no error flag |
| `tb_interface.sv` | 25 | **PASS** | Interface contract: register map defaults/writes/reserved; ERR_GAMMA_SAT/SV_ZERO/SV_OVERFLOW/GAMMA_ZERO/NUM_SAMPLES_ZERO; sticky hold; start-triggers-batch; start-outside-IDLE ignored; 2-sample batch |

**iverilog total: 13/13 PASS** (all tests revalidated after fixes 7–10; tb_interface vbatt ports connected)

---

## cocotb Testbenches

Run with:
```
make   # SIM=icarus, TOPLEVEL=svm_compute_core, COCOTB_TEST_MODULES=test_svm_compute_core
```

| Test | Sim Time | Result | What it covers |
|------|----------|--------|----------------|
| `test_reset_outputs` | 120 ns | **PASS** | `done`, `error`, `kernel_valid`, `qspi_ready` all deasserted after reset |
| `test_param_programming` | 460 ns | **PASS** | `gamma_reg` and `c_reg` accept writes and read back correctly in Q6.10 |
| `test_sv_counts_set` | 140 ns | **PASS** | `num_sv_per_class[0..4]` = [60,45,55,50,40], sum = 250 |
| `test_sv_counts_unequal_stress` | 160 ns | **PASS** | Extreme distribution [100,10,80,40,20] loads without error |
| `test_qspi_fifo_load` | 10420 ns | **PASS** | 256-word feature vector streams into FIFO without overflow |
| `test_qspi_backpressure` | 168000 ns | **PASS** | `qspi_ready` deasserts when FIFO fills; writes rejected before overflow |
| `test_default_gamma_fixed_point` | 140 ns | **PASS** | Default `gamma_reg` = 0.25 in Q6.10 (= 256 = 0x0100) |
| `test_full_pipeline_small_batch` | 71640 ns | **PASS** | Single-sample pipeline (2 SVs × 5 classes): `done` fires, no error |
| `test_kernel_output_range` | 71640 ns | **PASS** | All kernel outputs ∈ [0, 1] (RBF property holds for any input) |

**cocotb total: 9/9 PASS**

---

## Summary

| Suite | Tests | Passed | Failed |
|-------|-------|--------|--------|
| iverilog | 13 | 13 | 0 |
| cocotb | 9 | 9 | 0 |
| **Total** | **22** | **22** | **0** |
