# Testbench Results — ECE410 SVM Compute Core (LUT Kernel)

RTL file: `svm_compute_core.sv`  
Simulator: Icarus Verilog 13.0 / cocotb 2.0.1  
Date: 2026-05-05

---

## RTL Fixes Applied

| # | Fix | Symptom |
|---|-----|---------|
| 1 | **kernel_valid hold register** — changed from 1-cycle pulse to set/clear latch (held until `kernel_ready` handshake) | FSM stalled permanently if `kernel_ready=0` during the single cycle `kernel_valid` was high |
| 2 | **gamma shadow register** (`gamma_latched`) — captured from `gamma_int` at `start`, used by Horner engine throughout the batch | Mid-compute `param_write_en` could corrupt in-flight kernel values |
| 3 | **ERR_GAMMA_ZERO (0x6)** — added error code that fires when `gamma_int == 0` while FSM is not idle | gamma=0 silently produced all-1.0 kernels (constant classifier), no error was raised |

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
| `tb_svm_classifier.sv` | 5/5 correct | **PASS** | Full 5-class cardiac arrhythmia classification (Normal/PVC/AFib/VT/SVT); kernel MAE < 0.003; no error flag |

**iverilog total: 10/10 PASS**

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
| iverilog | 10 | 10 | 0 |
| cocotb | 9 | 9 | 0 |
| **Total** | **19** | **19** | **0** |
