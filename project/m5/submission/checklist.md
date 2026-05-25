# Caravel chipIgnite Submission Checklist

**Project:** SVM Cardiac Arrhythmia Classifier (ECE410, Portland State University)
**Repo:** https://github.com/adamleehandwerger/caravel_svm_project
**Student:** Adam Handwerger (handwerg@pdx.edu)

---

## Required Artifacts

| Item | File | Status |
|------|------|--------|
| Core GDS | `gds/svm_compute_core.gds` | ✅ 181 MB (Magic, OL2 job 91947) |
| Core LEF | `lef/svm_compute_core.lef` | ✅ 108 KB |
| Core GL netlist | `verilog/gl/svm_compute_core.v` | ✅ 13 MB |
| Wrapper GDS | `gds/user_project_wrapper.gds` | ⏳ Pending job 91948 |
| Wrapper LEF | `lef/user_project_wrapper.lef` | ⏳ Pending job 91948 |
| Wrapper GL netlist | `verilog/gl/user_project_wrapper.v` | ⏳ Pending job 91948 |
| RTL source | `verilog/rtl/svm_compute_core.sv` | ✅ Committed |
| Wrapper RTL | `verilog/rtl/user_project_wrapper.sv` | ✅ Committed |

## Timing & Physical Verification

| Check | Target | Result | Status |
|-------|--------|--------|--------|
| Core setup WNS (TT) | ≥ 0 ns | +7.923 ns | ✅ |
| Core hold WNS (TT) | ≥ 0 ns | +0.297 ns | ✅ |
| Core DRC violations | 0 | 0 | ✅ |
| Core power (avg, 80 bpm) | < 1 mW | 0.26 mW | ✅ |
| Wrapper setup WNS (TT) | ≥ 0 ns | TBD | ⏳ |
| Wrapper hold WNS (TT) | ≥ 0 ns | TBD | ⏳ |
| Wrapper DRC violations | 0 | TBD | ⏳ |
| mpw-precheck PASS | All gates | TBD | ⏳ |

## Functional Verification

| Test | Description | Status |
|------|-------------|--------|
| sklearn accuracy | 96.39% on MIT-BIH | ✅ (confusion_comparison.py) |
| Hardware accuracy | 0.00% gap vs. sklearn | ✅ (RTL simulation) |
| tb_top.sv | 5-heartbeat classification pipeline | ✅ |
| tb_error_codes.sv | All error codes + sticky latch + reset | ✅ |
| tb_backpressure.sv | FIFO backpressure handling | ✅ |
| tb_consecutive.sv | Back-to-back heartbeat processing | ✅ |
| tb_dist_boundary.sv | Accumulator saturation | ✅ |
| tb_dist_zero.sv | D=0 → kernel=1024 | ✅ |
| tb_gamma_zero.sv | γ=0 edge case | ✅ |
| tb_interface.sv | Port protocol compliance | ✅ |
| tb_min_sv.sv | 1 SV per class | ✅ |
| tb_multi_heartbeat.sv | num_samples=3 | ✅ |
| tb_param_write.sv | Runtime parameter write | ✅ |
| tb_power.sv | Clock-gate idle behavior | ✅ |
| tb_warmup.sv | Warmup state timing | ✅ |
| Caravel chip-level DV | RISC-V firmware + mprj_io pass | ⏳ Pending |

## Design Highlights

- **Accuracy:** 96.39% on MIT-BIH (5-class: Normal, PVC, AFib, VT, SVT)
- **Power:** ~0.26 mW average (14-day battery has 119-day headroom)
- **Feature vector:** 256-dim (128 single-beat + 64 10-beat + 64 100-beat context)
- **Fixed-point:** Q6.10, 0.00% quantization gap vs. sklearn float
- **SV RAM:** Off-chip via GPIO/LA (eliminates unavailable sky130 SRAM macro)
- **Clock:** 40 MHz (25 ns), TT corner clean with +7.9 ns margin
- **Cells:** 146,311 standard cells, 14.1% utilization in 2500×2500 µm

## Next Steps to Complete Submission

1. [ ] Wait for job 91948 to finish (wrapper hardening)
2. [ ] Copy wrapper GDS/LEF/GL to caravel repo
3. [ ] Commit and push (GDS via LFS or release asset)
4. [ ] Run `sbatch ~/ece410/precheck_run.sh`
5. [ ] Fix any precheck failures
6. [ ] Submit repo URL to Efabless chipIgnite portal
7. [ ] Submit to ECE410 class repo
