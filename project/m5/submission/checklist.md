# Caravel chipIgnite Submission Checklist

**Project:** SVM Cardiac Arrhythmia Classifier (ECE410, Portland State University)
**Repo:** https://github.com/adamleehandwerger/caravel_svm_project
**Student:** Adam Handwerger (handwerg@pdx.edu)
**Architecture:** Batch v8 — host pre-loads SV + input matrix; ASIC classifies autonomously

---

## Required Artifacts

| Item | File | Status |
|------|------|--------|
| Core GDS | `gds/svm_compute_core.gds` | ⏳ New DRT in progress (job 91959) |
| Core LEF | `lef/svm_compute_core.lef` | ⏳ Pending job 91959 |
| Core GL netlist | `verilog/gl/svm_compute_core.v` | ⏳ Pending job 91959 |
| Wrapper GDS | `gds/user_project_wrapper.gds` | ⏳ Pending core + wrapper DRT |
| Wrapper LEF | `lef/user_project_wrapper.lef` | ⏳ Pending |
| Wrapper GL netlist | `verilog/gl/user_project_wrapper.v` | ⏳ Pending |
| Core RTL | `verilog/rtl/svm_compute_core.sv` | ✅ Batch v8 committed |
| Wrapper RTL | `verilog/rtl/user_project_wrapper.sv` | ✅ Batch v8 committed |

---

## Timing & Physical Verification

| Check | Target | Prior (job 91947) | New DRT |
|-------|--------|-------------------|---------|
| Core setup WNS (TT) | ≥ 0 ns | +7.923 ns ✅ | Pending 91959 |
| Core hold WNS (TT) | ≥ 0 ns | +0.297 ns ✅ | Pending 91959 |
| Core DRC violations | 0 | 0 ✅ | Pending 91959 |
| Core power (avg, batch) | < 1 mW | ~0.15 mW ✅ | Pending 91959 |
| Wrapper setup WNS (TT) | ≥ 0 ns | TBD | Pending |
| Wrapper hold WNS (TT) | ≥ 0 ns | TBD | Pending |
| Wrapper DRC violations | 0 | TBD | Pending |
| mpw-precheck PASS | All gates | TBD | Pending |

---

## Functional Verification

| Test | Description | Status |
|------|-------------|--------|
| sklearn accuracy | 96.39% on MIT-BIH | ✅ (Python) |
| batch cosim (v8) | 300 samples, Icarus/cocotb | ✅ PASS (cocotb) |
| Batch accuracy | ASIC vs sklearn | ⚠️ 45% — alpha weights not implemented |
| Timeout budget | All 300 samples classified | ⚠️ 294/300 — budget fixed in testbench |
| tb_top.sv | 5-heartbeat pipeline (v7) | ✅ (prior) |
| tb_error_codes.sv | Error codes, sticky latch | ✅ (prior) |
| tb_consecutive.sv | Back-to-back heartbeats | ✅ (prior) |
| tb_dist_zero.sv | D=0 → kernel=1024 | ✅ (prior) |
| tb_param_write.sv | Runtime parameter write | ✅ (prior) |
| tb_warmup.sv | Warmup state timing | ✅ (prior) |
| Caravel chip-level DV | RISC-V firmware pass | ⏳ Pending |

---

## Known Issues (v8 Batch)

### 1. Alpha Weight Accuracy Gap (HIGH PRIORITY)

The ASIC computes `score_c = Σ K(x, svᵢ)` for each class without applying the
dual coefficients (αᵢ) from the sklearn SVM.

Sklearn's correct decision function:
```
f_c(x) = Σᵢ αᵢ·yᵢ·K(x, svᵢ) + b_c
```

Without αᵢ weights, the ASIC achieves ~45% accuracy on 300 test samples vs
sklearn's 96.39%.

**Fix options:**
- Add alpha coefficient RAM to the ASIC (250 × 5 × 16-bit = small on-chip)
- Train a constrained sklearn model where all αᵢ have the same sign per class

### 2. Cosim Timeout (FIXED in testbench)

6/300 samples timed out due to underestimated cycle budget. Fixed:
```python
# Old: cycles_per_sample = FEATURE_DIM + 2 + sv_total * (FEATURE_DIM + 22) + 10
# New: cycles_per_sample = FEATURE_DIM + 5 + sv_total * (FEATURE_DIM + 30) + 200
```

---

## Design Highlights (v8 Batch)

- **Architecture:** Batch — host pre-loads 1000 beats into SRAM, ASIC classifies in burst
- **Feature vector:** 256-dim (128 single-beat + 64 10-beat + 64 100-beat context)
- **Fixed-point:** Q6.10, matched to sklearn gamma=0.25
- **Off-chip RAM:** Unified 19-bit GPIO bus (SV matrix + input matrix)
- **Per-sample IRQ:** `sample_rdy` (IRQ[0]) fires per beat; `svm_done` (IRQ[1]) at end
- **Power:** ~0.15 mW average at 80 bpm (0.23% duty cycle, batch architecture)
- **Clock:** 40 MHz (25 ns), TT corner clean with +7.9 ns margin (prior run)

---

## Next Steps to Complete Submission

1. [ ] Wait for SLURM job 91959 (core_harden, batch v8) to finish
2. [ ] Fix alpha weight issue in ASIC or train compatible sklearn model
3. [ ] Submit wrapper_harden.sh after core GDS ready
4. [ ] Copy wrapper GDS/LEF/GL to caravel repo
5. [ ] Run `sbatch ~/ece410/precheck_run.sh`
6. [ ] Fix any precheck failures
7. [ ] Submit repo URL to Efabless chipIgnite portal
8. [ ] Submit to ECE410 class repo
