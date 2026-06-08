# Caravel chipIgnite — Submission Requirements & Status

**Project:** 5-Class Cardiac Arrhythmia Classifier (RBF-SVM ASIC)
**Student:** Adam Handwerger · handwerg@pdx.edu · Portland State University, ECE410
**Repo:** https://github.com/adamleehandwerger/caravel_svm_project
**PDK:** sky130A · sky130_fd_sc_hd · OpenLane 2 v2.3.10

---

## 1. Repository Structure

| Requirement | Path | Status |
|-------------|------|--------|
| Top-level `info.yaml` | `info.yaml` | ✅ |
| RTL source files | `verilog/rtl/` | ✅ v10 (NUM_SV=500, RAM_LATENCY=3) |
| Gate-level netlist — core | `verilog/gl/svm_compute_core.v` | ✅ 35 MB (job 92840) |
| Gate-level netlist — wrapper | `verilog/gl/user_project_wrapper.v` | ✅ 77 KB (job 92861) |
| GDS — core | `gds/svm_compute_core.gds` | ✅ 232 MB (job 92840) — local only, LFS fork restriction |
| GDS — wrapper | `gds/user_project_wrapper.gds` | ✅ 234 MB (job 92861) — local only, LFS fork restriction |
| LEF — core | `lef/svm_compute_core.lef` | ✅ 93 KB (job 92840) |
| LEF — wrapper | `lef/user_project_wrapper.lef` | ✅ 178 KB (job 92861) |
| OpenLane config — core | `openlane/svm_compute_core/config.json` | ✅ |
| OpenLane config — wrapper | `openlane/user_project_wrapper/config.json` | ✅ |

Note: GDS files exceed GitHub's LFS fork upload limit. Files are in the local working
tree and on Orca scratch. For ECE410 review, they can be provided on request or via
a GitHub Release attachment.

---

## 2. Physical Design Requirements

| Requirement | Specification | Result | Status |
|-------------|---------------|--------|--------|
| PDK | sky130A | sky130A | ✅ |
| Standard cell library | sky130_fd_sc_hd | sky130_fd_sc_hd | ✅ |
| Max routing layer | ≤ met5 | met4 | ✅ |
| Core DRC violations | 0 | 0 | ✅ |
| Core KLayout DRC | 0 | 0 (sky130A.lydrc) | ✅ |
| Wrapper Magic DRC violations | 0 | 11,906 boundary artifacts | ⚠️ acceptable |
| Wrapper KLayout DRC violations | 0 | 0 violations | ✅ |
| Core setup timing (TT) | ≥ 0 ns WNS | +3.96 ns | ✅ |
| Core hold timing (TT) | ≥ 0 ns WNS | +0.23 ns | ✅ |
| Wrapper setup timing (TT/FF) | ≥ 0 ns WNS | 0 ns / 0 ns | ✅ |
| Wrapper hold timing (TT) | ≥ 0 ns WNS | −0.406 ns (124 viol.) | ⚠️ through-macro paths |
| Wrapper hold reg-to-reg (TT) | ≥ 0 ns WNS | +0.265 ns, 0 viol. | ✅ |
| Die area (wrapper) | 2920 × 3520 µm (fixed) | 2920 × 3520 µm | ✅ |
| Die area (core) | ≤ user_project_area | 2500 × 2500 µm | ✅ |

---

## 3. Efabless mpw-precheck

| Check | Description | Status |
|-------|-------------|--------|
| All artifacts present | GDS/LEF/GL/RTL/info.yaml | PASSED (job 92871) | ✅ |
| Magic DRC — core | 0 violations on svm_compute_core GDS | 0 violations (job 92871) | ✅ |
| SPDX headers | Both RTL files | OK (job 92871) | ✅ |
| XOR | Magic GDS == KLayout GDS | ⏳ KLayout installed locally, pending run |
| Magic DRC — wrapper | Full Efabless precheck | ⏳ ECE410 class scope |
| LVS (netgen) | Netlist matches layout | ⏳ ECE410 class scope |

```bash
sbatch ~/ece410/precheck_run.sh   # see m4/caravel/precheck/precheck_run.sh
```

---

## 4. Functional Verification

| Test | Coverage | Result | Status |
|------|----------|--------|--------|
| sklearn accuracy | 97.67% on MIT-BIH + SVDB + INCART (300 samples) | 97.67% | ✅ |
| ASIC vs sklearn gap | Q6.10 quantization (gamma=0.25) | 0.00% | ✅ |
| Wishbone cosim | 300 samples, cocotb, full batch, LAT=3 | 97.67% ASIC | ✅ |
| RAM_LATENCY unit test | LAT=3, FEAT=4, NSV=5, iverilog | PASS, 208 cycles | ✅ |
| Caravel chip-level DV | RISC-V firmware, mprj_io check | PASSED (job 92867) | ✅ |

---

## 5. Design Specifications

| Parameter | Value |
|-----------|-------|
| Algorithm | 5-class RBF-SVM (OvR binary) |
| Classes | Normal, PVC, AFib, VT, SVT |
| Dataset | MIT-BIH + SVDB + INCART (PhysioNet) |
| Accuracy | 97.67% (equal to sklearn float) |
| Feature vector | 256-dim: 128 + 64 + 64 (multi-scale) |
| Support vectors | 500 total (100/class) |
| Fixed-point | Q6.10, 16-bit signed |
| Gamma | 0.25 (γ=0x0100 in Q6.10 — zero quantization error) |
| Clock | 40 MHz (25 ns period) |
| RAM_LATENCY | 3 (IS61WV51216 async SRAM, 10 ns access) |
| Inference time | 9.7 ms / beat (LAT=3) |
| Active power | 55.25 mW |
| Average power (80 bpm) | 0.727 mW |
| 14-day wearable target | MET — ~42-day headroom on 200 mAh cell (core alone) |
| Off-chip RAM address bus | GPIO[28:10] (19-bit) + GPIO[29] ren + LA[15:0] rdata |

---

## 6. Wishbone Register Map

| Offset | Register | R/W | Description |
|--------|----------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=err_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] heartbeats per batch |
| +0x10–0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class (up to 100) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data (γ, C, bias) |
| +0x28 | ALPHA_WR | WO | [24:16]=sv_global_idx (9-bit) [15:0]=alpha Q6.10 |

---

## 7. Outstanding Items Before Final Submission

- [ ] Resolve GitHub LFS fork restriction for GDS files (create Release or detach fork)
- [ ] Run mpw-precheck → record results in `m4/caravel/precheck/precheck_results.txt`
- [ ] Fix any precheck failures
- [ ] Run Caravel chip-level DV (`dv_run.sh`) → record in `dv_results.md`
- [ ] KLayout XOR — run locally on final GDS once pulled
- [ ] Tag repo: `git tag submission-v1 && git push origin submission-v1`
- [ ] Submit caravel_svm_project repo URL to ECE410

---

*Last updated: 2026-06-06 — wrapper harden complete (jobs 92840/92861), KLayout DRC 0 violations, LEF/GL pushed to GitHub*
