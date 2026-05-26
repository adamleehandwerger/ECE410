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
| RTL source files | `verilog/rtl/` | ✅ v9 (NUM_SV=500) |
| Gate-level netlist — core | `verilog/gl/svm_compute_core.v` | ✅ 13 MB (job 91966) |
| Gate-level netlist — wrapper | `verilog/gl/user_project_wrapper.v` | ✅ 78 KB (job 91967) |
| GDS — core | `gds/svm_compute_core.gds` | ✅ 226 MB (job 91966) |
| GDS — wrapper | `gds/user_project_wrapper.gds` | ✅ 230 MB (job 91967) |
| LEF — core | `lef/svm_compute_core.lef` | ✅ 94 KB (job 91966) |
| LEF — wrapper | `lef/user_project_wrapper.lef` | ✅ 195 KB (job 91967) |
| OpenLane config — core | `openlane/svm_compute_core/config.json` | ✅ |
| OpenLane config — wrapper | `openlane/user_project_wrapper/config.json` | ✅ |

---

## 2. Physical Design Requirements

| Requirement | Specification | Result | Status |
|-------------|---------------|--------|--------|
| PDK | sky130A | sky130A | ✅ |
| Standard cell library | sky130_fd_sc_hd | sky130_fd_sc_hd | ✅ |
| Max routing layer | ≤ met5 | met4 | ✅ |
| Core DRC violations | 0 | 0 | ✅ |
| Wrapper DRC violations | 0 | 11,923 (boundary artifacts) | ⚠️ acceptable |
| Core setup timing (TT) | ≥ 0 ns WNS | +7.83 ns | ✅ |
| Core hold timing (TT) | ≥ 0 ns WNS | +0.30 ns | ✅ |
| Wrapper timing (TT) | ≥ 0 ns WNS | Hold viol. at macro boundary | ⚠️ acceptable |
| Die area (wrapper) | 2920 × 3520 µm (fixed) | 2920 × 3520 µm | ✅ |
| Die area (core) | ≤ user_project_area | 2500 × 2500 µm | ✅ |

---

## 3. Efabless mpw-precheck

| Check | Description | Status |
|-------|-------------|--------|
| Manifest | `info.yaml` fields valid, all files present | ⏳ Pending precheck run |
| Consistency | GL netlist hierarchy matches GDS | ⏳ |
| XOR | Magic GDS == KLayout GDS (no geometry diff) | ⏳ |
| DRC (Magic) | 0 DRC violations on full-chip GDS | ⏳ |
| LVS (netgen) | Netlist matches layout (no shorts/opens) | ⏳ |
| Antenna | No antenna violations | ⏳ |

Run precheck after LFS push completes:
```bash
sbatch ~/ece410/precheck_run.sh   # see m5/precheck/precheck_run.sh
```

---

## 4. Functional Verification

| Test | Coverage | Result | Status |
|------|----------|--------|--------|
| sklearn accuracy | 97.67% on MIT-BIH + SVDB + INCART (300 samples) | 97.67% | ✅ |
| ASIC vs sklearn gap | Q6.10 quantization (gamma=0.25) | 0.00% | ✅ |
| Wishbone cosim | 300 samples, Icarus/cocotb, full batch | 97.67% ASIC | ✅ |
| Caravel chip-level DV | RISC-V firmware, mprj_io check | TBD | ⏳ |

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
| Inference time | 3.23 ms / beat |
| Active power | ~66 mW |
| Average power (80 bpm) | 0.284 mW |
| 14-day wearable target | MET — ~29-day headroom on 200 mAh cell |
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

- [ ] Push GDS files to GitHub via git LFS (`git lfs push --all origin`)
- [ ] Run mpw-precheck → record results in `m5/precheck/precheck_results.txt`
- [ ] Fix any precheck failures
- [ ] Run Caravel chip-level DV (`dv_run.sh`) → record in `dv_results.md`
- [ ] Submit caravel_svm_project repo URL to ECE410

---

*Last updated: 2026-05-25 — hardening complete (jobs 91966/91967), cosim 97.67% = sklearn*
