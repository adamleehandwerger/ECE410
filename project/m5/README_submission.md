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
| RTL source files | `verilog/rtl/` | ✅ |
| Gate-level netlist — core | `verilog/gl/svm_compute_core.v` | ✅ 13 MB |
| Gate-level netlist — wrapper | `verilog/gl/user_project_wrapper.v` | ⏳ job 91948 |
| GDS — core | `gds/svm_compute_core.gds` | ✅ 181 MB |
| GDS — wrapper | `gds/user_project_wrapper.gds` | ⏳ job 91948 |
| LEF — core | `lef/svm_compute_core.lef` | ✅ 108 KB |
| LEF — wrapper | `lef/user_project_wrapper.lef` | ⏳ job 91948 |
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
| Wrapper DRC violations | 0 | TBD | ⏳ |
| Core setup timing (TT) | ≥ 0 ns WNS | +7.923 ns | ✅ |
| Core hold timing (TT) | ≥ 0 ns WNS | +0.297 ns | ✅ |
| Wrapper setup timing (TT) | ≥ 0 ns WNS | TBD | ⏳ |
| Wrapper hold timing (TT) | ≥ 0 ns WNS | TBD | ⏳ |
| Power grid violations | 0 | 0 | ✅ |
| Floating nets | 0 | 0 | ✅ |
| Die area (wrapper) | 2920 × 3520 µm (fixed) | 2920 × 3520 µm | ✅ |
| Die area (core) | ≤ user_project_area | 2500 × 2500 µm | ✅ |

---

## 3. Efabless mpw-precheck

| Check | Description | Status |
|-------|-------------|--------|
| Manifest | `info.yaml` fields valid, all files present | ⏳ Pending wrapper GDS |
| Consistency | GL netlist hierarchy matches GDS | ⏳ Pending wrapper GDS |
| XOR | Magic GDS == KLayout GDS (no geometry diff) | ⏳ Pending wrapper GDS |
| DRC (Magic) | 0 DRC violations on full-chip GDS | ⏳ Pending wrapper GDS |
| LVS (netgen) | Netlist matches layout (no shorts/opens) | ⏳ Pending wrapper GDS |
| Antenna | No antenna violations | ⏳ Pending wrapper GDS |

Run precheck after job 91948 completes:
```bash
sbatch ~/ece410/precheck_run.sh   # see m5/precheck/precheck_run.sh
```

---

## 4. Functional Verification

| Test | Coverage | Result | Status |
|------|----------|--------|--------|
| sklearn accuracy | 96.39% on MIT-BIH 5-class | 96.39% | ✅ |
| Hardware vs. sklearn gap | Q6.10 quantization | 0.00% | ✅ |
| tb_top.sv | Full 5-heartbeat pipeline | PASS | ✅ |
| tb_error_codes.sv | All error codes, sticky latch, reset | PASS | ✅ |
| tb_backpressure.sv | FIFO backpressure | PASS | ✅ |
| tb_consecutive.sv | Back-to-back heartbeats | PASS | ✅ |
| tb_dist_boundary.sv | Accumulator saturation | PASS | ✅ |
| tb_dist_zero.sv | D=0 → K=1024 | PASS | ✅ |
| tb_gamma_zero.sv | γ=0 edge case | PASS | ✅ |
| tb_interface.sv | Port protocol compliance | PASS | ✅ |
| tb_min_sv.sv | 1 SV per class minimum | PASS | ✅ |
| tb_multi_heartbeat.sv | num_samples=3 batch | PASS | ✅ |
| tb_param_write.sv | Runtime param write | PASS | ✅ |
| tb_power.sv | Clock-gate idle | PASS | ✅ |
| tb_warmup.sv | Warmup state exit | PASS | ✅ |
| Caravel chip-level DV | RISC-V firmware, mprj_io=0xBB91 | TBD | ⏳ |

---

## 5. Design Specifications

| Parameter | Value | Status |
|-----------|-------|--------|
| Algorithm | 5-class RBF-SVM | ✅ |
| Classes | Normal, PVC, AFib, VT, SVT | ✅ |
| Dataset | MIT-BIH Arrhythmia Database | ✅ |
| Accuracy | 96.39% (equal to sklearn float) | ✅ |
| Feature vector | 256-dim: 128 + 64 + 64 (multi-scale) | ✅ |
| Support vectors | 250 (capped) | ✅ |
| Fixed-point | Q6.10, 16-bit signed | ✅ |
| Clock | 40 MHz (25 ns period) | ✅ |
| Active power | 66 mW | ✅ |
| Average power (80 bpm) | ~0.26 mW | ✅ |
| 14-day wearable target | ~0.26 mW → 119-day headroom | ✅ |
| SV RAM | Off-chip GPIO[24:10] + LA[15:0] | ✅ |
| Work RAM | On-chip 2 KB (2048 × 16-bit) | ✅ |
| Output | work_ram[0..N-1] class labels 0–4 | ✅ |
| Caravel Wishbone base | 0x30000000 | ✅ |

---

## 6. Wishbone Register Map

| Offset | Register | R/W | Description |
|--------|----------|-----|-------------|
| +0x00 | FIFO_DATA | WO | write 16-bit feature word to FIFO |
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn [3]=kern_ready |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class_out |
| +0x0C | NUM_SAMPLES | RW | [9:0] heartbeats per classification run |
| +0x10 | NUM_SV[0] | RW | [7:0] SVs for class 0 (Normal) |
| +0x14 | NUM_SV[1] | RW | [7:0] SVs for class 1 (PVC) |
| +0x18 | NUM_SV[2] | RW | [7:0] SVs for class 2 (AFib) |
| +0x1C | NUM_SV[3] | RW | [7:0] SVs for class 3 (VT) |
| +0x20 | NUM_SV[4] | RW | [7:0] SVs for class 4 (SVT) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data (γ, C, bias) |
| +0x38 | WORK_RD | WO | [10:0] address to latch from work_ram |
| +0x3C | STATUS2 | RO | [15:0] work_ram read data |

---

## 7. SVM Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0x0 | OK | No error |
| 0x1 | FIFO_OVERFLOW | Too many features written before start |
| 0x2 | BAD_NUM_SAMPLES | num_samples = 0 |
| 0x3 | BAD_NUM_SV | A class has 0 SVs |
| 0x4 | VBATT_LOW | vbatt_ok = 0 at start |
| 0x5 | VBATT_WARN | vbatt_warn asserted during compute |
| 0x8 | WARMING_UP | Heartbeat count < 100 (context window filling) |

---

## 8. Outstanding Items Before Final Submission

- [ ] job 91948 complete → copy wrapper GDS/LEF/GL → commit
- [ ] Run mpw-precheck → record results in `m5/precheck/precheck_results.txt`
- [ ] Fix any precheck failures
- [ ] Run Caravel chip-level DV (`dv_run.sh`) → record in `m4/tb/tb_results.md`
- [ ] Update `m5/pnr/` reports (area, timing, power, DRC) with job 91948 actuals
- [ ] Submit caravel_svm_project repo URL to ECE410

---

*Last updated: 2026-05-24 — wrapper hardening in progress (job 91948)*
