# Caravel Submission Checklist

**Project:** SVM Cardiac Arrhythmia Classifier (ECE410, Portland State University)
**Repo:** https://github.com/adamleehandwerger/caravel_svm_project
**Student:** Adam Handwerger (handwerg@pdx.edu)
**Architecture:** Batch v9 — host pre-loads SV + input matrix; ASIC classifies autonomously

---

## Required Artifacts

| Item | File | Size | Job | Status |
|------|------|------|-----|--------|
| Core GDS | `gds/svm_compute_core.gds` | 226 MB | 92840 | ✅ |
| Core LEF | `lef/svm_compute_core.lef` | 94 KB | 92840 | ✅ |
| Core GL netlist | `verilog/gl/svm_compute_core.v` | 13 MB | 92840 | ✅ |
| Wrapper GDS | `gds/user_project_wrapper.gds` | 235 MB | 92861 | ✅ |
| Wrapper LEF | `lef/user_project_wrapper.lef` | 178 KB | 92861 | ✅ |
| Wrapper GL netlist | `verilog/gl/user_project_wrapper.v` | 78 KB | 92861 | ✅ |
| Core RTL | `verilog/rtl/svm_compute_core.sv` | — | v10 | ✅ |
| Wrapper RTL | `verilog/rtl/user_project_wrapper.sv` | — | v10 | ✅ |

---

## Timing & Physical Verification

| Check | Target | Result | Status |
|-------|--------|--------|--------|
| Core setup WNS (TT) | ≥ 0 ns | +3.96 ns | ✅ |
| Core setup WNS (FF) | ≥ 0 ns | +11.24 ns | ✅ |
| Core setup WNS (SS) | ≥ 0 ns | −14.56 ns | ⚠️ 100°C/1.60V — see tapeout items |
| Core hold WNS (TT) | ≥ 0 ns | +0.23 ns | ✅ |
| Core DRC violations | 0 | 0 | ✅ |
| Core LVS | PASS | PASS | ✅ |
| Core antenna violations | 0 | 554 net / 808 pin | ⚠️ advisory (class); blocking (tapeout) |
| Core power (avg, 80 bpm) | < 1 mW | 0.727 mW (LAT=3) | ✅ |
| Wrapper setup WNS (TT) | ≥ 0 ns | PASS (checker clean) | ✅ |
| Wrapper hold WNS (TT) | ≥ 0 ns | Violations found | ⚠️ see tapeout items |
| Wrapper DRC violations | 0 | 11,906 boundary artifacts (Magic) | ⚠️ acceptable for class |
| Wrapper KLayout DRC | 0 | 0 violations (job 91967 GDS) | ✅ re-run on 92861 GDS |
| Wrapper LVS errors | 0 | 1,698 boundary artifacts | ⚠️ acceptable for class |
| mpw-precheck (custom) | PASS | PASSED (job 92871) — 0 DRC, SPDX OK | ✅ |

---

## Functional Verification

| Test | Description | Status |
|------|-------------|--------|
| sklearn accuracy | 97.67% on MIT-BIH + SVDB + INCART (300 test beats) | ✅ |
| Batch cosim (v9) | 300 samples, cocotb Wishbone, RAM_LATENCY=3 | ✅ 97.67% — zero gap |
| ASIC vs sklearn | 293/300 correct, zero accuracy gap | ✅ |
| Caravel chip-level DV | RISC-V firmware pass | PASSED (job 92867) | ✅ |

---

## ECE410 Submission Checklist

- [x] Core harden complete (job 92840, v10, RAM_LATENCY=3, 0 DRC)
- [x] Wrapper harden complete (job 92861, v10, GDS/LEF/GL in caravel repo)
- [x] Run precheck (`precheck/precheck_run.sh` on Orca) ✅ job 92871 — 0 DRC, SPDX OK, all files present
- [x] Run Caravel chip-level DV (`dv_run.sh`) ✅ job 92867 — "Monitor: SVM WB Test (RTL) Passed"
- [x] Push final LEF/GL to GitHub (commit 8008ee5); GDS attached as Release assets (v3.10-hardened)
- [x] Tag repo: `v3.10-hardened` — GitHub Release with GDS/LEF/GL artifacts
- [ ] Submit caravel_svm_project repo URL to ECE410

---

## Tapeout Requirements (prototype)

Required before submitting to an Efabless Caravel shuttle:

- [ ] **Wrapper hold violations** — `Checker.HoldViolations` reports hold violations at
      nom_tt_025C_1v80. The svm_compute_core hold is clean (+0.23 ns); wrapper
      violations likely originate in Wishbone controller registers synthesized without
      clock tree (`RUN_CTS=0`). Fix: enable `RUN_CTS=1`, re-run wrapper harden, verify
      hold WNS ≥ 0 in all corners. **Hold failures cause functional errors on silicon.**

- [ ] **IR drop analysis** — `OpenROAD.IRDropReport` is skipped (PSM-0069). Add
      `VSRC_LOC_FILES` pointing to vccd1/vssd1 supply entry points on the Caravel die
      boundary. Re-enable IRDropReport and verify < 5% voltage droop under worst-case
      switching activity. Caravel reference repo provides example VSRC files.

- [ ] **Antenna violations** — svm_compute_core has 554 net / 808 pin antenna
      violations. Re-harden core with `GRT_REPAIR_ANTENNAS=1` and
      `RUN_FILL_INSERTION=1` to insert diodes and tie-offs. Verify Magic antenna
      DRC = 0 before submitting.

- [x] **KLayout DRC** — 0 violations on job 91967 wrapper GDS (run 2026-06-06,
      KLayout 0.30.9, sky130A.lydrc, 24 min, report at `~/Desktop/wrapper_drc.xml`).
      Re-run on final job 92861 GDS once pulled from Orca to confirm clean.
      **KLayout XOR** still pending (requires two GDS inputs; run after final GDS pulled).

- [ ] **Wrapper DRC/LVS boundary artifacts** — 11,906 Magic DRC and 1,698 LVS errors
      at macro/wrapper boundary. Confirm with Efabless that these are expected for a
      hardened macro in the fixed DEF template, or adjust macro placement and power
      ring overlap rules to resolve.

- [ ] **Power BTERM workaround** — vccd2/vdda1/vdda2/vssa1/vssa2/vssd2 BTERMs are
      deleted from the routing database (DRT-0302 workaround). Verify with Efabless
      that this is acceptable for the Caravel SoC power delivery network.

- [ ] **SS corner timing** — WNS = −14.56 ns at nom_ss_100C_1v60 (163 violations).
      Document operating envelope. If prototype must operate at elevated temperature
      or low voltage, re-harden with tighter constraints or target a faster SCL.
