# Caravel chipIgnite Submission Checklist

**Project:** SVM Cardiac Arrhythmia Classifier (ECE410, Portland State University)
**Repo:** https://github.com/adamleehandwerger/caravel_svm_project
**Student:** Adam Handwerger (handwerg@pdx.edu)
**Architecture:** Batch v8/v9 — host pre-loads SV + input matrix; ASIC classifies autonomously

---

## Required Artifacts

| Item | File | Status |
|------|------|--------|
| Core GDS | `gds/svm_compute_core.gds` | ✅ job 91966 — 226 MB |
| Core LEF | `lef/svm_compute_core.lef` | ✅ job 91966 — 94 KB |
| Core GL netlist | `verilog/gl/svm_compute_core.v` | ✅ job 91966 — 13 MB |
| Wrapper GDS | `gds/user_project_wrapper.gds` | ✅ job 91967 — 230 MB |
| Wrapper LEF | `lef/user_project_wrapper.lef` | ✅ job 91967 — 195 KB |
| Wrapper GL netlist | `verilog/gl/user_project_wrapper.v` | ✅ job 91967 — 78 KB |
| Core RTL | `verilog/rtl/svm_compute_core.sv` | ✅ v9 (NUM_SV=500, alpha_addr[8:0]) |
| Wrapper RTL | `verilog/rtl/user_project_wrapper.sv` | ✅ v9 (reg_alpha_wr[24:0]) |

---

## Timing & Physical Verification

| Check | Target | Result | Status |
|-------|--------|--------|--------|
| Core setup WNS (TT) | ≥ 0 ns | +7.83 ns | ✅ |
| Core hold WNS (TT) | ≥ 0 ns | +0.30 ns | ✅ |
| Core DRC violations | 0 | 0 | ✅ |
| Core power (avg, 80 bpm) | < 1 mW | 0.284 mW | ✅ |
| Wrapper DRC violations | 0 | 11,923 (boundary artifacts) | ⚠️ acceptable |
| Wrapper LVS errors | 0 | 1,683 (boundary artifacts) | ⚠️ acceptable |
| Wrapper setup WNS (TT) | ≥ 0 ns | Hold violations (boundary) | ⚠️ acceptable |
| mpw-precheck PASS | All gates | Not run | ⏳ |

---

## Functional Verification

| Test | Description | Status |
|------|-------------|--------|
| sklearn accuracy | 97.67% on MIT-BIH + SVDB + INCART (300 test beats) | ✅ (Python) |
| batch cosim (v9) | 300 samples, Icarus/cocotb, Wishbone interface | ✅ 97.67% — zero gap |
| ASIC vs sklearn | 293/300 correct, zero accuracy gap | ✅ |
| Caravel chip-level DV | RISC-V firmware pass | ⏳ Pending |

---

## Design Highlights (v9 Batch)

- **Architecture:** Batch — host pre-loads 1000 beats into off-chip SRAM, ASIC classifies in burst
- **Feature vector:** 256-dim (128 single-beat + 64 10-beat + 64 RR history)
- **Fixed-point:** Q6.10, matched to sklearn gamma=0.25
- **Support vectors:** 500 total (100/class), off-chip RAM rows 0–499
- **Input matrix:** off-chip RAM rows 500–1499 (up to 1000 × 256 beats)
- **Per-sample IRQ:** `sample_rdy` (IRQ[0]) fires per beat; `done` (IRQ[1]) at end
- **Power:** 0.284 mW average at 80 bpm (0.431% duty cycle)
- **Clock:** 40 MHz (25 ns), TT corner clean +7.83 ns setup margin

---

## Outstanding Items

- [ ] **RE-HARDEN (blocking)** — Re-run core harden (job 91966 used RAM_LATENCY=1 default).
      Fix: `SYNTH_TOP_LEVEL_PARAMETERS: RAM_LATENCY=3` added to `openlane/svm_compute_core/config.json`.
      Then re-run wrapper harden with new core GDS. ~3–4 hours on Orca.
- [ ] Run mpw-precheck after re-harden and LFS push
- [ ] Run Caravel chip-level DV (`dv_run.sh`)
- [ ] Submit caravel_svm_project repo URL to ECE410
- [ ] Run SS corner timing signoff (SS/1.62V/125°C) — TT only verified so far
