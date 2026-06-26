# IHP SG13G2 Community Shuttle — Submission Checklist
## Project: 5-Class OVR RBF-SVM Cardiac Arrhythmia Classifier
**Design name:** `svm_top_ihp`  
**PDK:** IHP SG13G2 (130 nm BiCMOS)  
**Shuttle deadline:** August 23, 2026  
**GitHub submission repo:** https://github.com/adamleehandwerger/ihp-sg13g2-svm

---

## Physical Design

| Item | Status | Notes |
|------|--------|-------|
| Core harden (svm_compute_core) | ✅ DONE | Job 94594 — KLayout XOR 0 diffs, DRC 0 (2:03 runtime) |
| Top wrapper harden (svm_top_ihp) | ✅ DONE | Job 94595 — KLayout XOR 0 diffs, Magic DRC 0, KLayout DRC 0 (4:31 runtime) |
| Magic DRC: 0 violations | ✅ DONE | `drc.magic.rpt`: COUNT 0 (top + core) |
| KLayout DRC: 0 violations | ✅ DONE | `drc.klayout.lyrdb`: no items (top + core) |
| Setup timing met (typ 1.20V 25°C) | ✅ DONE | WNS = +12.93 ns |
| Hold timing met (typ 1.20V 25°C) | ✅ DONE | R2R WNS = +0.097 ns, 0 R2R violations |
| IR drop < 5% (VGND) | ✅ DONE | 0.57% drop (with vsrc pin locations) |
| Routing DRC: 0 violations | ✅ DONE | 0 after DRT convergence |
| Antenna violations: 0 | ✅ DONE | Diodes inserted, 0 remaining |
| GDS generated | ✅ DONE | `svm_top_ihp.gds` (171 MB, Jun 23 2026) |
| Abstract LEF generated | ✅ DONE | `svm_top_ihp.lef` |
| Gate-level netlist generated | ✅ DONE | `svm_top_ihp.v` |

## Design Specs

| Parameter | Value |
|-----------|-------|
| Die area | 2400 × 2400 µm |
| Core area | 2388 × 2366 µm |
| Technology | IHP SG13G2 130 nm |
| Supply voltage | 1.2 V |
| Clock | 40 MHz (25 ns period) |
| Classifier | 5-class OVR RBF-SVM (Normal/PVC/AFib/VT/SVT) |
| Support vectors | 500 total — [95,95,95,120,95] per class |
| Feature dimension | 256 (128 single-beat + 64 10-beat + 64 RR-interval) |
| RAM | Off-chip 1 MB async SRAM (IS62WV51216) |
| Configuration | SPI slave (CPOL=0, CPHA=0), nRF52840 MCU |
| Accuracy (ASIC Q6.10) | **98.33%** (295/300 PhysioNet test set) |
| Active power (est.) | ~24.5 mW (scaled from m5 at 1.2V vs 1.8V) |
| Avg power @ 80 bpm | ~0.52 mW |

## Submission Items

| Item | Status | Notes |
|------|--------|-------|
| GDS pushed to submission repo | ✅ DONE | Fresh Jun 23 GDS pushed to ihp-sg13g2-svm |
| LEF pushed to submission repo | ✅ DONE | |
| Gate-level netlist pushed | ✅ DONE | |
| DRC reports pushed | ✅ DONE | Magic + KLayout both 0 violations |
| Timing reports pushed | ✅ DONE | Post-PnR typ corner |
| README with project description | ✅ DONE | IHP__SVM5740 template structure |
| IHP submission issue opened | ✅ DONE | IHP-GmbH/Open-Silicon-MPW#40 (July-2026) |
| Repo restructured to IHP template | ✅ DONE | doc/, SVM5740-main/PlaceAndRoute/, rtl/, verification/ |

## Remaining Before Tape-Out

| Item | Priority | Notes |
|------|----------|-------|
| IHP response to issue #40 | 🔴 HIGH | Await IHP team review at IHP-GmbH/Open-Silicon-MPW#40 |
| LVS verification | 🟡 MED | ⚠️ Filler-cell mismatch only (job 94591): device count match, cell pins equivalent; fill/decap cells have VDD/VSS in SPICE but no pins in GL Verilog — standard PnR artifact, not functional |
| Fast-corner hold (1.65V/-40C) | 🟡 MED | ⚠️ WNS = -0.08 ns (1 path) — 1.65V is 37.5% above 1.2V nominal (op range 1.08–1.32V) |
| Update design_summary.md §m6 | 🟢 LOW | Document final harden metrics (jobs 94594/94595) |
| KLayout XOR (Magic vs. KLayout GDS) | ✅ DONE | Jobs 94594/94595 — Total XOR differences: 0 (all layers, both core + top) |

## Key File Locations

**GitHub:** https://github.com/adamleehandwerger/ihp-sg13g2-svm  
**ECE410 repo:** https://github.com/adamleehandwerger/ECE410 (`project/m6/`)  
**Orca artifacts:** `/scratch/funphin-openlane_svm/svm_m6_artifacts/`  
**Orca run dir:** `/scratch/funphin-openlane_svm/svm_m6/project/m6/synth/runs/top_harden/`

## Contacts

- **IHP Shuttle:** https://www.ihp-microelectronics.com/services/research-and-prototyping-services/mpw-prototyping/ihp-open-source-pdk  
- **ECE410 Instructor:** (for class submission)
- **Student:** Adam Handwerger — handwerg@pdx.edu
</content>
