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
| Core harden (svm_compute_core) | ✅ DONE | Job 93441 — 0 DRT violations |
| Top wrapper harden (svm_top_ihp) | ✅ DONE | Job 93553 — 0 DRC violations |
| Magic DRC: 0 violations | ✅ DONE | `drc.magic.rpt`: COUNT 0 |
| KLayout DRC: 0 violations | ✅ DONE | `drc.klayout.lyrdb`: no items |
| Setup timing met (typ 1.20V 25°C) | ✅ DONE | WNS = +12.93 ns |
| Hold timing met (typ 1.20V 25°C) | ✅ DONE | WNS = +0.35 ns |
| IR drop < 5% (VGND) | ✅ DONE | 0.03% drop |
| Routing DRC: 0 violations | ✅ DONE | 0 after 2 iterations |
| Antenna violations: 0 | ✅ DONE | 4 diodes inserted, 0 remaining |
| GDS generated | ✅ DONE | `svm_top_ihp.gds` (171 MB raw / 36 MB gz) |
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
| Active power | 55.25 mW |
| Avg power @ 80 bpm | 0.869 mW |

## Submission Items

| Item | Status | Notes |
|------|--------|-------|
| GDS pushed to submission repo | ✅ DONE | Gzip-compressed, ≤ 100 MB/file |
| LEF pushed to submission repo | ✅ DONE | |
| Gate-level netlist pushed | ✅ DONE | |
| DRC reports pushed | ✅ DONE | Magic + KLayout both 0 violations |
| Timing reports pushed | ✅ DONE | Post-PnR typ corner |
| README with project description | ✅ DONE | |

## Remaining Before Tape-Out

| Item | Priority | Notes |
|------|----------|-------|
| Register project with IHP shuttle | 🔴 HIGH | Contact IHP team at ihp-go.de before Aug 23 |
| Provide VSRC_LOC_FILES for accurate IR drop | 🟡 MED | Current IR drop advisory only (no supply pins specified) |
| LVS verification | 🟡 MED | Not run — Magic DRC only; LVS requires SPICE extraction |
| Confirm PDN macro power connection | 🟡 MED | u_svm VPWR/VGND connected via PDN_MACRO_CONNECTIONS; verify with LVS |
| Fast-corner hold timing (-0.127 ns pre-PnR) | 🟢 LOW | Post-PnR typ corner is clean; pre-PnR fast corner is pessimistic |
| Update design_summary.md §m6 | 🟢 LOW | Document final harden metrics |
| KLayout XOR (Magic vs. KLayout GDS) | 🟢 LOW | Set RUN_KLAYOUT_XOR: 1 and re-run signoff |

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
