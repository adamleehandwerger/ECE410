# ECE410 — 5-Class Cardiac Arrhythmia Classifier: RBF-SVM ASIC

**Student:** Adam Handwerger · handwerg@pdx.edu  
**Course:** ECE410, Portland State University  
**Technology:** sky130A · OpenLane 2 v2.3.10  
**Caravel repo:** [github.com/adamleehandwerger/caravel_svm_project](https://github.com/adamleehandwerger/caravel_svm_project)

---

## Project Summary

A custom ASIC implementation of a 5-class RBF-SVM cardiac arrhythmia classifier targeting
wearable deployment on sky130A. The classifier distinguishes Normal, PVC, AFib, VT, and SVT
rhythm classes from a 256-dimensional multi-scale ECG feature vector derived from the MIT-BIH
Arrhythmia Database.

| Metric | Value |
|--------|-------|
| Accuracy (Q6.10) | **98.33%** (295/300) — 0 quantization flips vs float |
| SV allocation | [95, 95, 95, 120, 95] = 500 total (VT at Q6.10 optimum) |
| Clock | 40 MHz (25 ns), 0 timing violations (TT/FF corners) |
| Active power | 55.25 mW (post-route STA) |
| Avg power (80 bpm, LAT=3) | **0.869 mW** |
| Battery life — SVM core | ~35.5 days (200 mAh @ 3.7V) |
| Battery life — full system | ~18.9 days |
| Die area (core) | 2500 × 2500 µm, 15.0% utilization, 157,991 cells |
| DRC violations | **0** |

---

## Repository Structure

```
project/
├── m0/   ← Project proposal: algorithm design, scope, SVM baseline
├── m1/   ← Architecture exploration and interface selection
├── m2/   ← RTL verification & synthesis (19/19 tests PASS)
├── m3/   ← Place-and-route: core GDS, 0 DRC, +3.96 ns WNS
└── m4/   ← Wrapper hardening, RAM_LATENCY, Caravel submission
```

---

## Milestone Deliverables

### m0 — Project Proposal
[`project/m0/`](project/m0/)

- Algorithm design: `Batch SVM Algorithm.pdf`
- System diagram, interface and storage trade studies

### m1 — Architecture Exploration
[`project/m1/`](project/m1/)

- Interface selection, data storage and transmission analysis

### m2 — RTL Verification & Synthesis
[`project/m2/`](project/m2/)

- Scope assessment: `scope_assessment.md`
- RTL source: `rt1/compute_core.sv`, `rt1/top.sv`, `rt1/interface.sv`
- 19/19 testbenches PASS (iverilog + cocotb)
- OpenLane synthesis: area, timing, power reports in `synth/`
- Confusion matrix and benchmark in `sim/` and `bench/`

### m3 — Place and Route
[`project/m3/`](project/m3/)

- Post-route GDS: `svm_compute_core.gds` (226 MB, job 91966)
- Setup WNS: +3.96 ns @ TT 25°C 1.8V — 0 violations
- Active power: 66 mW (post-route STA)
- Full unit + integration testbench suite: `tb/` (22 tests, 22 PASS)
- Roofline analysis: `bench/roofline_final.png`

### m4 — Caravel Wrapper & Submission
[`project/m4/`](project/m4/)

- `rt1/compute_core.sv` — v10 RTL (NUM_SV=500, RAM_LATENCY=3)
- `rt1/top.sv` — `user_project_wrapper` with Wishbone register map
- Wishbone cosim: 98.33% accuracy, 300 MIT-BIH+SVDB+INCART samples — `sim/`
- RAM_LATENCY unit test (LAT=3, 208 cycles/beat) — `tb/`
- Caravel chip-level DV: PASSED (job 92867) — `sim/final_run.log`
- Caravel submission artifacts (GDS, LEF, GL netlist) — `caravel/`
- **SV count sweep** (`sim/sv_sweep.py`, `sv_sweep.png`) — identifies Q6.10 optimum at N=120/class
- **Confusion matrices** (`sim/confusion_comparison_m5.py`, `confusion_comparison_m4.png`) — float vs Q6.10
- **Design summary** — `design_summary.md/pdf` (full analysis, Appendix A–C)
- **Study sheet** — `../study_sheet.tex/pdf` (4-page LaTeX reference)
- **Design justification report** — `report/design_justification.pdf`

---

## Key Design Decisions

- **Interface:** SPI → QSPI → Wishbone (Caravel native bus)
- **SV RAM:** moved off-chip to eliminate 256 KB SRAM macro from die
- **Precision:** Q6.10 fixed-point — minimum width with zero quantization gap vs float
- **SV allocation:** [95,95,95,120,95] — VT at Q6.10 per-class optimum; 0 flips
- **Kernel:** Horner LUT for RBF exp evaluation (range-reduces to avoid overflow)
- **Batch architecture:** MCU pre-loads SRAM; ASIC classifies autonomously
- **Next iteration (v11):** 600 SVs (alpha\_table[600], 10-bit addr) → 98.67% target

---

## Caravel Submission

Hardened GDS and all submission artifacts are in
[github.com/adamleehandwerger/caravel_svm_project](https://github.com/adamleehandwerger/caravel_svm_project).

| Artifact | Job | Size |
|----------|-----|------|
| `svm_compute_core.gds` | 92840 | 232 MB |
| `user_project_wrapper.gds` | 92861 | 234 MB |
| `svm_compute_core.lef` | 92840 | 93 KB |
| `user_project_wrapper.lef` | 92861 | 178 KB |

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Attribution required for reuse.

---

*ECE410 · Portland State University · Adam Handwerger · 2026-06-11 — v10 final (jobs 92840/92861)*
