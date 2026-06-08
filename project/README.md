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
| Accuracy | **97.67%** (293/300) — zero gap vs sklearn float |
| Clock | 40 MHz (25 ns), 0 timing violations |
| Active power | 66 mW |
| Avg power (80 bpm duty-cycled) | **0.284 mW** |
| Battery life — SVM core | ~108 days (200 mAh @ 3.7V) |
| Battery life — full system | ~29.6 days |
| Die area (core) | 2500 × 2500 µm, ~14% utilization |
| DRC violations | **0** |

---

## Repository Structure

```
project/
├── m0/   ← Project proposal: algorithm design, scope, SVM baseline
├── m1/   ← Architecture exploration and interface selection
├── m2/   ← RTL verification & synthesis (19/19 tests PASS)
├── m3/   ← Place-and-route: core GDS, 0 DRC, +7.83 ns WNS
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

- Post-route GDS: `svm_compute_core.gds` (232 MB, job 92840)
- Setup WNS: +3.96 ns @ TT 25°C 1.8V — 0 violations
- Active power: 55.25 mW (post-route STA)
- Full unit + integration testbench suite: `tb/` (22 tests, 22 PASS)
- Roofline analysis: `bench/roofline_final.png`

### m4 — Caravel Wrapper & Submission
[`project/m4/`](project/m4/)

- `rt1/compute_core.sv` — v10 RTL (NUM_SV=500, RAM_LATENCY=3)
- `rt1/top.sv` — `user_project_wrapper` with Wishbone register map
- Wishbone cosim: 97.67% accuracy, 300 MIT-BIH samples — `sim/`
- RAM_LATENCY unit test (LAT=3, 208 cycles/beat) — `tb/`
- Caravel chip-level DV: PASSED (job 92867) — `sim/final_run.log`
- Caravel submission artifacts (GDS, LEF, GL netlist) — `caravel/`
- **Design justification report** — `report/design_justification.pdf`
- **Figures** — `report/figures/`

---

## Key Design Decisions

- **Interface:** SPI → QSPI → Wishbone (Caravel native bus)
- **SV RAM:** moved off-chip to eliminate 256 KB SRAM macro from die
- **Precision:** Q6.10 fixed-point — minimum width with zero accuracy gap vs float
- **Kernel:** Horner LUT for RBF exp evaluation (range-reduces to avoid FP16 clamping)
- **Batch architecture:** MCU pre-loads SRAM; ASIC classifies autonomously

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

*ECE410 · Portland State University · Adam Handwerger · 2026-06-06 — v10 final (jobs 92840/92861)*
