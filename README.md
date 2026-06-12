# ECE410 — 5-Class Cardiac Arrhythmia Classifier: RBF-SVM ASIC

**Student:** Adam Handwerger · handwerg@pdx.edu  
**Course:** ECE410, Portland State University  
**Technology:** sky130A · OpenLane 2 v2.3.10  

---

## Latest Milestone → [project/m4/](project/m4/)

**m4** is the final submission milestone: Caravel wrapper hardening, full DV, and submission artifacts.

| Metric | Value |
|--------|-------|
| Accuracy (Q6.10) | **98.33%** (295/300) — 0 quantization flips vs float |
| SV allocation | [95, 95, 95, 120, 95] = 500 total (VT at Q6.10 optimum) |
| Clock | 40 MHz, 0 timing violations (TT/FF corners) |
| Active power | 55.25 mW → 0.869 mW avg at 80 bpm (LAT=3) |
| Core die | 2500 × 2500 µm, 15.0% utilization, 157,991 cells |
| DRC | 0 violations (KLayout) |
| Caravel DV | PASSED (job 92867) |

See [project/m4/README.md](project/m4/README.md) for the full milestone catalog, RTL source, testbenches, P&R reports, and Caravel submission artifacts.

---

## All Milestones

| Milestone | Directory | Description |
|-----------|-----------|-------------|
| m0 | [project/m0/](project/m0/) | Project proposal — algorithm design, SVM baseline |
| m1 | [project/m1/](project/m1/) | Architecture exploration — interface selection |
| m2 | [project/m2/](project/m2/) | RTL verification & synthesis — 19/19 tests PASS |
| m3 | [project/m3/](project/m3/) | Place-and-route — core GDS, 0 DRC, +3.96 ns WNS |
| **m4** | [**project/m4/**](project/m4/) | **Caravel wrapper, DV, submission ← latest** |

---

## Caravel Submission

Physical artifacts (GDS, LEF, GL netlist) live in the separate Caravel repo:  
[github.com/adamleehandwerger/caravel_svm_project](https://github.com/adamleehandwerger/caravel_svm_project)

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Attribution required for reuse.

---

*ECE410 · Portland State University · Adam Handwerger · 2026-06-11*
