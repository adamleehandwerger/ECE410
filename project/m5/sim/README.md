# m5 Simulation Results

Final simulation for m5 compares the ASIC output against two reference implementations:
1. **sklearn (float64)** — gold-standard Python SVM
2. **Numba-optimized Python** — optimized fixed-point software reference (Q6.10)

The goal is to show that the hardware confusion matrix is indistinguishable from the
optimized software implementation, confirming the fixed-point design is correct.

## Files (pending)

- `confusion_comparison_m5.png` — 3-way confusion matrix: sklearn vs. Numba vs. ASIC
- `cosim_run.log` — cocotb simulation log
- `numba_vs_hardware.csv` — per-sample prediction comparison

## How to generate

After job 91948 completes and the GL netlist is available:

```bash
# From m4/tb/
python3 ../confusion_comparison.py   # generates confusion matrices
make cosim                           # cocotb GL-level simulation
```

The `confusion_comparison.py` script already produces sklearn vs. hardware matrices.
A Numba reference column will be added in m5 to complete the 3-way comparison.
