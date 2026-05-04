# ECE410 Project — LUT Kernel SVM Compute Core

5-class cardiac arrhythmia detection (Normal / PVC / AFib / VT / SVT) using a
fixed-point RBF-SVM accelerator with a range-reduction LUT kernel (γ = 0.25,
256-dim multi-scale features, Q6.10 arithmetic).

---

## File Overview

| File | Purpose |
|------|---------|
| `svm_compute_core.sv` | Top-level RTL: FSM, FIFO, distance matrix, Horner engine |
| `svm_interfaces.sv` | SystemVerilog interface definitions (not needed for iverilog compile) |
| `tb_svm_classifier.sv` | Full-pipeline classification testbench (5 heartbeats, one per class) |
| `tb_svm_params.svh` | Auto-generated SV include: dual coefficients, intercepts, SV counts |
| `gen_tb_data.py` | Generates all `.hex` files and `tb_svm_params.svh` from MIT-BIH data |
| `hw_svm_comparison.py` | Standalone Python accuracy/comparison script (produces confusion matrices) |
| `sv_ram.hex` | SV feature RAM contents (generated) |
| `test_features.hex` | 5 test heartbeat feature vectors (generated) |
| `expected_kernels.hex` | Pre-computed reference kernel values (generated) |
| `test_labels.hex` | True class labels for the 5 test beats (generated) |
| `expected_preds.hex` | Expected OvO predictions (generated) |

---

## Dependencies

### Python 3.8+

```bash
pip install numpy scikit-learn wfdb matplotlib
```

| Package | Version tested | Notes |
|---------|---------------|-------|
| `numpy` | ≥ 1.21 | Fixed-point arithmetic |
| `scikit-learn` | ≥ 1.0 | SVC training and OvO classification |
| `wfdb` | ≥ 4.0 | Downloads MIT-BIH records directly from PhysioNet |
| `matplotlib` | ≥ 3.4 | Required by `hw_svm_comparison.py` only |

> **Note:** `wfdb` streams the MIT-BIH database from PhysioNet over the network
> on first run. An internet connection is required for `gen_tb_data.py`. Records
> are not cached locally — re-runs will re-download.

### Icarus Verilog (iverilog)

```bash
# macOS
brew install icarus-verilog

# Ubuntu / Debian
sudo apt install iverilog

# Fedora / RHEL
sudo dnf install iverilog
```

Verify installation:

```bash
iverilog -V
```

### GTKWave (optional — waveform viewer)

```bash
# macOS
brew install gtkwave

# Ubuntu / Debian
sudo apt install gtkwave
```

---

## Running the Testbench

### Step 1 — Generate testbench data

Run **once** before simulating. Trains the SVM on MIT-BIH data, computes
reference kernel values, and writes all `.hex` files and `tb_svm_params.svh`.

```bash
python3 gen_tb_data.py
```

Expected output (takes 1–3 minutes depending on network speed):

```
gen_tb_data.py — SVM Testbench Data Generator (LUT, γ=0.25)
[1/5] Building dataset (MIT-BIH + synthetic, multi-scale) ...
[2/5] Training SVM ...
[3/5] Computing reference kernels (Q6.10 LUT) ...
[4/5] Running OvO classification ...
[5/5] Writing data files ...
Done. All files written to <project dir>
```

### Step 2 — Compile

```bash
iverilog -g2012 -o tb_svm tb_svm_classifier.sv svm_compute_core.sv
```

### Step 3 — Run simulation

```bash
vvp tb_svm
```

The simulator prints per-heartbeat results to stdout and writes two output files:

| Output file | Contents |
|-------------|---------|
| `svm_classifier.log` | Per-heartbeat detail, kernel MAE, OvO predictions, final grade |
| `svm_waveform_hb0.vcd` | VCD waveform capture for heartbeat 0 (signal depth 2) |

### Step 4 — Check results

```bash
cat svm_classifier.log
```

Pass criterion: **≥ 4/5 heartbeats classified correctly AND the `error` flag
never asserted**.

### Step 5 — View waveform (optional)

```bash
gtkwave svm_waveform_hb0.vcd
```

---

## One-liner (Steps 2 + 3)

```bash
iverilog -g2012 -o tb_svm tb_svm_classifier.sv svm_compute_core.sv && vvp tb_svm
```

---

## Python-only Accuracy Check

To evaluate the full model accuracy on MIT-BIH without running RTL simulation:

```bash
python3 hw_svm_comparison.py
```

Produces a classification report, per-class accuracy breakdown, confusion
matrices (`confusion_matrices.png`), and a kernel error plot comparing the
Q6.10 LUT implementation against float64 reference values.

---

## Expected Accuracy

| Model | sklearn | HW Q6.10 | Notes |
|-------|---------|----------|-------|
| γ=0.25, LUT, causal RR | 98.67% | 98.33% | Current target |
| Kernel max \|err\| | — | 0.00055 | vs. float64, d²∈[0,60] |

---

## Testbench Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Clock | 50 MHz (20 ns) | `always #10 clk = ~clk` |
| Data format | Q6.10 | Scale = 1024 |
| γ | 0.25 → `0x0100` | Programmed via `param_write_en` |
| Feature dim | 256 | 128 beat + 64 mean + 64 RR |
| Test beats | 5 | One per class |
| Pass threshold | 4/5 (80%) | AND `error` flag never set |
| Kernel tolerance | ±2 LSB | For MAE reporting only |
| Watchdog timeout | 200 ms sim time | ~10M cycles at 50 MHz |
