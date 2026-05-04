# ECE410 Project — Updated SVM Compute Core

5-class cardiac arrhythmia detection (Normal / PVC / AFib / VT / SVT) using a
fixed-point RBF-SVM accelerator. γ = 0.01, single-beat 256-dim features, Q6.10
arithmetic with a single-stage Horner approximation of exp(−γD).

> **Note:** For the improved γ = 0.25 range-reduction LUT version (98.33% HW
> accuracy) see `ECE410_project_LUT/`.

---

## File Overview

| File | Purpose |
|------|---------|
| `svm_compute_core.sv` | Top-level RTL: FSM, FIFO, distance matrix, Horner engine |
| `svm_interfaces.sv` | SystemVerilog interface definitions |
| `tb_svm_classifier.sv` | Full-pipeline classification testbench (5 heartbeats, one per class) |
| `tb_svm_compute_core.sv` | Unit testbench for the compute core |
| `tb_svm_interfaces.sv` | Interface sanity testbenches (host / SV RAM / workspace RAM) |
| `tb_sample_waveform.sv` | Focused waveform capture (1 heartbeat, 5 SVs) |
| `tb_svm_params.svh` | Auto-generated SV include: dual coefficients, intercepts, SV counts |
| `gen_tb_data.py` | Generates all `.hex` files and `tb_svm_params.svh` from MIT-BIH data |
| `hw_svm_comparison.py` | Standalone Python accuracy/comparison script |
| `test_svm_compute_core.py` | cocotb test suite (9 tests) |
| `Makefile` | Runs cocotb tests via `make` |
| `sv_ram.hex` | SV feature RAM contents (generated) |
| `test_features.hex` | 5 test heartbeat feature vectors (generated) |
| `expected_kernels.hex` | Pre-computed reference kernel values (generated) |
| `test_labels.hex` | True class labels for the 5 test beats (generated) |
| `expected_preds.hex` | Expected OvO predictions (generated) |

---

## Dependencies

### Python 3.8+

```bash
pip install numpy scikit-learn wfdb matplotlib cocotb
```

| Package | Version tested | Notes |
|---------|---------------|-------|
| `numpy` | ≥ 1.21 | Fixed-point arithmetic |
| `scikit-learn` | ≥ 1.0 | SVC training and OvO classification |
| `wfdb` | ≥ 4.0 | Downloads MIT-BIH records from PhysioNet |
| `matplotlib` | ≥ 3.4 | Required by `hw_svm_comparison.py` only |
| `cocotb` | ≥ 1.8 | Required for `test_svm_compute_core.py` / `make` flow only |

> **Note:** `wfdb` streams the MIT-BIH database from PhysioNet over the network
> on first run. An internet connection is required for `gen_tb_data.py`.

### Icarus Verilog (iverilog)

```bash
# macOS
brew install icarus-verilog

# Ubuntu / Debian
sudo apt install iverilog

# Fedora / RHEL
sudo dnf install iverilog
```

Verify:

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

## Running the Testbenches

### Step 1 — Generate testbench data

Run **once** before simulating. Trains the SVM (γ = 0.01) on MIT-BIH data,
computes reference kernel values, and writes all `.hex` files and
`tb_svm_params.svh`.

```bash
python3 gen_tb_data.py
```

Expected output (takes 1–3 minutes depending on network speed):

```
[1/5] Building dataset (MIT-BIH + synthetic) ...
[2/5] Training RBF SVM (gamma=0.01, C=1.0) ...
[3/5] Computing reference kernels ...
[4/5] Running OvO classification ...
[5/5] Writing data files ...
```

---

### Testbench A — Full-pipeline classification (primary)

Uses `tb_svm_classifier.sv`. Tests 5 heartbeats (one per class) end-to-end.

```bash
iverilog -g2012 -o tb_svm tb_svm_classifier.sv svm_compute_core.sv
vvp tb_svm
```

**Outputs:**

| File | Contents |
|------|---------|
| `svm_classifier.log` | Per-heartbeat detail, kernel MAE, OvO predictions, PASS/FAIL grade |
| `svm_waveform_hb0.vcd` | VCD waveform capture for heartbeat 0 (depth 2) |

**Check results:**

```bash
cat svm_classifier.log
```

Pass criterion: **≥ 4/5 heartbeats correct AND `error` flag never asserted**.

---

### Testbench B — Compute core unit test

```bash
iverilog -g2012 -o tb_core tb_svm_compute_core.sv svm_compute_core.sv
vvp tb_core
```

---

### Testbench C — Interface sanity checks

Three top-level modules in one file. Run each separately with `-s`:

```bash
# Host interface
iverilog -g2012 -s tb_svm_host_if -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out

# SV RAM interface
iverilog -g2012 -s tb_svm_sv_ram_if -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out

# Workspace RAM interface
iverilog -g2012 -s tb_svm_work_ram_if -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out
```

---

### Testbench D — Focused waveform capture

1 heartbeat, 5 SVs (one per class). Useful for fast waveform inspection.

```bash
iverilog -g2012 -o tb_waveform tb_sample_waveform.sv svm_compute_core.sv
vvp tb_waveform
gtkwave sample_waveform.vcd
```

---

### Testbench E — cocotb suite (9 tests)

Requires `cocotb` and `icarus-verilog`.

```bash
make
```

The `Makefile` targets `svm_compute_core` as the DUT and runs all 9 tests in
`test_svm_compute_core.py`:

| Test | Description |
|------|-------------|
| `test_reset_outputs` | Verify all outputs are zero after reset |
| `test_param_programming` | Write γ via `param_write_en` interface |
| `test_sv_counts_set` | Set per-class SV counts |
| `test_sv_counts_unequal_stress` | Stress test with unequal SV distributions |
| `test_qspi_fifo_load` | Stream 256 feature words through QSPI |
| `test_qspi_backpressure` | Verify `qspi_ready` backpressure handshake |
| `test_default_gamma_fixed_point` | Check default γ = 0.01 in Q6.10 |
| `test_full_pipeline_small_batch` | End-to-end pipeline with small batch |
| `test_kernel_output_range` | Verify kernel output stays in [0, 1024] |

Results are written to `results.xml`. Console output shows pass/fail per test.

---

## Python-only Accuracy Check

To evaluate the full model without running RTL simulation:

```bash
python3 hw_svm_comparison.py
```

Produces a classification report, confusion matrices (`confusion_matrices.png`),
and a kernel error analysis comparing Q6.10 fixed-point to float64 reference.

---

## Expected Accuracy

| Model | sklearn | HW Q6.10 | Notes |
|-------|---------|----------|-------|
| γ = 0.01, single-beat | 96.33% | 96.33% | This version |

---

## Testbench Parameters (tb_svm_classifier.sv)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Clock | 50 MHz (20 ns) | `always #10 clk = ~clk` |
| Data format | Q6.10 | Scale = 1024 |
| γ | 0.01 → `0x000A` | Programmed via `param_write_en` |
| Feature dim | 256 | Single-beat morphology |
| Test beats | 5 | One per class |
| Pass threshold | 4/5 (80%) | AND `error` flag never set |
| Watchdog timeout | 200 ms sim time | ~10M cycles at 50 MHz |
