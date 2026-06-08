# SVM Compute Core — Design Summary (m4: svm_compute_core Hardened)

**Project:** Multi-Class Cardiac Arrhythmia Detection
**RTL:** `svm_compute_core.sv` (v9 — 256-dim, 500 SVs, Q6.10 fixed-point)
**Architecture:** Batch v9 — host pre-loads SV matrix + input matrix; ASIC classifies autonomously
**Accuracy:** 97.67% on PhysioNet (sklearn = hardware, 0.00% gap)
**Flow:** OpenLane 2 v2.3.10 Classic
**Status:** svm_compute_core hardened (job 91966). Wrapper in m4.

---

## P&R Results (OL2 job 91966, nom_tt_025C_1v80)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns), TT CLEAN — 0 violations |
| Setup WNS | +7.83 ns |
| Hold WNS | +0.30 ns |
| Active power | 66 mW |
| Inference time | 3.23 ms / beat (500 SVs × 256 dim) |
| Avg power (80 bpm) | 0.284 mW (0.431% duty cycle) |
| 14-day battery | 108 days on SVM core alone (200 mAh @ 3.7V) |
| Standard cells | ~146K |
| Die | 2500 × 2500 µm (6.25 mm²), ~14% utilization |
| DRC | **0 violations** |
| GDS | 226 MB |
| ASIC accuracy | **97.67%** (293/300) = sklearn exactly |

---

## Architecture

### Off-chip RAM Layout

| Rows | Content | Size |
|------|---------|------|
| 0 – 499 | SV matrix (500 × 256 × Q6.10) | 256 KB |
| 500 – 1499 | Input matrix (1000 × 256 × Q6.10) | 512 KB |

Address bus: `GPIO[28:10]` = `ram_addr[18:0]`, `GPIO[29]` = `ram_ren`, `LA[15:0]` = `ram_rdata`.

### Wishbone Register Map (base `0x3000_0000`)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in batch |
| +0x10–0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class (max 100) |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |
| +0x28 | ALPHA_WR | WO | [24:16]=sv_idx (9-bit) [15:0]=alpha Q6.10 |

---

## Feature Extraction (256-dim multi-scale)

| Group | Dims | Reference |
|-------|------|-----------|
| Single-beat morphology (±64 samples) | 128 | de Chazal et al., IEEE TBME 2004 |
| 10-beat mean template | 64 | de Chazal & Reilly, IEEE TBME 2006 |
| RR-interval history (99 intervals → 64 pts) | 64 | Llamedo & Martínez, IEEE TBME 2011 |

Standard: AAMI ANSI EC57:2012.
Dataset: PhysioNet MIT-BIH + SVDB + INCART.

---

## Acknowledgments

Hardening performed on **Orca**, Portland State University's HPC cluster,
using SLURM batch jobs with OpenLane 2 v2.3.10 inside a Singularity container.
Thanks to the PSU Research Computing team.

---

*Document version: m4/v9 · 2026-05-25 — svm_compute_core hardened (job 91966)*
