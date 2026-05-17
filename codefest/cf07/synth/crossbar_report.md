---
title: "OpenLane2 Timing & Area Report"
subtitle: "Design: `crossbar` | PDK: sky130A HD | Run: 2026-05-16"
date: "2026-05-16"
geometry: margin=2.5cm
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{xcolor}
  - \definecolor{violated}{RGB}{200,0,0}
  - \definecolor{passing}{RGB}{0,150,0}
---

# (a) Clock Period & Worst-Case Slack

| Parameter | Value |
|:---|:---|
| Clock period (configured) | 10.00 ns |
| Worst-case corner | `max_ss_100C_1v60` (slow silicon, 100 °C, 1.60 V) |
| Setup slack — worst corner | **−0.613 ns (VIOLATED)** |
| Setup slack — nominal (TT 25 °C 1.80 V) | +2.469 ns |
| Setup slack — best (FF −40 °C 1.95 V) | +3.689 ns |
| Total negative slack (TNS) | −1.695 ns across 16 endpoint violations |
| Hold slack (all corners) | ≥ +4.09 ns — no violations |

**Note:** The design is purely combinational (0 flip-flops). OpenROAD created a synthetic
`__VIRTUAL_CLK__` to bound the I/O paths. The −0.613 ns violation means the longest
combinational cone cannot complete within the 7.75 ns effective budget
(10 ns − 2 ns output delay − 0.25 ns uncertainty) at the worst PVT corner.
The design passes comfortably at nominal and fast corners.

---

# (b) Critical Path

**Source → Sink:** `v0[2]` (input port) → `i3[9]` (output port `i3`, bit 9)

> No source/sink *registers* exist — this is a pure combinational cone from primary input
> `v0[2]` to primary output `i3[9]`. The tool treats I/O ports as virtual register
> endpoints under `__VIRTUAL_CLK__`.

## Path Trace (max\_ss\_100C\_1v60, 8.363 ns arrival)

| Stage | Cell | Type | Delay (ns) | Cumulative (ns) |
|:---|:---|:---|---:|---:|
| Input buffer | `input3` | `clkbuf_4` | 0.296 | 2.305 |
| **OR-4 (dominant)** | `_525_` | `or4_4` | **1.122** | 3.429 |
| OR3-AND-1 | `_575_` | `o31a_1` | 0.700 | 4.133 |
| **OR-3b** | `_820_` | `or3b_4` | **0.817** | 4.951 |
| AND21-OI | `_836_` | `a21boi_2` | 0.358 | 5.310 |
| XOR-2 | `_839_` | `xor2_1` | 0.350 | 5.660 |
| XOR-2 | `_841_` | `xor2_1` | 0.444 | 6.104 |
| OR21-AND | `_842_` | `o21a_1` | 0.404 | 6.508 |
| AND-2b | `_861_` | `and2b_1` | 0.437 | 6.946 |
| **OR41-AND** | `_864_` | `o41a_1` | **0.767** | 7.713 |
| OR21-AND | `_865_` | `o21a_1` | 0.393 | 8.106 |
| Output buffer | `output72` | `buf_6` | 0.254 | 8.361 |
| | | | | |
| **Total arrival** | | | | **8.363 ns** |
| **Required time** | | | | **7.750 ns** |
| **Slack** | | | | **−0.613 ns** |

## Dominant Cell Types by Delay Contribution

| Rank | Cell Type | Path Delay | Notes |
|:---:|:---|:---:|:---|
| 1 | `or4_4` | 1.122 ns | High fanout (×7), large cap load, slow OR at SS corner |
| 2 | `or3b_4` | 0.817 ns | Drives 3 fanout at 100 °C / 1.60 V |
| 3 | `o41a_1` | 0.767 ns | Complex 4-input OR-AND gate |

The 4-input OR (`or4_4`) contributes **13.4%** of total path delay.
All three dominant stages are wide OR/AND-OR structures, consistent with the signed
addition logic (`s0+s1−s2−s3` etc.) resolving carry/sign bits through deep OR trees.

---

# (c) Cell Area — Top Contributors

| Metric | Value |
|:---|:---|
| Total standard cell area (post-PnR) | 4,200.28 µm² |
| Total instances (logic) | 764 |
| Core utilisation | 34.8% of 12,061.6 µm² |
| Sequential elements | 0 (100% combinational) |
| Die bounding box | 122.24 µm × 132.96 µm |

## Top 3 by Instance Count (Logic Cells from Synthesis)

| Rank | Cell | Count | Function |
|:---:|:---|:---:|:---|
| 1 | `sky130_fd_sc_hd__xnor2_2` | 68 | 2-input XNOR — sign/bit comparison in adder network |
| 2 | `sky130_fd_sc_hd__nand2_2` | 54 | 2-input NAND — complement logic |
| 3 | `sky130_fd_sc_hd__nor2_2`  | 52 | 2-input NOR — complement logic |

## Cell Class Breakdown (Post-PnR)

| Class | Count | % of logic instances |
|:---|:---:|:---:|
| Multi-input combinational | 496 | 64.9% |
| Timing repair buffers (added by OpenROAD) | 80 | 10.5% |
| Inverters | 20 | 2.6% |

---

# Summary & Recommendation

The −0.613 ns violation is a **timing budget artefact** of the 2 ns output-delay
assumption OpenROAD applies without an explicit SDC file. Adding a
`SIGNOFF_SDC_FILE` with realistic I/O constraints, or reducing `CLOCK_PERIOD` to
~7 ns, would give an accurate closure picture. At TT/nominal the design carries
+2.47 ns of positive slack — it is functionally fast.

If timing closure at the SS corner is required, the `or4_4` → `or3b_4` → `o41a_1`
chain is the primary target: restructure the adder tree (e.g. carry-select or
prefix-tree adder) or replace the deep OR fan-in with balanced two-level logic.

| Check | Result |
|:---|:---:|
| DRC | PASS |
| LVS | PASS |
| Antenna | PASS |
| Setup (TT nominal) | PASS |
| Setup (SS 100 °C worst) | **FAIL** (−0.613 ns, SDC artefact) |
| Hold (all corners) | PASS |
