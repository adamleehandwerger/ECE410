# Critical Path Analysis — svm_compute_core

**Tool:** OpenROAD STA (global routing parasitics)  
**Clock:** `clk`, 10.0 ns period (100 MHz target)  
**WNS:** −31.03 ns  **TNS:** −2,094,059 ns  
**Max achievable frequency:** ~24.4 MHz  (period = 10 + 31.03 = 41.03 ns)

---

## Worst-Slack Path

| # | Stage | Cell | Fanout | Delay (ns) | Cumul. (ns) |
|---|-------|------|-------:|----------:|------------:|
| 1 | Clock tree to source FF | `fanout21671` → 9× clkbuf chain | — | 7.32 | 7.32 |
| 2 | FF Q output | `_327208_` (dfrtp_1) `u_input_fifo.wr_ptr[8]` | 9 | 0.53 | 7.85 |
| 3 | **Bottleneck — NOR3B** | `_096543_` (nor3b_1) | 16 | **18.97** | 26.82 |
| 4 | **Bottleneck — NAND2** | `_103329_` (nand2_1) | **64** | **12.28** | 39.67 |
| 5 | NOR2 | `_103337_` (nor2_1) | 16 | 2.28 | 43.17 |
| 6 | FF data enable | `_298847_` (edfxtp_1) `/DE` | — | 0.14 | **43.31** |

**Data arrival:** 43.31 ns  **Data required:** 12.29 ns  **Slack:** −31.03 ns (VIOLATED)

---

## Description

The critical path starts at a rising-edge flip-flop carrying bit 8 of the
input FIFO write pointer (`u_input_fifo.wr_ptr[8]`). After a deep, heavily
buffered clock tree (7.32 ns of clock skew to reach this FF), the Q output
feeds a three-input NOR gate (`_096543_`, `sky130_fd_sc_hd__nor3b_1`) that
drives 16 downstream loads — causing a 26 ns output slew and an 18.97 ns
stage delay. The output of that NOR then drives a two-input NAND
(`_103329_`, `sky130_fd_sc_hd__nand2_1`) with a **fanout of 64** from a
minimum-drive-strength `nand2_1` cell, producing a further 12.28 ns delay
and a 2.36 ns output slew. After one more NOR stage, the path terminates at
the data-enable input of an edge-triggered D flip-flop (`_298847_`,
`edfxtp_1`). The total path delay of 43.31 ns against a 10 ns clock means
the design meets timing only at approximately **24 MHz**, not the 100 MHz
target.

## Root Cause

The timing violation is driven by two compounding problems in the synthesized
register-based FIFO (`u_input_fifo`):

1. **Extreme fanout (64×)** on a min-strength NAND2 — the synthesis tool
   did not insert repeaters on this net. Fixing with `MAX_FANOUT_CONSTRAINT`
   tighter than the current 10 (or by using `SYNTH_SIZING 1` with `repair_design`)
   would break this net into buffered segments.

2. **Excessive clock skew (7.32 ns)** between source and destination FFs —
   the placer placed these FFs far apart, requiring a long clock distribution
   chain. Physical-aware synthesis or tighter placement constraints would help.

## Recommended Fix (chipIgnite path)

The chipIgnite design already replaces `u_input_fifo` with a sky130 SRAM
macro (`svm_fifo_sram`), eliminating the 131,072-FF register bank and the
fanout-plagued FIFO pointer logic. This alone is expected to reduce the
critical path length by 15–20 ns and bring timing within reach of 50–60 MHz
on sky130A at the chipIgnite node.
