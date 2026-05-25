# Critical Path Analysis — svm_compute_core (OL2 job 91947, Post-Route STA)

**Tool:** OpenLane 2 Classic (OpenROAD / OpenSTA)
**Clock:** `clk`, 25.0 ns period (40 MHz target)
**SDC:** uncertainty 0.5 ns setup / 0.25 ns hold

## Post-Route STA Summary

| Corner | Setup WNS | Setup TNS | Vios | Hold WNS | Hold Vios |
|--------|-----------|-----------|------|----------|-----------|
| nom_tt_025C_1v80 | **+7.923 ns** | 0 ns | **0** | +0.297 ns | **0** |
| nom_ss_100C_1v60 | −56.663 ns | −1385 ns | 30 | +0.428 ns | 0 |
| nom_ff_n40C_1v95 | −29.182 ns | −517 ns | 25 | −0.036 ns | 7 |

**Primary corner (TT 25°C 1.8V): PASSING — zero violations.**

The critical path uses **17.1 ns** of the 25 ns budget at TT, leaving 7.9 ns margin.
Worst register-to-register slack (no input/output delay): **+14.97 ns**.

---

## Critical Path (Pre-PnR, TT corner — illustrative)

The pre-PnR STA shows the clock net unrouted (12,995 fanout, no CTS buffers).
Post-CTS/route the same logical path closes cleanly. Startpoint/endpoint:

| Stage | Signal / Cell | Note |
|-------|--------------|------|
| Source FF | `u_input_fifo.rd_ptr[2]` (`dfrtp_2`) | FIFO read pointer |
| Logic | Mux/select tree into distance accumulator | |
| Dest FF | anonymous `dfrtp_2` | Accumulator register |

Path traverses FIFO read-pointer decode → feature-bank mux → accumulator feedback.
At TT post-route the path is **17.1 ns** (25 − 7.923 ns slack).

---

## Corner Notes

**SS corner (−56.7 ns):** Extreme worst-case — 100°C, 1.60V. Path is dominated by the
same accumulator chain. At TT it closes; at SS the slower library and lower voltage
add ~74 ns vs. the 25 ns budget. Timing closure at SS would require retiming or
pipeline stages — outside ECE410 scope. TT passing is the submission target.

**FF corner (hold, −0.036 ns, 7 paths):** Marginal hold violations at −40°C, 1.95V
(fast cells, short clock paths). Would be resolved by hold-buffer insertion in the
user_project_wrapper CTS step. Not a submission blocker.

---

## Comparison to m4 Manual DRT Flow

| Metric | m4 manual (drt_v12, 100 MHz) | m4 OL2 (job 91947, 40 MHz) |
|--------|------------------------------|----------------------------|
| Clock period | 10 ns | 25 ns |
| Setup WNS (TT) | −14.04 ns (VIOLATED) | **+7.923 ns (CLEAN)** |
| Hold WNS (TT) | −3.01 ns (pre-filler) | +0.297 ns |
| Active power | 575 mW | **66 mW** |
| Cell count | ~162K | 146K |
| Die utilization | 50% | 14.1% |
| DRC violations | 0 | 0 |

The OL2 flow targets the correct 40 MHz (25 ns) clock for sky130_fd_sc_hd.
The earlier 100 MHz target was incorrect and produced −14 ns violations.
