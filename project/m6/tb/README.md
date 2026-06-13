# m6 Testbenches

**Target:** IHP SG13G2 standalone (`svm_top_ihp`) — SPI slave interface, no Caravel/Wishbone.

> **Rerun status:** All Wishbone-based testbenches must be rewritten for the SPI interface.
> Direct-RTL testbenches (compute_core port-level) are reusable with minor port updates.
> See status column below.

---

## Interface Coverage

There are two distinct test interfaces in m6:

| Testbench | Interface | Status | What it exercises |
|-----------|-----------|--------|-------------------|
| `tb_wb_cosim.py` | ~~Wishbone~~ | ⚠ NEEDS REWRITE | Was: full Caravel wrapper 300-sample cosim. Replace with `tb_spi_cosim.py`. |
| `tb_wb_unit.py` | ~~Wishbone~~ | ⚠ NEEDS REWRITE | Was: Wishbone register-level unit tests. Replace with SPI frame-level unit tests. |
| `tb_top.sv` | ~~Wishbone~~ | ⚠ NEEDS REWRITE | Was: `user_project_wrapper` integration TB. Replace with `svm_top_ihp` SPI TB. |
| `svm_ram_latency_tb.sv` | Direct RTL port | ✓ REUSABLE | Drives `svm_compute_core` directly — no wrapper. Verify RAM_LATENCY=3 at 600 SVs. |
| `tb_svm_classifier.sv` | Direct RTL port | ✓ REUSABLE | Core classify flow (update NUM_SV=600 param). |
| `tb_error_codes.sv` | Direct RTL port | ✓ REUSABLE | 13 error codes, no wrapper dependency. |
| `tb_param_write.sv` | Direct RTL port | ✓ REUSABLE | Kernel parameter write path. |
| `tb_power.sv` | Direct RTL port | ✓ REUSABLE | Clock gate enable/disable cycles. |
| `tb_backpressure.sv` | Direct RTL port | ✓ REUSABLE | Kernel backpressure logic. |
| `tb_consecutive.sv` | Direct RTL port | ✓ REUSABLE | Back-to-back classification. |
| `tb_multi_heartbeat.sv` | Direct RTL port | ✓ REUSABLE | Multi-beat batch. |
| `tb_warmup.sv` | Direct RTL port | ✓ REUSABLE | Warmup error-code sequence. |
| `tb_interface.sv` | Direct RTL port | ✓ REUSABLE | Interface-level, no Caravel. |
| `tb_dist_boundary.sv` | Direct RTL port | ✓ REUSABLE | Distance boundary case. |
| `tb_dist_zero.sv` | Direct RTL port | ✓ REUSABLE | Zero-distance kernel. |
| `tb_gamma_zero.sv` | Direct RTL port | ✓ REUSABLE | Gamma=0 degenerate case. |
| `tb_min_sv.sv` | Direct RTL port | ✓ REUSABLE | Minimum SV count. |

---

## Planned: SPI Testbench (`tb_spi_cosim.py`)

Write a new cocotb testbench targeting `svm_top_ihp`:

```
Step 1: Load alpha table — 600 SPI writes to ALPHA_WR (addr 0x0A)
Step 2: Write NUM_SV[0-4] = 120 via addrs 0x04–0x08
Step 3: Write NUM_SAMPLES = 1000 via addr 0x03 (once at startup)
Step 4: Assert start — write CONTROL 0x01 with bit[0]=1
Step 5: Poll STATUS 0x02 — wait for bit[0] (done) or sample irq_sample_rdy
Step 6: Read class_out[2:0] on each sample_rdy pulse
Step 7: Verify accuracy across 300-sample PhysioNet test set
```

SPI frame protocol (CPOL=0, CPHA=0, 40-bit, MSB first):
```
Write: CS# low | 0b0_addr[6:0] (8 bits) | data[31:0] | CS# high
Read:  CS# low | 0b1_addr[6:0] (8 bits) | data[31:0] clocked out on MISO | CS# high
```

---

## Simulation Cell Stubs

m5 used `sky130_stubs.v` for the `sky130_fd_sc_hd__dlclkp_1` ICG cell.

m6 needs an IHP stub for `sg13g2_dlclkp_1`:

```systemverilog
// ihp_stubs.v — IHP SG13G2 ICG cell stub for simulation
module sg13g2_dlclkp_1 (
    input  CLK,
    input  GATE,
    output GCLK
);
    // In simulation, top.sv uses `ifdef SIMULATION — this stub is not needed
    // unless compiling the `else branch. Use -DSIMULATION flag.
    assign GCLK = CLK & GATE;
endmodule
```

When compiling with `-DSIMULATION`, `top.sv` uses `assign svm_gclk = clk & svm_clk_en`
and the IHP ICG cell is not instantiated — no stub needed.

---

## How to Run Reusable Testbenches

```bash
# RAM_LATENCY unit test (standalone, iverilog, < 1 s)
cd project/m6/tb
iverilog -g2012 -DSIMULATION \
    -o /tmp/svm_lat_tb.out \
    ../rt1/compute_core.sv svm_ram_latency_tb.sv
/tmp/svm_lat_tb.out

# SPI confusion matrix (Python — runs Q6.10 hardware model, no RTL sim required)
cd project/m6/sim
python3 confusion_comparison_m6.py
# outputs: confusion_comparison_m6.png
```

---

## Testbench Analysis Status

`testbench_analysis.md` was carried over from m5 and references Wishbone register offsets
and Caravel DV framework entries.  It must be updated after the SPI cosim testbench is written:

- Replace WB cosim pass/fail entry with SPI cosim result
- Remove Caravel DV / `dv_run.sh` entries (no management SoC in standalone design)
- Update register offset references (Wishbone base 0x3000_0000 → SPI addr 0x01–0x0A)
- Update ICG stub reference (sky130 → IHP)
