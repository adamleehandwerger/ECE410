# ECE410 — Milestone 6: IHP SG13G2 Standalone Re-Target

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** IHP SG13G2 (130 nm BiCMOS open PDK, `sg13g2_stdcell`)  
**Flow:** OpenROAD (IHP open process design kit — FOSS VLSI tools)  
**Architecture:** Batch v11 — 600-SV uniform allocation, sticky NUM_SAMPLES, SPI slave interface  
**Status:** RTL complete; P&R synthesis pending (IHP PDK setup required)

---

## What Changed from m5 (Caravel sky130) to m6 (IHP SG13G2)

### 1 — 600 Support Vectors ([120, 120, 120, 120, 120])

m5 used 500 SVs ([95, 95, 95, 120, 95]).  The m5 allocation sweep found that VT required 120 SVs
to eliminate a quantization flip while the other four classes needed only 95.

m6 increases to 600 SVs with a **uniform** allocation:

| Allocation | Float acc. | Q6.10 acc. | Flips |
|------------|-----------|-----------|-------|
| [95,95,95,120,95] — m5 implemented | 98.33% | 98.33% | 0 |
| [120,120,120,120,120] — m6 target | 98.67% | **98.67%** | 0 |

At 600 SVs each class has enough high-|α| support vectors that concentrating more in any one
class does not help (confirmed by `alloc_sweep_600.py` — non-uniform splits match or degrade).
The 0.34 percentage-point accuracy gain (+1 sample per 300) justifies the extra storage.

**RTL impact:**
- `NUM_SV` parameter: 500 → 600
- `alpha_addr` port: 9-bit → 10-bit (indexes 0–599)
- `alpha_table` array: 500 → 600 entries
- SRAM requirement: ≥ 307.2 KB SV matrix + 512 KB input matrix ≈ 820 KB (IS62WV51216 1 MB recommended)

### 2 — Sticky NUM_SAMPLES (Write Once at Startup)

In m5, `NUM_SAMPLES` could be overwritten at any time during a batch, which risked an
in-flight sample-count change causing the FSM to stop at the wrong beat.

In m6 the firmware convention is: **write NUM_SAMPLES exactly once, before asserting start**.
The register itself is still read-write over SPI (no hardware lock), but the MCU firmware must
treat it as a startup-only configuration.

- Reset default: `10'd1000` (1000 beats per batch)
- Recommended firmware sequence: write NUM_SAMPLES → write NUM_SV[0-4] → write alpha table → assert start

### 3 — IHP SG13G2 Standalone Interface (Replaces Caravel)

**Why Caravel is no longer viable (see m5 Appendix D):**  
`ram_rdata[15:0]` was internally connected to `la_data_in[15:0]` — the Caravel logic
analyser bus — not to any pad.  All 30 available `mprj_io` pads were already consumed by outputs
(address + control + class signals), leaving zero pads for the 16-bit SRAM read-data bus.
Management SoC firmware relay was required but impossible within the 75 ns RAM access budget at
LAT=3 / 40 MHz.

**IHP SG13G2 selection rationale:**

| Criterion | IHP SG13G2 |
|-----------|-----------|
| Cost | Free for academic / non-commercial use |
| Pad ring | Own pad ring (~60–80 pads on 2 mm²), fully user-controlled |
| Technology | 130 nm BiCMOS (rf performance not needed here; digital flow used) |
| Shuttles | Quarterly open MPW shuttles |
| EDA support | OpenROAD, KLayout, Magic; open PDK at `ihp-open-pdk` |
| Prior art | Multiple academic tape-outs; well-documented standard cell library |

**Pad allocation (46 pads required — fits comfortably):**

| Signal | Direction | Pads | Notes |
|--------|-----------|------|-------|
| `ram_rdata_in[15:0]` | Input | 16 | Dedicated input pads — SRAM DQ[15:0] |
| `ram_addr_out[18:0]` | Output | 19 | SRAM A[18:0] |
| `ram_ren_out` | Output | 1 | SRAM OE# (active-low on SRAM side) |
| `class_out[2:0]` | Output | 3 | Per-sample predicted class |
| `sample_rdy` | Output | 1 | Pulses high when class_out is valid |
| `done` | Output | 1 | Batch complete |
| `error` / `error_code[3:0]` | Output | 5 | Error status |
| `irq_sample_rdy` / `irq_done` | Output | 2 | Hardware IRQ to MCU |
| `clk`, `rst_n` | Input | 2 | System clock and reset |
| SPI: `spi_csn/sclk/mosi/miso` | Mixed | 4 | MCU SPI master interface |
| **Total** | | **54** | Within IHP pad budget |

**New top-level module: `svm_top_ihp` (rt1/top.sv)**

Replaces `user_project_wrapper` from m5. Key differences:
- No Caravel Wishbone bus
- SPI slave (CPOL=0, CPHA=0) replaces Wishbone as the MCU configuration path
- `ram_rdata_in[15:0]` from dedicated input pads (no management SoC relay)
- IHP ICG cell: `sg13g2_dlclkp_1` (replaces `sky130_fd_sc_hd__dlclkp_1`)
- Reset defaults: NUM_SAMPLES = 1000, NUM_SV[0-4] = 120

**SPI register map (8-bit address, 32-bit data, 40-bit frames):**

| Addr | Access | Register | Bits | Description |
|------|--------|----------|------|-------------|
| 0x01 | RW | CONTROL | [0]=start (auto-clear), [1]=vbatt_ok, [2]=vbatt_warn | Start/status control |
| 0x02 | RO | STATUS | [0]=done, [1]=error, [5:2]=error_code, [8:6]=class, [9]=sample_rdy | Result readback |
| 0x03 | RW | NUM_SAMPLES | [9:0] default=1000 | Beats per batch (sticky — write once) |
| 0x04–0x08 | RW | NUM_SV[0–4] | [7:0] default=120 | SVs per class |
| 0x09 | WO | PARAM_WR | [19]=en, [18:16]=addr, [15:0]=data | Kernel parameter write |
| 0x0A | WO | ALPHA_WR | [25:16]=sv_idx (10-bit), [15:0]=alpha Q6.10 | Alpha table write |

**SPI protocol:**
```
CS# low → 8-bit address byte (MSB first) → 32-bit data (MSB first) → CS# high
Write: addr[7]=0, addr[6:0]=register, data[31:0]=value
Read:  addr[7]=1, addr[6:0]=register, MISO clocks out data[31:0] from bit 31
```

**MCU recommendation: nRF52840**
- 4 MHz SPI master — loads full 600-SV alpha table in < 1 ms per batch
- GPIO interrupt on `irq_sample_rdy` / `irq_done`
- 3.3 V I/O compatible with IHP SG13G2 pads

---

## Directory Structure

```
m6/
├── README.md                    ← this file
├── compute_core_math.tex/pdf    ← fixed-point RBF kernel derivation (Q6.10)
├── horner_lut_math.tex/pdf      ← LUT + Horner polynomial derivation
├── horner_errorplot.png         ← degree-6 Taylor error vs Q6.10 LSB
├── pipeline_drain.md            ← drain counter design note
│
├── rt1/                         ← RTL source (v11, IHP SG13G2 target)
│   ├── compute_core.sv          ← SVM core: NUM_SV=600, alpha_addr 10-bit, batch FSM
│   ├── top.sv                   ← IHP standalone top: SPI slave, ram_rdata direct pads,
│   │                                sg13g2_dlclkp_1 ICG, 600-SV/sticky defaults
│   └── interface.sv             ← SystemVerilog interface definitions
│
├── tb/                          ← Testbenches (⚠ see status below)
│   ├── README.md                ← Testbench overview and rerun status
│   ├── Makefile                 ← Build targets (update required for IHP)
│   ├── tb_wb_cosim.py           ← ⚠ NEEDS REWRITE — Wishbone cosim, not valid for SPI
│   ├── tb_wb_unit.py            ← ⚠ NEEDS REWRITE — Wishbone unit tests
│   ├── tb_top.sv                ← ⚠ NEEDS REWRITE — Wishbone integration TB
│   ├── svm_ram_latency_tb.sv    ← ✓ Reusable — drives compute_core directly (no wrapper)
│   ├── tb_svm_classifier.sv     ← ✓ Reusable — drives compute_core directly
│   ├── tb_error_codes.sv        ← ✓ Reusable — drives compute_core directly
│   ├── tb_param_write.sv        ← ✓ Reusable — drives compute_core directly
│   ├── tb_power.sv              ← ✓ Reusable — drives compute_core directly
│   ├── tb_backpressure.sv       ← ✓ Reusable after port update
│   ├── tb_consecutive.sv        ← ✓ Reusable after port update
│   ├── tb_multi_heartbeat.sv    ← ✓ Reusable after port update
│   ├── tb_warmup.sv             ← ✓ Reusable after port update
│   ├── tb_interface.sv          ← ✓ Reusable (interface-level, no Caravel)
│   ├── tb_dist_boundary.sv      ← ✓ Reusable
│   ├── tb_dist_zero.sv          ← ✓ Reusable
│   ├── tb_gamma_zero.sv         ← ✓ Reusable
│   ├── tb_min_sv.sv             ← ✓ Reusable
│   └── testbench_analysis.md    ← ⚠ Needs updating (Caravel refs, rerun status)
│
├── sim/                         ← Simulation outputs and sweep scripts
│   ├── sv_sweep.py              ← SV allocation sweep (500-SV, from m5)
│   ├── alloc_sweep_600.py       ← 600-SV allocation sweep (confirms uniform optimal)
│   ├── confusion_comparison_m6.py ← sklearn OVR float vs Q6.10 (600 SVs)
│   ├── confusion_comparison_m6.png ← generated figure
│   └── throughput_comparison.txt ← inference/power summary (m5 baseline; IHP pending)
│
├── bench/                       ← Benchmark artifacts (m5 baseline; IHP re-benchmark pending)
│   ├── benchmark.md
│   ├── benchmark_data.csv
│   ├── roofline_final.py/png
│
├── synth/                       ← P&R placeholder (IHP flow not yet run)
│   └── README_pending.md
│
├── pnr/                         ← P&R placeholder (IHP flow not yet run)
│   └── README_pending.md
│
└── report/                      ← Final project report (pending m6 updates)
```

---

## Key Design Parameters (v11 Target)

| Parameter | m5 (sky130 / Caravel) | m6 (IHP SG13G2) |
|-----------|----------------------|-----------------|
| Technology | sky130A (SkyWater 130 nm) | SG13G2 (IHP 130 nm BiCMOS) |
| Top module | `user_project_wrapper` | `svm_top_ihp` |
| Host interface | Caravel Wishbone | SPI slave (CPOL=0, CPHA=0) |
| SRAM data path | `la_data_in[15:0]` ⚠ BUG | `ram_rdata_in[15:0]` dedicated pads ✓ |
| Support vectors | 500 ([95,95,95,120,95]) | 600 ([120,120,120,120,120]) |
| `alpha_addr` width | 9-bit | 10-bit |
| ASIC accuracy (Q6.10) | 98.33% (295/300) | 98.67% (296/300) — target |
| Clock | 40 MHz (25 ns) | 40 MHz (25 ns, pending IHP STA) |
| RAM_LATENCY | 3 (IS61WV51216) | 3 (IS62WV51216) |
| ICG cell | `sky130_fd_sc_hd__dlclkp_1` | `sg13g2_dlclkp_1` |
| Fixed-point | Q6.10, 16-bit signed | Q6.10, 16-bit signed (unchanged) |
| Feature dimension | 256 | 256 (unchanged) |
| NUM_SAMPLES default | 1000 (from reset) | 1000 (sticky, write once) |

---

## Next Steps to Harden for IHP SG13G2

### 1. IHP PDK Setup
```bash
git clone https://github.com/IHP-GmbH/IHP-Open-PDK.git
# Set PDK_ROOT to the cloned path
export PDK_ROOT=/path/to/IHP-Open-PDK
export PDK=sg13g2
```

### 2. Write SPI Testbench (Replaces Wishbone Cosim)
Write `tb/tb_spi_cosim.py` (cocotb): replaces `tb_wb_cosim.py` for the new SPI interface.
- Drive `spi_csn`, `spi_sclk`, `spi_mosi` at the bit level
- Read `spi_miso` for STATUS readback
- Load alpha table via ALPHA_WR (0x0A), 600 writes
- Assert start via CONTROL (0x01), poll STATUS (0x02) for done
- Record `class_out[2:0]` on each `sample_rdy` pulse
- Verify 98.67% accuracy on 300-sample PhysioNet test set

### 3. Re-run Core Synthesis (IHP flow)
```bash
# OpenROAD flow with IHP PDK
cd synth/
# Create config.tcl targeting sg13g2_stdcell
# Target clock: 25 ns (40 MHz)
# Expected cell count: similar to m5 (~157 k cells at 15% util)
openroad -gui run_synth.tcl
```

### 4. Re-run Static Timing Analysis
- Target: TT WNS ≥ 0 ns at 40 MHz
- IHP 130 nm BiCMOS may have different timing arcs than sky130
- Verify hold margin ≥ 0 ns (enable CTS for wrapper)

### 5. Update Testbench Analysis
`tb/testbench_analysis.md` references Wishbone register offsets and Caravel DV framework.
After the SPI cosim testbench is written and passing, update:
- Test list (add SPI smoke test, remove WB cosim entries)
- Pass/fail table
- Caravel DV → n/a (standalone design, no management SoC)

### 6. IHP DRC and LVS
```bash
# KLayout DRC with IHP ruleset
klayout -b -r $PDK_ROOT/sg13g2/libs.tech/klayout/tech/drc/sg13g2.lydrc \
    -rd input=gds/svm_top_ihp.gds
# Magic LVS
magic -dnull -noconsole -rcfile $PDK_ROOT/sg13g2/libs.tech/magic/sg13g2.magicrc \
    << 'EOF'
gds read gds/svm_top_ihp.gds
flatten svm_top_ihp
extract all
ext2spice lvs
quit
EOF
netgen -batch lvs "svm_top_ihp.spice svm_top_ihp" \
    "netlist/svm_top_ihp.v svm_top_ihp" \
    $PDK_ROOT/sg13g2/libs.tech/netgen/sg13g2_setup.tcl
```

### 7. IHP MPW Shuttle Submission
- Register at `ihp-open-pdk` shuttle announcement channel
- Prepare GDS, LEF, abstract view, and design manifest per IHP checklist
- Target: next quarterly open MPW shuttle after P&R is clean

---

## Quick Start

```bash
# RAM_LATENCY unit test — compute_core directly (no wrapper dependency, iverilog)
cd tb
iverilog -g2012 -DSIMULATION \
    -o /tmp/svm_lat_tb.out \
    ../rt1/compute_core.sv svm_ram_latency_tb.sv
/tmp/svm_lat_tb.out

# 600-SV allocation sweep (Python, requires wfdb/PhysioNet)
cd sim
python3 alloc_sweep_600.py      # confirms [120,120,120,120,120] is optimal

# Confusion matrix (600-SV Q6.10 vs sklearn float)
cd sim
python3 confusion_comparison_m6.py   # outputs confusion_comparison_m6.png
```

Requires: `pip install cocotb scikit-learn wfdb matplotlib numpy`  
PhysioNet cache: `~/.physionet_cache/`  
NumPy cache: `/tmp/cosim_cache_ecg_n300_d256.npz`

---

## Differences from m5

m5 used Caravel (sky130A Wishbone wrapper).  m6 is a clean break:

| Item | m5 | m6 |
|------|----|----|
| Tapeout platform | Caravel / Efabless (shutdown 2025) | IHP SG13G2 open MPW |
| Host interface | Wishbone (Caravel) | SPI slave (standalone) |
| SRAM read data | `la_data_in[15:0]` (BUG — no pad path) | `ram_rdata_in[15:0]` (dedicated input pads) |
| Support vectors | 500 | 600 |
| `alpha_addr` | 9-bit | 10-bit |
| Q6.10 accuracy | 98.33% (295/300) | 98.67% target |
| ICG cell | `sky130_fd_sc_hd__dlclkp_1` | `sg13g2_dlclkp_1` |
| P&R status | Complete (jobs 92840/92861) | Pending (IHP PDK) |
| Testbench cosim | Wishbone (passing) | SPI (needs rewrite) |
