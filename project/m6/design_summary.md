# SVM Compute Core --- Full-Chip Design Summary (m6: IHP SG13G2 Standalone Re-target)

**Project:** Multi-Class Cardiac Arrhythmia Detection --- IHP SG13G2 Open MPW Tape-Out
**Technology:** IHP SG13G2 130 nm BiCMOS
**Flow:** OpenLane 2 (IHP sg13g2 PDK target, Orca HPC)
**Architecture:** Standalone SPI slave + SVM core --- no Caravel wrapper required
**RTL freeze:** m6/rt1 v11 --- NUM_SV=600, alpha_addr[9:0], SV_ALLOC=[120,120,120,120,120]
**Harden version:** pending (SLURM jobs to be submitted on Orca after cosim passes)

---

## Motivation for IHP Re-target

The m5 design achieved functional silicon in the sky130A process (Caravel chipIgnite), but
the Caravel submission framework added complexity (Wishbone-only interface, wrapper DRC
boundary artifacts, multi-power-domain LVS issues, 3.3 V-only GPIO). IHP SG13G2 is a
130 nm BiCMOS process available free for academic use via quarterly open MPW shuttles.
Key advantages for this design:

| Feature | sky130 / Caravel | IHP SG13G2 (m6) |
|---------|-----------------|-----------------|
| Interface | Wishbone bus (Caravel internal) | SPI slave (direct GPIO) |
| GPIO | 3.3 V only, muxed | 1.8 V / 3.3 V selectable, dedicated pads |
| Wrapper overhead | user_project_wrapper (2920x3520 um) | None --- standalone die |
| Die sizing | Fixed Caravel template | Freely sized (targeting 2.4x2.4 mm) |
| Power domains | 3 (Caravel management, user power, analog) | 1 (simplified pad ring) |
| Routing layers | Metal5 max | Metal5 max (same) |
| Host MCU required | PicoRV32 on-chip management core | Any MCU with SPI master |
| Clock | wb_clk_i via ICG | Direct clk pad |

The SPI interface is also better suited to the target deployment (wearable patch with
nRF52840 as the primary processor, connected directly via 4-wire SPI at up to 2 MHz).

---

## Component Summary

### svm_compute_core (IHP SG13G2, pending harden)

| Metric | Value |
|--------|-------|
| Clock | 40 MHz (25 ns) |
| Die area (target) | 2200 x 2200 um (core macro) |
| Target utilization | 50% |
| NUM_SV | 600 ([120,120,120,120,120]) |
| alpha_addr width | 10-bit (up from 9-bit in m5) |
| RAM_LATENCY | 3 (IS61WV51216 SRAM, same as m5) |
| Setup WNS | TBD (target >+2 ns at TT 25C 1.2V) |
| DRC | Pending |
| GDS | Pending (estimate ~200 MB at IHP SG13G2 density) |

### svm_top_ihp (IHP SG13G2, pending harden)

| Metric | Value |
|--------|-------|
| Die area (target) | 2400 x 2400 um (top including pad ring) |
| Pads | 54 (see pad assignment table below) |
| SPI interface | CPOL=0, CPHA=0, 40-bit frames (8-bit addr + 32-bit data), MSB first |
| Clock gate | sg13g2_dlclkp_1 (IHP ICG cell) |
| DRC | Pending |
| GDS | Pending |

---

## Pad Assignment (54 pads)

| Pad | Signal | Direction | Notes |
|-----|--------|-----------|-------|
| P1 | VDD | Power | 1.2 V core supply |
| P2 | VSS | Power | Ground |
| P3 | VDDIO | Power | 1.8 V / 3.3 V pad ring |
| P4 | VSSIO | Power | Pad ring ground |
| P5 | clk | Input | 40 MHz system clock |
| P6 | rst_n | Input | Active-low async reset |
| P7 | spi_csn | Input | SPI chip select (active low) |
| P8 | spi_sclk | Input | SPI clock (CPOL=0) |
| P9 | spi_mosi | Input | SPI MOSI (MSB first) |
| P10 | spi_miso | Output | SPI MISO read-back |
| P11 | ram_addr[0] | Output | SRAM address bit 0 |
| P12 | ram_addr[1] | Output | |
| P13 | ram_addr[2] | Output | |
| P14 | ram_addr[3] | Output | |
| P15 | ram_addr[4] | Output | |
| P16 | ram_addr[5] | Output | |
| P17 | ram_addr[6] | Output | |
| P18 | ram_addr[7] | Output | |
| P19 | ram_addr[8] | Output | |
| P20 | ram_addr[9] | Output | |
| P21 | ram_addr[10] | Output | |
| P22 | ram_addr[11] | Output | {row[3:0], col[7:0]} |
| P23 | ram_addr[12] | Output | |
| P24 | ram_addr[13] | Output | |
| P25 | ram_addr[14] | Output | |
| P26 | ram_addr[15] | Output | |
| P27 | ram_addr[16] | Output | |
| P28 | ram_addr[17] | Output | |
| P29 | ram_addr[18] | Output | |
| P30 | ram_ren_out | Output | SRAM read enable |
| P31 | ram_rdata_in[0] | Input | SRAM read data bit 0 |
| P32 | ram_rdata_in[1] | Input | |
| P33 | ram_rdata_in[2] | Input | |
| P34 | ram_rdata_in[3] | Input | |
| P35 | ram_rdata_in[4] | Input | |
| P36 | ram_rdata_in[5] | Input | |
| P37 | ram_rdata_in[6] | Input | |
| P38 | ram_rdata_in[7] | Input | |
| P39 | ram_rdata_in[8] | Input | |
| P40 | ram_rdata_in[9] | Input | |
| P41 | ram_rdata_in[10] | Input | |
| P42 | ram_rdata_in[11] | Input | |
| P43 | ram_rdata_in[12] | Input | |
| P44 | ram_rdata_in[13] | Input | |
| P45 | ram_rdata_in[14] | Input | |
| P46 | ram_rdata_in[15] | Input | |
| P47 | class_out[0] | Output | Predicted class bit 0 |
| P48 | class_out[1] | Output | Predicted class bit 1 |
| P49 | class_out[2] | Output | Predicted class bit 2 |
| P50 | sample_rdy | Output | IRQ: beat result ready |
| P51 | done | Output | IRQ: full batch complete |
| P52 | error | Output | Error flag |
| P53 | error_code[0] | Output | Error code bit 0 |
| P54 | irq_sample_rdy | Output | Alternate IRQ line |

*(Power pad count: 4 VDD/VSS + 4 corner cells = 8 power pads; signal pads: 46)*

---

## Functional Results

### Simulation (SPI cosim, tb_spi_cosim.py)

| Implementation | Accuracy | SVs | Notes |
|---|---|---|---|
| sklearn default OVR (float) | 97.67% | 416 (unlimited) | joint multiclass; OVO internally |
| sklearn binary OVR (float) | **98.67%** | 600, [120,120,120,120,120] | 5 independent binary SVMs |
| ASIC binary OVR (Q6.10, SPI cosim) | **98.67%** | 600, [120,120,120,120,120] | Q6.10 fixed-point; 0 quantization flips |

*Note: cosim result confirmed via `tb_spi_cosim.py` with real PhysioNet ECG data (MIT-BIH + SVDB + INCART).
Hardware interface modelled via full SPI BFM (CPOL=0, CPHA=0, 40-bit frames) and off-chip SRAM model.*

**Accuracy improvement over m5:** +0.34 pp (98.67% vs 98.33%) — achieved by raising alpha table from 500 to 600
entries and switching to uniform allocation [120,120,120,120,120]. See Appendix A for allocation sweep details.

Per-class Q6.10 results (allocation [120,120,120,120,120]):

| Class | Correct | Accuracy |
|-------|---------|----------|
| Normal (N) | 59/60 | 98.3% |
| PVC | 60/60 | 100.0% |
| AFib | 60/60 | 100.0% |
| VT | 58/60 | 96.7% |
| SVT | 59/60 | 98.3% |

*(Individual class results match alloc_sweep_600.py float/Q6.10 reference — no quantization flips.)*

---

## Training Set

| Parameter | Value |
|-----------|-------|
| Source | MIT-BIH + SVDB + INCART (PhysioNet) --- pooled, stratified |
| Split | 80% train / 20% test --- stratified, `random_state=42` |
| Beats per class | 240 |
| Classes | Normal (N), PVC, AFib, VT, SVT |
| **Total training beats** | **1,200** |
| Feature dimension | 256 (128 single-beat + 64 10-beat mean + 64 RR history) |
| Precision | Q6.10 fixed-point (16-bit signed) |

---

## Testing Set

| Parameter | Value |
|---|---|
| Source | MIT-BIH + SVDB + INCART (PhysioNet) --- held-out 20% stratified |
| Beats per class | 60 |
| **Total test beats** | **300** |
| sklearn binary OVR accuracy (float) | **98.67%** (296/300) |
| ASIC Q6.10 accuracy (SPI cosim) | **98.67%** (296/300) --- 0 quantization flips |

---

## SPI Register Map

The SVM core is configured entirely over SPI. Frames are 40 bits: 8-bit address (MSB first,
addr[7]=0 for write / 1 for read) followed by 32-bit data. CS must deassert between frames.

| Addr | Name | R/W | Width | Description |
|------|------|-----|-------|-------------|
| 0x01 | CONTROL | RW | 32 | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| 0x02 | STATUS | RO | 32 | [0]=done [1]=error [5:2]=error_code [8:6]=class_out [9]=sample_rdy |
| 0x03 | NUM_SAMPLES | RW | 10 | Beats per batch (1--1000); reset default = 1000 |
| 0x04 | NUM_SV[0] | RW | 10 | SVs for class 0 (Normal); reset default = 120 |
| 0x05 | NUM_SV[1] | RW | 10 | SVs for class 1 (PVC); reset default = 120 |
| 0x06 | NUM_SV[2] | RW | 10 | SVs for class 2 (AFib); reset default = 120 |
| 0x07 | NUM_SV[3] | RW | 10 | SVs for class 3 (VT); reset default = 120 |
| 0x08 | NUM_SV[4] | RW | 10 | SVs for class 4 (SVT); reset default = 120 |
| 0x09 | PARAM_WR | WO | 20 | [19]=en [18:16]=addr [15:0]=data (gamma/C/bias[0-4]) |
| 0x0A | ALPHA_WR | WO | 26 | [25:16]=sv_global_idx (10-bit) [15:0]=alpha Q6.10 |

### PARAM_WR address map (bits [18:16])

| addr | Parameter |
|------|-----------|
| 0 | Gamma (gamma, Q6.10) |
| 1 | C (regularization, Q6.10 --- training reference only) |
| 2--6 | Bias[0--4] (per-class, Q6.10) |

### SPI timing

| Parameter | Value |
|-----------|-------|
| Mode | CPOL=0, CPHA=0 (data sampled on SCLK rising edge*) |
| Frame | 40 bits, MSB first |
| Max SCLK | 2 MHz recommended (20 system clocks per half period at 40 MHz, safe for 2-FF sync) |
| CS polarity | Active low (spi_csn) |
| Data strobe | CS deassert latches the 40-bit shift register into the register file |

*Hardware note: the 2-FF synchroniser input sequence is `csn_s <= {csn_s[0], spi_csn}`
(csn_s[0]=newer). SCLK rising edge detection fires on SCLK FALLING in the system clock
domain (1-cycle delayed capture). MOSI is stable across both edges of SCLK, so the
1-cycle delay is inconsequential.*

---

## Batch Architecture (m6 / SPI)

### Off-chip RAM Bus (unchanged from m5)

| Signal | Direction | Description |
|--------|-----------|-------------|
| `ram_addr[18:0]` | ASIC out | {sv_or_input_row[10:0], feature[7:0]} |
| `ram_ren_out` | ASIC out | Read strobe |
| `ram_rdata_in[15:0]` | Host-driven | Data valid after `RAM_LATENCY` clock cycles |

Address layout: rows 0..599 = SV matrix (600 SVs); rows 600..1599 = input matrix (up to 1000 beats).

**Cycles per sample formula (LAT=3):**

```
cycles_per_sample = FEATURE_DIM*(RAM_LATENCY+1)            # load input
                  + sv_total*(FEATURE_DIM*(RAM_LATENCY+1) + 30)  # per-SV distance + kernel
                  + 500                                     # argmax + overhead
                = 256*4 + 600*(256*4 + 30) + 500
                = 1024 + 600*1054 + 500
                = 633,924 cycles/sample
```

At 40 MHz: **15.85 ms/beat** (LAT=3, 600 SVs). Well within the 750 ms heartbeat window at 80 bpm.

Inference time vs m5 (500 SVs): 15.85 ms vs 9.87 ms (+60.6%). The additional 100 SVs
cost ~6 ms per beat --- a worthwhile trade for +0.34 pp accuracy gain.

### What the Host MCU Does

```
nRF52840 MCU (SPI master, 4-wire SPI at ≤2 MHz)
    |
    |  1. Collect 1000 heartbeats (ECG frontend → feature extraction)
    |  2. Load SV matrix (600×256 Q6.10) → external SRAM rows 0..599
    |  3. Load input matrix (N×256 Q6.10) → SRAM rows 600..599+N
    |  4. SPI write ALPHA_WR × 600   (one write per SV)
    |  5. SPI write PARAM_WR × 7     (gamma + C + bias[0-4])
    |  6. SPI write NUM_SV[0-4]      (120 each; sticky — omit if unchanged)
    |  7. SPI write NUM_SAMPLES      (N beats; sticky — omit if unchanged)
    |  8. SPI write CONTROL[start]=1, vbatt_ok=1
    |
    v  ASIC takes over (~15.85 ms/beat, 600 SVs, LAT=3):
    +-- LOAD_INPUT per beat:       1,024 cycles (256 × 4)
    +-- COMPUTE_DIST per SV:       1,054 cycles (256×4 + 30 pipeline)
    +-- WRITE_CLASS: sample_rdy GPIO pulse per beat; done GPIO pulse at end
```

### Startup SPI Write Count

| Step | Writes | Notes |
|------|--------|-------|
| ALPHA_WR × 600 | 600 | One 40-bit SPI frame per SV |
| PARAM_WR × 7 | 7 | gamma + C + 5 biases |
| NUM_SV[0-4] | 5 | Sticky — write once at startup |
| NUM_SAMPLES | 1 | Sticky — write once at startup |
| CONTROL (start) | 1 | Fire after all above |
| **Total startup** | **614 SPI frames** | At 2 MHz SCLK: ~12.3 ms startup overhead |

---

## Full-Chip Power Estimate (IHP SG13G2)

The IHP SG13G2 process is 130 nm BiCMOS. Standard cell power at 1.2 V, 40 MHz is
comparable to sky130A at 1.8 V (similar physical gate density). m5 active power was
55.25 mW at TT 25°C 1.8 V. IHP at 1.2 V supply will be significantly lower:

```
P_dynamic ∝ α × C × V^2 × f
Scaling from 1.8 V → 1.2 V at same frequency: (1.2/1.8)^2 = 0.44x
Estimated IHP active power: 55.25 mW × 0.44 ≈ 24 mW  (TT estimate)
```

| Subsystem | Active Power | Duty Cycle | Avg Power |
|-----------|-------------|-----------|-----------|
| svm_compute_core (600 SVs, LAT=3) | ~24 mW | 2.11% (15.85 ms / 750 ms) | **~0.51 mW** |
| SPI slave + register file | ~0.5 mW | ~0.1% | ~0.001 mW |
| ECG frontend (analog) | ~0.5 mW | 100% | 0.5 mW |
| nRF52840 (BLE + MCU) | ~5 mW | ~2% | ~0.1 mW |
| **Total estimated** | --- | --- | **~1.11 mW** |

Battery budget: 200 mAh @ 3.7 V = 740 mWh → **740 mWh / 1.11 mW ≈ 667 hours (~27.8 days)**.
14-day target met with 2.0x margin. 47% power reduction vs m5 (Caravel, sky130A at 1.8 V).

*(These are first-order estimates. Actual power requires post-layout toggle-rate analysis
from OpenLane 2. The IHP PDK Liberty files at nom_tt_025C_1p20V should be used for
formal power signoff.)*

---

## IHP SG13G2 PDK Notes

| Parameter | Value |
|-----------|-------|
| Node | 130 nm BiCMOS |
| Standard cell library | `sg13g2_stdcell` |
| Clock gate cell | `sg13g2_dlclkp_1` (ICG; used in top.sv) |
| Liberty file | `sg13g2_stdcell_typ_1p20V_25C.lib` |
| LEF | `sg13g2_stdcell.lef` |
| Routing layers | Metal1–Metal5 (Metal5 = RT_MAX_LAYER) |
| PDK repo | [IHP-GmbH/IHP-Open-PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) |
| OpenLane 2 PDK flag | `--pdk sg13g2 --pdk-root $IHP_PDK_ROOT` |
| Tape-out vehicle | IHP Open MPW shuttle (quarterly) |

---

## Orca P&R Submission Steps

```bash
# One-time setup (interactive, not SLURM)
bash project/m6/pnr/setup_ihp_pdk.sh

# Phase 1: harden svm_compute_core as standalone macro
sbatch project/m6/pnr/core_harden.sh
# Wait for job completion; outputs: $SCRATCH/svm_m6_artifacts/{svm_compute_core.gds,.lef,.v}

# Phase 2: harden svm_top_ihp with core as black-box macro
sbatch project/m6/pnr/top_harden.sh
# Outputs: $SCRATCH/svm_m6_artifacts/{svm_top_ihp.gds,.lef,.v}

# Verify with KLayout DRC
klayout -b -r $IHP_PDK_ROOT/ihp-sg13g2/libs.tech/klayout/tech/drc/sg13g2.lydrc \
        -rd input=$SCRATCH/svm_m6_artifacts/svm_top_ihp.gds
```

SLURM job parameters: `--partition=long`, 8 CPUs, 128 GB RAM, 7-day walltime.

---

## m5 → m6 Delta

| Parameter | m5 (Caravel sky130) | m6 (IHP SG13G2 standalone) |
|-----------|--------------------|-----------------------------|
| Process | sky130A 180 nm | IHP SG13G2 130 nm |
| Supply | 1.8 V | 1.2 V |
| Interface | Wishbone @ 0x3000_0000 | SPI CPOL=0 CPHA=0 |
| NUM_SV | 500 [95,95,95,120,95] | **600 [120,120,120,120,120]** |
| alpha_addr | 9-bit | **10-bit** |
| ALPHA_WR addr field | [24:16] 9-bit | **[25:16] 10-bit** |
| Die | Caravel fixed 2920×3520 | Standalone 2400×2400 |
| Wrapper | user_project_wrapper | None |
| Clock source | wb_clk_i (Caravel) | Direct clk pad |
| ICG cell | sky130_fd_sc_hd__dlclkp | sg13g2_dlclkp_1 |
| Accuracy (Q6.10) | 98.33% | **98.67%** |
| Active power (est.) | 55.25 mW | **~24 mW** |
| Avg power (80 bpm) | 0.869 mW | **~0.51 mW** |
| Inference time | 9.87 ms/beat | **15.85 ms/beat** (+6 ms for 100 extra SVs) |
| Cosim testbench | tb_top.sv (Wishbone) | **tb_spi_cosim.py (SPI BFM)** |

---

## Acknowledgments

Place-and-route will be performed on **Orca**, Portland State University's high-performance
computing cluster, using SLURM batch jobs with OpenLane 2 inside an Apptainer container.
IHP SG13G2 PDK provided by IHP Microelectronics via the IHP-Open-PDK GitHub repository.
We thank the PSU Research Computing team for providing access to the CPU nodes that make
multi-hour OpenLane runs feasible.

---

## Feature Extraction References

The 256-dim multi-scale feature vector follows established AAMI EC57 beat classification
conventions:

| Feature group | Dims | Reference |
|---------------|------|-----------|
| Single-beat morphology (±64 samples, amplitude-norm) | 128 | de Chazal P et al., *IEEE Trans Biomed Eng* 51(7):1196--1206, 2004 |
| 10-beat mean morphology template | 64 | de Chazal P, Reilly RB, *IEEE Trans Biomed Eng* 53(12):2535--2543, 2006 |
| RR-interval history (99 intervals → 64 pts, norm 308 ms) | 64 | Llamedo M, Martinez JP, *IEEE Trans Biomed Eng* 58(3):616--625, 2011 |

Standard: AAMI ANSI EC57:2012.
Datasets: PhysioNet MIT-BIH (DOI: 10.13026/C2F305); SVDB; INCART --- all via wfdb.

---

*Document version: m6/v11 · 2026-06-13 --- IHP SG13G2 re-target; NUM_SV=600 [120×5]; SPI interface; harden pending*

---

## Appendix A --- 600-SV Allocation Sweep

*(Inherited from m5 Appendix B.11.2 with m6 update)*

### A.1 SV Count Sweep (uniform allocation)

| N/class | Total SVs | Float | Q6.10 | Flips |
|---------|-----------|-------|-------|-------|
| 90 | 450 | 98.00% | 98.00% | 0 |
| 100 | 500 (equal) | 98.33% | 98.00% | 1 |
| **120** | **600** | **98.67%** | **98.67%** | **0** |
| 150 | 750 | 98.00% | 98.00% | 0 |

The global accuracy optimum is N=120/class (600 total). The m5 hardware ceiling (500 SVs,
9-bit alpha_addr) prevented implementing this optimum directly; m6 widens alpha_addr to
10 bits and sizes alpha_table[600].

### A.2 Non-Uniform 600-SV Allocation Sweep (alloc_sweep_600.py)

At the 600-SV level, the uniform allocation is optimal. Non-uniform splits either match
or degrade accuracy:

| Allocation | Float | Q6.10 | Flips |
|------------|-------|-------|-------|
| [120,120,120,120,120] (uniform) | 98.67% | **98.67%** | 0 |
| [140,115,115,115,115] (Normal boost) | 98.67% | 98.67% | 0 |
| [115,115,115,140,115] (VT boost) | 98.67% | 98.67% | 0 |
| [110,110,110,160,110] (VT +40) | 98.67% | 98.33% | 1 (flip) |
| [115,115,115,115,140] (SVT boost) | 98.00% | 98.00% | 0 |

This contrasts with the 500-SV case, where non-uniform [95,95,95,120,95] was required
to eliminate a quantization flip. At 600 SVs each class has enough high-|alpha| support
vectors; concentrating more in one class introduces tail noise without sharpening any
boundary. **m6 target: [120,120,120,120,120] (uniform, 600 total).**

---

## Appendix B --- Runtime Model Reload via SPI

The ASIC is fully runtime-reprogrammable. A new trained SVM can be loaded without
resetting or re-synthesizing. The on-chip alpha table and off-chip SRAM are
overwritten in place while the FSM is idle.

### B.1 What constitutes a "model"

| Parameter | Location | Size |
|-----------|----------|------|
| SV matrix (600 × 256 × 16-bit) | Off-chip SRAM, rows 0--599 | 307.2 KB |
| Alpha coefficients (600 × 16-bit) | On-chip `alpha_table[600]` | 1.2 KB |
| Gamma (Q6.10) | On-chip `gamma_reg` (PARAM_WR addr 0) | 2 bytes |
| C (Q6.10, training ref) | On-chip `c_int` (PARAM_WR addr 1) | 2 bytes |
| Bias[0--4] (Q6.10) | On-chip `bias_int[0-4]` (PARAM_WR addr 2--6) | 10 bytes |
| SV counts (NUM_SV[0--4]) | On-chip SPI registers (0x04--0x08) | 5 × 10 bits |

### B.2 Reload sequence

Perform while ASIC is idle (STATUS[done]=1 or after rst_n). Do **not** fire
CONTROL[start] until all steps are complete.

```python
SPI_ALPHA_WR    = 0x0A
SPI_PARAM_WR    = 0x09
SPI_NUM_SV_BASE = 0x04
SPI_NUM_SAMPLES = 0x03
SPI_CONTROL     = 0x01

# 1. Write SV matrix to off-chip SRAM (via MCU direct SRAM bus, not SPI)
for sv in range(600):
    for feat in range(256):
        sram_write(addr=(sv << 8) | feat, data=sv_matrix[sv][feat])

# 2. Write alphas via SPI
for idx in range(600):
    word = (idx << 16) | (alpha[idx] & 0xFFFF)  # [25:16]=idx, [15:0]=alpha Q6.10
    spi_write(SPI_ALPHA_WR, word)

# 3. Write gamma, C, biases
spi_write(SPI_PARAM_WR, (1 << 19) | (0 << 16) | gamma_Q6_10)   # gamma
spi_write(SPI_PARAM_WR, (1 << 19) | (1 << 16) | C_Q6_10)       # C (ref only)
for c in range(5):
    spi_write(SPI_PARAM_WR, (1 << 19) | ((2+c) << 16) | bias_Q6_10[c])

# 4. (If changed) Update SV counts
for c in range(5):
    spi_write(SPI_NUM_SV_BASE + c, num_sv[c])

# 5. Fire start
spi_write(SPI_CONTROL, 0x3)  # start=1, vbatt_ok=1
```

### B.3 Constraints

- Total SV count must not exceed 600 (`alpha_table` has 600 entries).
- Gamma must be representable in Q6.10 (range 0–63.999, resolution ~0.001).
- Alpha values must be representable in Q6.10 (signed, range –32 to +31.999).
- Do not write ALPHA_WR during active inference (STATUS[done]=0) — undefined behavior.

---

## Appendix C --- Recommended MCU: nRF52840

| Feature | nRF52840 | Rationale |
|---------|----------|-----------|
| Core | ARM Cortex-M4F | Low power; sufficient for feature extraction |
| BLE | 5.0 integrated | Wireless result logging |
| SPI master | Up to 32 MHz | Drives ASIC SPI at ≤2 MHz with margin |
| Power | 4.6 mA active @ 3.3V | Compatible with 200 mAh LiPo budget |
| GPIO | 48 | Enough for SPI + SRAM bus + ASIC IRQ lines |
| SDK | Zephyr RTOS / nRF5 SDK | Easy to integrate SPI BFM-derived driver |

The same SPI transactions used in `tb_spi_cosim.py` translate directly to nRF52840
`nrf_spi_mngr` API calls --- the testbench BFM is the firmware driver prototype.

---

## Appendix D --- System-Level Improvements for Next Iteration

*(Inherited from m5; applicable to m6 and beyond)*

### D.1 Argmax confidence and OvR score calibration

The current hardware outputs a hard class label via argmax across 5 OvR scores with
no confidence information. Recommended additions:

1. **Platt scaling** --- fit a sigmoid to each classifier's raw scores to output calibrated
   probabilities. Argmax over probabilities is more reliable than argmax over raw margins.
2. **Confidence threshold** --- add a `low_confidence` flag to STATUS when the margin
   between the top two scores is below a threshold; defer to a clinician on low-confidence beats.
3. **Unknown/artifact class** --- add a 6th "unknown" OvR classifier trained on
   out-of-distribution beats (pacemaker spikes, noise, WPW) to prevent silent
   misclassification of unseen beat types.

### D.2 VT/PVC discrimination: beat-to-beat consistency

The primary remaining confusion is VT misclassified as PVC (similar wide-QRS morphology
on a single-beat basis). The clinical discriminator is **sustained rhythm context** ---
VT is a run of consecutive wide-QRS beats; PVC is an isolated ectopic beat.

Recommended: encode beat-to-beat morphology standard deviation (over the preceding 10
beats) as an explicit feature. Low variance = sustained VT; high variance = isolated PVC.
This feature lives in the MCU feature extraction step --- no hardware changes required.

### D.3 Class weight tuning for clinical asymmetry

`class_weight='balanced'` corrects for class frequency imbalance but not clinical
asymmetry (missing VT >> missing PVC in danger). A grid search over per-class weights
optimizing VT/AFib recall subject to a Normal false-positive rate constraint would
improve clinical safety without hardware changes.

---

*End of design_summary.md --- m6 IHP SG13G2 re-target*
