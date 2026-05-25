# ECE410 — Milestone 5: Wrapper Hardening & Final Submission (Batch v8)

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**Architecture:** Batch v8 — host pre-loads SV + input matrix; ASIC classifies autonomously
**RTL:** m4/rt1 (batch v8 — frozen for hardening)
**Status:** New DRT in progress for svm_compute_core (batch v8); wrapper DRT follows

The m5 milestone completes the Caravel chipIgnite submission package:
1. **svm_compute_core re-harden** — updated batch v8 RTL (removed FIFO + work_ram)
2. **user_project_wrapper hardening** — full-chip GDS/LEF/GL with updated macro
3. **Efabless mpw-precheck** — design rule and LVS checks
4. **Final submission** — caravel_svm_project repo in submission-ready state

---

## Milestone Status

| Task | Status |
|------|--------|
| Batch v8 RTL (svm_compute_core) | ✅ Complete — m4/rt1/ |
| Batch v8 RTL (user_project_wrapper) | ✅ Complete — m4/rt1/ |
| cosim (batch protocol) | 🔄 Running |
| svm_compute_core re-harden (DRT) | 🔄 New SLURM job submitted |
| user_project_wrapper hardening | ⏳ After core DRT completes |
| Wrapper GDS committed | ⏳ Pending wrapper DRT |
| Efabless mpw-precheck | ⏳ Pending wrapper GDS |
| Final repo submission | ⏳ Pending precheck pass |

---

## Batch Architecture (v8)

The key architectural change from v7 (streaming) to v8 (batch):

| | v7 | v8 |
|--|----|----|
| Input path | Stream 256 words per beat via WB FIFO | Pre-load input matrix in off-chip SRAM |
| SV path | GPIO[25:10] 15-bit sv_ram_addr | GPIO[28:10] 19-bit unified ram_addr |
| Per-beat result | Poll work_ram after `done` | `sample_rdy` IRQ[0] per beat |
| Batch done | IRQ[0] | IRQ[1] |
| Removed | — | FIFO (512 regs), work_ram (64 regs), FIFO_DATA/WORK_RD/STATUS2 WB regs |

Off-chip RAM address: `{row[10:0], col[7:0]}` = 19-bit.
Rows 0..249 = SV matrix. Rows 250..1249 = input matrix.

---

## GPIO / LA Pin Assignments (v8)

| Signal | Direction | Pin |
|--------|-----------|-----|
| `class_out[2:0]` | out | GPIO[2:0] |
| `sample_rdy` | out | GPIO[3] / IRQ[0] |
| `svm_done` | out | GPIO[4] / IRQ[1] |
| `svm_error` | out | GPIO[5] |
| `error_code[3:0]` | out | GPIO[9:6] |
| `ram_addr[18:0]` | out | GPIO[28:10] |
| `ram_ren` | out | GPIO[29] |
| `ram_rdata[15:0]` | in | LA[15:0] (host-driven) |

---

## Caravel Wrapper Architecture (v8)

```
user_project_wrapper (2920 × 3520 µm fixed die)
├── u_svm : svm_compute_core macro (2500 × 2500 µm) at (253, 554) N
│   ├── feature_bank        256 × 16-bit registers (LOAD_INPUT → COMPUTE_DIST)
│   ├── distance_engine     RBF: Σ(xᵢ−svᵢ)² accumulator
│   ├── horner_engine       exp(−γd²) Horner LUT approximation
│   ├── argmax              5-class winner-take-all
│   └── FSM                 IDLE → LOAD_INPUT → COMPUTE_DIST → COMPUTE_KERNEL
│                                → OUTPUT_RESULT → WRITE_CLASS → (loop)
├── batch_active reg        Keeps ICG open for entire ~70M-cycle batch
├── clock gate              sky130_fd_sc_hd__dlclkp_1 ICG
└── Wishbone decoder        base 0x3000_0000 (5 registers)
```

---

## Wishbone Register Map (v8)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| +0x04 | CONTROL | RW | [0]=start [1]=vbatt_ok [2]=vbatt_warn |
| +0x08 | STATUS | RO | [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy |
| +0x0C | NUM_SAMPLES | RW | [9:0] beats in batch |
| +0x10–+0x20 | NUM_SV[0–4] | RW | [7:0] SVs per class |
| +0x24 | PARAM_WR | WO | [19]=en [18:16]=addr [15:0]=data |

---

## Directory Structure

```
m5/
├── README.md                  ← this file
├── README_submission.md       ← Efabless submission requirements
├── design_summary.md          ← full-chip design summary (batch v8)
├── block_diagram.png          ← architecture block diagram
├── generate_block_diagram.py  ← renders block_diagram.png
├── rt1/                       ← RTL snapshot for hardening (= m4/rt1/)
│   ├── svm_compute_core.sv    ← batch v8 compute core
│   └── user_project_wrapper.sv ← batch v8 wrapper
├── pnr/                       ← wrapper P&R scripts and configs
│   ├── wrapper_config.json    ← OL2 config: 2920×3520 µm, RT_MAX_LAYER=met4
│   ├── wrapper_harden.sh      ← SLURM: full OL2 flow on Orca (long partition)
│   ├── base_user_project_wrapper.sdc ← timing constraints
│   ├── macro.cfg              ← u_svm at (253, 554) N
│   ├── area_report.txt        ← pending new DRT
│   ├── timing_report.txt      ← pending new DRT
│   ├── power_report.txt       ← pending new DRT
│   └── drc_report.txt         ← pending new DRT
├── precheck/
│   ├── precheck_run.sh        ← Efabless mpw-precheck on Orca
│   └── precheck_results.txt   ← pending
├── submission/
│   └── checklist.md           ← Caravel submission checklist
└── sim/
    ├── Makefile               ← cocotb sim (RTL_DIR → m5/rt1/)
    ├── tb_wb_cosim.py         ← batch protocol testbench (v8)
    ├── sky130_stubs.v         ← ICG behavioral stub
    ├── confusion_comparison_m5.py ← sklearn vs. ASIC confusion matrix
    └── README.md
```

---

## Running on Orca

```bash
# Step 1: Re-harden svm_compute_core with batch v8 RTL
sbatch ~/ece410/core_harden.sh

# Step 2: After core GDS ready, harden wrapper
sbatch ~/ece410/wrapper_harden.sh

# Step 3: Run Efabless precheck
sbatch ~/ece410/precheck_run.sh
```
