# ECE410 — Milestone 5: Wrapper Hardening, Efabless Precheck & Final Submission

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys 0.46 + OpenROAD + TritonRoute)
**RTL:** Frozen at m4 — see `../m4/rt1/` for source
**Status:** user_project_wrapper hardening running (SLURM job 91948, long partition)

The m5 milestone completes the Caravel chipIgnite submission package:
1. **user_project_wrapper hardening** — full-chip GDS/LEF/GL with svm_compute_core macro
2. **Efabless mpw-precheck** — official design rule and LVS checks
3. **Final submission** — caravel_svm_project repo in submission-ready state

---

## Milestone Status

| Task | Status |
|------|--------|
| svm_compute_core GDS/LEF/GL | ✅ Complete (m4, job 91947) |
| user_project_wrapper hardening | 🔄 Running (job 91948, orcaga11) |
| Wrapper GDS committed | ⏳ Pending job 91948 |
| Efabless mpw-precheck | ⏳ Pending wrapper GDS |
| Final repo submission | ⏳ Pending precheck pass |

---

## Directory Structure

```
m5/
├── README.md                  ← this file
├── design_summary.md          ← full-chip design summary and submission package
├── pnr/                       ← wrapper P&R scripts, configs, and reports
│   ├── wrapper_config.json    ← OL2 config for user_project_wrapper (job 91948)
│   │                              die 2920×3520 µm, FP_DEF_TEMPLATE, RT_MAX_LAYER=met4
│   ├── wrapper_harden.sh      ← SLURM script: full OL2 flow on Orca (long partition)
│   ├── base_user_project_wrapper.sdc ← timing constraints for wrapper
│   ├── macro.cfg              ← svm_compute_core macro at (253, 554) N
│   ├── area_report.txt        ← wrapper area, cell count, utilization (post-job 91948)
│   ├── timing_report.txt      ← wrapper timing: setup/hold WNS, violations
│   ├── power_report.txt       ← full-chip power estimate
│   ├── drc_report.txt         ← wrapper DRC violations
│   └── gds/
│       └── user_project_wrapper.gds  ← full-chip GDS (pending, ~1-2 GB expected)
├── precheck/
│   ├── precheck_run.sh        ← script to run efabless mpw-precheck on Orca
│   └── precheck_results.txt   ← mpw-precheck output (pending)
├── submission/
│   └── checklist.md           ← Caravel submission checklist
└── sim/
    └── (DV simulation logs, waveforms — pending dv_run.sh)
```

---

## Caravel Wrapper Architecture

```
user_project_wrapper (2920×3520 µm fixed die)
├── u_svm : svm_compute_core macro (2500×2500 µm) at (253, 554) N
│   ├── input_fifo          256-word × 16-bit FIFO (register-based)
│   ├── feature_bank        256 × 16-bit registers
│   ├── distance_engine     RBF kernel: Horner LUT + Q6.10 fixed-point
│   ├── argmax              5-class winner-take-all
│   └── FSM                 WARMUP → RECEIVE → PROCESS → WRITE_CLASS
├── work_ram                2048 × 16-bit (register array, Wishbone-accessible)
├── clock gate              sky130_fd_sc_hd__dlclkp_1 ICG
└── Wishbone decoder        base 0x30000000
```

## GPIO / LA Pin Assignments

| Signal | Direction | Assignment |
|--------|-----------|------------|
| `class_out[2:0]` | out | GPIO[2:0] |
| `done` | out | GPIO[3] |
| `error` | out | GPIO[4] |
| `error_code[3:0]` | out | GPIO[8:5] |
| `fifo_ready` | out | GPIO[9] |
| `sv_ram_addr[14:0]` | out | GPIO[24:10] |
| `sv_ram_ren` | out | GPIO[25] |
| `sv_ram_rdata[15:0]` | in | LA[15:0] |
| `done` (IRQ) | out | user_irq[0] |

## Running on Orca

```bash
# Wrapper hardening (job 91948 already running):
sbatch ~/ece410/wrapper_harden.sh

# After wrapper completes, run precheck:
sbatch ~/ece410/precheck_run.sh

# DV regression test (after dv_setup.sh):
sbatch ~/ece410/dv_run.sh
```

## Differences from m4

| | m4 | m5 |
|--|----|----|
| Scope | svm_compute_core hardened | user_project_wrapper hardened |
| GDS | core only (181 MB) | full chip (expected ~1–2 GB) |
| Precheck | Not run | mpw-precheck pass required |
| Submission | Artifacts staged | Repo in submission-ready state |
| DV | Unit tests + cocotb | + Caravel chip-level RISC-V DV |
