# ECE410 — Milestone 5: Caravel Wrapper Hardening & Submission

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd  
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys + OpenROAD + TritonRoute)  
**Architecture:** Batch v9 — host pre-loads SV + input matrix; ASIC classifies autonomously  
**Status:** Core hardened (job 91966, 0 DRC), wrapper hardened (job 91967, boundary DRC acceptable)

---

## Directory Structure

```
m5/
├── README.md                    ← this file — full m5 catalog
├── README_errorcodes.md         ← 13 error codes, sticky latch, reset-clear reference
├── README_mcu.md                ← MCU integration guide (batch pre-load protocol)
├── block_diagram.png            ← hardware block diagram (v9, batch architecture)
├── generate_block_diagram.py    ← renders block_diagram.png (matplotlib)
├── design_summary.md            ← full design: area, power, timing, RAM_LATENCY,
│                                    Appendix A (model reload), Appendix B (hospital design)
├── design_summary.pdf           ← compiled PDF of design_summary.md
├── horner_lut_math.tex          ← LaTeX: fixed-point RBF kernel derivation
│                                    (range-reduction LUT + Horner, γ=0.25, Q6.10)
├── horner_lut_math.pdf          ← compiled PDF of horner_lut_math.tex
│
├── rt1/                         ← RTL source (v9, final)
│   ├── compute_core.sv          ← SVM core: NUM_SV=500, RAM_LATENCY param, batch FSM
│   ├── top.sv                   ← Caravel wrapper: Wishbone decode, clock gate,
│   │                                reg_alpha_wr[24:0], GPIO/LA pin assignments
│   └── interface.sv             ← SystemVerilog interface definitions (svm_data_if,
│                                    svm_ctrl_if) used by the compute core
│
├── tb/                          ← Testbenches and verification
│   ├── README.md                ← testbench overview and how to run
│   ├── Makefile                 ← `make sim` runs Wishbone cocotb cosim
│   ├── tb_wb_cosim.py           ← cocotb testbench: full 300-sample Wishbone cosim
│   ├── svm_ram_latency_tb.sv    ← unit test: RAM_LATENCY parameter (LAT=3 → PASS,
│   │                                208 cycles/beat; FEAT=4, NSV=5, iverilog)
│   ├── sky130_stubs.v           ← sky130 cell stubs for Icarus simulation
│   ├── confusion_comparison_m5.py ← generates confusion matrix comparison plot
│   ├── testbench_summary.md     ← summary of all m5 testbenches and pass/fail results
│   └── dv_run.sh                ← Caravel DV RTL simulation run script
│
├── sim/                         ← Simulation outputs
│   ├── final_run.log            ← Wishbone cosim log (300 samples, 97.67% accuracy)
│   ├── final_waveform.png       ← timing diagram: wb_stb, ram_ren, sample_rdy,
│   │                                STATUS.done, class bus — 5 representative beats
│   ├── confusion_comparison_m5.png ← sklearn vs ASIC confusion matrix comparison
│   ├── asic_preds.csv           ← 300 ASIC predictions (last cosim run)
│   └── throughput_comparison.txt ← inference time and power summary
│
├── synth/                       ← Place-and-route summary (OL2 jobs 91966 / 91967)
│   ├── config.json              ← OpenLane 2 wrapper config (25 ns clock, Caravel die)
│   ├── openlane_run.log         ← P&R run log summary (SLURM job 91967)
│   ├── timing_report.txt        ← STA: core WNS +7.83 ns, 0 violations
│   ├── area_report.txt          ← core 2500×2500 µm; wrapper 2920×3520 µm (Caravel fixed)
│   ├── power_report.txt         ← 66 mW active, 0.284 mW avg @ 80 bpm; 14-day target met
│   ├── drc_report.txt           ← core 0 violations; wrapper 11,923 boundary artifacts
│   └── critical_path.md         ← critical path through dist_acc; wrapper paths trivial
│
├── bench/                       ← Benchmark: ASIC vs optimized Python
│   ├── benchmark.md             ← accuracy, throughput, power, energy efficiency tables
│   ├── benchmark_data.csv       ← raw measurements (ASIC measured; CPU estimated)
│   ├── roofline_final.png       ← dual-panel roofline + power-efficiency chart
│   └── roofline_final.py        ← script that generates roofline_final.png (matplotlib)
│
├── caravel/                     ← Caravel chipIgnite submission artifacts
│   ├── README_caravel.md        ← Caravel submission overview and repo layout
│   ├── README_submission.md     ← submission requirements and status checklist
│   ├── checklist.md             ← item-by-item submission checklist (all tracked)
│   └── precheck/                ← Efabless mpw-precheck
│       ├── precheck_run.sh      ← SLURM script to run precheck on Orca
│       └── precheck_results.txt ← results (pending wrapper GDS precheck run)
│
├── report/                      ← Final project report
│   ├── final_report.md          ← 10-section design justification report (markdown)
│   ├── final_report.pdf         ← compiled PDF (pandoc + xelatex)
│   ├── design_justification.pdf ← copy of final_report.pdf for Caravel submission
│   └── figures/                 ← embedded report figures
│       ├── fig_A1_block_diagram.png  ← hardware block diagram (referenced §4.1)
│       ├── fig_A2_confusion_matrix.png ← confusion matrix (referenced §8.1)
│       └── fig_A3_roofline.png       ← roofline chart (referenced §5.2, §8.2)
│
└── pnr/                         ← Full P&R artifacts (scripts, configs, GDS, logs)
    ├── wrapper_config.json      ← OL2 config for user_project_wrapper
    ├── wrapper_harden.sh        ← SLURM script: hardens user_project_wrapper on Orca
    ├── base_user_project_wrapper.sdc ← timing constraints (wrapper)
    ├── macro.cfg                ← u_svm macro placement (253, 554) N
    ├── timing_report.txt        ← STA report (jobs 91966 / 91967)
    ├── area_report.txt          ← area/utilization (jobs 91966 / 91967)
    ├── power_report.txt         ← power report (jobs 91966 / 91967)
    ├── drc_report.txt           ← DRC/LVS (core 0 viol; wrapper boundary artifacts)
    ├── gds/                     ← GDS placeholder (230 MB — lives in caravel repo)
    └── logs/                    ← SLURM job logs
        ├── core_harden_91966.out    ← core harden SLURM output
        ├── wrapper_harden_91963.out ← wrapper harden SLURM output
        ├── mpw_precheck_91986.err   ← mpw-precheck stderr
        └── mpw_precheck_91986.out   ← mpw-precheck stdout
```

---

## Key Design Parameters

| Parameter | Value |
|-----------|-------|
| Feature dimension | 256 (128 single-beat + 64 10-beat + 64 RR history) |
| Support vectors | 500 total (100 per class, 5 classes) |
| Fixed-point | Q6.10, 16-bit signed |
| Gamma / C | 0.25 / 1.0 |
| Clock | 40 MHz (25 ns) |
| RAM_LATENCY | 1 (cosim default) / 3 (IS61WV51216 async SRAM) |
| Core setup WNS | +7.83 ns (TT, 0 violations) |
| Core hold WNS | +0.30 ns (TT, 0 violations) |
| Core DRC | 0 violations |
| Wrapper DRC | 11,923 boundary artifacts (acceptable) |
| Active power | 66 mW → 0.869 mW avg at 80 bpm (1.316% duty cycle, LAT=3) |
| Core die | 2500 × 2500 µm, ~14% utilization, ~146K cells |
| Wrapper die | 2920 × 3520 µm (Caravel fixed), 230 MB GDS |
| ASIC accuracy | 97.67% (293/300) — exact match with sklearn, zero gap |

## Quick Start

```bash
# Full 300-sample Wishbone cosim (~96 min)
cd tb && PYTHONUNBUFFERED=1 make sim

# Quick subset (25 samples)
cd tb && COSIM_N_EVAL=25 COSIM_GAMMA=0.25 PYTHONUNBUFFERED=1 make sim

# RAM_LATENCY unit test (iverilog standalone, <1 s)
cd tb
iverilog -g2012 -DSIMULATION -o /tmp/svm_lat_tb.out \
    ../rt1/compute_core.sv svm_ram_latency_tb.sv
/tmp/svm_lat_tb.out

# Regenerate block diagram
python3 generate_block_diagram.py

# Regenerate confusion matrix
cd tb && python3 confusion_comparison_m5.py
```

Requires: `pip install cocotb scikit-learn wfdb matplotlib numpy`  
PhysioNet cache: `~/.physionet_cache/`  
NumPy cache: `/tmp/cosim_cache_ecg_n300_d256.npz`

## Caravel Repo (`caravel_svm_project`)

Physical artifacts (GDS, LEF, GL netlist) live in the separate Caravel repo at  
`https://github.com/adamleehandwerger/caravel_svm_project`.  
See `caravel/README_caravel.md` for the full repo layout.

## Differences from m4

m4 hardened the svm_compute_core macro only. m5 adds:
- user_project_wrapper hardening (job 91967) — Caravel fixed die, macro placement
- RAM_LATENCY parameter — configurable wait-states for IS61WV51216 async SRAM
- svm_ram_latency_tb.sv — unit test: LAT=3 → PASS, 208 cycles/beat
- horner_lut_math.tex/pdf — fixed-point RBF kernel derivation document
- Caravel submission artifacts (caravel/ folder)
- report/ folder (final project report, to be filled)

## Next Step

MCU design — seeking clinical input from Dr. Eric Stecker (OHSU Knight  
Cardiovascular / Insight Health AI) on arrhythmia patterns and temporal window  
requirements for the batch classification interface.
