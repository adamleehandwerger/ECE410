# ECE410 — Milestone 5: Caravel Wrapper Hardening & Submission

**Design:** 5-class Cardiac Arrhythmia Classifier (RBF-SVM accelerator)  
**Technology:** sky130A (SkyWater 130 nm open-PDK), sky130_fd_sc_hd  
**Flow:** OpenLane 2 v2.3.10 Classic (Yosys + OpenROAD + TritonRoute)  
**Architecture:** Batch v10 — host pre-loads SV + input matrix; ASIC classifies autonomously  
**Status:** Core + wrapper hardened ✅ (jobs 92840/92861, v10, 0 DRC core, KLayout DRC 0 wrapper)

---

## Directory Structure

```
m5/
├── README.md                    ← this file — full m5 catalog
├── README_errorcodes.md         ← 13 error codes, sticky latch, reset-clear reference
├── README_mcu.md                ← MCU integration guide (batch pre-load protocol)
├── block_diagram.png            ← hardware block diagram (v10, batch architecture)
├── generate_block_diagram.py    ← renders block_diagram.png (matplotlib)
├── design_summary.md            ← full design: area, power, timing, RAM_LATENCY,
│                                    Appendix A (model reload), Appendix B (hospital design),
│                                    Appendix C (MCU task sequence)
├── design_summary.pdf           ← compiled PDF of design_summary.md
├── horner_lut_math.tex          ← LaTeX: fixed-point RBF kernel derivation
│                                    (range-reduction LUT + Horner, γ=0.25, Q6.10)
├── horner_lut_math.pdf          ← compiled PDF of horner_lut_math.tex
│
├── rt1/                         ← RTL source (v10, final)
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
├── synth/                       ← Place-and-route summary (OL2 jobs 92840 / 92861)
│   ├── config.json              ← OpenLane 2 wrapper config (25 ns clock, Caravel die)
│   ├── openlane_run.log         ← P&R run log summary (SLURM job 92861)
│   ├── timing_report.txt        ← STA: TT WNS +3.96 ns; FF +11.24 ns; SS -14.56 ns (expected)
│   ├── area_report.txt          ← core 2500×2500 µm, 15.0% util; wrapper 2920×3520 µm
│   ├── power_report.txt         ← 55.25 mW active, 0.727 mW avg @ 80 bpm (LAT=3)
│   ├── drc_report.txt           ← core 0 DRC/LVS; antenna 554 nets (advisory); wrapper boundary artifacts
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
    ├── timing_report.txt        ← STA report (jobs 92840 / 92861)
    ├── area_report.txt          ← area/utilization (jobs 92840 / 92861)
    ├── power_report.txt         ← power report (jobs 92840 / 92861)
    ├── drc_report.txt           ← DRC/LVS (core 0 viol; wrapper boundary artifacts)
    ├── gds/                     ← GDS placeholder (230 MB — lives in caravel repo)
    └── logs/                    ← SLURM job logs
        ├── core_harden_92840.out    ← core harden SLURM output (v10)
        ├── wrapper_harden_92861.out ← wrapper harden SLURM output (v10)
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
| Core setup WNS | +3.96 ns TT / +11.24 ns FF — 0 violations; −14.56 ns SS (expected) |
| Core hold WNS | +0.23 ns TT — 0 violations all corners |
| Core DRC | 0 violations ✅ |
| Wrapper Magic DRC | 11,906 boundary artifacts (acceptable) |
| Wrapper KLayout DRC | 0 violations ✅ |
| Active power | 55.25 mW → 0.727 mW avg at 80 bpm (LAT=3) |
| Core die | 2500 × 2500 µm, 15.0% utilization, 157,991 cells |
| Wrapper die | 2920 × 3520 µm (Caravel fixed), 234 MB GDS, 707 std cells |
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
- user_project_wrapper hardening (job 92861, v10) — Caravel fixed die, macro placement
- RAM_LATENCY=3 parameter — configurable wait-states for IS61WV51216 async SRAM
- RUN_MCSTA=1 — multi-corner STA (SS/FF/TT) via post-PNR OpenSTA
- svm_ram_latency_tb.sv — unit test: LAT=3 → PASS, 208 cycles/beat
- compute_core_math.tex/pdf — LaTeX mathematical description of dist + horner subfunctions
- design_summary.md/pdf — full design summary with Appendices A/B/C (model reload, hospital design, MCU sequence)
- design_rationale.md — architectural decision record (8 decisions with alternatives)
- testbench_analysis.md/pdf — 5-level testbench analysis (25 tests)
- horner_errorplot.png — degree-6 Taylor error vs Q6.10 LSB threshold
- Caravel submission artifacts (caravel/ folder)

## Outstanding (pre-submission)

- mpw-precheck — run on Orca; see `caravel/precheck/precheck_run.sh`
- Caravel chip-level DV — run `tb/dv_run.sh`
- GDS files on GitHub — resolve LFS fork restriction (create Release or detach fork)
- KLayout XOR — run locally on final 234 MB GDS
- git tag `submission-v1` in caravel_svm_project repo
- Submit caravel_svm_project URL to ECE410

## Tapeout Requirements (prototype)

Required before submitting to an Efabless Caravel shuttle or equivalent:

- **IR drop analysis (`VSRC_LOC_FILES`)** — `OpenROAD.IRDropReport` is currently
  skipped (PSM-0069, "Check connectivity failed on vccd1"). PSM requires
  `VSRC_LOC_FILES` specifying the vccd1/vssd1 supply entry points on the Caravel
  die boundary so it can trace current paths from the package into the power ring.
  The Caravel reference repository provides example VSRC files for the user project
  area. Re-enable `OpenROAD.IRDropReport` with those files and verify acceptable
  voltage droop (target: < 5% of VDD under worst-case switching activity).

- **Antenna violations** — svm_compute_core has 554 net / 808 pin antenna
  violations (advisory for class, blocking for tapeout). Re-harden with
  `GRT_REPAIR_ANTENNAS=1` and `RUN_FILL_INSERTION=1` to insert diodes and
  tie-offs. Verify Magic antenna DRC = 0 before submitting.

- **Wrapper DRC/LVS boundary artifacts** — 11,923 Magic DRC violations and 1,683
  LVS errors on the wrapper are reported as macro-boundary artifacts from the
  svm_compute_core power ring meeting the Caravel template geometry. Confirm with
  Efabless support that these are expected for a hardened macro instantiated inside
  the fixed DEF template, or resolve by adjusting the macro placement and power ring
  overlap rules in `config.json`.

- **Power BTERM handling** — vccd2/vdda1/vdda2/vssa1/vssa2/vssd2 BTERMs are
  deleted from the routing database at DetailedRouting time (DRT-0302 workaround).
  For tapeout, verify with Efabless that unconnected Caravel power domains are
  properly handled in the PDN and that deleting their routing BTERMs does not affect
  the Caravel SoC power delivery network for adjacent user projects.

- **KLayout XOR / DRC** — `KLayout.XOR` and `KLayout.DRC` are skipped on Orca
  (Ruby not available on compute nodes). XOR checks for phantom geometry between
  the GDS and the routed DEF; KLayout DRC is a second DRC signoff layer. Both
  must pass on a Ruby-capable machine (local install or a machine with KLayout +
  Ruby) before submitting to a shuttle. Run `klayout -b -r $PDK_ROOT/.../drc.lydrc
  -rd input=user_project_wrapper.gds` and `klayout -b -r xor.lydrc` on the final
  GDS produced by Magic.

- **Wrapper hold violations (TT corner)** — `Checker.HoldViolations` reports hold
  violations at nom_tt_025C_1v80 in the wrapper (checker skipped to allow flow
  completion). svm_compute_core hold is clean (+0.23 ns). Wrapper violations likely
  originate in the Wishbone controller registers synthesized without clock tree
  (`RUN_CTS=0`). For tapeout: enable `RUN_CTS=1`, re-run wrapper harden, and verify
  hold WNS ≥ 0 in all corners. Hold failures cause functional errors on silicon
  (data captured too early on the wrong clock edge).

- **SS corner timing** — WNS = −14.56 ns at nom_ss_100C_1v60 (163 violations).
  This corner is 100 °C / 1.60 V; if the prototype will only operate at room
  temperature with nominal supply this is not a risk. Document the operating
  envelope explicitly for any silicon bringup plan.

## Next Step

MCU design — nRF52840 recommended for ECG sampling, feature extraction, and
Wishbone/GPIO control of the ASIC. See `design_summary.md` Appendix C for
the full MCU task sequence.
