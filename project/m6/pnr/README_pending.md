# m6 Place-and-Route — Pending (IHP SG13G2)

P&R for `svm_top_ihp` has not been run yet.  This directory will hold the
IHP OpenROAD flow outputs once synthesis completes.

## Expected Artifacts (after P&R)

```
pnr/
├── config.tcl           ← OpenROAD flow config (IHP PDK, 25 ns clock)
├── harden.sh            ← run script
├── timing_report.txt    ← STA: TT/FF/SS corners
├── area_report.txt      ← die size, utilization
├── power_report.txt     ← active and average power
├── drc_report.txt       ← KLayout DRC (IHP sg13g2 ruleset)
├── gds/                 ← GDS output
│   └── svm_top_ihp.gds
└── logs/                ← OpenROAD run logs
```

## IHP MPW Submission Checklist (future)

- [ ] IHP PDK installed and verified (`sg13g2_stdcell` cells resolve)
- [ ] Synthesis clean (0 unresolved instances)
- [ ] P&R complete (routing 100%, 0 DRC)
- [ ] KLayout DRC: 0 violations with IHP `sg13g2.lydrc` ruleset
- [ ] Magic LVS: CLEAN
- [ ] SPI cosim testbench passing (300 samples, ≥98.67% accuracy)
- [ ] IHP shuttle registration submitted
- [ ] GDS + design manifest uploaded to IHP shuttle portal
