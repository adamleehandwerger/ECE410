# m6 Synthesis — Pending (IHP SG13G2)

Synthesis and place-and-route for `svm_top_ihp` have not been run yet.
The m5 P&R used OpenLane 2 with sky130A. m6 targets IHP SG13G2 with OpenROAD
and the `ihp-open-pdk` PDK.

## Setup Required

```bash
git clone https://github.com/IHP-GmbH/IHP-Open-PDK.git
export PDK_ROOT=/path/to/IHP-Open-PDK
export PDK=sg13g2
```

## Expected Configuration Changes (vs m5 config.json)

| Setting | m5 (sky130) | m6 (IHP SG13G2) |
|---------|------------|-----------------|
| PDK | `sky130A` | `sg13g2` |
| Standard cell library | `sky130_fd_sc_hd` | `sg13g2_stdcell` |
| ICG cell | `sky130_fd_sc_hd__dlclkp_1` | `sg13g2_dlclkp_1` |
| Clock period | 25 ns (40 MHz) | 25 ns (40 MHz, pending STA) |
| Die size | 2500×2500 µm (Caravel fixed) | TBD — IHP standalone |
| Wrapper | Caravel `user_project_wrapper` | None — `svm_top_ihp` is the top |

## Timing Notes

IHP SG13G2 130 nm BiCMOS has similar feature size to sky130 but different cell timing.
Expect setup timing to be comparable; verify TT WNS ≥ 0 at 40 MHz before committing.
Enable CTS (`RUN_CTS=1`) to avoid hold violations in the SPI register bank.

## m5 Reference Numbers (sky130, for comparison)

| Metric | Value |
|--------|-------|
| Core die | 2500×2500 µm, 15.0% util, 157,991 cells |
| TT WNS | +3.96 ns |
| FF WNS | +11.24 ns |
| SS WNS | −14.56 ns (expected at 100°C/1.60V) |
| Active power | 55.25 mW |
| Avg power @ 80 bpm, LAT=3 | 0.869 mW |
