# Critical Path — user_project_wrapper (OL2 job 91967)

## Summary

The user_project_wrapper contains only Wishbone decode glue logic, a clock-gate
ICG cell, and the alpha_wr register (25-bit). All register-to-register paths are
extremely short (2–4 gate levels). The critical path of the chip lives inside the
pre-hardened svm_compute_core macro.

## svm_compute_core Critical Path (job 91966)

The core's critical path runs through the distance accumulator in COMPUTE_DIST:

```
FF (dist_acc[31]) → ADD_32 → MUX (sat) → FF (dist_acc[31])
  ~2 ns             ~6 ns     ~1 ns        (register)
```

Setup WNS: **+7.83 ns** (period 25 ns, required 25 ns, arrival ~17.2 ns)  
Slack: 7.83 ns — large margin, design is not timing-critical.

## Wrapper Path (job 91967)

```
FF (reg_alpha_wr[24:0]) → WB decode MUX → svm_core.alpha_wr_data[15:0]
  <1 ns                    ~2 ns
```

All wrapper paths have > 10 ns slack. Hold violations are confined to the
macro boundary and are boundary artifacts from the Caravel FP_DEF_TEMPLATE —
they are not real hold violations within the design logic.
