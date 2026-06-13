# Distance Pipeline — Cycle-by-Cycle at End of Feature Vector

Three registered stages in the distance accumulator:

```
Stage 1:  diff_reg  ←  x[i] − sv[i]          (subtract)
Stage 2:  sq_reg    ←  diff_reg²              (square)
Stage 3:  acc       ←  acc + sq_reg           (accumulate)
```

Each stage is clocked, so after the last feature index (i = 255) two drain
cycles are needed to flush `diff_reg` and `sq_reg` through to `acc`.

---

## Cycle Table

| Cycle | dim_idx  | diff_reg  | sq_reg       | acc update           |
|-------|----------|-----------|--------------|----------------------|
| 1     | 0        | δ[0]      | —            | —                    |
| 2     | 1        | δ[1]      | δ[0]²        | —                    |
| 3     | 2        | δ[2]      | δ[1]²        | D += δ[0]²           |
| …     | …        | …         | …            | …                    |
| 252   | 251      | δ[251]    | δ[250]²      | D += δ[249]²         |
| 253   | 252      | δ[252]    | δ[251]²      | D += δ[250]²         |
| 254   | 253      | δ[253]    | δ[252]²      | D += δ[251]²         |
| 255   | 254      | δ[254]    | δ[253]²      | D += δ[252]²         |
| 256   | **255**  | δ[255]    | δ[254]²      | D += δ[253]²         |
| 257   | drain 1  | —         | **δ[255]²**  | D += δ[254]²         |
| 258   | drain 2  | —         | —            | **D += δ[255]²  ✓**  |

D is valid after cycle 258. FSM holds `ACCUMULATE` for cycles 257–258 before
transitioning to `KERNEL`.

---

## Why 2 Drain Cycles

After dim_idx hits 255, two values are still in-flight in registered stages:

- `diff_reg` holds δ[255] — needs one cycle to become `sq_reg`
- `sq_reg` will hold δ[255]² — needs one more cycle to reach `acc`

Skipping the drain (v6 bug) caused accuracy collapse because the last two
squared differences were never accumulated into D.

---

## General Formula

| LAT | Cycles per SV |
|-----|---------------|
| 1   | 256 × 1 + 2 = **258** |
| 2   | 256 × 2 + 2 = **514** |
| 3   | 256 × 3 + 2 = **770** |

`LAT` = SRAM read latency (number of cycles to fetch one feature word).
The +2 is always the fixed pipeline drain, independent of LAT.
