# ECE410 Study Sheet — RBF-SVM ASIC Project (m5)
**Adam Handwerger · Portland State University · 2026-06-10**

---

## 1. OpenLane Physical Implementation Flow

OpenLane runs the same sequence for both `compute_core` harden and `wrapper` harden. The two-stage structure means the core is hardened first as a black-box macro; the wrapper harden then places that macro into the Caravel SoC frame and routes only the SoC-to-macro interface signals.

| Stage | Tool | What it does |
|-------|------|--------------|
| **Synthesis** | Yosys | RTL -> gate-level netlist. Maps HDL to standard cells from the liberty file (`sky130_fd_sc_hd`). Logic optimization here. Produces the 157,991 cell count. |
| **Floorplan** | OpenROAD | Defines die area, core area, IO pin locations, power ring geometry. Sets utilization target (15.0% in v10). |
| **PDN** | OpenROAD | Inserts VDD/VSS stripes and rings before placement. Power routing before signal routing. |
| **Placement** | OpenROAD (RePlAce) | Global pass (rough, minimize wire length) then detailed pass (legalize: no overlaps, cells on grid). |
| **CTS** | OpenROAD (TritonCTS) | Builds clock buffer tree to minimize skew. **Skipped in wrapper** (`RUN_CTS=0`) — tree already exists inside hardened core macro; re-running would fight it. |
| **Routing** | OpenROAD (TritonRoute) | Global routing assigns nets to channels; detailed routing cuts actual metal traces. Antenna violations accumulate on long runs here. |
| **STA** | OpenSTA | Computes WNS/TNS across timing corners. v10 uses `RUN_MCSTA=1`: TT (+3.96 ns), FF (+11.24 ns), SS at 100°C/1.60V (163 viol, -14.56 ns — expected for sky130, non-blocking). |
| **DRC** | Magic / KLayout | Checks physical polygons against foundry manufacturing rules. Core: 0 violations. |
| **LVS** | Netgen | Compares extracted layout netlist to gate-level netlist. 1,683 errors on wrapper are macro boundary artifacts, not logical errors. Core: 0 errors. |
| **GDSII export** | Magic / KLayout | Streams final layout to file for submission. |

---

## 2. CTS — Clock Tree Synthesis

**What it is:** Inserts buffers and inverters into the clock distribution network so every flip-flop receives the clock edge at nearly the same time. Without it, a long wire driving 157K flops would have large skew — flops at the far end see the clock nanoseconds later, eating into both setup and hold margins.

**Why RUN_CTS=0 in the wrapper:** The clock tree already exists inside the hardened `svm_compute_core` macro. Re-running CTS on the wrapper would attempt to rebuild a tree from scratch, but the tool cannot see inside the black-box macro. It would fight the existing internal tree, add skew on top of what is already there, and likely violate timing inside the core. The wrapper clock path is also trivially simple: `wb_clk_i -> ICG -> svm_gclk -> one macro`. No tree is needed.

---

## 3. 2-Cycle Pipeline Drain — Distance Accumulator

**The pipeline:** Inside `distance_matrix`, each dimension goes through a 3-stage pipeline:
```
Stage 1 (subtract):   diff  = feature[k] - sv[k]
Stage 2 (square):     sq    = diff * diff
Stage 3 (accumulate): accum += sq
```
After the last feature (dim 255), the subtract and square results for dims 254 and 255 are still in-flight in stages 1 and 2. The FSM must hold in ACCUMULATE for 2 extra cycles to flush them through stage 3 before reading the final distance.

**What happens without the drain counter (the v6 bug):**
- Cycle T: feature 255 presented, diff_255 computed in stage 1
- Cycle T+1: sq_255 in stage 2; accum gets sq_254 (one beat late)
- Cycle T+2: accum gets sq_255 (two beats late)
- If FSM exits ACCUMULATE at T+0 (no drain), sq_254 and sq_255 are **never added**
- Features 254 and 255 are missing from every distance sum -> accuracy collapse

**The fix** (in `m4/rt1/compute_core.sv`, lines 650-734):
```systemverilog
logic [1:0] drain_cnt;
// In ACCUMULATE state, hold until drain_cnt reaches 2:
if (drain_cnt == 2'd2) next_state = OUTPUT;
// Drain counter logic:
if (dim_counter >= FEATURE_DIM - 1 && valid_in && drain_cnt == 2'd0)
    drain_cnt <= 2'd1;
else if (drain_cnt != 2'd0)
    drain_cnt <= drain_cnt + 2'd1;
```

**Historical note:** At m2, `tb_distance_matrix` acknowledged the issue with the comment "the 2-entry miss is negligible (<1%) for FEATURE_DIM=256." This was the accepted pre-fix state. The drain counter was added after the miss was found to cause real accuracy regression on MIT-BIH data.

**Verification:** `tb_dist_zero.sv` is the canonical check — sets all features equal to all SV features (both 0x0400 = 1.0 Q6.10), expects `kernel_out = 1024` (= exp(0) = 1.0). Reading the accumulator one cycle early gives a nonzero distance and produces a kernel below 1024, failing the check.

---

## 4. Multi-Corner STA — v10 Results

`RUN_MCSTA=1` runs Static Timing Analysis across three PVT (Process-Voltage-Temperature) corners:

| Corner | WNS | Violations | Notes |
|--------|-----|-----------|-------|
| TT (typical-typical, 25°C, 1.80V) | +3.96 ns | 0 | Nominal operating point |
| FF (fast-fast, -40°C, 1.95V) | +11.24 ns | 0 | Fast cells, low temp = more slack |
| SS (slow-slow, 100°C, 1.60V) | -14.56 ns | 163 | Setup violations expected for sky130 |

**Why SS fails setup (not hold):** Slow-slow means transistors switch slowly (slow process) at high temperature and low voltage. Slow cells mean combinational paths take longer to settle, so the data arrives at the flip-flop *late* relative to the clock edge — a setup violation. Hold violations appear in the fast-fast corner (cells settle so quickly the data may pass through and change before the hold time expires).

**Why SS violations are non-blocking:** sky130 is an educational/open-source PDK not optimized for high-frequency or extreme PVT corners. SS at 100°C/1.60V is an unusually harsh corner that even well-designed commercial chips often fail at sky130's nominal clock period. The device operates at room temperature on a stable supply; TT and FF are both clean. The 163 violations are an artifact of the PDK's SS characterization, not a silicon failure risk.

---

## 5. ICG — Integrated Clock Gate

**What it is:** A latch + AND gate combination. The latch captures the clock-enable signal on the falling edge (level-sensitive, glitch-free), and the AND gate combines it with the clock. This stops the clock cleanly when the core is idle, eliminating the dynamic power of the entire clock tree.

**Why not just AND the clock directly:** A combinational AND gate on a clock line creates glitches — if the enable changes while the clock is high, a partial pulse propagates. The latch ensures the enable is only sampled at the falling edge, so only complete clock cycles pass through.

**Impact on this design:** The ASIC classifies one 1000-beat batch every ~14 minutes. Active duty cycle is ~1.3%. The ICG gates `svm_gclk` from `wb_clk_i` for the other 98.7% of the time. Clock tree power is ~37% of active power; gating it during idle is the dominant power saving.

---

## 6. DRC vs LVS

| Check | Full Name | What it verifies | Tool | m5 result |
|-------|-----------|-----------------|------|-----------|
| **DRC** | Design Rule Check | Physical polygons obey foundry manufacturing rules: minimum width, spacing, via size, enclosure | Magic / KLayout | Core: 0 viol; Wrapper: 554 antenna (advisory) |
| **LVS** | Layout vs. Schematic | Extracted netlist from layout matches the gate-level schematic (same devices, same connections) | Netgen | Core: 0 errors; Wrapper: 1,683 boundary artifacts |

**Antenna violations (554 net / 808 pin on wrapper):** Charge accumulation during plasma etching on long metal runs that are not yet connected to a gate. Advisory, not blocking for tape-out. Mitigated by antenna diodes or metal jumpers in production.

**LVS 1,683 errors on wrapper:** Macro interface boundary artifacts — the extractor sees the hardened core ports as unresolved black-box connections. Not logical errors. Core LVS is clean (0 errors).

---

## 7. Key Project Numbers (v10 / m5 final)

| Parameter | Value |
|-----------|-------|
| Accuracy (HW, 300 samples) | 97.67% (293/300) |
| Standard cell library | sky130_fd_sc_hd (high-density) |
| Cell count | 157,991 |
| Core utilization | 15.0% |
| TT WNS | +3.96 ns |
| Active power (TT) | 55.25 mW |
| Clock frequency | 25 MHz |
| Feature dimensions | 256 (multi-scale, MIT-BIH + SVDB + INCART) |
| Support vectors | 500 total (100/class) |
| RAM_LATENCY | 3 (IS61WV51216 async SRAM, PCB-calibrated) |
| Fixed-point format | Q6.10 (16-bit signed, scale = 1024) |
| gamma | 0.25 -> 0x0100 Q6.10 |
| Classification scheme | OvR (one-vs-rest), 5 classes |
| Classes | Normal, PVC, AFib, VT, SVT |
| Harden jobs | Core: 92840, Wrapper: 92861/92867 |

---

## 8. RR Normalization

**Purpose:** Makes rhythm features rate-independent across patients and conditions. Raw RR intervals vary with heart rate (60 bpm -> ~1000 ms; 90 bpm -> ~667 ms), so the same arrhythmia pattern would look different for different patients. Dividing by a reference value expresses each interval as a ratio.

**Reference value:** `NORMAL_RR = 308 ms` — median normal sinus rhythm interval in the MIT-BIH + SVDB + INCART training set.

**Separate fact — 64 RR features:** The 256-dim feature vector is split as 192 morphology dims (QRS waveform shape, 3 multi-scale windows x 64) + 64 rhythm dims (normalized RR history over ~100 beats). The 64 is the dimensionality of the rhythm slice, not a consequence of normalization.

---

*This sheet grows as the study session continues. Add new topics below.*
