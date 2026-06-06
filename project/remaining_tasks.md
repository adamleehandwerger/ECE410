# Next Steps — SVM Cardiac Classifier ASIC

*Adam Handwerger · ECE410 · Portland State University · 2026-05-29*

---

## 1. Caravel Submission Checklist

### 1.1 Artifacts Already Complete

| Artifact | Job | Status |
|---|---|---|
| `svm_compute_core.gds` | 91966 | Done — 226 MB |
| `user_project_wrapper.gds` | 91967 | Done — 230 MB |
| `svm_compute_core.lef` | 91966 | Done — 94 KB |
| `user_project_wrapper.lef` | 91967 | Done — 195 KB |
| GL netlist | 91967 | Done |
| DRC violations | — | **0** |
| Timing violations | — | **0** (WNS +7.83 ns) |
| Accuracy vs. float baseline | — | **0 gap** (97.67%) |

### 1.2 Remaining Submission Tasks

- [ ] **Verify Wishbone register map** conforms to Caravel spec — base address `0x3000_0000`, no address collisions with management SoC
- [ ] **Confirm `io_oeb` direction bits** are set correctly for all user GPIO in `user_project_wrapper.v`
- [ ] **Submit precheck** — run Efabless `mpw-precheck` locally against `user_project_wrapper.gds` to catch any last DRC/LVS issues before uploading
- [ ] **Upload to Efabless** — submit GDS + LEF + GL netlist via the Efabless project submission portal at [efabless.com](https://efabless.com)
- [ ] **Tag the caravel repo** — `git tag submission-v1` and push so the submitted commit is pinned
- [ ] **Confirm license** — Apache 2.0 in `caravel_svm_project` repo (required for open shuttle acceptance)
- [ ] **Re-harden with RAM_LATENCY=3 (blocking)** — job 91966 was synthesized with default RAM_LATENCY=1; silicon will fail with IS61WV51216 SRAM. Fix already applied to `caravel/openlane/svm_compute_core/config.json` (`SYNTH_TOP_LEVEL_PARAMETERS: RAM_LATENCY=3`). Re-run core + wrapper harden on Orca (~3–4 hrs). Then re-run mpw-precheck and DV.
- [ ] **SS corner timing signoff** — re-run OpenSTA at SS/1.62V/125°C; TT corner (+7.83 ns WNS) is the only corner verified so far

### 1.3 Next Shuttle

Efabless chipIgnite shuttles run **approximately quarterly**. As of spring 2026 the next available window should be **mid to late summer 2026**, but dates shift. Check the current schedule at:

> **efabless.com/open_shuttle_program**

Key lead times to account for:
- Precheck + submission review: ~1–2 weeks
- Fab turnaround (sky130A MPW): ~3–4 months after shuttle close
- Parts back in hand: **~Q4 2026 / Q1 2027** if you make the next summer shuttle

---

## 2. Next Phase — MCU Design

### 2.1 Role of the MCU

The ASIC classifies — it does not acquire. The MCU must:

1. **Sample ECG** — interface to an analog front-end (AFE) such as the ADS1299 or MAX30003
2. **Extract features** — compute the 256-dimensional feature vector (1-beat morphology, 10-beat mean template, 64-dim RR history) in real time
3. **Load SRAM** — write the 256 Q6.10 features into external SRAM via SPI or direct bus before triggering the ASIC
4. **Trigger and read** — assert `CONTROL.start` over Wishbone (or GPIO), wait for `STATUS.done`, read back the 3-bit class label
5. **Log / alert** — store classifications, timestamp arrhythmia events, alert via BLE if needed

### 2.2 Off-the-Shelf MCU Options

A custom MCU is not necessary for an initial prototype. Several commercial parts satisfy the requirements:

| MCU | Core | Clock | Flash | SRAM | BLE | Notes |
|---|---|---|---|---|---|---|
| **nRF52840** | Cortex-M4F | 64 MHz | 1 MB | 256 KB | Yes | Best for wearable BLE; FPU for feature math |
| **STM32L4+** | Cortex-M4F | 120 MHz | 2 MB | 640 KB | No | Low power, good DSP, add BLE module |
| **ESP32-S3** | Xtensa LX7 | 240 MHz | — | 512 KB | Yes | Fastest; higher idle power |
| **RP2040** | Cortex-M0+ | 133 MHz | — | 264 KB | No | Cheapest; no FPU, feature math in SW |

**Recommendation:** nRF52840 (e.g., Nordic DK or Adafruit Feather nRF52840) — integrated BLE, FPU for fixed-point feature extraction, proven low-power SDK, and widely used in medical wearable prototypes.

### 2.3 Feature Extraction Pipeline on MCU

The 256-dim vector must be computed on every beat:

| Block | Description | Cost (approximate) |
|---|---|---|
| R-peak detection | Pan-Tompkins or threshold | ~50 µs |
| 128-dim morphology | Windowed ECG sample around R-peak, Q6.10 | ~200 µs |
| 64-dim 10-beat mean | Running mean of last 10 morphology vectors | ~100 µs |
| 64-dim RR history | Circular buffer of last 100 RR intervals | ~10 µs |
| SRAM write (256 × 2 B) | SPI @ 8 MHz → ~64 µs | ~64 µs |
| **Total** | | **~430 µs/beat** |

At 80 bpm one beat occurs every 750 ms — the MCU has ~300× more time than it needs. Most of the 750 ms window can be spent in deep sleep.

### 2.4 System Block Diagram (Text)

```
ECG electrode
     │
  AFE (MAX30003 / ADS1299)
     │ SPI
  nRF52840 MCU
     │── Feature extraction (FPU, Q6.10)
     │── External SRAM (512 KB, SPI/QSPI)
     │── Wishbone / GPIO ──► user_project_wrapper (sky130A ASIC)
     │                            └─ svm_compute_core
     │◄── class label (3-bit)
     │── BLE (notify on arrhythmia)
  Smartphone app / cloud log
```

---

## 3. Expert Advice

### 3.1 Commercialization Path

The student shuttle **does not restrict commercial use** of your own IP. Apache 2.0 requires attribution but permits proprietary products. The practical path:

1. **Validate silicon** — test the returned die against the cocotb suite; confirm 97.67% accuracy on real ECG
2. **Prototype wearable** — MCU + ASIC + AFE + coin cell on a flex PCB; demonstrate 29-day battery life
3. **Regulatory** — FDA Class II medical device (cardiac monitor) requires **510(k) clearance**; engage a regulatory consultant early; budget 12–18 months and ~$50–200k
4. **IP protection** — file a provisional patent before any public disclosure beyond the ECE410 submission; provisional costs ~$1,500–3,000 with a patent attorney
5. **Funding** — NSF I-Corps (~$50k, no equity) is the right first step for PSU students; pitch the power efficiency story (30,000× vs. CPU)

### 3.2 Immediate Technical Priorities

| Priority | Action | Why |
|---|---|---|
| High | Run `mpw-precheck` | Catch any submission blockers now |
| High | Tag `submission-v1` in caravel repo | Pinned reference for silicon |
| Medium | Port feature extractor to nRF52840 | Proves full system works end-to-end |
| Medium | Order MAX30003 eval board | Real ECG acquisition for system demo |
| Low | Custom PCB (MCU + ASIC breakout) | After silicon validated |

### 3.3 Key People to Contact

- **Efabless support** — precheck failures and submission questions: community.efabless.com
- **PSU TechTransfer** (OTT) — provisional patent and licensing before commercializing PSU-affiliated work
- **NSF I-Corps Pacific Northwest node** — free customer discovery + $50k grant for commercialization feasibility
- **FDA Digital Health Center of Excellence** — pre-submission meetings are free and clarify 510(k) scope early

---

*For silicon timeline: submit by the next Efabless shuttle close date → parts back ~Q1 2027 → system demo Q2 2027.*
