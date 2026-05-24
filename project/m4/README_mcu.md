# MCU Integration Guide — SVM Cardiac Arrhythmia Classifier

This document describes how the Caravel RISC-V management core (MCU) interacts
with the `user_project_wrapper` SVM classifier, and how to build beat-sequence
analysis on top of the per-beat classification results.

---

## 1. System Overview

```
ECG frontend
     │  128 × 16-bit features (one heartbeat)
     ▼
[Wishbone FIFO]  ──►  svm_compute_core  ──►  argmax  ──►  class (0–4)
                            │                                   │
                     kernel scores                        user_irq[0]
                     cs0–cs3 on LA bus                         │
                                                          RISC-V MCU
                                                               │
                                                     higher-level analysis
                                                     alert / telemetry
```

The five arrhythmia classes follow the AAMI EC57 standard mapping:

| Class | Label | Description                        |
|-------|-------|------------------------------------|
|   0   |   N   | Normal / non-ectopic               |
|   1   |   S   | Supraventricular ectopic (SVEB)    |
|   2   |   V   | Ventricular ectopic (VEB / PVC)    |
|   3   |   F   | Fusion beat                        |
|   4   |   Q   | Unclassifiable / unknown           |

---

## 2. Wishbone Register Map

Base address: `0x3000_0000`

| Offset | Access | Name        | Bits      | Description                                   |
|--------|--------|-------------|-----------|-----------------------------------------------|
| 0x00   | WO     | FIFO_DATA   | [15:0]    | Write one 16-bit feature word; 128 words = one beat |
| 0x04   | RW     | CONTROL     | [0]       | `start` — pulse high to begin classification  |
|        |        |             | [1]       | `vbatt_ok` — battery voltage acceptable       |
|        |        |             | [2]       | `vbatt_warn` — battery low warning            |
|        |        |             | [3]       | `kern_ready` — enable kernel score readback on LA |
| 0x08   | RO     | STATUS      | [0]       | `done` — classification complete              |
|        |        |             | [1]       | `error` — fault detected                     |
|        |        |             | [5:2]     | `error_code` — error detail (4 bits)          |
|        |        |             | [8:6]     | `class` — winning class (0–4)                 |
| 0x0C   | RW     | NUM_SAMPLES | [9:0]     | Feature vector length (default 128)           |
| 0x10   | RW     | NUM_SV_0   | [7:0]     | Support vectors for class 0                   |
| 0x14   | RW     | NUM_SV_1   | [7:0]     | Support vectors for class 1                   |
| 0x18   | RW     | NUM_SV_2   | [7:0]     | Support vectors for class 2                   |
| 0x1C   | RW     | NUM_SV_3   | [7:0]     | Support vectors for class 3                   |
| 0x20   | RW     | NUM_SV_4   | [7:0]     | Support vectors for class 4                   |
| 0x24   | WO     | PARAM_WR   | [19]      | Write enable                                  |
|        |        |             | [18:16]   | Parameter address (gamma, C, bias[0–4])       |
|        |        |             | [15:0]    | Parameter value                               |
| 0x38   | WO     | WORK_RD    | [10:0]    | Work-RAM address to latch for readback        |
| 0x3C   | RO     | STATUS2    | [15:0]    | Work-RAM data at address written to WORK_RD   |

---

## 3. Per-Beat Classification Flow

### 3.1 Feature Streaming

Write 128 consecutive 16-bit feature words to `FIFO_DATA`. The FIFO signals
readiness via `GPIO[9]` (`fifo_ready`). Writes should be gated on this signal
to avoid overrun.

```c
for (int i = 0; i < 128; i++) {
    while (!(gpio_read() & (1 << 9)));   // wait fifo_ready
    wb_write(0x30000000, features[i]);
}
```

### 3.2 Triggering Classification

Pulse the `start` bit in CONTROL. The hardware auto-clears it on the next clock.

```c
wb_write(0x30000004, 0x0009);  // start=1, kern_ready=1 (bit3 enables LA score output)
```

### 3.3 Interrupt-Driven Result Readout

`user_irq[0]` fires when `svm_done` goes high. Register an ISR:

```c
void irq_handler(void) {
    uint32_t status = wb_read(0x30000008);
    uint8_t  class  = (status >> 6) & 0x7;
    bool     error  = (status >> 1) & 0x1;
    record_beat(class, error);
    arm_next_beat();  // re-enable frontend for next heartbeat
}
```

Alternatively poll `STATUS[0]` (`done` bit) if interrupts are not used.

### 3.4 GPIO Hardware Path

The classification result is also available directly on IO pads — no MCU
involvement required:

| GPIO   | Signal        | Description                        |
|--------|---------------|------------------------------------|
| [2:0]  | class_out     | Winning class (0–4)                |
| [3]    | done          | Pulses high when result is valid   |
| [4]    | error         | Asserted on fault                  |
| [8:5]  | error_code    | 4-bit fault code                   |
| [9]    | fifo_ready    | FIFO accepting feature data        |
| [24:10]| sv_ram_addr   | Support-vector RAM address (output)|
| [25]   | sv_ram_ren    | Support-vector RAM read enable     |

An external MCU or display controller can read `class_out` and `done` directly
from the pads without going through Wishbone at all.

---

## 4. Raw Class Scores via Logic Analyzer Bus

When `CONTROL[3]` (`kern_ready`) is set, the four lowest-indexed class scores
are exposed on the Logic Analyzer bus so the MCU can inspect the *margin* of
each decision:

| LA bits   | Signal | Description                        |
|-----------|--------|------------------------------------|
| [31:0]    | cs0    | Accumulated kernel score, class 0  |
| [63:32]   | cs1    | Accumulated kernel score, class 1  |
| [95:64]   | cs2    | Accumulated kernel score, class 2  |
| [127:96]  | cs3    | Accumulated kernel score, class 3  |

Scores are signed 32-bit integers; a larger value means stronger membership
in that class. Class 4 (Q) is not exposed on the LA bus but its contribution
is implicit: if all four visible scores are low, class 4 won the argmax.

**Why margins matter:**  
A beat classified as Normal (class 0) with cs0 = +12000 and cs2 = +11800 is
a much less confident prediction than cs0 = +12000 and cs2 = −5000. The MCU
can use this spread to flag uncertain beats for closer inspection rather than
treating all predictions equally.

Reading scores in C (Caravel firmware):

```c
int32_t cs0 = (int32_t)la_read_bits(31, 0);
int32_t cs1 = (int32_t)la_read_bits(63, 32);
int32_t cs2 = (int32_t)la_read_bits(95, 64);
int32_t cs3 = (int32_t)la_read_bits(127, 96);
int32_t margin = cs_winner - cs_runner_up;  // confidence proxy
```

---

## 5. Beat-Sequence Analysis

The hardware classifier delivers one label per beat. The MCU is responsible
for all analysis that spans multiple beats. Caravel's RISC-V core has 256 KB
of SRAM — enough to store several hundred beats of history with scores.

### 5.1 Suggested Data Structure

```c
#define HISTORY_LEN 128

typedef struct {
    uint8_t  class;       // 0–4
    int32_t  scores[4];   // cs0–cs3 from LA bus
    uint32_t rr_ticks;    // inter-beat interval (MCU timer ticks)
    bool     error;
} beat_t;

beat_t history[HISTORY_LEN];
uint8_t head = 0;         // circular buffer index
```

### 5.2 Heart Rate and RR Interval

The MCU controls when `start` is pulsed and therefore knows the time between
consecutive beats. Store the RR interval alongside each label to separate
rate-dependent from rate-independent arrhythmias.

```c
uint32_t last_beat_time = 0;

void record_beat(uint8_t class, bool error) {
    uint32_t now = timer_read();
    history[head].rr_ticks = now - last_beat_time;
    history[head].class    = class;
    history[head].error    = error;
    last_beat_time         = now;
    head = (head + 1) % HISTORY_LEN;
}
```

### 5.3 Pattern Detection (100-Beat Window)

Run the following checks after every `N` beats (e.g. N = 100):

**Burden count** — fraction of abnormal beats:

```c
uint8_t pvc_count = 0;
for (int i = 0; i < HISTORY_LEN; i++)
    if (history[i].class == CLASS_V) pvc_count++;

if (pvc_count > 10)          // >10% PVC burden
    raise_alert(ALERT_HIGH_PVC_BURDEN);
```

**Sustained runs** — consecutive identical abnormal beats:

```c
uint8_t run = 1, max_run = 1;
for (int i = 1; i < HISTORY_LEN; i++) {
    if (history[i].class == history[i-1].class &&
        history[i].class != CLASS_N)
        run++;
    else run = 1;
    if (run > max_run) max_run = run;
}
if (max_run >= 3)
    raise_alert(ALERT_SUSTAINED_ARRHYTHMIA);
```

**Bigeminy / trigeminy** — alternating or repeating patterns:

```c
// Bigeminy: N-V-N-V...
bool bigeminy = true;
for (int i = 0; i < 8; i++) {
    uint8_t expected = (i % 2 == 0) ? CLASS_N : CLASS_V;
    if (history[i].class != expected) { bigeminy = false; break; }
}
if (bigeminy) raise_alert(ALERT_BIGEMINY);
```

**Transition matrix** — track how often each class follows each other class
to detect evolving rhythms:

```c
uint16_t transitions[5][5] = {0};
for (int i = 1; i < HISTORY_LEN; i++)
    transitions[history[i-1].class][history[i].class]++;
```

### 5.4 Confidence-Weighted Decisions

Low-confidence beats (small winner–runner-up margin) should be down-weighted
in sequence counts to reduce false alerts:

```c
bool is_confident(beat_t *b) {
    int32_t scores[4] = {b->scores[0], b->scores[1],
                         b->scores[2], b->scores[3]};
    int32_t top = INT32_MIN, second = INT32_MIN;
    for (int i = 0; i < 4; i++) {
        if (scores[i] > top)    { second = top; top = scores[i]; }
        else if (scores[i] > second) second = scores[i];
    }
    return (top - second) > CONFIDENCE_THRESHOLD;  // tune empirically
}
```

Only count confident beats toward the burden/run thresholds to avoid noise
beats inflating arrhythmia counts.

### 5.5 Heart Rate Variability (HRV)

Low HRV is an independent risk marker. Track mean and variance of RR intervals:

```c
uint32_t rr_sum = 0, rr_sq_sum = 0;
for (int i = 0; i < HISTORY_LEN; i++) {
    rr_sum    += history[i].rr_ticks;
    rr_sq_sum += (history[i].rr_ticks * history[i].rr_ticks);
}
uint32_t mean_rr   = rr_sum / HISTORY_LEN;
uint32_t variance  = (rr_sq_sum / HISTORY_LEN) - (mean_rr * mean_rr);
// SDNN (std dev of NN intervals) = sqrt(variance) in ticks
```

---

## 6. Alert and Output Strategy

### 6.1 Alert Levels

| Level    | Condition                                   | Action                        |
|----------|---------------------------------------------|-------------------------------|
| INFO     | Isolated ectopic (1–2 in 100 beats)         | Log only                      |
| WARN     | Burden 5–15%, or run of 3–5 beats           | Log + LED flash               |
| CRITICAL | Burden >15%, sustained run ≥6, or VTach     | Interrupt host, transmit data |

### 6.2 Telemetry Payload

When a CRITICAL alert fires, transmit a compact summary:

```
[timestamp 4B] [alert_type 1B] [class_counts 5B] [max_run 1B]
[last_8_classes 1B each] [mean_rr 2B] [rr_variance 2B]
```

This fits in a single BLE notification (≤20 bytes without compression).

---

## 7. Battery-Aware Operation

The hardware exposes two battery signals writable via CONTROL:

- `vbatt_ok` (`CONTROL[1]`): normal voltage — full classification enabled
- `vbatt_warn` (`CONTROL[2]`): battery low — core reduces activity

The MCU should monitor the battery ADC and update these bits accordingly.
Under `vbatt_warn`, the SVM core can gate its clock more aggressively, and
the MCU might reduce analysis depth (e.g., skip HRV computation, increase
the burst threshold for alerts).

---

## 8. Startup and Parameter Loading

Before the first beat, the MCU must:

1. Write `NUM_SV_0`–`NUM_SV_4` with the per-class SV counts from the trained model.
2. Write `NUM_SAMPLES` with the feature vector length (default 128).
3. Write `PARAM_WR` entries to load gamma, C, and the five bias values.
4. Set `vbatt_ok` in CONTROL.
5. Supply SV RAM data on `LA[15:0]` in response to `GPIO[25]` (sv_ram_ren) and `GPIO[24:10]` (sv_ram_addr).

The support vector data itself lives in host-side flash and is streamed to the
core one word at a time via `LA[15:0]` during inference — the MCU acts as the
SV RAM controller for each classification run.

---

## 9. Complete ISR Sketch

```c
#include <stdint.h>
#include <stdbool.h>

#define WB_BASE        0x30000000
#define REG_FIFO_DATA  (WB_BASE + 0x00)
#define REG_CONTROL    (WB_BASE + 0x04)
#define REG_STATUS     (WB_BASE + 0x08)
#define CLASS_N 0
#define CLASS_S 1
#define CLASS_V 2
#define CLASS_F 3
#define CLASS_Q 4

static beat_t  history[HISTORY_LEN];
static uint8_t head = 0;

void user_irq0_handler(void) {
    uint32_t status = wb_read(REG_STATUS);
    uint8_t  class  = (status >> 6) & 0x7;
    bool     error  = (status >> 1) & 0x1;

    // Read raw scores from LA bus while kern_ready is still set
    beat_t *b = &history[head];
    b->class     = class;
    b->error     = error;
    b->rr_ticks  = timer_elapsed();
    b->scores[0] = (int32_t)la_read(31,  0);
    b->scores[1] = (int32_t)la_read(63, 32);
    b->scores[2] = (int32_t)la_read(95, 64);
    b->scores[3] = (int32_t)la_read(127, 96);
    head = (head + 1) % HISTORY_LEN;

    // Run sequence analysis every 100 beats
    if (head % 100 == 0)
        analyse_sequence();

    timer_reset();
    wb_write(REG_CONTROL, 0x0002);  // clear start, keep vbatt_ok
}
```

---

*Part of ECE410 — Portland State University, 2024.  
Design: Adam Handwerger.*
