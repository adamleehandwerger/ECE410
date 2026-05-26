# MCU Integration Guide — SVM Cardiac Arrhythmia Classifier

This document describes how the Caravel RISC-V management core (MCU) interacts
with the `user_project_wrapper` SVM classifier.

---

## 1. System Overview

```
ECG frontend
     │
     │  Feature extraction (three temporal scales)
     │  ├─ 128 single-beat morphology features (current beat)
     │  ├─  64 10-beat context features  (mean of previous 9 beats)
     │  └─  64 100-beat context features (rhythm statistics, previous 99 beats)
     │       └── concatenated → 256 × 16-bit feature vector
     ▼
RISC-V MCU
     │  256 × 16-bit words via Wishbone FIFO_DATA
     ▼
[Wishbone FIFO]  ──►  svm_compute_core  ──►  internal argmax  ──►  class (0–4)
                            │                       │                    │
                     kernel scores          work_ram[beat_i]       user_irq[0]
                     cs0–cs3 on LA bus    (after all beats done)
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

## 2. Multi-Scale Feature Vector

Each beat is classified using a single 256-dimensional feature vector assembled
by the MCU from three temporal scales:

| Dimensions | Scale   | Content                                          |
|------------|---------|--------------------------------------------------|
| [0..127]   | 1-beat  | 128 morphology features of the current beat      |
| [128..191] | 10-beat | 64 context features summarising previous 9 beats |
| [192..255] | 100-beat| 64 context features summarising previous 99 beats|

The MCU extracts these features from its local beat buffer, concatenates them,
and streams the resulting 256-word vector to the SVM core via Wishbone.

**Warm-up:** The 10-beat and 100-beat slices are only fully populated after the
device has processed at least 99 beats. During cold start, `ERR_WARMING_UP` is
raised (advisory, non-sticky) to signal that results from the first 99 beats may
be less reliable. The core still classifies normally; the MCU may choose to flag
or discard those early results.

---

## 3. Wishbone Register Map

Base address: `0x3000_0000`

| Offset | Access | Name        | Bits      | Description                                   |
|--------|--------|-------------|-----------|-----------------------------------------------|
| 0x00   | WO     | FIFO_DATA   | [15:0]    | Write one 16-bit feature word; 256 words = one beat |
| 0x04   | RW     | CONTROL     | [0]       | `start` — pulse high to begin classification  |
|        |        |             | [1]       | `vbatt_ok` — battery voltage acceptable       |
|        |        |             | [2]       | `vbatt_warn` — battery low warning            |
| 0x08   | RO     | STATUS      | [0]       | `done` — all beats in batch classified        |
|        |        |             | [1]       | `error` — fault detected                     |
|        |        |             | [5:2]     | `error_code` — error detail (4 bits)          |
|        |        |             | [8:6]     | `class` — class label of beat 0 (work_ram[0]) |
| 0x0C   | RW     | NUM_SAMPLES | [9:0]     | Number of beats in this batch (1–1000)        |
| 0x10   | RW     | NUM_SV_0   | [7:0]     | Support vectors for class 0                   |
| 0x14   | RW     | NUM_SV_1   | [7:0]     | Support vectors for class 1                   |
| 0x18   | RW     | NUM_SV_2   | [7:0]     | Support vectors for class 2                   |
| 0x1C   | RW     | NUM_SV_3   | [7:0]     | Support vectors for class 3                   |
| 0x20   | RW     | NUM_SV_4   | [7:0]     | Support vectors for class 4                   |
| 0x24   | WO     | PARAM_WR   | [19]      | Write enable                                  |
|        |        |             | [18:16]   | Parameter address (gamma, C, bias[0–4])       |
|        |        |             | [15:0]    | Parameter value                               |
| 0x38   | WO     | WORK_RD    | [10:0]    | work_ram address to latch for readback        |
| 0x3C   | RO     | STATUS2    | [15:0]    | work_ram data at address written to WORK_RD   |

---

## 4. Classification Flow

### 4.1 Feature Streaming

The MCU assembles the 256-dim multi-scale feature vector and writes it word by
word to `FIFO_DATA`. The FIFO signals readiness via `GPIO[9]` (`fifo_ready`).

```c
for (int i = 0; i < 256; i++) {
    while (!(gpio_read() & (1 << 9)));   // wait fifo_ready
    wb_write(0x30000000, features[i]);   // [0..127] morph, [128..191] 10-beat, [192..255] 100-beat
}
```

### 4.2 Triggering a Batch

Pulse `start` in CONTROL. The hardware auto-clears it the next clock cycle.
`num_samples` sets how many consecutive beats to classify in this run.

```c
wb_write(0x30000004, 0x0002);  // start=1, vbatt_ok=1
```

### 4.3 Interrupt-Driven Result Readout

`user_irq[0]` fires when the entire batch of `num_samples` beats is classified.
The class labels are stored in `work_ram[0..num_samples-1]` — one 3-bit label
per beat, readable via WORK_RD / STATUS2.

```c
void user_irq0_handler(void) {
    uint32_t num = reg_num_samples;

    for (uint32_t i = 0; i < num; i++) {
        wb_write(0x30000038, i);                        // latch work_ram[i]
        uint8_t class = wb_read(0x3000003C) & 0x7;     // read label
        history[i].class = class;
    }
    analyse_sequence(num);
    arm_next_batch();
}
```

Alternatively poll `STATUS[0]` (`done` bit) if interrupts are not used.

### 4.4 GPIO Hardware Path

Classification results are also available directly on IO pads:

| GPIO   | Signal        | Description                        |
|--------|---------------|------------------------------------|
| [2:0]  | class_out     | Class label for beat 0 (work_ram[0])|
| [3]    | done          | Pulses high when batch is complete |
| [4]    | error         | Asserted on fault                  |
| [8:5]  | error_code    | 4-bit fault code                   |
| [9]    | fifo_ready    | FIFO accepting feature data        |
| [24:10]| sv_ram_addr   | Support-vector RAM address (output)|
| [25]   | sv_ram_ren    | Support-vector RAM read enable     |

---

## 5. Raw Class Scores via Logic Analyzer Bus

The four lowest-indexed class scores (accumulated kernel sums) are exposed on the
Logic Analyzer bus directly from the SVM core after each beat is classified:

| LA bits   | Signal | Description                        |
|-----------|--------|------------------------------------|
| [31:0]    | cs0    | Accumulated kernel score, class 0  |
| [63:32]   | cs1    | Accumulated kernel score, class 1  |
| [95:64]   | cs2    | Accumulated kernel score, class 2  |
| [127:96]  | cs3    | Accumulated kernel score, class 3  |

Scores are unsigned 32-bit integers (accumulated Q6.10 kernel values plus bias).
A larger value means stronger membership in that class.

**Why margins matter:**  
A beat classified Normal with cs0 = 42000 and cs2 = 41800 is far less confident
than cs0 = 42000 and cs2 = 5000. The MCU can use this spread to flag uncertain
beats for closer inspection.

Reading scores in C (Caravel firmware):

```c
uint32_t cs0 = la_read_bits(31,  0);
uint32_t cs1 = la_read_bits(63, 32);
uint32_t cs2 = la_read_bits(95, 64);
uint32_t cs3 = la_read_bits(127, 96);
```

Note: scores are only valid in the one-cycle window while the core is in
`WRITE_CLASS` state. Latch them immediately after `user_irq[0]` fires.

---

## 6. Beat-Sequence Analysis

After reading the batch's class labels from work_ram the MCU runs sequence
analysis on the classification vector. Caravel's RISC-V core has 256 KB of
SRAM — enough to hold 1000 beats of history with scores.

### 6.1 Suggested Data Structure

```c
#define HISTORY_LEN 1000

typedef struct {
    uint8_t  class;       // 0–4
    uint32_t rr_ticks;    // inter-beat interval (MCU timer ticks)
} beat_t;

beat_t history[HISTORY_LEN];
uint16_t head = 0;        // circular buffer index
```

### 6.2 Heart Rate and RR Interval

The MCU knows the time between `start` pulses and can record the RR interval
alongside each class label to separate rate-dependent from rate-independent
arrhythmias.

### 6.3 Pattern Detection (100-Beat Window)

**Burden count** — fraction of abnormal beats:

```c
uint16_t pvc_count = 0;
for (int i = 0; i < 100; i++)
    if (history[i].class == CLASS_V) pvc_count++;
if (pvc_count > 10)
    raise_alert(ALERT_HIGH_PVC_BURDEN);
```

**Sustained runs** — consecutive identical abnormal beats:

```c
uint8_t run = 1, max_run = 1;
for (int i = 1; i < 100; i++) {
    if (history[i].class == history[i-1].class &&
        history[i].class != CLASS_N)
        run++;
    else run = 1;
    if (run > max_run) max_run = run;
}
if (max_run >= 3)
    raise_alert(ALERT_SUSTAINED_ARRHYTHMIA);
```

**Bigeminy** — alternating N-V-N-V pattern:

```c
bool bigeminy = true;
for (int i = 0; i < 8; i++) {
    uint8_t expected = (i % 2 == 0) ? CLASS_N : CLASS_V;
    if (history[i].class != expected) { bigeminy = false; break; }
}
if (bigeminy) raise_alert(ALERT_BIGEMINY);
```

### 6.4 Heart Rate Variability (HRV)

```c
uint32_t rr_sum = 0, rr_sq_sum = 0;
for (int i = 0; i < HISTORY_LEN; i++) {
    rr_sum    += history[i].rr_ticks;
    rr_sq_sum += (history[i].rr_ticks * history[i].rr_ticks);
}
uint32_t mean_rr  = rr_sum / HISTORY_LEN;
uint32_t variance = (rr_sq_sum / HISTORY_LEN) - (mean_rr * mean_rr);
// SDNN ≈ sqrt(variance) in timer ticks
```

---

## 7. Alert and Output Strategy

### 7.1 Alert Levels

| Level    | Condition                                   | Action                        |
|----------|---------------------------------------------|-------------------------------|
| INFO     | Isolated ectopic (1–2 in 100 beats)         | Log only                      |
| WARN     | Burden 5–15%, or run of 3–5 beats           | Log + LED flash               |
| CRITICAL | Burden >15%, sustained run ≥6, or VTach     | Interrupt host, transmit data |

### 7.2 Telemetry Payload

When a CRITICAL alert fires, transmit a compact summary:

```
[timestamp 4B] [alert_type 1B] [class_counts 5B] [max_run 1B]
[last_8_classes 1B each] [mean_rr 2B] [rr_variance 2B]
```

This fits in a single BLE notification (≤20 bytes without compression).

---

## 8. Battery-Aware Operation

The hardware exposes two battery signals writable via CONTROL:

- `vbatt_ok` (`CONTROL[1]`): normal voltage — full classification enabled
- `vbatt_warn` (`CONTROL[2]`): battery low — core raises ERR_LOW_BATTERY advisory

The MCU should monitor the battery ADC and update these bits accordingly.
Under low battery the MCU might increase the burst threshold for alerts or
skip non-critical processing.

---

## 9. Startup and Parameter Loading

Before the first beat, the MCU must:

1. Write `NUM_SV_0`–`NUM_SV_4` with per-class SV counts from the trained model.
2. Write `NUM_SAMPLES` with the number of beats per batch.
3. Write `PARAM_WR` entries to load gamma, C, and the five bias values.
4. Set `vbatt_ok` in CONTROL.
5. Pre-load the off-chip SRAM before asserting `start`:
   - Rows 0–499: SV matrix (500 SVs × 256 features, Q6.10)
   - Rows 500–1499: input matrix (up to 1000 beats × 256 features, Q6.10)

The ASIC reads both matrices autonomously via `GPIO[28:10]` (`ram_addr[18:0]`),
`GPIO[29]` (`ram_ren`), and `LA[15:0]` (`ram_rdata[15:0]`). The `RAM_LATENCY`
RTL parameter configures how many cycles after `ram_ren` the data must be valid.
With the IS61WV51216 async SRAM (10 ns access) at 40 MHz, set `RAM_LATENCY=3`.
The core inserts wait states automatically — no MCU involvement during classification.

---

## 10. Complete Batch Processing Sketch

```c
#include <stdint.h>
#include <stdbool.h>

#define WB_BASE        0x30000000
#define REG_FIFO_DATA  (WB_BASE + 0x00)
#define REG_CONTROL    (WB_BASE + 0x04)
#define REG_STATUS     (WB_BASE + 0x08)
#define REG_NUM_SAMPLES (WB_BASE + 0x0C)
#define REG_WORK_RD    (WB_BASE + 0x38)
#define REG_STATUS2    (WB_BASE + 0x3C)
#define CLASS_N 0
#define CLASS_S 1
#define CLASS_V 2
#define CLASS_F 3
#define CLASS_Q 4
#define BATCH_SIZE 100

static uint8_t labels[BATCH_SIZE];

// Called by the host to stream one beat's feature vector to the core.
// features[0..127]   = single-beat morphology
// features[128..191] = 10-beat context
// features[192..255] = 100-beat context
void stream_beat(uint16_t features[256]) {
    for (int i = 0; i < 256; i++) {
        while (!(gpio_read() & (1 << 9)));  // wait fifo_ready
        wb_write(REG_FIFO_DATA, features[i]);
    }
}

void start_batch(uint16_t num_beats) {
    wb_write(REG_NUM_SAMPLES, num_beats);
    wb_write(REG_CONTROL, 0x0002);  // start=1, vbatt_ok=1
}

void user_irq0_handler(void) {
    uint32_t n = wb_read(REG_NUM_SAMPLES) & 0x3FF;

    for (uint32_t i = 0; i < n; i++) {
        wb_write(REG_WORK_RD, i);
        labels[i] = wb_read(REG_STATUS2) & 0x7;
    }

    analyse_sequence(labels, n);
}
```

---

*Part of ECE410 — Portland State University, 2024.  
Design: Adam Handwerger.*
