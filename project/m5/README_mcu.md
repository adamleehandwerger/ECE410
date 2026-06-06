# MCU Integration Guide — SVM Cardiac Arrhythmia Classifier

This document describes how a host MCU interacts with the `user_project_wrapper`
SVM classifier using the **v9 batch architecture**.  The MCU pre-loads both the
SV matrix and the input matrix into off-chip SRAM, fires a single `start` pulse,
and the ASIC classifies the entire batch autonomously.

---

## 1. System Overview

```
ECG frontend
     │
     │  Feature extraction (three temporal scales)
     │  ├─ 128 single-beat morphology features (current beat)
     │  ├─  64 10-beat context features  (mean of previous 9 beats)
     │  └─  64 100-beat context features (RR history, previous 99 beats)
     │       └── concatenated → 256 × 16-bit Q6.10 feature vector
     ▼
Host MCU
     │  Pre-load off-chip SRAM via SPI/GPIO:
     │  ├─ Rows   0–499   SV matrix     (500 SVs × 256 features)
     │  └─ Rows 500–1499  Input matrix  (up to 1000 beats × 256 features)
     │
     │  Write Wishbone config registers, then assert CONTROL[start=1]
     ▼
user_project_wrapper  (Caravel SoC)
     │
     svm_compute_core
     │  Reads off-chip SRAM autonomously via GPIO[29:10] + LA[15:0]
     │  Classifies all beats in batch without MCU involvement
     │
     │  Per-beat output:
     │  ├─ GPIO[3]   sample_rdy  (one-cycle pulse per classified beat)
     │  └─ GPIO[2:0] class_out   (3-bit label, stable when sample_rdy high)
     │
     │  Batch done:
     ├─ GPIO[4]     svm_done    (one-cycle pulse after last beat)
     └─ user_irq[1] svm_done
```

The five arrhythmia classes:

| code | Label  | Description                        |
|------|--------|------------------------------------|
|  0   | Normal | Normal sinus rhythm                |
|  1   | PVC    | Premature Ventricular Contraction  |
|  2   | AFib   | Atrial Fibrillation                |
|  3   | VT     | Ventricular Tachycardia            |
|  4   | SVT    | Supraventricular Tachycardia       |

---

## 2. Multi-Scale Feature Vector

Each beat is classified using a single 256-dimensional feature vector assembled
by the MCU from three temporal scales:

| Dimensions | Scale    | Content                                           |
|------------|----------|---------------------------------------------------|
| [0..127]   | 1-beat   | 128 morphology features of the current beat       |
| [128..191] | 10-beat  | 64 context features summarising previous 9 beats  |
| [192..255] | 100-beat | 64 RR-history features summarising previous 99 beats |

**Warm-up:** The 10-beat and 100-beat slices are only fully populated after the
device has processed at least 99 beats. During cold start, `ERR_WARMING_UP` is
raised (advisory, code 0x8, non-sticky) to signal that results from the first 99
beats may be less reliable. The core still classifies normally.

---

## 3. Wishbone Register Map

Base address: `0x3000_0000`

| Offset | Access | Name         | Bits      | Description                                     |
|--------|--------|--------------|-----------|-------------------------------------------------|
| 0x04   | RW     | CONTROL      | [0]       | `start` — write 1 to begin batch classification |
|        |        |              | [1]       | `vbatt_ok` — battery voltage acceptable         |
|        |        |              | [2]       | `vbatt_warn` — battery low warning              |
| 0x08   | RO     | STATUS       | [0]       | `done` — batch complete                         |
|        |        |              | [1]       | `error` — sticky fault detected                 |
|        |        |              | [5:2]     | `error_code` — 4-bit fault code                 |
|        |        |              | [8:6]     | `class_out` — label of most recently classified beat |
|        |        |              | [9]       | `sample_rdy` — pulsed high when a beat completes |
| 0x0C   | RW     | NUM_SAMPLES  | [9:0]     | Number of beats in this batch (1–1000)          |
| 0x10   | RW     | NUM_SV_0     | [7:0]     | Support vectors for class 0 (Normal)            |
| 0x14   | RW     | NUM_SV_1     | [7:0]     | Support vectors for class 1 (PVC)               |
| 0x18   | RW     | NUM_SV_2     | [7:0]     | Support vectors for class 2 (AFib)              |
| 0x1C   | RW     | NUM_SV_3     | [7:0]     | Support vectors for class 3 (VT)                |
| 0x20   | RW     | NUM_SV_4     | [7:0]     | Support vectors for class 4 (SVT)               |
| 0x24   | WO     | PARAM_WR     | [19]      | Write enable                                    |
|        |        |              | [18:16]   | Parameter address (0=gamma, 1=C, 2–6=bias[0–4]) |
|        |        |              | [15:0]    | Parameter value (Q6.10)                         |
| 0x28   | WO     | ALPHA_WR     | [24:16]   | Alpha coefficient index (0–499)                 |
|        |        |              | [15:0]    | Alpha value (Q6.10 signed)                      |

There is no FIFO_DATA register. Feature data lives in off-chip SRAM, pre-loaded
by the MCU before asserting `start`.

---

## 4. Classification Flow

### 4.1 Pre-Load Off-Chip SRAM

Before firing `start`, the MCU writes both matrices into the off-chip IS61WV51216
SRAM via its own SPI or parallel GPIO interface.

```
Off-chip SRAM address map (row × FEATURE_DIM, FEATURE_DIM = 256):
  Rows   0 .. 499   SV matrix     (500 SVs × 256 features × 2 B  = 256 KB)
  Rows 500 .. 1499  Input matrix  (1000 beats × 256 features × 2 B = 512 KB)
  Maximum address: 1500 × 256 − 1 = 383 999  →  19-bit address bus
```

The ASIC drives GPIO[28:10] = `ram_addr[18:0]` and GPIO[29] = `ram_ren`.
The MCU (or SRAM interface hardware) must return the requested word on
`LA[15:0]` (`ram_rdata`) within `RAM_LATENCY` cycles after `ram_ren` asserts.
With the IS61WV51216 at 40 MHz, `RAM_LATENCY = 3`.

### 4.2 Write Configuration Registers

```c
// SV counts (must match the trained model)
wb_write(0x30000010, 100);  // NUM_SV_0 (Normal)
wb_write(0x30000014, 100);  // NUM_SV_1 (PVC)
wb_write(0x30000018, 100);  // NUM_SV_2 (AFib)
wb_write(0x3000001C, 100);  // NUM_SV_3 (VT)
wb_write(0x30000020, 100);  // NUM_SV_4 (SVT)

// Batch size
wb_write(0x3000000C, 300);  // NUM_SAMPLES = 300 beats

// Gamma = 0.25 = 0x0100 in Q6.10
wb_write(0x30000024, (1 << 19) | (0 << 16) | 0x0100);  // PARAM_WR, addr=0, gamma

// Alpha coefficients (500 total, one per SV)
for (int i = 0; i < 500; i++)
    wb_write(0x30000028, (i << 16) | alpha_q10[i]);     // ALPHA_WR
```

### 4.3 Start Batch

```c
wb_write(0x30000004, 0x0003);  // CONTROL: start=1, vbatt_ok=1
```

**Important:** `start` is NOT auto-cleared. After `done` asserts, the FSM returns
to IDLE. If `start` is still high, it immediately re-fires the next batch. The
MCU must clear `start` (write `CONTROL = 0x0002`) after the done pulse:

```c
// In the svm_done ISR:
wb_write(0x30000004, 0x0002);  // clear start, keep vbatt_ok
```

### 4.4 Collect Per-Beat Results

`sample_rdy` (GPIO[3], `user_irq[0]`) pulses once per classified beat.
`class_out[2:0]` (GPIO[2:0]) is stable when `sample_rdy` is high.

Collect via polling:
```c
uint8_t labels[BATCH_SIZE];
int n = 0;
while (n < num_samples) {
    uint32_t io = gpio_read();
    if ((io >> 3) & 1) {           // sample_rdy (GPIO[3])
        labels[n++] = io & 0x7;    // class_out  (GPIO[2:0])
    }
}
```

Or via `user_irq[0]` (sample_rdy) interrupt handler:
```c
static volatile int beat_count = 0;
static uint8_t labels[BATCH_SIZE];

void user_irq0_handler(void) {    // fires on sample_rdy
    uint32_t io = gpio_read();
    labels[beat_count++] = io & 0x7;
}
```

Or read STATUS[8:6] and STATUS[9] via Wishbone (same information on the bus).

### 4.5 Batch Done

`svm_done` (GPIO[4], `user_irq[1]`) pulses once when the last beat completes.

```c
void user_irq1_handler(void) {    // fires on svm_done
    wb_write(0x30000004, 0x0002); // clear start
    analyse_sequence(labels, num_samples);
    load_next_batch();
}
```

### 4.6 GPIO Signal Map

| GPIO   | Direction | Signal           | Description                               |
|--------|-----------|------------------|-------------------------------------------|
| [2:0]  | Output    | class_out[2:0]   | Class label of most recently classified beat |
| [3]    | Output    | sample_rdy       | Pulses high for one cycle per beat        |
| [4]    | Output    | svm_done         | Pulses high for one cycle at batch end    |
| [5]    | Output    | svm_error        | Asserted on sticky fault                  |
| [9:6]  | Output    | svm_error_code   | 4-bit fault code                          |
| [28:10]| Output    | ram_addr[18:0]   | Off-chip SRAM address driven by ASIC      |
| [29]   | Output    | ram_ren          | Off-chip SRAM read enable driven by ASIC  |

`LA[15:0]` is the return path: the MCU or SRAM interface drives `la_data_in[15:0]`
with the requested SRAM word within `RAM_LATENCY` cycles of `ram_ren`.

---

## 5. IRQ Summary

| IRQ        | Signal     | When fires                |
|------------|------------|---------------------------|
| user_irq[0]| sample_rdy | Once per classified beat  |
| user_irq[1]| svm_done   | Once at end of batch      |

---

## 6. Beat-Sequence Analysis

After collecting `labels[]`, the MCU runs sequence analysis on the classification
vector. Caravel's RISC-V core has 256 KB of SRAM — enough to hold 1000 beats.

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

The MCU knows the time between SRAM pre-load cycles and can record the RR interval
alongside each class label to separate rate-dependent from rate-independent
arrhythmias.

### 6.3 Pattern Detection (100-Beat Window)

**Burden count** — fraction of abnormal beats:

```c
uint16_t pvc_count = 0;
for (int i = 0; i < 100; i++)
    if (history[i].class == CLASS_PVC) pvc_count++;
if (pvc_count > 10)
    raise_alert(ALERT_HIGH_PVC_BURDEN);
```

**Sustained runs** — consecutive identical abnormal beats:

```c
uint8_t run = 1, max_run = 1;
for (int i = 1; i < 100; i++) {
    if (history[i].class == history[i-1].class &&
        history[i].class != CLASS_NORMAL)
        run++;
    else run = 1;
    if (run > max_run) max_run = run;
}
if (max_run >= 3)
    raise_alert(ALERT_SUSTAINED_ARRHYTHMIA);
```

**Bigeminy** — alternating N-PVC-N-PVC pattern:

```c
bool bigeminy = true;
for (int i = 0; i < 8; i++) {
    uint8_t expected = (i % 2 == 0) ? CLASS_NORMAL : CLASS_PVC;
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

Before the first batch, the MCU must:

1. Pre-load the off-chip SRAM:
   - Rows 0–499: SV matrix (500 SVs × 256 features, Q6.10)
   - Rows 500+: Input matrix (up to 1000 beats × 256 features, Q6.10)
2. Write `NUM_SV_0`–`NUM_SV_4` with per-class SV counts.
3. Write `NUM_SAMPLES` with the number of beats per batch.
4. Write `PARAM_WR` entries for gamma, C, and the five bias values.
5. Write 500 `ALPHA_WR` entries with trained alpha coefficients.
6. Set `CONTROL = 0x0003` (start=1, vbatt_ok=1).

The ASIC reads both matrices autonomously via GPIO[28:10] and GPIO[29].
`RAM_LATENCY` (synthesised as 3 for IS61WV51216 at 40 MHz) configures wait states.

---

## 10. Complete Batch Processing Sketch

```c
#include <stdint.h>
#include <stdbool.h>

#define WB_BASE        0x30000000
#define REG_CONTROL    (WB_BASE + 0x04)
#define REG_STATUS     (WB_BASE + 0x08)
#define REG_NUM_SAMPLES (WB_BASE + 0x0C)
#define REG_NUM_SV0    (WB_BASE + 0x10)
#define REG_PARAM_WR   (WB_BASE + 0x24)
#define REG_ALPHA_WR   (WB_BASE + 0x28)

#define CLASS_NORMAL 0
#define CLASS_PVC    1
#define CLASS_AFIB   2
#define CLASS_VT     3
#define CLASS_SVT    4
#define BATCH_SIZE   300

static volatile uint8_t  labels[BATCH_SIZE];
static volatile uint16_t beat_count = 0;

// Called by user_irq[0] ISR (sample_rdy, fires once per classified beat)
void user_irq0_handler(void) {
    uint32_t io = gpio_read();
    if (beat_count < BATCH_SIZE)
        labels[beat_count++] = io & 0x7;   // class_out = GPIO[2:0]
}

// Called by user_irq[1] ISR (svm_done, fires once at batch end)
void user_irq1_handler(void) {
    wb_write(REG_CONTROL, 0x0002);          // clear start, keep vbatt_ok
    analyse_sequence((uint8_t*)labels, beat_count);
    beat_count = 0;
    // Pre-load next batch into SRAM then restart:
    // load_next_sram_batch();
    // wb_write(REG_CONTROL, 0x0003);
}

void init_svm(uint8_t num_sv[5], float gamma, float *alphas) {
    // SV counts
    for (int c = 0; c < 5; c++)
        wb_write(REG_NUM_SV0 + c * 4, num_sv[c]);
    // Batch size
    wb_write(REG_NUM_SAMPLES, BATCH_SIZE);
    // Gamma = 0.25 → 0x0100 in Q6.10
    uint16_t gamma_q = (uint16_t)(gamma * 1024 + 0.5f);
    wb_write(REG_PARAM_WR, (1 << 19) | (0 << 16) | gamma_q);
    // Alpha coefficients
    int total_sv = 0;
    for (int c = 0; c < 5; c++) total_sv += num_sv[c];
    for (int i = 0; i < total_sv; i++) {
        int16_t a_q = (int16_t)(alphas[i] * 1024);
        wb_write(REG_ALPHA_WR, (i << 16) | (uint16_t)a_q);
    }
}

void start_batch(void) {
    beat_count = 0;
    wb_write(REG_CONTROL, 0x0003);          // start=1, vbatt_ok=1
}
```

---

*Part of ECE410 — Portland State University, 2026.
Design: Adam Handwerger.*
