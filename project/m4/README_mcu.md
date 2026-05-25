# MCU Integration Guide — SVM Cardiac Arrhythmia Classifier (v8 Batch)

This document describes how the Caravel RISC-V management core (MCU) interacts
with the `user_project_wrapper` SVM classifier using the **batch architecture**.

---

## 1. System Overview

```
ECG frontend (low-power continuous)
     │
     │  Feature extraction — 256-dim multi-scale per beat
     ▼
RISC-V MCU  ←─ collects 1000 beats at low power
     │
     │  1. Pre-load SVs into off-chip SRAM (rows 0..249)
     │  2. Pre-load input matrix into off-chip SRAM (rows 250..1249)
     │  3. Write NUM_SAMPLES, fire CONTROL[start]
     ▼
ASIC (svm_compute_core, 40 MHz burst)
     │  Autonomously classifies all N beats back-to-back
     │  Drives 19-bit ram_addr + ram_ren via GPIO[28:10], GPIO[29]
     │  Reads SV data and input data from LA[15:0] (host serves)
     │
     ├─► sample_rdy (GPIO[3] / IRQ[0]) — pulses once per beat classified
     │       class_out[2:0] (GPIO[2:0]) stable when sample_rdy fires
     │
     └─► svm_done (GPIO[4] / IRQ[1]) — pulses once at end of batch
```

The five arrhythmia classes follow the AAMI EC57 standard mapping:

| Class | Label | Description                      |
|-------|-------|----------------------------------|
|   0   |   N   | Normal / non-ectopic             |
|   1   |   S   | Supraventricular ectopic (SVEB)  |
|   2   |   V   | Ventricular ectopic (VEB / PVC)  |
|   3   |   F   | Fusion beat                      |
|   4   |   Q   | Unclassifiable / unknown         |

---

## 2. Multi-Scale Feature Vector (256-dim)

Each beat uses a single 256-dimensional feature vector:

| Dimensions | Scale    | Content                                           |
|------------|----------|---------------------------------------------------|
| [0..127]   | 1-beat   | 128 morphology features of the current beat       |
| [128..191] | 10-beat  | 64 context features summarising previous 9 beats  |
| [192..255] | 100-beat | 64 context features summarising previous 99 beats |

**Warm-up:** Advisory `ERR_WARMING_UP` fires during the first 100 classified
beats. Results are still valid; the MCU may flag them as lower-confidence.

---

## 3. Off-Chip RAM Layout

The ASIC reads both SVs and input vectors from a single host-side SRAM over the
GPIO/LA bus. The host must populate this SRAM before firing `start`.

**Address encoding:** `addr[18:0] = {row[10:0], col[7:0]}`

| Row range       | Content                  | Size               |
|-----------------|--------------------------|--------------------|
| 0 .. NUM_SV-1   | SV matrix (250 × 256 × 2 B) | 128 KB          |
| NUM_SV .. 1249  | Input matrix (1000 × 256 × 2 B) | 512 KB max  |

Maximum address: (250 + 1000) × 256 − 1 = 319 999 → fits in 19 bits.

**Bus assignment:**

| Signal          | Pin         | Direction | Description                         |
|-----------------|-------------|-----------|-------------------------------------|
| `ram_addr[18:0]`| GPIO[28:10] | output    | Row×256 + column address            |
| `ram_ren`       | GPIO[29]    | output    | Read enable (active high)           |
| `ram_rdata[15:0]`| LA[15:0]  | input     | Data from host SRAM (1-cycle latency)|

The host MCU must respond to every `ram_ren` pulse by driving `LA[15:0]` with
`SRAM[ram_addr]` on the **following clock cycle**.

---

## 4. Wishbone Register Map

Base address: `0x3000_0000`

| Offset | Access | Name        | Bits    | Description                                      |
|--------|--------|-------------|---------|--------------------------------------------------|
| 0x04   | RW     | CONTROL     | [0]     | `start` — pulse high to begin batch; auto-clears |
|        |        |             | [1]     | `vbatt_ok` — battery voltage acceptable          |
|        |        |             | [2]     | `vbatt_warn` — battery low warning               |
| 0x08   | RO     | STATUS      | [0]     | `done` — entire batch classified                 |
|        |        |             | [1]     | `error` — fault detected                         |
|        |        |             | [5:2]   | `error_code` — 4-bit fault detail                |
|        |        |             | [8:6]   | `class` — current/last class label               |
|        |        |             | [9]     | `sample_rdy` — per-beat ready pulse              |
| 0x0C   | RW     | NUM_SAMPLES | [9:0]   | Number of input beats in this batch (1–1000)     |
| 0x10   | RW     | NUM_SV_0   | [7:0]   | Support vectors for class 0                      |
| 0x14   | RW     | NUM_SV_1   | [7:0]   | Support vectors for class 1                      |
| 0x18   | RW     | NUM_SV_2   | [7:0]   | Support vectors for class 2                      |
| 0x1C   | RW     | NUM_SV_3   | [7:0]   | Support vectors for class 3                      |
| 0x20   | RW     | NUM_SV_4   | [7:0]   | Support vectors for class 4                      |
| 0x24   | WO     | PARAM_WR   | [19]    | Write enable (auto-clears)                       |
|        |        |             | [18:16] | Parameter address (0=gamma, 1=C, 2–6=bias[0..4]) |
|        |        |             | [15:0]  | Parameter value (Q6.10 fixed-point)              |

**Removed vs. v7:** `FIFO_DATA` (0x00), `WORK_RD` (0x38), and `STATUS2` (0x3C)
are no longer present. There is no feature FIFO — input data is pre-loaded into
off-chip SRAM before start.

---

## 5. GPIO Pin Assignments

| GPIO    | Signal         | Direction | Description                          |
|---------|----------------|-----------|--------------------------------------|
| [2:0]   | `class_out`    | output    | Class label, stable when sample_rdy  |
| [3]     | `sample_rdy`   | output    | Pulses one cycle per beat classified |
| [4]     | `svm_done`     | output    | Pulses one cycle when batch finishes |
| [5]     | `svm_error`    | output    | Asserted on fault                    |
| [9:6]   | `error_code`   | output    | 4-bit fault code                     |
| [28:10] | `ram_addr[18:0]`| output   | 19-bit off-chip SRAM address         |
| [29]    | `ram_ren`      | output    | Off-chip SRAM read enable            |

**Removed vs. v7:** `fifo_ready` (GPIO[9]) and `sv_ram_addr/sv_ram_ren`
(GPIO[25:10]) are replaced by the unified 19-bit `ram_addr` bus.

---

## 6. Interrupt Lines

| IRQ   | Signal       | Description                            |
|-------|--------------|----------------------------------------|
| IRQ[0]| `sample_rdy` | Fires once per classified beat         |
| IRQ[1]| `svm_done`   | Fires once at end of batch             |

---

## 7. Batch Protocol

### 7.1 One-Time Startup Configuration

```c
// Set SV counts from trained model
wb_write(0x30000010, sv_count[0]);
wb_write(0x30000014, sv_count[1]);
wb_write(0x30000018, sv_count[2]);
wb_write(0x3000001C, sv_count[3]);
wb_write(0x30000020, sv_count[4]);

// Load gamma (Q6.10: 0.25 → 0x0100)
wb_write(0x30000024, (1<<19) | (0<<16) | 0x0100);

// Assert vbatt_ok
wb_write(0x30000004, 0x0002);
```

### 7.2 Per-Batch Flow

```c
void run_batch(uint16_t n_beats,
               uint16_t sv_matrix[NUM_SV][256],   // host SRAM rows 0..249
               uint16_t input_matrix[n_beats][256]) // host SRAM rows 250..
{
    // Step 1: populate off-chip SRAM (host side, before start)
    sram_load(sv_matrix,    n_rows=NUM_SV,    base_row=0);
    sram_load(input_matrix, n_rows=n_beats,   base_row=NUM_SV);

    // Step 2: tell ASIC how many beats
    wb_write(0x3000000C, n_beats);

    // Step 3: fire start (and keep vbatt_ok)
    wb_write(0x30000004, 0x0003);   // start=1 + vbatt_ok=1
}
```

### 7.3 SRAM Serving (must run concurrently with ASIC)

While the ASIC is running, the MCU must respond to every `ram_ren` pulse:

```c
// Polling loop (or use GPIO interrupt on GPIO[29])
void sram_server_loop(void) {
    while (!done_flag) {
        if (gpio_read() & (1<<29)) {          // ram_ren asserted
            uint32_t addr = (gpio_read() >> 10) & 0x7FFFF;  // bits [28:10]
            la_write_bits(15, 0, sram[addr]);  // drive LA[15:0] next cycle
        }
    }
}
```

### 7.4 Per-Sample Result Capture (IRQ[0])

```c
uint8_t labels[MAX_BATCH];
uint16_t sample_idx = 0;

void irq0_handler(void) {
    // sample_rdy pulsed — class_out[2:0] is stable right now
    labels[sample_idx++] = gpio_read() & 0x7;
}
```

### 7.5 Batch Done (IRQ[1])

```c
void irq1_handler(void) {
    // svm_done — all labels captured
    done_flag = 1;
    analyse_sequence(labels, sample_idx);
}
```

---

## 8. Raw Class Scores via Logic Analyzer Bus

After all beats in a batch are classified, the four lowest-indexed class scores
are available on the LA bus:

| LA bits  | Signal | Description                       |
|----------|--------|-----------------------------------|
| [31:0]   | cs0    | Accumulated kernel score, class 0 |
| [63:32]  | cs1    | Accumulated kernel score, class 1 |
| [95:64]  | cs2    | Accumulated kernel score, class 2 |
| [127:96] | cs3    | Accumulated kernel score, class 3 |

Scores are unsigned 32-bit integers (Q6.10 accumulated sums plus bias). A wider
margin between the top score and the runner-up indicates higher confidence.

---

## 9. Battery-Aware Operation

| Bit           | Register  | Action when asserted              |
|---------------|-----------|-----------------------------------|
| `vbatt_ok`    | CONTROL[1]| Normal operation — required to start |
| `vbatt_warn`  | CONTROL[2]| Raises `ERR_LOW_BATTERY` advisory |

If `vbatt_ok` is deasserted mid-batch, the core raises `ERR_POWER_FAIL` and
stops. Re-assert `vbatt_ok` and re-run the batch.

---

## 10. Complete Sketch

```c
#define WB_CONTROL     0x30000004
#define WB_STATUS      0x30000008
#define WB_NUM_SAMPLES 0x3000000C
#define WB_NUM_SV_BASE 0x30000010
#define WB_PARAM_WR    0x30000024

#define NUM_SV   250
#define BATCH_N  1000
#define FEAT_DIM 256

// Host SRAM: sv_sram[row][col], row 0..249 = SVs; row 250..1249 = inputs
static uint16_t host_sram[1250][FEAT_DIM];

static volatile uint8_t  labels[BATCH_N];
static volatile uint16_t label_idx;
static volatile bool     batch_done;

void irq0_handler(void) {
    labels[label_idx++] = gpio_read() & 0x7;
}

void irq1_handler(void) {
    batch_done = true;
}

// MCU must respond to ram_ren on every cycle (call from tight poll loop)
static inline void sram_serve(void) {
    uint32_t gp = gpio_read();
    if (gp & (1u << 29)) {
        uint32_t addr = (gp >> 10) & 0x7FFFF;
        la_write_bits(15, 0, host_sram[addr / FEAT_DIM][addr % FEAT_DIM]);
    }
}

void classify_batch(void) {
    // Startup config (once)
    for (int c = 0; c < 5; c++)
        wb_write(WB_NUM_SV_BASE + c * 4, sv_count[c]);
    wb_write(WB_PARAM_WR, (1<<19)|(0<<16)|0x0100);  // gamma = 0.25
    wb_write(WB_CONTROL, 0x0002);                     // vbatt_ok

    // Load SVs and input matrix into host SRAM (app-specific)
    load_sv_matrix(host_sram);
    load_input_matrix(host_sram + NUM_SV, beat_buffer, BATCH_N);

    // Fire batch
    label_idx  = 0;
    batch_done = false;
    wb_write(WB_NUM_SAMPLES, BATCH_N);
    wb_write(WB_CONTROL, 0x0003);   // start + vbatt_ok

    // Serve SRAM while ASIC runs; IRQ handlers capture results
    while (!batch_done)
        sram_serve();

    process_results(labels, BATCH_N);
}
```

---

*Part of ECE410 — Portland State University, 2024.
Design: Adam Handwerger.*
