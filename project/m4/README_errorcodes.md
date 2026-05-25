# SVM Compute Core — Error Code Reference (v8 Batch Architecture)

**RTL:** `user_project_wrapper.sv` → `svm_compute_core.sv`
**Milestone:** m4/m5 (batch architecture, Caravel chipIgnite, sky130A)

Error codes are exposed exclusively via Wishbone STATUS register and GPIO pins —
no raw RTL ports are visible outside `user_project_wrapper`.

---

## Wishbone Register Map (base `0x3000_0000`)

| Address | Name | R/W | Description |
|---------|------|-----|-------------|
| `+0x04` | CONTROL | RW | `[0]`=start `[1]`=vbatt_ok `[2]`=vbatt_warn |
| `+0x08` | STATUS  | RO | `[0]`=done `[1]`=error `[5:2]`=error_code `[8:6]`=class `[9]`=sample_rdy |

### STATUS register bit layout (`0x30000008`)

```
 31       10   9       8   6  5    2  1     0
 ┌──────────┬──────┬──────┬──────┬──────┬──────┐
 │  (zero)  │ srdy │ cls  │ ecode│ err  │ done │
 └──────────┴──────┴──────┴──────┴──────┴──────┘
               [9]   [8:6]  [5:2]  [1]    [0]
```

- **`done` [0]**: pulses one cycle when entire batch finishes; sticky until next `start`
- **`error` [1]**: sticky for faults 0x1–0x7; self-clearing for advisories 0x8–0xB
- **`error_code` [5:2]**: 4-bit code (see table); holds first fault that fired
- **`class` [8:6]**: 3-bit predicted class (0–4); updated per beat
- **`sample_rdy` [9]**: pulses one cycle per beat classified; mirrors IRQ[0]

### CONTROL register bit layout (`0x30000004`)

```
 31        3    2             1          0
 ┌──────────┬──────────────┬───────────┬───────┐
 │  (zero)  │  vbatt_warn  │  vbatt_ok │ start │
 └──────────┴──────────────┴───────────┴───────┘
```

- **`start` [0]**: write 1 to begin batch; auto-clears after one cycle
- **`vbatt_ok` [1]**: drive 1 when supply is above operational threshold; required to start
- **`vbatt_warn` [2]**: drive 1 when supply is below soft warning threshold

**Removed vs. v7:** `kern_ready` (bit 3) is not present in v8 — the batch
architecture does not use a feature FIFO or separate kernel-ready handshake.

---

## Error Code Reference

| Code | Name | Category | Clears on | Blocks start? |
|------|------|----------|-----------|---------------|
| `0x0` | ERR_NONE | — | — | No |
| `0x1` | ERR_SV_ZERO | Sticky | `rst_n` | Yes |
| `0x2` | ERR_SV_OVERFLOW | Sticky | `rst_n` | Yes |
| `0x3` | ERR_ILLEGAL_STATE | Sticky | `rst_n` | Yes |
| `0x4` | ERR_GAMMA_SAT | Sticky | `rst_n` | No |
| `0x5` | *(reserved)* | — | — | — |
| `0x6` | ERR_GAMMA_ZERO | Sticky | `rst_n` | No |
| `0x7` | ERR_NUM_SAMPLES_ZERO | Sticky | `rst_n` | Yes |
| `0x8` | ERR_WARMING_UP | Advisory | Beat 100 | No |
| `0x9` | ERR_INTERRUPTED | Advisory | Beat 100 | No |
| `0xA` | ERR_LOW_BATTERY | Advisory | `vbatt_warn` deasserts | No |
| `0xB` | ERR_POWER_FAIL | Advisory | `vbatt_ok` reasserts | Yes (new starts only) |

**0x5 removed:** `ERR_FIFO_OVERFLOW` no longer exists — the input FIFO was
removed in v8. Code 0x5 is reserved and will never fire.

Codes `0x1–0x7` set `error [1]` and latch until `rst_n`. Codes `0x8–0xB` set
`error [1]` while active and auto-clear when the condition resolves.

---

## Error Priority Encoder (from RTL)

The following priority order is implemented in `svm_compute_core.sv`:

```
1. ERROR_STATE FSM default branch → ERR_ILLEGAL_STATE (0x3)
2. total_sv_count == 0            → ERR_SV_ZERO       (0x1)
3. total_sv_count > NUM_SV        → ERR_SV_OVERFLOW   (0x2)
4. num_samples == 0               → ERR_NUM_SAMPLES_ZERO (0x7)
5. gamma > 8.0 (Q6.10: > 8192)   → ERR_GAMMA_SAT     (0x4)
6. gamma == 0                     → ERR_GAMMA_ZERO    (0x6)
7. !vbatt_ok                      → ERR_POWER_FAIL    (0xB)
8. vbatt_warn                     → ERR_LOW_BATTERY   (0xA)
9. heartbeat_count < 100          → ERR_WARMING_UP    (0x8)
10. (none)                        → ERR_NONE          (0x0)
```

Sticky errors (0x1–0x7) only latch if the current code is ERR_NONE or advisory
(≥0x8). They cannot be overwritten by a lower-priority sticky code.

---

## GPIO Error Signals

| GPIO  | Signal       | Description                    |
|-------|--------------|--------------------------------|
| [5]   | `svm_error`  | Asserted while error is active |
| [9:6] | `error_code` | 4-bit code, direct from RTL    |

These mirror the Wishbone STATUS register and can be read without the Wishbone
bus for low-overhead polling.

---

## Port-to-Register Mapping (v8)

| RTL signal | Wishbone register / bit |
|------------|-------------------------|
| `start` | CONTROL `[0]` — write 1 to pulse |
| `vbatt_ok` | CONTROL `[1]` — held high by firmware |
| `vbatt_warn` | CONTROL `[2]` — driven by firmware from ADC |
| `done` (batch) | STATUS `[0]` |
| `error` | STATUS `[1]` |
| `error_code[3:0]` | STATUS `[5:2]` |
| `class_out[2:0]` | STATUS `[8:6]` |
| `sample_rdy` | STATUS `[9]` / IRQ[0] |
| `svm_done` (batch) | STATUS `[0]` / IRQ[1] |
| `num_sv_per_class[5][7:0]` | NUM_SV registers `+0x10`–`+0x20` |
| `num_samples[9:0]` | NUM_SAMPLES `+0x0C` |
| `ram_addr[18:0]` | GPIO[28:10] (output) |
| `ram_ren` | GPIO[29] (output) |
| `ram_rdata[15:0]` | LA[15:0] (input, host-driven) |

---

## Firmware Pattern (RISC-V C, v8 Batch)

```c
#define SVM_BASE        0x30000000
#define REG_CONTROL     (*(volatile uint32_t*)(SVM_BASE + 0x04))
#define REG_STATUS      (*(volatile uint32_t*)(SVM_BASE + 0x08))
#define REG_NUM_SAMPLES (*(volatile uint32_t*)(SVM_BASE + 0x0C))

// 1. Configure SVs, gamma, num_samples
REG_NUM_SAMPLES = 1000;
// ... write NUM_SV0-4, PARAM_WR ...

// 2. Assert vbatt_ok, fire start
REG_CONTROL = (1<<1) | (1<<0);   // vbatt_ok=1, start=1

// 3. Serve SRAM and wait for batch done (IRQ[1] or poll)
while (!(REG_STATUS & 0x1)) {
    serve_sram_one_cycle();
}

// 4. Decode status
uint32_t s        = REG_STATUS;
uint8_t  err_flag = (s >> 1) & 0x1;
uint8_t  err_code = (s >> 2) & 0xF;
uint8_t  class_w  = (s >> 6) & 0x7;

if (err_flag) {
    if (err_code <= 0x7) {
        // Sticky fault — pulse rst_n before retrying
        reg_mprj_reset = 1;
        for (volatile int i = 0; i < 100; i++);
        reg_mprj_reset = 0;
    }
    // Advisories (0x8–0xB) auto-clear; log and continue
}
```

---

## Notes Specific to v8 (Batch Architecture)

- **`ERR_FIFO_OVERFLOW` removed**: There is no input FIFO in v8. Input data lives
  in host-side SRAM and is read autonomously — overflow is architecturally impossible.
- **`kern_ready` removed**: The batch FSM does not require a separate kernel-ready
  handshake. The core transitions from OUTPUT_RESULT to COMPUTE_DIST/WRITE_CLASS
  autonomously.
- **`sample_rdy` (STATUS[9] / IRQ[0])**: New in v8. Fires once per classified beat
  so the host can capture labels in real time without polling STATUS after `done`.
- **`vbatt_ok` is required before `start`**: If `vbatt_ok` is low, `start` is
  ignored and the core stays in IDLE.

---

*Part of ECE410 — Portland State University, 2024.
Design: Adam Handwerger.*
