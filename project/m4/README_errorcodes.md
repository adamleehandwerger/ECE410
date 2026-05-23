# SVM Compute Core — Error Code Reference (m4: Caravel / Wishbone)

**RTL:** `user_project_wrapper.sv` → `svm_compute_core` (blackbox GL)  
**Verification status:** 13/13 unit testbenches PASS; Caravel DV: svm_wb_test  
**Milestone:** m4 (Caravel chipIgnite, sky130A, DRT 0 violations)

Error codes are the same 4-bit values as m3. In m4 they are accessed exclusively
through Wishbone memory-mapped registers — there are no direct RTL ports exposed
outside the wrapper. See `m3/README_errorcodes.md` for full per-code detail.

---

## Wishbone Register Map (base `0x30000000`)

| Address | Name | R/W | Description |
|---------|------|-----|-------------|
| `+0x04` | CONTROL | RW | `[0]`=start `[1]`=vbatt_ok `[2]`=vbatt_warn `[3]`=kern_ready |
| `+0x08` | STATUS  | RO | `[0]`=done `[1]`=error `[5:2]`=error_code `[8:6]`=class_out |

### STATUS register bit layout (`0x30000008`)

```
 31        9  8   6  5    2  1     0
 ┌──────────┬─────┬───────┬──────┬──────┐
 │  (zero)  │ cls │ ecode │ err  │ done │
 └──────────┴─────┴───────┴──────┴──────┘
              [8:6]  [5:2]   [1]    [0]
```

- **`done` [0]**: pulses one cycle when classification finishes; `class_out` holds result
- **`error` [1]**: sticky latch for faults 0x1–0x7; self-clearing for advisories 0x8–0xB
- **`error_code` [5:2]**: 4-bit code (see table below); holds first fault that fired
- **`class_out` [8:6]**: 3-bit predicted class (0–4); valid after `done`

### CONTROL register bit layout (`0x30000004`)

```
 31       4   3            2             1          0
 ┌─────────┬─────────────┬──────────────┬───────────┬───────┐
 │ (zero)  │ kern_ready  │  vbatt_warn  │  vbatt_ok │ start │
 └─────────┴─────────────┴──────────────┴───────────┴───────┘
```

- **`start` [0]**: write 1 to begin classification; auto-clears after one cycle
- **`vbatt_ok` [1]**: drive 1 when supply is above operational threshold
- **`vbatt_warn` [2]**: drive 1 when supply is below soft warning threshold
- **`kern_ready` [3]**: set 1 after writing NUM_SV; tells argmax to count kernel outputs

---

## Error Code Quick Reference

| Code | Name | Category | Clears on | Blocks start? |
|------|------|----------|-----------|---------------|
| `0x0` | ERR_NONE | — | — | No |
| `0x1` | ERR_SV_ZERO | Sticky | `rst_n` | Yes |
| `0x2` | ERR_SV_OVERFLOW | Sticky | `rst_n` | Yes |
| `0x3` | ERR_ILLEGAL_STATE | Sticky | `rst_n` | Yes |
| `0x4` | ERR_GAMMA_SAT | Sticky | `rst_n` | No |
| `0x5` | ERR_FIFO_OVERFLOW | Sticky | `rst_n` | No |
| `0x6` | ERR_GAMMA_ZERO | Sticky | `rst_n` | No |
| `0x7` | ERR_NUM_SAMPLES_ZERO | Sticky | `rst_n` | Yes |
| `0x8` | ERR_WARMING_UP | Advisory | Beat 100 | No |
| `0x9` | ERR_INTERRUPTED | Advisory | Beat 100 | No |
| `0xA` | ERR_LOW_BATTERY | Advisory | `vbatt_warn` deasserts | No |
| `0xB` | ERR_POWER_FAIL | Advisory | `vbatt_ok` reasserts | Yes (new starts only) |

Codes `0x1–0x7` set `error [1]` and latch until `rst_n`. Codes `0x8–0xB` set
`error [1]` while active and auto-clear when the condition resolves.

---

## Port-to-Register Mapping (m3 RTL → m4 Wishbone)

| m3 RTL port | m4 Wishbone register / bit |
|-------------|---------------------------|
| `start` | CONTROL `[0]` — write 1 to pulse |
| `vbatt_ok` | CONTROL `[1]` — held high by firmware |
| `vbatt_warn` | CONTROL `[2]` — driven by firmware from ADC |
| `kernel_ready` | CONTROL `[3]` — set after loading NUM_SV |
| `done` | STATUS `[0]` |
| `error` | STATUS `[1]` |
| `error_code[3:0]` | STATUS `[5:2]` |
| `class_out[2:0]` | STATUS `[8:6]` |
| `num_sv_per_class[5][7:0]` | NUM_SV registers `+0x10`–`+0x20` |
| `num_samples[9:0]` | NUM_SAMPLES `+0x0C` |

---

## Firmware Pattern (RISC-V C)

```c
#define SVM_BASE        0x30000000
#define REG_CONTROL     (*(volatile uint32_t*)(SVM_BASE + 0x04))
#define REG_STATUS      (*(volatile uint32_t*)(SVM_BASE + 0x08))
#define REG_NUM_SAMPLES (*(volatile uint32_t*)(SVM_BASE + 0x0C))
#define REG_NUM_SV0     (*(volatile uint32_t*)(SVM_BASE + 0x10))
// ... NUM_SV1–4 at +0x14, +0x18, +0x1C, +0x20

// 1. Configure
REG_NUM_SAMPLES = 100;
REG_NUM_SV0 = 30; /* ... set all 5 classes */

// 2. Start (vbatt_ok=1, kern_ready=1, start=1)
REG_CONTROL = (1<<3) | (1<<1) | (1<<0);

// 3. Poll done
uint32_t s;
do { s = REG_STATUS; } while (!(s & 0x1));

// 4. Decode result
uint8_t err_flag  = (s >> 1) & 0x1;
uint8_t err_code  = (s >> 2) & 0xF;
uint8_t class_out = (s >> 6) & 0x7;

// 5. Handle errors
if (err_flag) {
    if (err_code <= 0x7) {
        // sticky fault — must reset core before retrying
        // pulse rst_n via housekeeping SPI or management GPIO
    } else {
        // advisory — log and continue; code auto-clears
    }
}
```

---

## Clearing a Sticky Fault in Caravel

On bare silicon, `rst_n` to the user project comes from the management SoC.
The RISC-V firmware can trigger it via the housekeeping register:

```c
// Pulse user-area reset (management SoC housekeeping)
reg_mprj_reset = 1;
// small delay
for (volatile int i = 0; i < 100; i++);
reg_mprj_reset = 0;
```

After the pulse, STATUS reads `0x0` and the core is back in the warming-up state
(`error_code = 0x8`, advisory).

---

## Notes Specific to m4

- **`vbatt_ok` / `vbatt_warn`** are firmware-driven bits in CONTROL, not analog inputs.
  In the Caravel context there is no on-chip analog comparator; the RISC-V firmware
  is responsible for reading the battery rail (e.g., via an external I2C fuel gauge
  on a Wishbone peripheral) and writing the bits accordingly.
- **`kern_ready`** (CONTROL `[3]`) must be set before issuing `start`, or the argmax
  accumulator will not count kernel outputs and `class_out` will always read 0.
- The wrapper resets `start` to 0 after one clock cycle automatically — firmware
  does not need to write CONTROL again to clear it.
