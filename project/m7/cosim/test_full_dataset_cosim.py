"""
test_full_dataset_cosim.py — full-dataset RTL cosim of svm_compute_core.

Streams the entire 300-sample PhysioNet test set through the ACTUAL RTL (the
signed-off 600-SV pipelined compute_core), driving the off-chip RAM with the
support-vector matrix (rows 0..599) and the test features (rows 600..899),
loading the Q6.10 model (gamma, 5 OVR biases, 600 alphas), then capturing
class_out at every sample_rdy pulse. Writes the hardware predictions to
/tmp/svm_cosim_preds.npy for confusion-matrix scoring.

Model + data come from export_model.py (/tmp/svm_cosim_model.npz), which reuses
the m6 pipeline that realizes 98.67%.
"""
import os, cocotb, numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

M       = np.load("/tmp/svm_cosim_model.npz")
X_te    = M["X_te"].astype(np.int64)      # (Ntest,256) Q6.10
y_te    = M["y_te"].astype(int)
SV      = M["SV"].astype(np.int64)        # (600,256) Q6.10
ALPHA   = M["alpha"].astype(np.int64)     # (600,)  Q6.10
BIAS    = M["bias"].astype(np.int64)      # (5,)    Q6.10
GAMMA_Q = int(M["gamma"])
COUNTS  = [int(c) for c in M["counts"]]
NUM_SV  = SV.shape[0]
NTEST   = min(X_te.shape[0], int(os.environ.get("COSIM_NTEST", X_te.shape[0])))  # smoke-test override

def u16(v):
    return int(v) & 0xFFFF

def sv_flat(counts):
    return sum(int(c) << (8 * i) for i, c in enumerate(counts))

async def write_param_raw(dut, addr, qval):
    """Write a RAW Q6.10 value to a parameter register (addr: 0=gamma, 2..6=bias0..4)."""
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 1
    dut.param_addr.value     = addr
    dut.param_data.value     = u16(qval)
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 0
    await RisingEdge(dut.clk)

async def ram_model(dut):
    """Serve the unified off-chip RAM: rows 0..NUM_SV-1 = SV matrix,
       rows NUM_SV.. = test features. addr = {row[10:0], col[7:0]}."""
    while True:
        await RisingEdge(dut.clk)
        addr = int(dut.ram_addr.value)
        row  = addr >> 8
        col  = addr & 0xFF
        if row < NUM_SV:
            dut.ram_rdata.value = u16(SV[row][col])
        else:
            s = row - NUM_SV
            dut.ram_rdata.value = u16(X_te[s][col]) if 0 <= s < NTEST else 0

@cocotb.test()
async def full_dataset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # ---- reset + static inputs ----
    dut.rst_n.value          = 0
    dut.vbatt_warn.value     = 0
    dut.vbatt_ok.value       = 1
    dut.start.value          = 0
    dut.num_samples.value    = NTEST          # stream all test beats in one batch
    dut.kernel_ready.value   = 1
    dut.param_write_en.value = 0
    dut.param_addr.value     = 0
    dut.param_data.value     = 0
    dut.ram_rdata.value      = 0
    dut.alpha_write_en.value = 0
    dut.alpha_addr.value     = 0
    dut.alpha_data.value     = 0
    dut.num_sv_per_class_flat.value = sv_flat(COUNTS)
    for _ in range(6):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    cocotb.start_soon(ram_model(dut))

    # ---- load Q6.10 model ----
    await write_param_raw(dut, 0, GAMMA_Q)              # gamma
    for c in range(5):
        await write_param_raw(dut, 2 + c, BIAS[c])     # OVR bias per class
    for a in range(NUM_SV):                            # 600 alpha coefficients
        await RisingEdge(dut.clk)
        dut.alpha_write_en.value = 1
        dut.alpha_addr.value     = a
        dut.alpha_data.value     = u16(ALPHA[a])
    await RisingEdge(dut.clk)
    dut.alpha_write_en.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)

    dut._log.info(f"model loaded: NUM_SV={NUM_SV} counts={COUNTS} gamma={GAMMA_Q} "
                  f"bias={list(BIAS)} streaming {NTEST} samples")

    # ---- stream the batch; capture class_out at each sample_rdy ----
    preds = []
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    TIMEOUT = 400_000_000
    prev_rdy = 0
    for _ in range(TIMEOUT):
        await RisingEdge(dut.clk)
        rdy = int(dut.sample_rdy.value)
        if rdy and not prev_rdy:                       # rising edge of sample_rdy
            preds.append(int(dut.class_out.value))
            if len(preds) % 50 == 0:
                dut._log.info(f"  classified {len(preds)}/{NTEST}")
        prev_rdy = rdy
        if int(dut.done.value) and len(preds) >= NTEST:
            break

    preds = np.array(preds[:NTEST], dtype=np.int32)
    np.save("/tmp/svm_cosim_preds.npy", preds)
    acc = float((preds == y_te[:len(preds)]).mean()) if len(preds) else 0.0
    dut._log.info(f"COSIM DONE: captured {len(preds)}/{NTEST}  hw_acc={acc:.4f}")
    assert len(preds) == NTEST, f"captured {len(preds)} != {NTEST} predictions"
