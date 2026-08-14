"""
test_svm_compute_core.py — Level 2 cocotb integration tests for svm_compute_core (m6)

Target: svm_compute_core (m6 unified RAM interface)
Run via:  make level2   (see Makefile)

Uses default parameters: NUM_SV=500, FEATURE_DIM=256, RAM_LATENCY=3, MAX_BATCH_SIZE=1000
Tests use sv_counts=[2,2,2,2,2] (10 SVs total, well below 500) for manageable runtime.

Address layout: ram_addr = {row[10:0], col[7:0]}
  rows 0..NUM_SV-1           → SV matrix
  rows NUM_SV..NUM_SV+N-1   → N input samples
"""

import cocotb
import math
import ctypes
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ---------------------------------------------------------------------------
# Fixed-point helpers  (Q6.10 format)
# ---------------------------------------------------------------------------
FRAC = 10
SCALE = 1 << FRAC   # 1024


def to_q(value: float) -> int:
    return ctypes.c_uint16(int(round(value * SCALE))).value


def from_q(raw) -> float:
    return ctypes.c_int16(int(raw)).value / SCALE


def sv_flat(counts) -> int:
    """Pack list of 5 sv_counts into the 40-bit num_sv_per_class_flat register."""
    return sum(int(c) << (8 * i) for i, c in enumerate(counts))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
NUM_SV_DEFAULT = 500   # must match compile-time parameter


async def reset_dut(dut, sv_counts=None):
    """Full synchronous reset; sv_counts default = [2,2,2,2,2]."""
    if sv_counts is None:
        sv_counts = [2, 2, 2, 2, 2]
    dut.rst_n.value           = 0
    dut.vbatt_warn.value      = 0
    dut.vbatt_ok.value        = 1
    dut.start.value           = 0
    dut.num_samples.value     = 1
    dut.kernel_ready.value    = 1
    dut.param_write_en.value  = 0
    dut.param_addr.value      = 0
    dut.param_data.value      = 0
    dut.ram_rdata.value       = to_q(1.0)   # safe idle value
    dut.alpha_write_en.value  = 0
    dut.alpha_addr.value      = 0
    dut.alpha_data.value      = 0
    dut.num_sv_per_class_flat.value = sv_flat(sv_counts)
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)


async def write_param(dut, addr: int, value: float):
    """Write a fixed-point value to a parameter register."""
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 1
    dut.param_addr.value     = addr
    dut.param_data.value     = to_q(value)
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 0
    await RisingEdge(dut.clk)


async def ram_model(dut, data_fn, num_sv=NUM_SV_DEFAULT):
    """
    Drive ram_rdata one clock after ram_addr is presented (LAT=1 effective).
    DUT's RAM_LATENCY wait-state counter absorbs any additional latency.
    data_fn(row, col) → uint16 value.
    """
    while True:
        await RisingEdge(dut.clk)
        addr = int(dut.ram_addr.value)
        row  = addr >> 8
        col  = addr & 0xFF
        dut.ram_rdata.value = data_fn(row, col)


async def run_batch(dut, timeout=200_000):
    """Pulse start and await done; return True if done fired before timeout."""
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.done.value:
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset_outputs(dut):
    """All outputs are de-asserted immediately after reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert dut.done.value        == 0, f"done={dut.done.value} expected 0"
    assert dut.error.value       == 0, f"error={dut.error.value} expected 0"
    assert dut.kernel_valid.value == 0, f"kernel_valid={dut.kernel_valid.value} expected 0"
    assert dut.sample_rdy.value  == 0, f"sample_rdy={dut.sample_rdy.value} expected 0"
    assert dut.ram_ren.value     == 0, f"ram_ren={dut.ram_ren.value} expected 0 in IDLE"


@cocotb.test()
async def test_param_programming(dut):
    """Gamma and C registers accept writes and read back correctly in Q6.10."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await write_param(dut, 0, 0.125)
    got = from_q(dut.gamma_reg.value)
    assert abs(got - 0.125) < 1e-3, f"gamma readback {got:.6f} != 0.125"

    await write_param(dut, 1, 2.0)
    got = from_q(dut.c_reg.value)
    assert abs(got - 2.0) < 0.01, f"C readback {got:.6f} != 2.0"

    # Restore
    await write_param(dut, 0, 0.25)
    await write_param(dut, 1, 1.0)
    assert abs(from_q(dut.gamma_reg.value) - 0.25) < 1e-3
    assert abs(from_q(dut.c_reg.value)    - 1.0)  < 0.01


@cocotb.test()
async def test_sv_counts_flat(dut):
    """num_sv_per_class_flat encoding: valid counts → no error; sum > NUM_SV → ERR_SV_OVERFLOW."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut, sv_counts=[2, 2, 2, 2, 2])

    # Valid counts (sum=10 << NUM_SV=500): start should not immediately error
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    assert int(dut.error_code.value) not in (0x1, 0x2), \
        f"Unexpected error 0x{int(dut.error_code.value):X} with valid sv_counts"

    # Overflow: sum = 5×200 = 1000 > NUM_SV=500 → ERR_SV_OVERFLOW (0x2)
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.num_sv_per_class_flat.value = sv_flat([200, 200, 200, 200, 200])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    code = int(dut.error_code.value)
    assert code == 0x2, f"ERR_SV_OVERFLOW expected (0x2), got 0x{code:X}"


@cocotb.test()
async def test_default_gamma(dut):
    """Default gamma register encodes 0.25 = 0x0100 in Q6.10 immediately after reset."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    gamma = from_q(dut.gamma_reg.value)
    assert abs(gamma - 0.25) < 1e-3, \
        f"Default gamma {gamma:.6f} deviates from 0.25"


@cocotb.test()
async def test_full_pipeline(dut):
    """Single-sample batch: SV=input=1.0 → dist=0 → all kernels=1024; done fires."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Constant SRAM: both SV and input = to_q(1.0) = 0x0400 → dist=0 → kernel=1024
    def const_ram(row, col):
        return to_q(1.0)

    cocotb.start_soon(ram_model(dut, const_ram))
    await reset_dut(dut, sv_counts=[2, 2, 2, 2, 2])  # 10 SVs
    await write_param(dut, 0, 0.25)                    # gamma = 0.25

    kernel_vals = []
    async def collect_kernels():
        while True:
            await RisingEdge(dut.clk)
            if dut.kernel_valid.value and dut.kernel_ready.value:
                kernel_vals.append(int(dut.kernel_out.value))

    cocotb.start_soon(collect_kernels())

    fired = await run_batch(dut)
    assert fired, "Timeout: done never asserted for single-sample batch"
    assert dut.error.value == 0 or int(dut.error_code.value) >= 0x8, \
        f"Real fault 0x{int(dut.error_code.value):X} asserted"

    assert len(kernel_vals) == 10, \
        f"Expected 10 kernel outputs (2 SVs × 5 classes), got {len(kernel_vals)}"
    for i, k in enumerate(kernel_vals):
        assert k == 1024, f"kernel[{i}]={k} expected 1024 (dist=0)"


@cocotb.test()
async def test_kernel_range(dut):
    """RBF kernel outputs are in [0, 1024] (Q6.10 equivalent of [0, 1.0])."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Sine-wave SV data (bounded [-0.3, 0.3]); input rows = 0.0 → non-trivial distance
    def sine_ram(row, col):
        if row < NUM_SV_DEFAULT:
            return to_q(math.sin(row * 0.3 + col * 0.05) * 0.3)
        else:
            return to_q(0.0)

    cocotb.start_soon(ram_model(dut, sine_ram))
    await reset_dut(dut, sv_counts=[2, 2, 2, 2, 2])
    await write_param(dut, 0, 0.25)

    out_of_range = 0
    async def check_kernels():
        nonlocal out_of_range
        while True:
            await RisingEdge(dut.clk)
            if dut.kernel_valid.value:
                k = int(dut.kernel_out.value)
                if k < 0 or k > 1024:
                    out_of_range += 1

    cocotb.start_soon(check_kernels())

    fired = await run_batch(dut)
    assert fired, "Timeout: done never asserted in kernel_range test"
    assert out_of_range == 0, f"{out_of_range} kernel(s) outside [0, 1024]"


@cocotb.test()
async def test_multi_sample(dut):
    """num_samples=3: done fires exactly once; kernel count = 3 × 10 = 30."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    def const_ram(row, col):
        return to_q(1.0)

    cocotb.start_soon(ram_model(dut, const_ram))
    await reset_dut(dut, sv_counts=[2, 2, 2, 2, 2])
    await write_param(dut, 0, 0.25)

    dut.num_samples.value = 3

    kernel_count = 0
    done_count   = 0

    async def count_outputs():
        nonlocal kernel_count, done_count
        while True:
            await RisingEdge(dut.clk)
            if dut.kernel_valid.value:
                kernel_count += 1
            if dut.done.value:
                done_count += 1

    cocotb.start_soon(count_outputs())

    fired = await run_batch(dut, timeout=500_000)
    # Let counters settle for a few extra cycles
    for _ in range(10):
        await RisingEdge(dut.clk)

    assert fired, "Timeout: done never asserted for 3-sample batch"
    assert done_count == 1, f"done fired {done_count} times (expected 1)"
    assert kernel_count == 30, \
        f"Expected 30 kernels (3 samples × 10 SVs), got {kernel_count}"
