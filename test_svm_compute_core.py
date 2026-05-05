import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, First
import ctypes
import math

# ---------------------------------------------------------------------------
# Fixed-point helpers  (Q6.10 format, DATA_WIDTH=16, FRAC_BITS=10)
# ---------------------------------------------------------------------------
FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS   # 1024


def to_fixed(value: float) -> int:
    """Convert float to Q6.10 unsigned int (16-bit two's complement)."""
    raw = int(round(value * SCALE))
    return ctypes.c_uint16(raw).value


def from_fixed(raw) -> float:
    """Convert Q6.10 unsigned int back to signed float."""
    signed = ctypes.c_int16(int(raw)).value
    return signed / SCALE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.qspi_valid.value     = 0
    dut.qspi_data.value      = 0
    dut.start.value          = 0
    dut.num_samples.value    = 0
    dut.kernel_ready.value   = 1
    dut.param_write_en.value = 0
    dut.param_addr.value     = 0
    dut.param_data.value     = 0
    dut.sv_ram_rdata.value   = 0
    dut.work_ram_rdata.value = 0

    # Realistic unequal class distribution (60+45+55+50+40 = 250)
    sv_counts = [60, 45, 55, 50, 40]
    for i, n in enumerate(sv_counts):
        dut.num_sv_per_class[i].value = n

    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)


async def send_feature(dut, value: float):
    """Send one fixed-point feature word over the QSPI interface."""
    await RisingEdge(dut.clk)
    dut.qspi_valid.value = 1
    dut.qspi_data.value  = to_fixed(value)
    await RisingEdge(dut.clk)
    # Wait until chiplet is ready (back-pressure)
    while not dut.qspi_ready.value:
        await RisingEdge(dut.clk)
    dut.qspi_valid.value = 0


async def program_param(dut, addr: int, value: float):
    """Write a fixed-point value to a parameter register."""
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 1
    dut.param_addr.value     = addr
    dut.param_data.value     = to_fixed(value)
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


# ---------------------------------------------------------------------------
# SV RAM model — responds to sv_ram_ren with a simple sine-pattern value
# ---------------------------------------------------------------------------

async def sv_ram_model(dut):
    while True:
        await RisingEdge(dut.clk)
        # Use == 1 (not truthiness) so Logic('X') safely evaluates to False
        if dut.sv_ram_ren.value == 1:
            addr = int(dut.sv_ram_addr.value)
            val  = math.sin(addr * 0.01) * 0.5
            dut.sv_ram_rdata.value = to_fixed(val)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_reset_outputs(dut):
    """All outputs are de-asserted immediately after reset."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    assert dut.done.value        == 0, f"done should be 0, got {dut.done.value}"
    assert dut.error.value       == 0, f"error should be 0, got {dut.error.value}"
    assert dut.kernel_valid.value == 0, f"kernel_valid should be 0, got {dut.kernel_valid.value}"
    assert dut.qspi_ready.value  == 0, "qspi_ready should be 0 outside LOAD_FIFO state"


@cocotb.test()
async def test_param_programming(dut):
    """Gamma and C registers accept writes and read back correctly."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    # Write gamma = 0.005
    await program_param(dut, 0, 0.005)
    readback = from_fixed(dut.gamma_reg.value.to_unsigned())
    assert abs(readback - 0.005) < 1e-3, f"gamma readback {readback:.6f} != 0.005"

    # Write C = 2.0
    await program_param(dut, 1, 2.0)
    readback = from_fixed(dut.c_reg.value.to_unsigned())
    assert abs(readback - 2.0) < 0.01, f"C readback {readback:.6f} != 2.0"

    # Restore defaults
    await program_param(dut, 0, 0.25)
    await program_param(dut, 1, 1.0)
    assert abs(from_fixed(dut.gamma_reg.value.to_unsigned()) - 0.25) < 1e-3
    assert abs(from_fixed(dut.c_reg.value.to_unsigned())    - 1.0)  < 0.01


@cocotb.test()
async def test_sv_counts_set(dut):
    """num_sv_per_class values are applied and total to 250."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    expected = [60, 45, 55, 50, 40]
    total    = 0
    for i, exp in enumerate(expected):
        got    = int(dut.num_sv_per_class[i].value)
        total += got
        assert got == exp, f"Class {i}: expected {exp} SVs, got {got}"

    assert total == 250, f"Total SVs should be 250, got {total}"


@cocotb.test()
async def test_sv_counts_unequal_stress(dut):
    """Extreme imbalance (100/10/80/40/20) is accepted without error."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    stressed = [100, 10, 80, 40, 20]
    for i, n in enumerate(stressed):
        dut.num_sv_per_class[i].value = n

    await RisingEdge(dut.clk)
    assert dut.error.value == 0, "error asserted after loading extreme SV distribution"

    total = sum(int(dut.num_sv_per_class[i].value) for i in range(5))
    assert total == 250, f"Stressed total should be 250, got {total}"


@cocotb.test()
async def test_qspi_fifo_load(dut):
    """FIFO accepts 256 feature words without asserting full."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    # num_samples must be set BEFORE start so LOAD_FIFO doesn't exit immediately
    dut.num_samples.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Send one full feature vector (256 features)
    for i in range(256):
        val = math.cos(i * 0.05) * 0.8
        await send_feature(dut, val)

    # FIFO should not have overflowed (no error)
    assert dut.error.value == 0, "error asserted after loading 256 features"


@cocotb.test()
async def test_qspi_backpressure(dut):
    """qspi_ready de-asserts when FIFO is full (overflow protection)."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    # Large num_samples keeps FSM in LOAD_FIFO long enough to fill the FIFO
    dut.num_samples.value = 1000
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Fill FIFO to depth (8192) and count rejected writes
    rejected = 0
    for i in range(8192 + 200):
        await RisingEdge(dut.clk)
        if dut.qspi_ready.value:
            dut.qspi_valid.value = 1
            dut.qspi_data.value  = to_fixed(i * 0.001)
        else:
            rejected            += 1
            dut.qspi_valid.value = 0

    dut.qspi_valid.value = 0
    assert rejected > 0, "FIFO should have signalled full before 8392 entries"


@cocotb.test()
async def test_default_gamma_fixed_point(dut):
    """Default gamma register encodes 0.25 correctly in Q6.10."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset_dut(dut)

    gamma_val = from_fixed(dut.gamma_reg.value.to_unsigned())
    assert abs(gamma_val - 0.25) < 1e-3, \
        f"Default gamma {gamma_val:.6f} deviates from 0.25 by more than 1e-3"


@cocotb.test()
async def test_full_pipeline_small_batch(dut):
    """Single-sample pipeline: load → start → wait for done, no error."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    cocotb.start_soon(sv_ram_model(dut))
    await reset_dut(dut)

    # Use equal distribution for a clean small test (2 SVs × 5 classes = 10 total)
    for i in range(5):
        dut.num_sv_per_class[i].value = 2

    # Start BEFORE loading features so the FSM is in LOAD_FIFO when features arrive
    dut.num_samples.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Load one feature vector while FSM is in LOAD_FIFO
    for i in range(256):
        await send_feature(dut, math.sin(i * 0.05) * 0.5)

    # Wait for done with a timeout of 100,000 cycles
    for _ in range(100_000):
        await RisingEdge(dut.clk)
        if dut.done.value:
            break
    else:
        assert False, "Timeout: done never asserted for single-sample batch"

    assert dut.error.value == 0, "error asserted during single-sample pipeline"


@cocotb.test()
async def test_kernel_output_range(dut):
    """All kernel outputs are in [0, 1] (RBF property)."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    cocotb.start_soon(sv_ram_model(dut))
    await reset_dut(dut)

    for i in range(5):
        dut.num_sv_per_class[i].value = 2

    # Start BEFORE loading so FSM is in LOAD_FIFO when features arrive
    dut.num_samples.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for i in range(256):
        await send_feature(dut, (i % 10) * 0.1)

    out_of_range = 0
    for _ in range(100_000):
        await RisingEdge(dut.clk)
        if dut.kernel_valid.value:
            k = from_fixed(dut.kernel_out.value.to_unsigned())
            if k < -0.01 or k > 1.01:
                out_of_range += 1
        if dut.done.value:
            break

    assert out_of_range == 0, f"{out_of_range} kernel outputs were outside [0, 1]"
