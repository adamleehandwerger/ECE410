# cocotb test for the GF180 chip_core SPI-slave bridge (wafer.space).
# Verifies the real chip_core register protocol (not the IHP svm_top_ihp cosim):
#   - byte-oriented SPI, mode 0, MSB first; header = {rd(1)/wr(0), addr[6:0]}
#   - PARAM write (0x03): [param_addr, data_hi, data_lo] -> pulses param_write_en
#   - GAMMA read  (0x41): 2 bytes, MSB first
#   - STATUS read (0x40)
# Pad map (chip_core): input_in[0]=sclk [1]=cs_n [2]=mosi ; bidir_out[36]=miso
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

SCLK, CSN, MOSI = 0, 1, 2      # input_in bit indices
MISO = 36                      # bidir_out bit index
CLKS_PER_EDGE = 6              # clk cycles per SPI half-bit (>= 2-FF sync depth)

ADDR_CTRL, ADDR_NSAMP, ADDR_NSVPC, ADDR_PARAM, ADDR_ALPHA = 0x00, 0x01, 0x02, 0x03, 0x04
ADDR_STATUS, ADDR_GAMMA = 0x40, 0x41

_INBUF = {"v": (1 << CSN)}      # Python shadow of input_in (avoids RMW read-back race)

def _set_in(dut, bit, val):
    _INBUF["v"] = (_INBUF["v"] | (1 << bit)) if val else (_INBUF["v"] & ~(1 << bit))
    dut.input_in.value = _INBUF["v"]

def _miso(dut):
    return (int(dut.bidir_out.value) >> MISO) & 1

async def _spi_byte(dut, tx):
    """Clock one byte MSB-first; return the byte read back on MISO (mode 0)."""
    rx = 0
    for i in range(8):
        _set_in(dut, MOSI, (tx >> (7 - i)) & 1)
        _set_in(dut, SCLK, 0)                        # sclk low: slave presented on entering fall
        await ClockCycles(dut.clk, CLKS_PER_EDGE)
        rx = (rx << 1) | _miso(dut)                  # sample at end of stable low period
        _set_in(dut, SCLK, 1)                        # rising edge
        await ClockCycles(dut.clk, CLKS_PER_EDGE)
    _set_in(dut, SCLK, 0)
    return rx

async def spi_write(dut, addr, payload):
    _set_in(dut, CSN, 0); await ClockCycles(dut.clk, CLKS_PER_EDGE)
    await _spi_byte(dut, addr & 0x7F)               # header, bit7=0 (write)
    for b in payload:
        await _spi_byte(dut, b & 0xFF)
    await ClockCycles(dut.clk, CLKS_PER_EDGE)
    _set_in(dut, CSN, 1); await ClockCycles(dut.clk, CLKS_PER_EDGE)

async def spi_read(dut, addr, nbytes):
    _set_in(dut, CSN, 0); await ClockCycles(dut.clk, CLKS_PER_EDGE)
    await _spi_byte(dut, 0x80 | (addr & 0x7F))      # header, bit7=1 (read)
    out = [await _spi_byte(dut, 0x00) for _ in range(nbytes)]
    await ClockCycles(dut.clk, CLKS_PER_EDGE)
    _set_in(dut, CSN, 1); await ClockCycles(dut.clk, CLKS_PER_EDGE)
    return out

async def _reset(dut):
    _INBUF["v"] = (1 << CSN)             # cs_n idle high, sclk/mosi low
    dut.input_in.value = _INBUF["v"]
    dut.bidir_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 8)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 8)

@cocotb.test()
async def test_default_gamma(dut):
    """After reset, GAMMA reads the Q6.10 default 0.25 = 0x0100."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    try:
        dut._log.info(f"[debug] internal gamma_reg = 0x{int(dut.gamma_reg.value):04x}")
    except Exception as e:
        dut._log.info(f"[debug] gamma_reg not accessible: {e}")
    # manual read with internal probes to localize the read path
    _set_in(dut, CSN, 0); await ClockCycles(dut.clk, CLKS_PER_EDGE)
    await _spi_byte(dut, 0x80 | ADDR_GAMMA)      # read header
    def g(name):
        try: return int(getattr(dut, name).value)
        except Exception: return -1
    dut._log.info(f"[dbg] after hdr: have_hdr={g('have_hdr')} rd_nwr={g('rd_nwr')} addr=0x{g('addr'):02x} tx_shift=0x{g('tx_shift'):02x} miso_q={g('miso_q')}")
    hi = await _spi_byte(dut, 0)
    dut._log.info(f"[dbg] byte0=0x{hi:02x} tx_shift=0x{g('tx_shift'):02x}")
    lo = await _spi_byte(dut, 0)
    _set_in(dut, CSN, 1); await ClockCycles(dut.clk, CLKS_PER_EDGE)
    dut._log.info(f"[dbg] SPI GAMMA = 0x{hi:02x} 0x{lo:02x}")
    assert (hi << 8 | lo) == 0x0100, f"default gamma via SPI = 0x{(hi<<8|lo):04x}, expected 0x0100"

@cocotb.test()
async def test_param_write_readback(dut):
    """SPI PARAM write of gamma is reflected in the GAMMA read (SPI wr+rd path)."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    new_g = 0x0200                      # 0.5 in Q6.10
    # PARAM payload: [param_addr=0 (gamma), data_hi, data_lo]
    await spi_write(dut, ADDR_PARAM, [0x00, (new_g >> 8) & 0xFF, new_g & 0xFF])
    await ClockCycles(dut.clk, 20)
    hi, lo = await spi_read(dut, ADDR_GAMMA, 2)
    g = (hi << 8) | lo
    assert g == new_g, f"gamma after SPI write = 0x{g:04x}, expected 0x{new_g:04x}"

@cocotb.test()
async def test_status_readable(dut):
    """STATUS is readable over SPI and idle after reset (done=0, no error)."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    hi, lo = await spi_read(dut, ADDR_STATUS, 2)
    status = (hi << 8) | lo
    # status_word bit layout: [15]=sample_rdy [14]=done [13]=error ...
    done = (status >> 14) & 1
    error = (status >> 13) & 1
    assert done == 0 and error == 0, f"idle STATUS unexpected: 0x{status:04x}"
