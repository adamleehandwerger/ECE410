// ===========================================================================
// SVM Compute Core — SystemVerilog Interfaces
// ===========================================================================
// Three interfaces map to the three physical boundaries of svm_compute_core:
//
//   svm_host_if   — MCU ↔ Core  (QSPI features, parameters, control/status,
//                                 num_sv_per_class, kernel output stream)
//   svm_sv_ram_if — Core ↔ Support-Vector SRAM  (128 KB, read-only at runtime)
//   svm_work_ram_if — Core ↔ Workspace SRAM     (≤500 KB, read/write)
//
// Each interface declares modports for both sides so direction errors are
// caught at elaboration time when the interface is bound to a module port.
//
// ===========================================================================
// QSPI PROTOCOL — clock polarity, phase, and word framing
// ===========================================================================
//
//   Mode        : SPI Mode 0  (CPOL = 0, CPHA = 0)
//   CPOL = 0    : SCK idles LOW between transfers.
//   CPHA = 0    : Data is CAPTURED on the RISING SCK edge;
//                 data is SHIFTED OUT on the FALLING SCK edge.
//
//   Physical pins (chiplet boundary):
//     SCK        — serial clock driven by MCU master
//     CS_N       — active-low chip select driven by MCU master
//     IO[3:0]    — quad data bus; bidirectional, half-duplex
//                  (MCU drives IO[3:0] during feature/command writes;
//                   chiplet drives IO[3:0] during kernel output reads)
//
//   Word framing (feature input, 16-bit Q6.10):
//     Each 16-bit word occupies 4 SCK cycles on 4 data lanes.
//     Bit order: MSB first; IO[3] carries the MSB of each nibble.
//
//       SCK cycle :  1            2            3            4
//       IO[3:0]   :  bits[15:12]  bits[11:8]   bits[7:4]    bits[3:0]
//                    ──────────── ──────────── ──────────── ────────────
//                    captured on rising SCK (CPHA=0)
//
//   Deserialization:
//     A QSPI deserializer (not part of this RTL) assembles the 4 nibbles
//     into a 16-bit word and presents it as a single-cycle pulse on the
//     qspi_valid / qspi_data / qspi_ready ready-valid bus.
//     The chiplet's internal logic sees only the deserialized bus.
//
//   Transfer rate:
//     4 MHz SCK × 4 lanes ÷ 16 bits/word = 1 M words/sec = 2 MB/s
//     256 features/heartbeat → one heartbeat's features arrive in 256 µs.
//
// ===========================================================================
// REGISTER MAP  (param_addr / param_data interface, svm_host_if)
// ===========================================================================
//
//   Access protocol:
//     Assert param_write_en = 1 for exactly one cycle.
//     param_addr selects the register; param_data carries the new value.
//     Both registers are readable at any time via the combinational
//     readback outputs gamma_reg and c_reg.
//
//   Address  Register        Width   Format   Reset default    Notes
//   ───────────────────────────────────────────────────────────────────────
//   2'b00    gamma_reg       16-bit  Q6.10    10  (≈ 0.01)     RBF bandwidth γ
//   2'b01    c_reg           16-bit  Q6.10    1024 (= 1.0)     SVM penalty C
//   2'b10    (reserved)      —       —        —                writes ignored
//   2'b11    (reserved)      —       —        —                writes ignored
//
//   Q6.10 encoding reminder:
//     raw integer = round(real_value × 1024)
//     gamma = 0.01  →  raw = 10    (actual stored ≈ 0.009766)
//     C     = 1.0   →  raw = 1024  (exact)
//     Range: −32.000 to +31.999; resolution ≈ 0.000977
//
//   Non-programmable control fields (set in svm_host_if, not via param_addr):
//
//   Field              Width   Format     Notes
//   ───────────────────────────────────────────────────────────────────────
//   num_sv_per_class   5×8-bit uint8      SV count for each of 5 classes.
//                                         Latched on start; sum must be
//                                         1–250 or error flag is set.
//   num_samples        10-bit  uint10     Heartbeats in this batch (1–1000).
//                                         Latched on start pulse.
//   start              1-bit   pulse      One-cycle high pulse in IDLE only.
//   done               1-bit   pulse      One-cycle high after last kernel.
//   error              1-bit   sticky     Set on illegal state or SV count
//                                         violation; cleared by rst_n only.
//
// ===========================================================================

// ---------------------------------------------------------------------------
// svm_host_if
// Connects the host MCU to svm_compute_core.
// Modports:
//   .host  — signals as driven/read by the MCU
//   .core  — signals as driven/read by svm_compute_core
// ---------------------------------------------------------------------------
interface svm_host_if #(
    parameter int DATA_WIDTH = 16
) (
    input logic clk,
    input logic rst_n
);

    // --- Parameter programming ---
    logic                   param_write_en;
    logic [1:0]             param_addr;
    logic [DATA_WIDTH-1:0]  param_data;
    logic [DATA_WIDTH-1:0]  gamma_reg;     // readback
    logic [DATA_WIDTH-1:0]  c_reg;         // readback

    // --- Per-class SV counts (set before asserting start) ---
    logic [7:0]             num_sv_per_class [5];

    // --- QSPI feature stream ---
    logic                   qspi_valid;
    logic [DATA_WIDTH-1:0]  qspi_data;
    logic                   qspi_ready;

    // --- Batch control ---
    logic                   start;
    logic [9:0]             num_samples;

    // --- Status outputs ---
    logic                   done;
    logic                   error;

    // --- Kernel output stream ---
    logic [DATA_WIDTH-1:0]  kernel_out;
    logic                   kernel_valid;
    logic                   kernel_ready;

    // MCU drives: param writes, QSPI stream, sv_counts, start/num_samples,
    //             kernel_ready. MCU reads: gamma_reg, c_reg, done, error,
    //             kernel_out/valid, qspi_ready.
    modport host (
        output param_write_en,
        output param_addr,
        output param_data,
        input  gamma_reg,
        input  c_reg,
        output num_sv_per_class,
        output qspi_valid,
        output qspi_data,
        input  qspi_ready,
        output start,
        output num_samples,
        input  done,
        input  error,
        input  kernel_out,
        input  kernel_valid,
        output kernel_ready
    );

    // Core drives: gamma_reg, c_reg, qspi_ready, done, error, kernel_out,
    //             kernel_valid. Core reads everything the host drives.
    modport core (
        input  param_write_en,
        input  param_addr,
        input  param_data,
        output gamma_reg,
        output c_reg,
        input  num_sv_per_class,
        input  qspi_valid,
        input  qspi_data,
        output qspi_ready,
        input  start,
        input  num_samples,
        output done,
        output error,
        output kernel_out,
        output kernel_valid,
        input  kernel_ready
    );

endinterface : svm_host_if


// ---------------------------------------------------------------------------
// svm_sv_ram_if
// Connects svm_compute_core to the off-chip Support Vector SRAM.
// 250 SVs × 256 features × 2 bytes = 128 KB; 18-bit address, read-only.
// Modports:
//   .core — as seen by the compute core (initiates reads)
//   .ram  — as seen by the SRAM (responds to reads)
// ---------------------------------------------------------------------------
interface svm_sv_ram_if #(
    parameter int DATA_WIDTH  = 16,
    parameter int ADDR_WIDTH  = 18
) ();

    logic [ADDR_WIDTH-1:0]  addr;
    logic [DATA_WIDTH-1:0]  rdata;
    logic                   ren;

    // Core asserts addr/ren; SRAM returns rdata one cycle later.
    modport core (
        output addr,
        output ren,
        input  rdata
    );

    // SRAM sees addr/ren as inputs and drives rdata.
    modport ram (
        input  addr,
        input  ren,
        output rdata
    );

endinterface : svm_sv_ram_if


// ---------------------------------------------------------------------------
// svm_work_ram_if
// Connects svm_compute_core to the off-chip Workspace SRAM.
// 1000 samples × 250 SVs × 2 bytes = 500 KB max; 18-bit address, R/W.
// Modports:
//   .core — as seen by the compute core (initiates reads and writes)
//   .ram  — as seen by the SRAM
// ---------------------------------------------------------------------------
interface svm_work_ram_if #(
    parameter int DATA_WIDTH  = 16,
    parameter int ADDR_WIDTH  = 18
) ();

    logic [ADDR_WIDTH-1:0]  addr;
    logic [DATA_WIDTH-1:0]  wdata;
    logic [DATA_WIDTH-1:0]  rdata;
    logic                   wen;
    logic                   ren;

    modport core (
        output addr,
        output wdata,
        input  rdata,
        output wen,
        output ren
    );

    modport ram (
        input  addr,
        input  wdata,
        output rdata,
        input  wen,
        input  ren
    );

endinterface : svm_work_ram_if


// ===========================================================================
// Instantiation guide — how to wire the three interfaces to svm_compute_core
// in a testbench or SoC top-level (Questa / VCS / Xcelium; Icarus 13 does
// not support interface types in module port lists).
//
//   logic clk, rst_n;
//
//   svm_host_if    #(.DATA_WIDTH(16))  host    (.clk(clk), .rst_n(rst_n));
//   svm_sv_ram_if  #(.DATA_WIDTH(16))  sv_ram  ();
//   svm_work_ram_if#(.DATA_WIDTH(16))  work_ram();
//
//   svm_compute_core u_core (
//       .clk             (clk),
//       .rst_n           (rst_n),
//       .param_write_en  (host.param_write_en),
//       .param_addr      (host.param_addr),
//       .param_data      (host.param_data),
//       .gamma_reg       (host.gamma_reg),
//       .c_reg           (host.c_reg),
//       .num_sv_per_class(host.num_sv_per_class),
//       .qspi_valid      (host.qspi_valid),
//       .qspi_data       (host.qspi_data),
//       .qspi_ready      (host.qspi_ready),
//       .start           (host.start),
//       .num_samples     (host.num_samples),
//       .done            (host.done),
//       .error           (host.error),
//       .kernel_out      (host.kernel_out),
//       .kernel_valid    (host.kernel_valid),
//       .kernel_ready    (host.kernel_ready),
//       .sv_ram_addr     (sv_ram.addr),
//       .sv_ram_rdata    (sv_ram.rdata),
//       .sv_ram_ren      (sv_ram.ren),
//       .work_ram_addr   (work_ram.addr),
//       .work_ram_wdata  (work_ram.wdata),
//       .work_ram_rdata  (work_ram.rdata),
//       .work_ram_wen    (work_ram.wen),
//       .work_ram_ren    (work_ram.ren)
//   );
// ===========================================================================
