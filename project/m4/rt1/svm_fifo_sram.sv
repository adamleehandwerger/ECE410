// SPDX-License-Identifier: Apache-2.0
// svm_fifo_sram.v — 8192×16 synchronous FIFO backed by sky130 SRAM macros
//
// Replaces the register-based input_fifo in svm_compute_core to cut area
// from ~2 mm² (131K FFs) to ~0.18 mm² (two 8KB SRAM macros).
//
// Uses two sky130_sram_8kbyte_1rw1r_32x2048_8 macros (32-bit × 2048 words
// each = 8KB each). We pack two 16-bit FIFO words per 32-bit SRAM word,
// so each macro holds 4096 16-bit entries. Two macros → 8192 entries total.
//
// Each macro has one read-write port (port 0) and one read-only port (port 1).
// We use port 0 for writes and port 1 for reads (simultaneous R/W safe).
//
// Interface: identical to the existing input_fifo module so it's a drop-in
// replacement inside svm_compute_core.sv (just swap the instantiation).

`default_nettype none

module svm_fifo_sram #(
    parameter DATA_WIDTH = 16,
    parameter DEPTH      = 8192,    // must be 8192 for this macro config
    parameter ADDR_WIDTH = 13
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // Write port
    input  wire                  wr_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    output wire                  full,
    // Read port
    input  wire                  rd_en,
    output reg  [DATA_WIDTH-1:0] rd_data,
    output wire                  empty,
    // Status
    output wire [ADDR_WIDTH:0]   count
);

    // -------------------------------------------------------------------------
    // Pointers: 14-bit (bit 13 = wrap flag, bits 12:0 = address)
    // -------------------------------------------------------------------------
    reg [ADDR_WIDTH:0] wr_ptr, rd_ptr;
    wire [ADDR_WIDTH-1:0] wr_addr = wr_ptr[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] rd_addr = rd_ptr[ADDR_WIDTH-1:0];

    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) &&
                   (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);
    assign count = wr_ptr - rd_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            rd_data <= 0;
        end else begin
            if (wr_en && !full)  wr_ptr <= wr_ptr + 1;
            if (rd_en && !empty) rd_ptr <= rd_ptr + 1;
        end
    end

    // -------------------------------------------------------------------------
    // SRAM addressing
    // Address 0-4095   → macro 0, words 0-2047, both halves
    // Address 4096-8191 → macro 1, words 0-2047, both halves
    // Within each macro: word_addr = entry[12:1], half = entry[0]
    // -------------------------------------------------------------------------
    wire        wr_macro_sel   = wr_addr[12];    // 0=macro0, 1=macro1
    wire [10:0] wr_word_addr   = wr_addr[11:1];  // 11-bit word index in macro
    wire        wr_half        = wr_addr[0];     // 0=lower 16b, 1=upper 16b

    wire        rd_macro_sel   = rd_addr[12];
    wire [10:0] rd_word_addr   = rd_addr[11:1];
    wire        rd_half        = rd_addr[0];

    // Write mask and data for 32-bit SRAM port
    wire [3:0]  wr_wmask0 = (!wr_macro_sel && wr_en && !full) ?
                             (wr_half ? 4'b1100 : 4'b0011) : 4'b0000;
    wire [3:0]  wr_wmask1 = ( wr_macro_sel && wr_en && !full) ?
                             (wr_half ? 4'b1100 : 4'b0011) : 4'b0000;

    wire [31:0] wr_data32 = wr_half ? {wr_data, 16'h0} : {16'h0, wr_data};

    // Read data from macros
    wire [31:0] rd_data32_0, rd_data32_1;
    wire [31:0] rd_data32 = rd_macro_sel ? rd_data32_1 : rd_data32_0;

    always @(posedge clk)
        rd_data <= rd_half ? rd_data32[31:16] : rd_data32[15:0];

    // -------------------------------------------------------------------------
    // SRAM macro instantiations
    // sky130_sram_8kbyte_1rw1r_32x2048_8:
    //   Port 0 (RW): clk0, csb0, web0, wmask0[3:0], addr0[10:0], din0[31:0], dout0[31:0]
    //   Port 1 (RO): clk1, csb1, addr1[10:0], dout1[31:0]
    // -------------------------------------------------------------------------
    // Macro 0: entries 0–4095
    sky130_sram_8kbyte_1rw1r_32x2048_8 macro0 (
        // Port 0: write
        .clk0  (clk),
        .csb0  (wr_macro_sel),           // active-low chip select
        .web0  (~(wr_en && !full)),       // active-low write enable
        .wmask0(wr_wmask0),
        .addr0 (wr_word_addr),
        .din0  (wr_data32),
        .dout0 (),
        // Port 1: read
        .clk1  (clk),
        .csb1  (rd_macro_sel),           // select macro 0 only when rd_macro_sel=0
        .addr1 (rd_word_addr),
        .dout1 (rd_data32_0)
    );

    // Macro 1: entries 4096–8191
    sky130_sram_8kbyte_1rw1r_32x2048_8 macro1 (
        .clk0  (clk),
        .csb0  (~wr_macro_sel),
        .web0  (~(wr_en && !full)),
        .wmask0(wr_wmask1),
        .addr0 (wr_word_addr),
        .din0  (wr_data32),
        .dout0 (),
        .clk1  (clk),
        .csb1  (~rd_macro_sel),
        .addr1 (rd_word_addr),
        .dout1 (rd_data32_1)
    );

endmodule
`default_nettype wire
