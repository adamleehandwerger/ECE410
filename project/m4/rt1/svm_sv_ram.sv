// SPDX-License-Identifier: Apache-2.0
// svm_sv_ram.sv — Support vector RAM for 256 SVs × 128 features × 16-bit
//
// Total storage: 256 × 128 × 2 bytes = 65,536 bytes = 64 KB
//
// Implementation: four sky130_sram_16kbyte_1rw1r_32x4096_8 macros
//   Each macro: 4096 × 32-bit = 16KB → stores 8192 × 16-bit entries
//   4 macros × 8192 = 32,768 entries (covers 256 × 128 = 32,768 exactly)
//
// Address layout: sv_ram_addr[14:0] (15 bits = 32,768 entries)
//   sv_ram_addr[0]     = half-word select (0=lower 16b, 1=upper 16b)
//   sv_ram_addr[12:1]  = row address within macro (12 bits → 4096 rows)
//   sv_ram_addr[14:13] = macro select (2 bits → macros 0–3)
//
// NOTE: bits 12 and 14:13 do NOT overlap — this is the corrected decode.
//
// SV weights are loaded by management SoC via Wishbone before inference.
// Port 0 (RW): used by Wishbone loader
// Port 1 (RO): used by svm_compute_core read port

`default_nettype none

module svm_sv_ram (
    input  wire        clk,
    // svm_compute_core read port
    input  wire [14:0] rd_addr,
    input  wire        rd_en,
    output reg  [15:0] rd_data,
    // Wishbone write port (load SV weights)
    input  wire [14:0] wr_addr,
    input  wire [15:0] wr_data,
    input  wire        wr_en
);

    // -------------------------------------------------------------------------
    // Address decode — non-overlapping bit fields
    //   [0]     = half-word select
    //   [12:1]  = row address within macro (4096 rows)
    //   [14:13] = macro select (4 macros)
    // -------------------------------------------------------------------------
    wire [1:0]  rd_macro = rd_addr[14:13];
    wire [11:0] rd_word  = rd_addr[12:1];
    wire        rd_half  = rd_addr[0];

    wire [1:0]  wr_macro = wr_addr[14:13];
    wire [11:0] wr_word  = wr_addr[12:1];
    wire        wr_half  = wr_addr[0];

    // -------------------------------------------------------------------------
    // Per-macro chip selects
    // -------------------------------------------------------------------------
    wire [3:0] rd_csb, wr_csb;
    genvar i;
    generate
        for (i = 0; i < 4; i = i+1) begin : macro_cs
            assign rd_csb[i] = ~(rd_en && (rd_macro == i[1:0]));
            assign wr_csb[i] = ~(wr_en && (wr_macro == i[1:0]));
        end
    endgenerate

    wire [3:0]  wr_wmask  = wr_half ? 4'b1100 : 4'b0011;
    wire [31:0] wr_data32 = wr_half ? {wr_data, 16'h0} : {16'h0, wr_data};

    // -------------------------------------------------------------------------
    // Macro outputs — mux read data
    // -------------------------------------------------------------------------
    wire [31:0] macro_dout [0:3];
    reg  [1:0]  rd_macro_r;
    reg         rd_half_r;
    always @(posedge clk) begin
        rd_macro_r <= rd_macro;
        rd_half_r  <= rd_half;
        rd_data    <= rd_half_r ? macro_dout[rd_macro_r][31:16]
                                : macro_dout[rd_macro_r][15:0];
    end

    // -------------------------------------------------------------------------
    // Four sky130_sram_16kbyte_1rw1r_32x4096_8 macros
    // -------------------------------------------------------------------------
    generate
        for (i = 0; i < 4; i = i+1) begin : sv_macros
            sky130_sram_16kbyte_1rw1r_32x4096_8 macro (
                // Port 0: Wishbone write
                .clk0  (clk),
                .csb0  (wr_csb[i]),
                .web0  (~wr_en),
                .wmask0(wr_wmask),
                .addr0 (wr_word),
                .din0  (wr_data32),
                .dout0 (),
                // Port 1: svm_compute_core read
                .clk1  (clk),
                .csb1  (rd_csb[i]),
                .addr1 (rd_word),
                .dout1 (macro_dout[i])
            );
        end
    endgenerate

endmodule
`default_nettype wire
