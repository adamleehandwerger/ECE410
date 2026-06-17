// SPDX-FileCopyrightText: 2026 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// svm_top_ihp — Standalone IHP SG13G2 top-level wrapper
// ECE 410 Project  |  Milestone: m6
//
// Replaces Caravel user_project_wrapper from m5.  Key differences:
//   - No Caravel Wishbone bus; MCU configuration via SPI slave (CPOL=0, CPHA=0)
//   - ram_rdata[15:0] sourced from ram_rdata_in[15:0] (dedicated input pads)
//     — eliminates the m5 la_data_in relay through the management SoC
//   - 500-SV alpha_table: alpha_addr 10-bit, ALPHA_WR sv_idx field [25:16]
//   - NUM_SV reset defaults [95,95,95,120,95] (VT-boosted 500-SV)
//   - NUM_SAMPLES reset default 10'd1000 (sticky — write once at startup)
//   - IHP SG13G2 ICG cell: sg13g2_dlclkp_1 (replaces sky130 dlclkp)
//
// SPI register map (address byte [6:0]; bit[7]=0 write, bit[7]=1 read):
//   0x01 RW  CONTROL      [0]=start(auto-clear) [1]=vbatt_ok [2]=vbatt_warn
//   0x02 RO  STATUS       [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy
//   0x03 RW  NUM_SAMPLES  [9:0]  default=1000  (sticky — write once at startup)
//   0x04 RW  NUM_SV[0]    [7:0]  default=95   (Normal)
//   0x05 RW  NUM_SV[1]    [7:0]  default=95   (PVC)
//   0x06 RW  NUM_SV[2]    [7:0]  default=95   (AFib)
//   0x07 RW  NUM_SV[3]    [7:0]  default=120  (VT — boosted)
//   0x08 RW  NUM_SV[4]    [7:0]  default=95   (SVT)
//   0x09 WO  PARAM_WR     [19]=en [18:16]=addr [15:0]=data
//   0x0A WO  ALPHA_WR     [25:16]=sv_global_idx(10-bit) [15:0]=alpha Q6.10
//
// SPI protocol (CPOL=0, CPHA=0, MSB first):
//   CS# low -> clock in 8-bit address byte -> clock in/out 32-bit data -> CS# high
//   Write: addr[7]=0, addr[6:0]=reg, data[31:0]=value
//   Read:  addr[7]=1, addr[6:0]=reg, MISO clocks out data[31:0] from bit 31 down
//
// Off-chip RAM pin assignment (46 pads):
//   ram_rdata_in[15:0]  — 16 dedicated input pads (SRAM DQ[15:0])
//   ram_addr_out[18:0]  — 19 output pads  (SRAM A[18:0])
//   ram_ren_out         —  1 output pad   (SRAM OE#, active-low, inverted here)
//   class_out[2:0]      —  3 output pads
//   sample_rdy          —  1 output pad
//   done                —  1 output pad
//   error               —  1 output pad
//   error_code[3:0]     —  4 output pads
//   Total: 46 pads (16 in + 30 out)
//
// Address map (SV allocation [95,95,95,120,95], FEATURE_DIM=256):
//   Rows 0..499   SV matrix     (500 x 256 x 2 B = 256 KB)
//   Rows 500..1499 input matrix (1000 x 256 x 2 B = 512 KB)
//   SRAM required: >= 768 KB (IS62WV51216 1MB async SRAM recommended)
// ============================================================================

`default_nettype none

module svm_top_ihp (
    input  logic        clk,
    input  logic        rst_n,

    // SPI slave interface (nRF52840 SPI master, CPOL=0 CPHA=0)
    input  logic        spi_csn,
    input  logic        spi_sclk,
    input  logic        spi_mosi,
    output logic        spi_miso,

    // Off-chip SRAM — dedicated input pads for read data (no management SoC relay)
    input  logic [15:0] ram_rdata_in,

    // Off-chip SRAM — output pads
    output logic [18:0] ram_addr_out,
    output logic        ram_ren_out,    // active-high to core; invert for SRAM OE#

    // Per-sample result GPIO
    output logic [2:0]  class_out,
    output logic        sample_rdy,
    output logic        done,
    output logic        error,
    output logic [3:0]  error_code,

    // Hardware interrupt lines to MCU
    output logic        irq_sample_rdy,
    output logic        irq_done
);

    // =========================================================================
    // Register file
    // =========================================================================
    reg [31:0] reg_control;
    reg [9:0]  reg_num_samples;
    reg [7:0]  reg_num_sv [0:4];
    reg [19:0] reg_param_wr;
    reg [25:0] reg_alpha_wr;   // [25:16]=sv_global_idx (10-bit), [15:0]=alpha Q6.10
    reg        alpha_wr_en_r;

    // =========================================================================
    // Core wires
    // =========================================================================
    wire [3:0] svm_error_code;
    wire       svm_done;
    wire       sample_rdy_w;
    wire       svm_error;
    wire [2:0] class_out_w;

    // =========================================================================
    // Clock gate (IHP SG13G2 ICG cell)
    // =========================================================================
    wire core_warming = (svm_error_code == 4'h8);

    reg batch_active;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)              batch_active <= 1'b0;
        else if (reg_control[0]) batch_active <= 1'b1;
        else if (svm_done)       batch_active <= 1'b0;
    end

    reg [5:0] drain_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            drain_cnt <= 6'd0;
        else if (reg_control[0] || core_warming)
            drain_cnt <= 6'd0;
        else if (svm_done)
            drain_cnt <= 6'd32;
        else if (drain_cnt > 0)
            drain_cnt <= drain_cnt - 6'd1;
    end

    wire svm_clk_en = !rst_n | batch_active | reg_control[0] | core_warming
                    | (drain_cnt > 0) | reg_param_wr[19] | alpha_wr_en_r;

    wire svm_gclk;
`ifdef SIMULATION
    assign svm_gclk = clk & svm_clk_en;
`else
    // IHP SG13G2 integrated clock gating cell (replaces sky130 dlclkp)
    /* verilator lint_off MODMISSING */
    sg13g2_dlclkp_1 u_icg (
        .CLK(clk), .GATE(svm_clk_en), .GCLK(svm_gclk)
    );
    /* verilator lint_on MODMISSING */
`endif

    // =========================================================================
    // SPI slave — CPOL=0, CPHA=0, 8-bit address + 32-bit data = 40-bit frame
    // =========================================================================
    // Synchronise SPI inputs to clk domain (2-FF)
    reg [1:0] csn_s, sclk_s, mosi_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csn_s  <= 2'b11; sclk_s <= 2'b00; mosi_s <= 2'b00;
        end else begin
            csn_s  <= {csn_s[0],  spi_csn};
            sclk_s <= {sclk_s[0], spi_sclk};
            mosi_s <= {mosi_s[0], spi_mosi};
        end
    end
    wire csn_r   = csn_s[1];
    wire sclk_r  = sclk_s[1];
    wire mosi_r  = mosi_s[1];
    wire sclk_rise = (sclk_s[1] & ~sclk_s[0]) ? 1'b1 : 1'b0;  // sclk fell (samples MOSI 1 period late — stable, OK)
    wire csn_rise  = (~csn_s[1] &  csn_s[0])  ? 1'b1 : 1'b0;  // CS deassert (csn: 0→1)

    reg [5:0]  spi_bit_cnt;   // counts 0..39 within a frame
    reg [7:0]  spi_addr;      // address byte
    reg [31:0] spi_rx;        // incoming data shift register
    reg [31:0] spi_tx;        // outgoing data shift register
    reg        spi_addr_done; // address byte captured

    // Status readback (combinational)
    reg [31:0] spi_rdata;
    always @(*) case (spi_addr[6:0])
        7'h01: spi_rdata = reg_control;
        7'h02: spi_rdata = {22'd0, sample_rdy_w, class_out_w,
                             svm_error_code, svm_error, svm_done};
        7'h03: spi_rdata = {22'd0, reg_num_samples};
        7'h04: spi_rdata = {24'd0, reg_num_sv[0]};
        7'h05: spi_rdata = {24'd0, reg_num_sv[1]};
        7'h06: spi_rdata = {24'd0, reg_num_sv[2]};
        7'h07: spi_rdata = {24'd0, reg_num_sv[3]};
        7'h08: spi_rdata = {24'd0, reg_num_sv[4]};
        default: spi_rdata = 32'd0;
    endcase

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_bit_cnt   <= 6'd0;
            spi_addr      <= 8'd0;
            spi_rx        <= 32'd0;
            spi_tx        <= 32'd0;
            spi_addr_done <= 1'b0;
        end else begin
            if (csn_r) begin
                // CS deasserted — reset frame state
                spi_bit_cnt   <= 6'd0;
                spi_addr_done <= 1'b0;
            end else if (sclk_rise) begin
                if (spi_bit_cnt < 8) begin
                    // Address byte phase
                    spi_addr <= {spi_addr[6:0], mosi_r};
                    if (spi_bit_cnt == 7) begin
                        spi_addr_done <= 1'b1;
                        spi_tx <= spi_rdata;    // pre-load TX for read transactions
                    end
                end else begin
                    // Data phase — shift in MOSI, shift out MISO
                    spi_rx <= {spi_rx[30:0], mosi_r};
                    if (spi_addr_done)
                        spi_tx <= {spi_tx[30:0], 1'b0};
                end
                spi_bit_cnt <= spi_bit_cnt + 6'd1;
            end
        end
    end
    assign spi_miso = spi_tx[31];

    // =========================================================================
    // Register write — latch on CS rising edge after complete 40-bit frame
    // =========================================================================
    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_control     <= 32'd8;       // CONTROL reset: vbatt_warn=1 (safe)
            reg_num_samples <= 10'd1000;    // sticky default — write once at startup
            reg_param_wr    <= 20'd0;
            reg_alpha_wr    <= 26'd0;
            alpha_wr_en_r   <= 1'b0;
            reg_num_sv[0] <= 8'd95;    // [95,95,95,120,95] VT-boosted 500-SV
            reg_num_sv[1] <= 8'd95;
            reg_num_sv[2] <= 8'd95;
            reg_num_sv[3] <= 8'd120;   // VT class gets extra SVs
            reg_num_sv[4] <= 8'd95;
        end else begin
            reg_control[0]   <= 1'b0;       // start auto-clears
            reg_param_wr[19] <= 1'b0;       // param_write_en auto-clears
            alpha_wr_en_r    <= 1'b0;
            if (csn_rise && spi_bit_cnt == 40 && !spi_addr[7]) begin
                // Complete write transaction (40 bits, addr[7]=0)
                case (spi_addr[6:0])
                    7'h01: reg_control        <= spi_rx;
                    7'h03: reg_num_samples    <= spi_rx[9:0];
                    7'h04: reg_num_sv[0]      <= spi_rx[7:0];
                    7'h05: reg_num_sv[1]      <= spi_rx[7:0];
                    7'h06: reg_num_sv[2]      <= spi_rx[7:0];
                    7'h07: reg_num_sv[3]      <= spi_rx[7:0];
                    7'h08: reg_num_sv[4]      <= spi_rx[7:0];
                    7'h09: reg_param_wr       <= spi_rx[19:0];
                    7'h0A: begin
                        reg_alpha_wr  <= spi_rx[25:0]; // [25:16]=sv_idx(10-bit) [15:0]=alpha
                        alpha_wr_en_r <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    // =========================================================================
    // Off-chip RAM interface — direct pad path, no management SoC relay
    // ram_rdata sourced from dedicated input pads (SRAM DQ[15:0])
    // =========================================================================
    wire [18:0] ram_addr_w;
    wire        ram_ren_w;

    // =========================================================================
    // svm_compute_core instantiation
    // =========================================================================
    wire        svm_kernel_valid;
    wire [15:0] svm_kernel_out;

    svm_compute_core u_svm (
        .clk             (svm_gclk),
        .rst_n           (rst_n),
        .param_write_en  (reg_param_wr[19]),
        .param_addr      (reg_param_wr[18:16]),
        .param_data      (reg_param_wr[15:0]),
        .gamma_reg       (),
        .c_reg           (),
        .num_sv_per_class_flat ({reg_num_sv[4], reg_num_sv[3], reg_num_sv[2],
                                 reg_num_sv[1], reg_num_sv[0]}),
        .ram_addr        (ram_addr_w),
        .ram_rdata       (ram_rdata_in),  // direct from input pads — no relay
        .ram_ren         (ram_ren_w),
        .vbatt_warn      (reg_control[2]),
        .vbatt_ok        (reg_control[1]),
        .start           (reg_control[0]),
        .num_samples     (reg_num_samples),
        .sample_rdy      (sample_rdy_w),
        .class_out       (class_out_w),
        .done            (svm_done),
        .error           (svm_error),
        .error_code      (svm_error_code),
        .kernel_out      (svm_kernel_out),
        .kernel_valid    (svm_kernel_valid),
        .kernel_ready    (1'b1),
        .class_scores_la (),
        .alpha_write_en  (alpha_wr_en_r),
        .alpha_addr      (reg_alpha_wr[25:16]),
        .alpha_data      (reg_alpha_wr[15:0])
    );

    // =========================================================================
    // Output assignments
    // =========================================================================
    assign ram_addr_out   = ram_addr_w;
    assign ram_ren_out    = ram_ren_w;
    assign class_out      = class_out_w;
    assign sample_rdy     = sample_rdy_w;
    assign done           = svm_done;
    assign error          = svm_error;
    assign error_code     = svm_error_code;
    assign irq_sample_rdy = sample_rdy_w;
    assign irq_done       = svm_done;

endmodule
`default_nettype wire
