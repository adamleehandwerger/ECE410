// SPDX-FileCopyrightText: 2024 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// Caravel user_project_wrapper — 5-class RBF-SVM Cardiac Arrhythmia Classifier
// ECE410, Portland State University  |  Milestone: m5  (batch architecture)
//
// Batch architecture (v8):
//   Host collects 1000 beats at low power, extracts 256-dim features,
//   pre-loads both SVs and the input matrix into off-chip SRAM, then fires
//   start.  ASIC reads both datasets autonomously over the same GPIO/LA bus.
//
// Off-chip RAM (shared bus, host serves from SRAM):
//   Address layout: {row[10:0], col[7:0]} = 19-bit
//   Rows  0..249        SV matrix      (250 × 256 × 2 B = 128 KB)
//   Rows  250..1249     input matrix   (1000 × 256 × 2 B = 512 KB)
//   GPIO[28:10] = ram_addr[18:0]  (output)
//   GPIO[29]    = ram_ren          (output)
//   LA[15:0]    = ram_rdata[15:0] (input from host)
//
// Per-sample output (fires every WRITE_CLASS):
//   GPIO[2:0] = class_out    GPIO[3] = sample_rdy    IRQ[0] = sample_rdy
//
// Batch done (fires once at end of batch):
//   GPIO[4] = done    IRQ[1] = done
//
// GPIO[5] = error    GPIO[9:6] = error_code
//
// ┌──────────────────────────────────────────────────────────┐
// │  Wishbone Memory Map  (base 0x3000_0000)                 │
// │  0x04 RW  CONTROL      [0]=start [1]=vbatt_ok            │
// │                         [2]=vbatt_warn                    │
// │  0x08 RO  STATUS       [0]=done(batch) [1]=error         │
// │                         [5:2]=error_code [8:6]=class      │
// │                         [9]=sample_rdy                    │
// │  0x0C RW  NUM_SAMPLES  [9:0]                             │
// │  0x10–0x20 RW  NUM_SV_0–4  [7:0] SVs per class          │
// │  0x24 WO  PARAM_WR    [19]=en [18:16]=addr [15:0]=data   │
// │  0x28 WO  ALPHA_WR    [23:16]=sv_global_idx [15:0]=alpha │
// └──────────────────────────────────────────────────────────┘

`default_nettype none

module user_project_wrapper #(
    parameter BITS = 32
) (
    inout vccd1, inout vssd1,
    input  wb_clk_i, input  wb_rst_i,
    input  wbs_stb_i, input  wbs_cyc_i, input  wbs_we_i,
    input  [3:0]  wbs_sel_i,
    input  [31:0] wbs_dat_i,
    input  [31:0] wbs_adr_i,
    output wbs_ack_o,
    output [31:0] wbs_dat_o,
    input  [127:0] la_data_in,
    output [127:0] la_data_out,
    output [127:0] la_oenb,
    input  [`MPRJ_IO_PADS-1:0] io_in,
    output [`MPRJ_IO_PADS-1:0] io_out,
    output [`MPRJ_IO_PADS-1:0] io_oeb,
    inout  [`MPRJ_IO_PADS-10:0] analog_io,
    input  user_clock2,
    output [2:0] user_irq
);

    wire clk   = wb_clk_i;
    wire rst_n = ~wb_rst_i;

    // =========================================================================
    // Register declarations  (must precede clock gate logic)
    // =========================================================================
    reg [31:0] reg_control;
    reg [9:0]  reg_num_samples;
    reg [7:0]  reg_num_sv [0:4];
    reg [19:0] reg_param_wr;
    reg [24:0] reg_alpha_wr;   // [24:16]=sv_global_idx (9-bit), [15:0]=alpha Q6.10
    reg        alpha_wr_en_r;  // 1-cycle write-enable pulse to core

    // =========================================================================
    // Clock gate
    // =========================================================================
    wire [3:0] svm_error_code;
    wire       svm_done;
    wire       sample_rdy_w;

    wire core_warming = (svm_error_code == 4'h8);

    // batch_active: set on start pulse, held until batch done
    reg batch_active;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)              batch_active <= 1'b0;
        else if (reg_control[0]) batch_active <= 1'b1;
        else if (svm_done)       batch_active <= 1'b0;
    end

    reg [5:0] drain_cnt;
    always @(posedge clk) begin
        if (!rst_n)
            drain_cnt <= 6'd0;
        else if (reg_control[0] || core_warming)
            drain_cnt <= 6'd0;
        else if (svm_done)
            drain_cnt <= 6'd32;
        else if (drain_cnt > 0)
            drain_cnt <= drain_cnt - 6'd1;
    end

    wire svm_clk_en = !rst_n | batch_active | reg_control[0] | core_warming | (drain_cnt > 0)
                   | reg_param_wr[19] | alpha_wr_en_r;

    wire svm_gclk;
`ifdef SIMULATION
    assign svm_gclk = clk & svm_clk_en;
`else
    sky130_fd_sc_hd__dlclkp_1 u_icg (
        .CLK(clk), .GATE(svm_clk_en), .GCLK(svm_gclk)
    );
`endif

    // =========================================================================
    // Wishbone decode
    // =========================================================================
    wire       wb_valid = wbs_cyc_i && wbs_stb_i && (wbs_adr_i[31:8] == 24'h300000);
    wire       wb_wr    = wb_valid && wbs_we_i;
    wire [5:0] wb_reg   = wbs_adr_i[7:2];

    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_control     <= 32'd8;
            reg_num_samples <= 0;
            reg_param_wr    <= 0;
            reg_alpha_wr    <= 0;
            alpha_wr_en_r   <= 1'b0;
            for (c = 0; c < 5; c = c+1) reg_num_sv[c] <= 8'd50;
        end else begin
            reg_control[0]   <= 1'b0;          // start auto-clears
            reg_param_wr[19] <= 1'b0;          // param_write_en auto-clears
            alpha_wr_en_r    <= 1'b0;          // alpha_write_en auto-clears
            if (wb_wr) case (wb_reg)
                6'h01: reg_control        <= wbs_dat_i;
                6'h03: reg_num_samples    <= wbs_dat_i[9:0];
                6'h04: reg_num_sv[0]      <= wbs_dat_i[7:0];
                6'h05: reg_num_sv[1]      <= wbs_dat_i[7:0];
                6'h06: reg_num_sv[2]      <= wbs_dat_i[7:0];
                6'h07: reg_num_sv[3]      <= wbs_dat_i[7:0];
                6'h08: reg_num_sv[4]      <= wbs_dat_i[7:0];
                6'h09: reg_param_wr       <= wbs_dat_i[19:0];
                6'h0A: begin
                    reg_alpha_wr  <= wbs_dat_i[24:0]; // [24:16]=sv_idx [15:0]=alpha
                    alpha_wr_en_r <= 1'b1;
                end
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Off-chip RAM interface  (19-bit address, via GPIO + LA)
    // GPIO[28:10] = ram_addr[18:0]  GPIO[29] = ram_ren
    // LA[15:0]    = ram_rdata[15:0] (driven by host)
    // =========================================================================
    wire [18:0] ram_addr_w;
    wire        ram_ren_w;
    wire [15:0] ram_rdata_w = la_data_in[15:0];

    // =========================================================================
    // Core outputs
    // =========================================================================
    wire        svm_error;
    wire [2:0]  class_out_w;
    wire [15:0] svm_kernel_out;
    wire        svm_kernel_valid;
    wire [127:0] la_scores_w;

    // =========================================================================
    // Wishbone read mux
    // =========================================================================
    reg [31:0] wb_rdata;
    always @(*) case (wb_reg)
        6'h01: wb_rdata = reg_control;
        6'h02: wb_rdata = {22'd0, sample_rdy_w, class_out_w,
                            svm_error_code, svm_error, svm_done};
        6'h03: wb_rdata = {22'd0, reg_num_samples};
        6'h04: wb_rdata = {24'd0, reg_num_sv[0]};
        6'h05: wb_rdata = {24'd0, reg_num_sv[1]};
        6'h06: wb_rdata = {24'd0, reg_num_sv[2]};
        6'h07: wb_rdata = {24'd0, reg_num_sv[3]};
        6'h08: wb_rdata = {24'd0, reg_num_sv[4]};
        default: wb_rdata = 32'd0;
    endcase

    reg wb_ack_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) wb_ack_r <= 1'b0;
        else        wb_ack_r <= wb_valid;
    end

    assign wbs_ack_o = wb_ack_r;
    assign wbs_dat_o = wb_rdata;

    // =========================================================================
    // svm_compute_core
    // =========================================================================
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
        .ram_rdata       (ram_rdata_w),
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
        .class_scores_la (la_scores_w),
        .alpha_write_en  (alpha_wr_en_r),
        .alpha_addr      (reg_alpha_wr[24:16]),
        .alpha_data      (reg_alpha_wr[15:0])
    );

    // =========================================================================
    // GPIO / LA outputs
    //
    // [2:0]   class_out       per-sample result
    // [3]     sample_rdy      pulses once per heartbeat classified
    // [4]     svm_done        pulses once at end of batch
    // [5]     svm_error
    // [9:6]   svm_error_code
    // [28:10] ram_addr[18:0]  19-bit off-chip address (output)
    // [29]    ram_ren          off-chip read enable   (output)
    // =========================================================================
    assign io_out  = {{`MPRJ_IO_PADS-30{1'b0}},
                       ram_ren_w,
                       ram_addr_w[18:0],
                       svm_error_code,
                       svm_error,
                       svm_done,
                       sample_rdy_w,
                       class_out_w};

    assign io_oeb  = {{`MPRJ_IO_PADS-30{1'b1}}, 30'b0};

    assign la_data_out = la_scores_w;
    assign la_oenb     = 128'h0000_0000_0000_0000_0000_0000_0000_FFFF;

    assign user_irq = {1'b0, svm_done, sample_rdy_w};

endmodule
`default_nettype wire
