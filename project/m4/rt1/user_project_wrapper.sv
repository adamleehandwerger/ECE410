// SPDX-FileCopyrightText: 2024 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// Caravel user_project_wrapper — 5-class RBF-SVM Cardiac Arrhythmia Classifier
// ECE410, Portland State University
//
// SV RAM architecture (off-chip):
//   The 64 KB support vector store (256 SVs × 128 features × 2 B) is held in
//   host-side flash/SRAM.  The svm_compute_core drives sv_ram_addr[14:0] and
//   sv_ram_ren out via GPIO[25:10]; the host responds with sv_ram_rdata[15:0]
//   on LA[15:0] the following cycle.  This eliminates the need for the
//   sky130_sram_16kbyte macro (not available in this PDK snapshot).
//
// Clock gating strategy:
//   - svm_compute_core receives a gated clock (svm_gclk) via sky130 ICG cell
//   - Gate is OPEN when: warming up, receiving data, start pulsed, post-done drain
//   - Gate is CLOSED when: fully idle between heartbeats
//
// ┌─────────────────────────────────────────────────────────┐
// │  Wishbone Memory Map  (base 0x3000_0000)                │
// │  0x00 WO  FIFO_DATA    write 16-bit feature word        │
// │  0x04 RW  CONTROL      [0]=start [1]=vbatt_ok           │
// │                         [2]=vbatt_warn [3]=kern_ready    │
// │  0x08 RO  STATUS       [0]=done [1]=error               │
// │                         [5:2]=error_code [8:6]=class     │
// │  0x0C RW  NUM_SAMPLES  [9:0]                            │
// │  0x10–0x20 RW  NUM_SV_0–4  [7:0] SVs per class         │
// │  0x24 WO  PARAM_WR    [19]=en [18:16]=addr [15:0]=data  │
// │  0x38 WO  WORK_RD     [10:0]=addr to read from work_ram │
// │  0x3C RO  STATUS2     work_ram read data [15:0]         │
// │  SV RAM (off-chip):                                      │
// │    GPIO[24:10] = sv_ram_addr[14:0]  (output)            │
// │    GPIO[25]    = sv_ram_ren          (output)            │
// │    LA[15:0]    = sv_ram_rdata[15:0] (input from host)   │
// └─────────────────────────────────────────────────────────┘
//
// GPIO[2:0] = class_out    GPIO[3] = done    GPIO[4] = error
// GPIO[8:5] = error_code   GPIO[9] = fifo_ready
// GPIO[24:10] = sv_ram_addr[14:0]   GPIO[25] = sv_ram_ren
// IRQ[0] = done pulse

`default_nettype none

module user_project_wrapper #(
    parameter BITS = 32
) (
`ifdef USE_POWER_PINS
    inout vdda1, inout vdda2, inout vssa1, inout vssa2,
    inout vccd1, inout vccd2, inout vssd1, inout vssd2,
`endif
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
    inout  [`MPRJ_IO_PADS-10-1:0] analog_io,
    input  user_clock2,
    output [2:0] user_irq
);

    wire clk   = wb_clk_i;
    wire rst_n = ~wb_rst_i;

    // =========================================================================
    // Clock gate
    // =========================================================================
    wire [3:0] svm_error_code;
    wire       svm_done;

    wire core_warming = (svm_error_code == 4'h8);

    reg [5:0] drain_cnt;

    always @(posedge clk) begin
        if (!rst_n)
            drain_cnt <= 6'd0;
        else if (qspi_valid_r || reg_control[0] || core_warming)
            drain_cnt <= 6'd0;
        else if (svm_done)
            drain_cnt <= 6'd32;
        else if (drain_cnt > 0)
            drain_cnt <= drain_cnt - 6'd1;
    end

    wire svm_clk_en = !rst_n | qspi_valid_r | reg_control[0] | core_warming | (drain_cnt > 0);

    wire svm_gclk;
`ifdef SIMULATION
    assign svm_gclk = clk & svm_clk_en;
`else
    sky130_fd_sc_hd__dlclkp_1 u_icg (
  `ifdef USE_POWER_PINS
        .VPWR(vccd1), .VPB(vccd1), .VGND(vssd1), .VNB(vssd1),
  `endif
        .CLK(clk), .GATE(svm_clk_en), .GCLK(svm_gclk)
    );
`endif

    // =========================================================================
    // Wishbone decode
    // =========================================================================
    wire       wb_valid = wbs_cyc_i && wbs_stb_i && (wbs_adr_i[31:8] == 24'h300000);
    wire       wb_wr    = wb_valid && wbs_we_i;
    wire [5:0] wb_reg   = wbs_adr_i[7:2];

    // =========================================================================
    // Registers
    // =========================================================================
    reg [31:0] reg_control;
    reg [9:0]  reg_num_samples;
    reg [7:0]  reg_num_sv [0:4];
    reg [19:0] reg_param_wr;
    reg        qspi_valid_r;
    reg [15:0] qspi_data_r;
    reg [15:0] work_rd_latch;

    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_control     <= 32'd8;
            reg_num_samples <= 0;
            reg_param_wr    <= 0;
            qspi_valid_r    <= 0;
            qspi_data_r     <= 0;
            for (c = 0; c < 5; c = c+1) reg_num_sv[c] <= 8'd50;
        end else begin
            qspi_valid_r     <= 1'b0;
            reg_control[0]   <= 1'b0;
            reg_param_wr[19] <= 1'b0;
            if (wb_wr) case (wb_reg)
                6'h00: begin
                    qspi_data_r  <= wbs_dat_i[15:0];
                    qspi_valid_r <= 1'b1;
                end
                6'h01: reg_control        <= wbs_dat_i;
                6'h03: reg_num_samples    <= wbs_dat_i[9:0];
                6'h04: reg_num_sv[0]      <= wbs_dat_i[7:0];
                6'h05: reg_num_sv[1]      <= wbs_dat_i[7:0];
                6'h06: reg_num_sv[2]      <= wbs_dat_i[7:0];
                6'h07: reg_num_sv[3]      <= wbs_dat_i[7:0];
                6'h08: reg_num_sv[4]      <= wbs_dat_i[7:0];
                6'h09: reg_param_wr       <= wbs_dat_i[19:0];
                6'h0E: work_rd_latch      <= work_ram[wbs_dat_i[10:0]];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // SV RAM — off-chip interface via GPIO + LA
    // =========================================================================
    wire [17:0] sv_ram_addr_w;
    wire        sv_ram_ren_w;
    wire [15:0] sv_ram_rdata_w = la_data_in[15:0];

    // =========================================================================
    // work_ram: 2KB scratch (register-based)
    // =========================================================================
    (* ram_style = "registers" *) reg [15:0] work_ram [0:2047];
    wire [18:0] work_ram_addr_w;
    wire [15:0] work_ram_wdata_w;
    wire        work_ram_wen_w, work_ram_ren_w;
    reg  [15:0] work_ram_rdata_r;

    always @(posedge clk) begin
        if (work_ram_wen_w && work_ram_addr_w[10:0] < 2048)
            work_ram[work_ram_addr_w[10:0]] <= work_ram_wdata_w;
        if (work_ram_ren_w && work_ram_addr_w[10:0] < 2048)
            work_ram_rdata_r <= work_ram[work_ram_addr_w[10:0]];
    end

    // =========================================================================
    // Argmax
    // =========================================================================
    wire [15:0] svm_kernel_out;
    wire        svm_kernel_valid;
    wire        fifo_ready;
    wire        svm_error;

    reg [31:0] class_score [0:4];
    reg [2:0]  class_out_r;
    reg [2:0]  kern_class_idx;
    reg [7:0]  kern_sv_idx;
    reg [7:0]  sv_snap [0:4];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kern_class_idx <= 0; kern_sv_idx <= 0;
            for (c = 0; c < 5; c = c+1) begin
                class_score[c] <= 0; sv_snap[c] <= 50;
            end
        end else begin
            if (reg_control[0]) begin
                kern_class_idx <= 0; kern_sv_idx <= 0;
                for (c = 0; c < 5; c = c+1) begin
                    class_score[c] <= 0; sv_snap[c] <= reg_num_sv[c];
                end
            end else if (svm_kernel_valid && reg_control[3]) begin
                class_score[kern_class_idx] <=
                    class_score[kern_class_idx] + {{16{svm_kernel_out[15]}}, svm_kernel_out};
                if (kern_sv_idx >= sv_snap[kern_class_idx] - 1) begin
                    kern_sv_idx    <= 0;
                    kern_class_idx <= kern_class_idx + 1;
                end else kern_sv_idx <= kern_sv_idx + 1;
            end
        end
    end

    // Flatten array to wires — Yosys mem2reg can't handle variable index
    // (class_score[argmax_comb]) in a combinational always block
    wire [31:0] cs0 = class_score[0];
    wire [31:0] cs1 = class_score[1];
    wire [31:0] cs2 = class_score[2];
    wire [31:0] cs3 = class_score[3];
    wire [31:0] cs4 = class_score[4];

    reg [2:0]  argmax_comb;
    reg [31:0] argmax_best;
    always @(*) begin
        argmax_comb = 3'd0; argmax_best = cs0;
        if ($signed(cs1) > $signed(argmax_best)) begin argmax_comb = 3'd1; argmax_best = cs1; end
        if ($signed(cs2) > $signed(argmax_best)) begin argmax_comb = 3'd2; argmax_best = cs2; end
        if ($signed(cs3) > $signed(argmax_best)) begin argmax_comb = 3'd3; argmax_best = cs3; end
        if ($signed(cs4) > $signed(argmax_best)) begin argmax_comb = 3'd4; argmax_best = cs4; end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) class_out_r <= 3'd0;
        else if (svm_done) class_out_r <= argmax_comb;
    end

    // =========================================================================
    // Wishbone read mux
    // =========================================================================
    reg [31:0] wb_rdata;
    always @(*) case (wb_reg)
        6'h01: wb_rdata = reg_control;
        6'h02: wb_rdata = {23'd0, class_out_r, svm_error_code, svm_error, svm_done};
        6'h03: wb_rdata = {22'd0, reg_num_samples};
        6'h04: wb_rdata = {24'd0, reg_num_sv[0]};
        6'h05: wb_rdata = {24'd0, reg_num_sv[1]};
        6'h06: wb_rdata = {24'd0, reg_num_sv[2]};
        6'h07: wb_rdata = {24'd0, reg_num_sv[3]};
        6'h08: wb_rdata = {24'd0, reg_num_sv[4]};
        6'h0F: wb_rdata = {16'd0, work_rd_latch};
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
        .bias_reg_flat   (),
        .num_sv_per_class_flat({reg_num_sv[4], reg_num_sv[3], reg_num_sv[2],
                                reg_num_sv[1], reg_num_sv[0]}),
        .qspi_valid      (qspi_valid_r),
        .qspi_data       (qspi_data_r),
        .qspi_ready      (fifo_ready),
        .sv_ram_addr     (sv_ram_addr_w),
        .sv_ram_rdata    (sv_ram_rdata_w),
        .sv_ram_ren      (sv_ram_ren_w),
        .work_ram_addr   (work_ram_addr_w),
        .work_ram_wdata  (work_ram_wdata_w),
        .work_ram_rdata  (work_ram_rdata_r),
        .work_ram_wen    (work_ram_wen_w),
        .work_ram_ren    (work_ram_ren_w),
        .vbatt_warn      (reg_control[2]),
        .vbatt_ok        (reg_control[1]),
        .start           (reg_control[0]),
        .num_samples     (reg_num_samples),
        .done            (svm_done),
        .error           (svm_error),
        .error_code      (svm_error_code),
        .kernel_out      (svm_kernel_out),
        .kernel_valid    (svm_kernel_valid),
        .kernel_ready    (reg_control[3])
    );

    // =========================================================================
    // GPIO / LA outputs
    // =========================================================================
    assign io_out  = {{`MPRJ_IO_PADS-26{1'b0}},
                       sv_ram_ren_w,
                       sv_ram_addr_w[14:0],
                       fifo_ready,
                       svm_error_code,
                       svm_error,
                       svm_done,
                       class_out_r};

    assign io_oeb  = {{`MPRJ_IO_PADS-26{1'b1}}, 26'b0};

    assign la_data_out = {cs3, cs2, cs1, cs0};
    assign la_oenb     = 128'h0000_0000_0000_0000_0000_0000_0000_FFFF;

    assign user_irq = {2'b0, svm_done};

endmodule
`default_nettype wire
