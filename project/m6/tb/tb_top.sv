// ============================================================================
// tb_top.sv  —  ECE410 m5  Top-level Wishbone integration testbench
// ============================================================================
// Instantiates user_project_wrapper (top.sv + compute_core.sv) and drives the
// complete Wishbone interface: register config → SRAM model → classification.
//
// Final cosim results (tb_wb_cosim.py, job 92867, 300 MIT-BIH samples):
//   Overall accuracy : 97.67% (293/300)
//   Normal  : 60/60 (100.0%)    PVC  : 60/60 (100.0%)
//   AFib    : 60/60 (100.0%)    VT   : 56/60 ( 93.3%)
//   SVT     : 57/60 ( 95.0%)
//   Accuracy gap vs sklearn float: 0.0000 (exact match in Q6.10)
//
// This file: structural integration check of the Caravel wrapper.
//   NUM_SV=[1,1,1,1,1]  NUM_SAMPLES=5  SRAM=all-zeros
//   All-zero SV + input features → dist(x,sv)=0 → kernel=exp(0)=1.0
//     (Q6.10 = 0x0400) for every SV in every class.
//   Alpha:  sv_idx=0 (class 0, Normal)  = 0x0200  (0.5)
//           sv_idx=1..4 (classes 1-4)   = 0x0040  (0.0625)
//   gamma=0.25, C=1.0, bias[0-4]=0.0  (all reset defaults — no explicit write)
//   Scores: class 0 = 0.5 × 1.0 = 0.5
//           class 1-4 = 0.0625 × 1.0 = 0.0625
//   Expected: class_out=0 (Normal) all 5 beats, no error
//
// Pass criterion: 5/5 beats → class_out=0, no sticky error
// ERR_WARMING_UP (errcode=0x8) is advisory — fires for beats 1-99 across the
// core lifetime. Classification result is valid; treat as non-fatal here.
//
// Run (from project/m5/tb/):
//   iverilog -g2012 -DSIMULATION -DMPRJ_IO_PADS=38 -o /tmp/tb_top.out \
//       ../rt1/top.sv ../rt1/compute_core.sv sky130_stubs.v tb_top.sv
//   /tmp/tb_top.out
// ============================================================================

`timescale 1ns/1ps
`default_nettype none

// Pass -DMPRJ_IO_PADS=38 on the iverilog command line (see run comment above).

module tb_top;

    // ── parameters ───────────────────────────────────────────────────────────
    localparam int N_BEATS  = 5;
    localparam int TIMEOUT  = 500_000;   // watchdog cycles

    // ── clock: 40 MHz ────────────────────────────────────────────────────────
    logic clk = 0;
    always #12.5 clk = ~clk;

    // ── Wishbone master ──────────────────────────────────────────────────────
    logic        wb_rst_i  = 1;
    logic        wbs_stb_i = 0, wbs_cyc_i = 0, wbs_we_i = 0;
    logic [3:0]  wbs_sel_i = 4'hF;
    logic [31:0] wbs_dat_i = 0, wbs_adr_i = 0;
    logic        wbs_ack_o;
    logic [31:0] wbs_dat_o;

    // ── LA / GPIO ─────────────────────────────────────────────────────────────
    // la_data_in[15:0] = ram_rdata; all-zero SRAM → tie to 0.
    wire  [127:0] la_data_in = 128'd0;
    logic [127:0] la_data_out, la_oenb;
    logic [`MPRJ_IO_PADS-1:0] io_in = '0, io_out, io_oeb;
    logic [2:0] user_irq;

    // ── power supplies (inout ports require net drivers, not literals) ────────
    wire vccd1 = 1'b1;
    wire vssd1 = 1'b0;

    // ── DUT ──────────────────────────────────────────────────────────────────
    user_project_wrapper dut (
        .vccd1       (vccd1),      .vssd1       (vssd1),
        .wb_clk_i    (clk),        .wb_rst_i    (wb_rst_i),
        .wbs_stb_i   (wbs_stb_i), .wbs_cyc_i   (wbs_cyc_i),
        .wbs_we_i    (wbs_we_i),  .wbs_sel_i   (wbs_sel_i),
        .wbs_dat_i   (wbs_dat_i), .wbs_adr_i   (wbs_adr_i),
        .wbs_ack_o   (wbs_ack_o), .wbs_dat_o   (wbs_dat_o),
        .la_data_in  (la_data_in), .la_data_out (la_data_out),
        .la_oenb     (la_oenb),
        .io_in       (io_in),     .io_out      (io_out),
        .io_oeb      (io_oeb),
        .analog_io   (),
        .user_clock2 (1'b0),
        .user_irq    (user_irq)
    );

    // ── off-chip SRAM model: all zeros, LAT=1 ────────────────────────────────
    // RAM_LATENCY=1 (compute_core default): ram_beat fires every cycle.
    // la_data_in is a wire tied to 0 above — no further assignment needed.

    // ── GPIO aliases ─────────────────────────────────────────────────────────
    wire [2:0] gpio_class   = io_out[2:0];
    wire       gpio_srdy    = io_out[3];
    wire       gpio_done    = io_out[4];
    wire       gpio_error   = io_out[5];
    wire [3:0] gpio_errcode = io_out[9:6];

    // ── beat capture ─────────────────────────────────────────────────────────
    int beat_idx = 0;
    int pass_cnt = 0;

    // advisory = errcode >= 0x8; does not invalidate the classification result
    wire advisory = gpio_error && (gpio_errcode >= 4'h8);

    always @(posedge clk) begin
        if (gpio_srdy) begin
            $display("[t=%8.0f ns] beat %0d  class=%0d  errcode=0x%0h  %s",
                $realtime, beat_idx, gpio_class, gpio_errcode,
                (gpio_class == 3'd0 && (!gpio_error || advisory)) ? "PASS" : "FAIL");
            if (gpio_class == 3'd0 && (!gpio_error || advisory))
                pass_cnt++;
            beat_idx++;
        end
    end

    // ── Wishbone write task ───────────────────────────────────────────────────
    // Ack is registered 1 cycle after valid; drive on negedge, sample on posedge.
    task automatic wb_write(input [31:0] addr, data);
        @(posedge clk); #1;
        wbs_adr_i = addr; wbs_dat_i = data;
        wbs_cyc_i = 1; wbs_stb_i = 1; wbs_we_i = 1; wbs_sel_i = 4'hF;
        @(posedge clk);           // cycle 1: wb_valid=1, write captured, ack_r set
        @(posedge clk); #1;       // cycle 2: wbs_ack_o=1
        wbs_cyc_i = 0; wbs_stb_i = 0; wbs_we_i = 0;
    endtask

    // ── stimulus ─────────────────────────────────────────────────────────────
    localparam logic [31:0] BASE = 32'h3000_0000;

    initial begin
        $dumpfile("/tmp/tb_top.vcd");
        $dumpvars(0, tb_top);

        // ── reset ─────────────────────────────────────────────────────────
        wb_rst_i = 1;
        repeat (4) @(posedge clk);
        wb_rst_i = 0;
        repeat (2) @(posedge clk);

        // ── alpha coefficients ────────────────────────────────────────────
        // ALPHA_WR (0x28): [24:16]=sv_global_idx  [15:0]=alpha Q6.10
        // sv_idx=0 → class 0 (Normal):   alpha = 0.5    → 0x0200
        // sv_idx=1 → class 1 (PVC):      alpha = 0.0625 → 0x0040
        // sv_idx=2 → class 2 (AFib):     alpha = 0.0625 → 0x0040
        // sv_idx=3 → class 3 (VT):       alpha = 0.0625 → 0x0040
        // sv_idx=4 → class 4 (SVT):      alpha = 0.0625 → 0x0040
        wb_write(BASE + 32'h28, {7'd0, 9'd0, 16'h0200});
        wb_write(BASE + 32'h28, {7'd0, 9'd1, 16'h0040});
        wb_write(BASE + 32'h28, {7'd0, 9'd2, 16'h0040});
        wb_write(BASE + 32'h28, {7'd0, 9'd3, 16'h0040});
        wb_write(BASE + 32'h28, {7'd0, 9'd4, 16'h0040});

        // ── NUM_SV[0-4] = 1 per class ────────────────────────────────────
        wb_write(BASE + 32'h10, 32'd1);   // class 0
        wb_write(BASE + 32'h14, 32'd1);   // class 1
        wb_write(BASE + 32'h18, 32'd1);   // class 2
        wb_write(BASE + 32'h1C, 32'd1);   // class 3
        wb_write(BASE + 32'h20, 32'd1);   // class 4

        // ── NUM_SAMPLES = 5 ───────────────────────────────────────────────
        wb_write(BASE + 32'h0C, 32'd5);

        // ── fire: start=1, vbatt_ok=1, vbatt_warn=0 → CONTROL=0x03 ──────
        wb_write(BASE + 32'h04, 32'h0000_0003);

        // ── wait for done (IRQ[1]) or watchdog ───────────────────────────
        fork
            begin : wait_done
                wait (user_irq[1]);
                @(posedge clk); #1;
                wb_write(BASE + 32'h04, 32'h0000_0002);  // clear start bit
            end
            begin : watchdog
                repeat (TIMEOUT) @(posedge clk);
                $display("WATCHDOG: done not seen within %0d cycles", TIMEOUT);
                $finish;
            end
        join_any
        disable fork;

        repeat (4) @(posedge clk);

        // ── final report ──────────────────────────────────────────────────
        $display("");
        $display("=====================================================");
        $display("tb_top: %0d/%0d beats correct (expected class 0)", pass_cnt, N_BEATS);
        $display("Verdict : %s", (pass_cnt == N_BEATS) ? "PASS" : "FAIL");
        $display("=====================================================");
        $display("");
        $display("Reference — cocotb cosim (job 92867, 300 MIT-BIH samples):");
        $display("  Overall: 97.67%%  (293/300)");
        $display("  Normal=100%%  PVC=100%%  AFib=100%%  VT=93.3%%  SVT=95.0%%");
        $display("  sklearn gap: 0.0000");
        $finish;
    end

endmodule
