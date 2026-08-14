// tb_num_samples.sv — m6 RAM interface
// Dedicated coverage for the num_samples parameter.
//
//  T1  num_samples=1         — baseline: done×1, sample_rdy×1, kernels=NSV
//  T2  num_samples=4         — done×1, sample_rdy×4, class_out valid each pulse
//  T3  num_samples=MAX=8     — boundary: done×1, sample_rdy×8, kernels=8×NSV
//  T4  num_samples=0         — ERR_NUM_SAMPLES_ZERO (0x7) fires AND is sticky:
//                              error_code held for 50 idle cycles; rst_n clears
//  T5  Change between batches — batch1 N=2 then batch2 N=3 without intervening reset;
//                              each batch produces correct sample_rdy and kernel counts
//
// SRAM model: constant 0x0400 (SV=input=1.0); dist=0 → kernel=1024 for all SVs.
// Address layout: {row[10:0], col[7:0]} — addr_r[18:8] < NSV selects SV rows.
`timescale 1ns/1ps
`default_nettype none
module tb_num_samples;

localparam int DW        = 16;
localparam int FD        = 16;
localparam int NSV       = 5;    // 1 SV per class, 5 classes
localparam int MAX_BATCH = 8;
localparam int LAT       = 1;

logic clk = 0; always #5 clk = ~clk;

logic        rst_n=0, param_write_en=0;
logic [2:0]  param_addr=0;
logic [DW-1:0] param_data=0, gamma_reg, c_reg;
logic [39:0] num_sv_per_class_flat;
logic [18:0] ram_addr;
logic [DW-1:0] ram_rdata;
logic        ram_ren;
logic        start=0;
logic [9:0]  num_samples=10'd1;
logic        sample_rdy, done, error;
logic [2:0]  class_out;
logic [3:0]  error_code;
logic [DW-1:0] kernel_out;
logic        kernel_valid;
logic [127:0] class_scores_la;

assign num_sv_per_class_flat = {8'd1,8'd1,8'd1,8'd1,8'd1};

logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = 16'h0400;  // SV = input = 1.0 → dist=0 → kernel=1024

svm_compute_core #(.DATA_WIDTH(DW), .FEATURE_DIM(FD), .NUM_SV(NSV),
                   .MAX_BATCH_SIZE(MAX_BATCH), .RAM_LATENCY(LAT)) dut (
    .clk(clk), .rst_n(rst_n),
    .param_write_en(param_write_en), .param_addr(param_addr),
    .param_data(param_data), .gamma_reg(gamma_reg), .c_reg(c_reg),
    .num_sv_per_class_flat(num_sv_per_class_flat),
    .ram_addr(ram_addr), .ram_rdata(ram_rdata), .ram_ren(ram_ren),
    .vbatt_warn(1'b0), .vbatt_ok(1'b1),
    .start(start), .num_samples(num_samples),
    .sample_rdy(sample_rdy), .class_out(class_out),
    .done(done), .error(error), .error_code(error_code),
    .kernel_out(kernel_out), .kernel_valid(kernel_valid), .kernel_ready(1'b1),
    .class_scores_la(class_scores_la),
    .alpha_write_en(1'b0), .alpha_addr(10'd0), .alpha_data(16'd0)
);

// ── Running counters ──────────────────────────────────────────────────────
int total_kernels = 0;
int total_srdy    = 0;
int total_done    = 0;
int class_out_bad = 0;  // sample_rdy pulse with class_out outside [0,4]

always @(posedge clk) begin
    if (kernel_valid) total_kernels++;
    if (sample_rdy) begin
        total_srdy++;
        if (class_out > 3'd4) class_out_bad++;
    end
    if (done) total_done++;
end

// ── Scoreboard ────────────────────────────────────────────────────────────
int pass_count=0, fail_count=0;

task automatic chk(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-44s got=%0d", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-44s got=%0d  exp=%0d", name, got, exp); fail_count++;
    end
endtask

task automatic chk_code(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-44s code=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-44s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

// ── Helpers ───────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'h0100;  // gamma=0.25
    @(posedge clk); #1; param_write_en=0;
endtask

// Pulse start with given num_samples; wait for done; return per-batch deltas.
task automatic run_batch(
    input  int  n_samples,
    input  int  timeout_cyc,
    output int  k_delta,
    output int  sr_delta,
    output int  dn_delta
);
    automatic int k0  = total_kernels;
    automatic int sr0 = total_srdy;
    automatic int dn0 = total_done;
    int t;

    num_samples = 10'(n_samples);
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;

    t = timeout_cyc;
    while (!done && t>0) begin @(posedge clk); #1; t--; end
    if (t==0) $fatal(1,"[FATAL] run_batch(%0d): timeout", n_samples);
    if (error && error_code < 4'h8)
        $fatal(1,"[FATAL] run_batch(%0d): real fault 0x%0h", n_samples, error_code);

    @(posedge clk); #1;  // let counters latch

    k_delta  = total_kernels - k0;
    sr_delta = total_srdy    - sr0;
    dn_delta = total_done    - dn0;
endtask

// ── Main ──────────────────────────────────────────────────────────────────
initial begin
    int kd, srd, dnd;
    $display("=== tb_num_samples ===");
    $display("    FD=%0d  NSV=%0d  MAX_BATCH=%0d  LAT=%0d", FD, NSV, MAX_BATCH, LAT);

    // ── T1: num_samples=1 (baseline) ──────────────────────────────────────
    $display("T1: num_samples=1 (baseline)");
    do_reset();
    run_batch(1, 5000, kd, srd, dnd);
    chk("T1 done fires once",          dnd, 1);
    chk("T1 sample_rdy fires once",    srd, 1);
    chk("T1 kernel count = NSV",       kd,  NSV);

    // ── T2: num_samples=4 ─────────────────────────────────────────────────
    $display("T2: num_samples=4");
    do_reset();
    run_batch(4, 20000, kd, srd, dnd);
    chk("T2 done fires once",          dnd,          1);
    chk("T2 sample_rdy fires 4×",      srd,          4);
    chk("T2 kernel count = 4×NSV",     kd,           4*NSV);
    chk("T2 class_out always valid",   class_out_bad, 0);

    // ── T3: num_samples=MAX_BATCH (boundary) ──────────────────────────────
    $display("T3: num_samples=%0d (MAX_BATCH boundary)", MAX_BATCH);
    do_reset();
    run_batch(MAX_BATCH, 50000, kd, srd, dnd);
    chk("T3 done fires once",          dnd, 1);
    chk("T3 sample_rdy = MAX_BATCH",   srd, MAX_BATCH);
    chk("T3 kernel count = MAX×NSV",   kd,  MAX_BATCH*NSV);

    // ── T4: num_samples=0 → ERR_NUM_SAMPLES_ZERO, sticky ─────────────────
    $display("T4: num_samples=0 — ERR_NUM_SAMPLES_ZERO (0x7), must be sticky");
    do_reset();
    num_samples = 10'd0;
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("T4 ERR_NUM_SAMPLES_ZERO fires", error_code, 4'h7);
    chk("T4 error flag set",                  error,      1);
    // Sticky: hold error code for 50 idle cycles without triggering start or reset
    repeat(50) @(posedge clk); #1;
    chk_code("T4 sticky after 50 cycles",     error_code, 4'h7);
    chk("T4 error still set after 50 cy",     error,      1);
    // Reset must clear
    rst_n=0;
    repeat(4) @(posedge clk); #1;
    chk_code("T4 code clears during rst_n=0", error_code, 4'h0);
    chk("T4 error clears during rst_n=0",     error,      1'b0);
    rst_n=1;
    repeat(2) @(posedge clk); #1;
    chk_code("T4 code clear after rst_n=1",   error_code, 4'h0);

    // ── T5: Change num_samples between batches (no reset) ─────────────────
    $display("T5: Change num_samples 2→3 between consecutive batches");
    do_reset();
    run_batch(2, 10000, kd, srd, dnd);
    chk("T5 batch1 done fires once",       dnd, 1);
    chk("T5 batch1 sample_rdy×2",          srd, 2);
    chk("T5 batch1 kernels = 2×NSV",       kd,  2*NSV);
    run_batch(3, 15000, kd, srd, dnd);   // no reset between
    chk("T5 batch2 done fires once",       dnd, 1);
    chk("T5 batch2 sample_rdy×3",          srd, 3);
    chk("T5 batch2 kernels = 3×NSV",       kd,  3*NSV);

    // ── Summary ───────────────────────────────────────────────────────────
    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_num_samples: FAIL");
    else $display("tb_num_samples: PASS");
    $finish;
end

initial begin #5_000_000; $fatal(1,"[FATAL] tb_num_samples: watchdog timeout"); end
endmodule
`default_nettype wire
