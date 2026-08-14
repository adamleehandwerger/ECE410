// tb_error_codes.sv — m6 RAM interface
// Tests error codes, sticky-latch behavior, and reset-clear guarantee.
//
//  T1  ERR_SV_ZERO         (0x1)  all sv_counts = 0
//  T2  ERR_SV_OVERFLOW     (0x2)  sum(sv_counts) > NUM_SV
//  T3  ERR_GAMMA_SAT       (0x4)  gamma_int > 8192
//  T4  ERR_NUM_SAMPLES_ZERO(0x7)  num_samples = 0 at start  [replaces FIFO_OVERFLOW]
//  T5  Sticky              error_code holds first fault across 50 idle cycles
//  T6  Reset clears        rst_n pulse → error=0, error_code=0x0
`timescale 1ns/1ps
`default_nettype none
module tb_error_codes;

localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 10;
localparam int LAT = 1;

logic clk = 0; always #5 clk = ~clk;

logic        rst_n=0, param_write_en=0;
logic [2:0]  param_addr=0;
logic [DW-1:0] param_data=0, gamma_reg, c_reg;
logic [39:0] num_sv_per_class_flat=0;
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

logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = 16'h0400;

svm_compute_core #(.DATA_WIDTH(DW), .FEATURE_DIM(FD), .NUM_SV(NSV),
                   .MAX_BATCH_SIZE(4), .RAM_LATENCY(LAT)) dut (
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

task automatic do_reset();
    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    num_sv_per_class_flat = {8'd2,8'd2,8'd2,8'd2,8'd2};  // sum=10=NSV
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic prog_gamma(input logic [15:0] g);
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=g;
    @(posedge clk); #1; param_write_en=0;
endtask

task automatic pulse_start();
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
endtask

int pass_count=0, fail_count=0;
task automatic chk(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-32s got=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-32s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

initial begin
    $display("=== tb_error_codes ===");

    // T1: ERR_SV_ZERO
    $display("T1: ERR_SV_ZERO");
    do_reset(); prog_gamma(16'h0100);
    num_sv_per_class_flat = 40'd0;
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T1 error",      error,      1);
    chk("T1 error_code", error_code, 4'h1);

    // T2: ERR_SV_OVERFLOW — sum=11 > NSV=10
    $display("T2: ERR_SV_OVERFLOW");
    do_reset(); prog_gamma(16'h0100);
    num_sv_per_class_flat = {8'd2,8'd2,8'd2,8'd2,8'd3};  // sum=11
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T2 error",      error,      1);
    chk("T2 error_code", error_code, 4'h2);

    // T3: ERR_GAMMA_SAT
    $display("T3: ERR_GAMMA_SAT");
    do_reset(); prog_gamma(16'd9000);
    num_sv_per_class_flat = {8'd2,8'd2,8'd2,8'd2,8'd2};  // sum=10=NSV, valid
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T3 error",      error,      1);
    chk("T3 error_code", error_code, 4'h4);

    // T4: ERR_NUM_SAMPLES_ZERO (0x7) — num_samples=0
    $display("T4: ERR_NUM_SAMPLES_ZERO");
    do_reset(); prog_gamma(16'h0100);
    num_sv_per_class_flat = {8'd2,8'd2,8'd2,8'd2,8'd2};
    num_samples = 10'd0;
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T4 error",      error,      1);
    chk("T4 error_code", error_code, 4'h7);

    // T5: Sticky — error_code holds first fault for 50 idle cycles
    $display("T5: Sticky behavior");
    do_reset(); prog_gamma(16'h0100);
    num_sv_per_class_flat = 40'd0;  // ERR_SV_ZERO condition
    pulse_start();
    repeat(2) @(posedge clk); #1;
    chk("T5 initial latch", error_code, 4'h1);
    repeat(50) @(posedge clk); #1;
    chk("T5 sticky @+50",   error_code, 4'h1);
    chk("T5 sticky error",  error,      1);

    // T6: Reset clears
    $display("T6: Reset clears");
    rst_n=0;
    repeat(4) @(posedge clk); #1;
    chk("T6 code during rst_n=0", error_code, 4'h0);
    chk("T6 err  during rst_n=0", error,      1'b0);
    rst_n=1;
    repeat(2) @(posedge clk); #1;
    chk("T6 code after  rst_n=1", error_code, 4'h0);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_error_codes: FAIL");
    else $display("tb_error_codes: PASS");
    $finish;
end

initial begin #500_000; $fatal(1,"[FATAL] tb_error_codes: timeout"); end
endmodule
`default_nettype wire
