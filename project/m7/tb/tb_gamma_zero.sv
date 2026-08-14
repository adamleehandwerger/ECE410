// tb_gamma_zero.sv — m6 RAM interface
// gamma=0 → P=0 for all distances → all kernels=1024; ERR_GAMMA_ZERO fires
`timescale 1ns/1ps
`default_nettype none
module tb_gamma_zero;
localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 5;
localparam int LAT = 1;
localparam int EXPECTED_KERNEL = 1024;

logic clk = 0; always #5 clk = ~clk;

logic        rst_n=0, param_write_en=0;
logic [2:0]  param_addr=0;
logic [DW-1:0] param_data=0, gamma_reg, c_reg;
logic [39:0] num_sv_per_class_flat;
logic [18:0] ram_addr;
logic [DW-1:0] ram_rdata;
logic        ram_ren;
logic        start=0;
logic [9:0]  num_samples;
logic        sample_rdy, done, error;
logic [2:0]  class_out;
logic [3:0]  error_code;
logic [DW-1:0] kernel_out;
logic        kernel_valid;
logic [127:0] class_scores_la;

assign num_sv_per_class_flat = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
assign num_samples = 10'd1;

// SV=0x0200 (0.5), input=0x0400 (1.0) — distinct to prove gamma=0 ignores distance
// Address layout: {row[10:0], col[7:0]} — SV rows: row < NSV; input row: row = NSV
logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = (addr_r[18:8] < 11'(NSV)) ? 16'h0200 : 16'h0400;

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

int kernel_vals [NSV]; int kernel_count=0; int wrong_count=0;
always @(posedge clk)
    if (kernel_valid) begin
        kernel_vals[kernel_count] = int'(kernel_out);
        if (int'(kernel_out) !== EXPECTED_KERNEL) wrong_count++;
        kernel_count++;
    end

initial begin
    int timeout;
    $display("=== tb_gamma_zero ===");
    repeat(4) @(posedge clk); #1; rst_n = 1;
    repeat(2) @(posedge clk); #1;
    // Override gamma with 0
    param_write_en=1; param_addr=3'h0; param_data=16'h0000; @(posedge clk); #1;
    param_write_en=0;
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    timeout=2000;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    if (timeout==0) $fatal(1,"[FAIL] timeout");
    if (error && error_code===4'h6) $display("  [PASS] ERR_GAMMA_ZERO (0x6) asserted");
    else if (!error) $fatal(1,"[FAIL] error not raised — ERR_GAMMA_ZERO missing");
    else $fatal(1,"[FAIL] wrong code: got=0x%0h expected=0x6", error_code);
    if (kernel_count !== NSV) $fatal(1,"[FAIL] kernel_count=%0d expected %0d", kernel_count, NSV);
    if (wrong_count===0) $display("  [PASS] All %0d kernels = %0d", NSV, EXPECTED_KERNEL);
    else $fatal(1,"[FAIL] %0d kernel(s) != %0d", wrong_count, EXPECTED_KERNEL);
    $display("tb_gamma_zero: PASS"); $finish;
end
initial begin #500_000; $fatal(1,"[FATAL] tb_gamma_zero: timeout"); end
endmodule
`default_nettype wire
