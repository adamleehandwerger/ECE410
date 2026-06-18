// tb_dist_boundary.sv — m6 RAM interface
// SV=0x8000 (-32768), input=0x7FFF (+32767) → max dist → kernel=0
`timescale 1ns/1ps
`default_nettype none
module tb_dist_boundary;
localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 5;
localparam int LAT = 1;
localparam int EXPECTED_KERNEL = 0;

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

// Address layout: {row[10:0], col[7:0]} — rows are 256 apart regardless of FD
// SV rows: row < NSV (addr[18:8] < NSV); input row: row = NSV
logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = (addr_r[18:8] < 11'(NSV)) ? 16'h8000 : 16'h7FFF;

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

int kernel_values [NSV]; int kernel_count = 0;
always @(posedge clk)
    if (kernel_valid) begin kernel_values[kernel_count] = int'(kernel_out); kernel_count++; end

initial begin
    int timeout;
    $display("=== tb_dist_boundary ===");
    repeat(4) @(posedge clk); #1; rst_n = 1;
    repeat(2) @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'h0100; @(posedge clk); #1;
    param_write_en=0;
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    timeout=2000;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    if (timeout==0) $fatal(1,"[FAIL] timeout");
    if (error)      $fatal(1,"[FAIL] error=0x%0h", error_code);
    if (kernel_count !== NSV) $fatal(1,"[FAIL] kernel_count=%0d expected %0d", kernel_count, NSV);
    for (int k=0; k<NSV; k++) begin
        if (kernel_values[k]===EXPECTED_KERNEL) $display("  [PASS] kernel[%0d]=%0d (saturated)", k, kernel_values[k]);
        else $fatal(1,"[FAIL] kernel[%0d]=%0d expected %0d", k, kernel_values[k], EXPECTED_KERNEL);
    end
    $display("tb_dist_boundary: PASS"); $finish;
end
initial begin #500_000; $fatal(1,"[FATAL] tb_dist_boundary: timeout"); end
endmodule
`default_nettype wire
