// tb_consecutive.sv — m6 RAM interface
// Two back-to-back classification batches without intervening reset.
// After batch 1 done fires FSM returns to IDLE; batch 2 must start cleanly
// and produce the same kernel outputs.
`timescale 1ns/1ps
`default_nettype none
module tb_consecutive;

localparam int DW   = 16;
localparam int FD   = 16;
localparam int NSV  = 5;
localparam int LAT  = 1;
localparam int N_SVS = 5;  // 1 SV per class

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

// SV and input both 0x0400 → dist=0 → kernel=1024
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

int batch_kernel_count = 0;
int batch_kernel_sum   = 0;
always @(posedge clk)
    if (kernel_valid) begin
        batch_kernel_count++;
        batch_kernel_sum += int'(kernel_out);
    end

task automatic run_batch(input int batch_id, output int kcount, output int ksum);
    int timeout;
    automatic int kc_before = batch_kernel_count;
    automatic int ks_before = batch_kernel_sum;

    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;

    timeout=2000;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    if (timeout==0) $fatal(1,"[FAIL] batch %0d: timeout", batch_id);
    if (error && error_code < 4'h8)
        $fatal(1,"[FAIL] batch %0d: error_code=0x%0h", batch_id, error_code);

    @(posedge clk); #1;
    kcount = batch_kernel_count - kc_before;
    ksum   = batch_kernel_sum   - ks_before;
    $display("  Batch %0d: kernel_count=%0d  kernel_sum=%0d", batch_id, kcount, ksum);
endtask

initial begin
    int kcount1, ksum1, kcount2, ksum2;
    $display("=== tb_consecutive ===");

    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;

    param_write_en=1; param_addr=3'h0; param_data=16'h0100;
    @(posedge clk); #1; param_write_en=0;

    run_batch(1, kcount1, ksum1);
    run_batch(2, kcount2, ksum2);  // no reset between batches

    if (kcount1===N_SVS)
        $display("  [PASS] batch1 kernel_count=%0d", kcount1);
    else
        $fatal(1,"[FAIL] batch1 kernel_count=%0d expected %0d", kcount1, N_SVS);

    if (kcount2===N_SVS)
        $display("  [PASS] batch2 kernel_count=%0d", kcount2);
    else
        $fatal(1,"[FAIL] batch2 kernel_count=%0d expected %0d", kcount2, N_SVS);

    if (ksum1===ksum2)
        $display("  [PASS] kernel sums equal (%0d)", ksum1);
    else
        $fatal(1,"[FAIL] kernel sums differ: batch1=%0d  batch2=%0d", ksum1, ksum2);

    $display("tb_consecutive: PASS"); $finish;
end

initial begin #2_000_000; $fatal(1,"[FATAL] tb_consecutive: timeout"); end
endmodule
`default_nettype wire
