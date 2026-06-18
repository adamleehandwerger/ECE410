// tb_param_write.sv — m6 RAM interface
// Verifies the gamma shadow-register fix for mid-compute param writes.
//
// SV rows (addr < NSV*FD): 0x0400; input row: 0x0600 → non-zero distance
// so gamma actually affects kernel_out.
//
//   diff=512/dim, diff²>>10=256; FD=16 dims: accum=16×256=4096
//   P = gamma(256) × 4096 = 0x100000 → I=1 → lut=377, F_q=0 → Horner=1024
//   kernel_out = (377×1024)>>10 = 377 per SV
//
// Sequence:
//   Run 1 baseline:  gamma=256 throughout → kernel=377
//   Run 2 mid-write: gamma=256 at start, write gamma=9000 after 50cy
//     With shadow-register fix: gamma_latched=256 → kernel still 377
//     Without fix: Horner would use gamma=9000 → P overflows → kernel=0
//   Verify ERR_GAMMA_SAT fires; both runs produce equal kernel output
`timescale 1ns/1ps
`default_nettype none
module tb_param_write;

localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 5;
localparam int LAT = 1;
localparam int EXPECTED_KERNEL = 377;

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

// Address layout: {row[10:0], col[7:0]} — SV rows: row < NSV; input row: row = NSV
logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = (addr_r[18:8] < 11'(NSV)) ? 16'h0400 : 16'h0600;

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

int kernel_vals [NSV];
int kernel_count=0;
always @(posedge clk)
    if (kernel_valid && kernel_count < NSV) begin
        kernel_vals[kernel_count] = int'(kernel_out);
        kernel_count++;
    end

int pass_count=0, fail_count=0;
task automatic chk(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-42s got=%0d", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-42s got=%0d  exp=%0d", name, got, exp); fail_count++;
    end
endtask

task automatic do_reset();
    rst_n=0; start=0; param_write_en=0;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic run_batch(input bit do_midwrite, output int kernel_sum);
    int timeout;
    // Program gamma=0.25
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'h0100;
    @(posedge clk); #1; param_write_en=0;
    // Start
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    if (do_midwrite) begin
        // Wait 50 cycles to land inside the distance/Horner pipeline
        repeat(50) @(posedge clk); #1;
        $display("    Injecting gamma=9000 mid-pipeline...");
        param_write_en=1; param_addr=3'h0; param_data=16'd9000;
        @(posedge clk); #1; param_write_en=0;
    end
    timeout=3000;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    if (timeout==0) $fatal(1,"[FAIL] timeout in run_batch");
    kernel_sum=0;
    for (int k=0; k<NSV; k++) kernel_sum += int'(kernel_vals[k]);
endtask

initial begin
    int ksum_baseline, ksum_midwrite;
    $display("=== tb_param_write ===");
    $display("  sv=0x0400, input=0x0600 → expected kernel=%0d per SV", EXPECTED_KERNEL);

    $display("  Run 1: baseline (no gamma write)");
    do_reset(); kernel_count=0;
    run_batch(0, ksum_baseline);
    $display("    kernel sum = %0d", ksum_baseline);

    $display("  Run 2: mid-pipeline gamma=9000 write");
    do_reset(); kernel_count=0;
    run_batch(1, ksum_midwrite);
    $display("    kernel sum = %0d", ksum_midwrite);

    repeat(3) @(posedge clk); #1;
    chk("ERR_GAMMA_SAT after mid-write",       error_code,    4'h4);
    chk("error flag set",                       error,         1);
    chk("kernel sums equal (shadow-reg fix)",   ksum_midwrite, ksum_baseline);
    chk("baseline kernel sum = 5×EXPECTED",     ksum_baseline, NSV * EXPECTED_KERNEL);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_param_write: FAIL");
    else $display("tb_param_write: PASS");
    $finish;
end

initial begin #2_000_000; $fatal(1,"[FATAL] tb_param_write: timeout"); end
endmodule
`default_nettype wire
