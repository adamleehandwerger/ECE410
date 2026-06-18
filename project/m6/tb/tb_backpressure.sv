// tb_backpressure.sv — m6 RAM interface
// Verifies kernel_ready / kernel_valid handshake.
// FSM reads SVs and input autonomously from SRAM; kernel_valid must be held
// until kernel_ready acknowledges it.
//
//  A  Baseline:    kernel_ready=1 always → done fires after all SVs
//  B  Same-cycle:  kernel_ready released on posedge kernel_valid each time
//  C  Late (3cy):  kernel_ready released 3 cycles after kernel_valid — fixed RTL
`timescale 1ns/1ps
`default_nettype none
module tb_backpressure;

localparam int DW    = 16;
localparam int FD    = 16;
localparam int NSV   = 5;
localparam int LAT   = 1;
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
logic        kernel_ready=1;
logic [127:0] class_scores_la;

assign num_sv_per_class_flat = {8'd1,8'd1,8'd1,8'd1,8'd1};

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
    .kernel_out(kernel_out), .kernel_valid(kernel_valid),
    .kernel_ready(kernel_ready),
    .class_scores_la(class_scores_la),
    .alpha_write_en(1'b0), .alpha_addr(10'd0), .alpha_data(16'd0)
);

task automatic do_reset();
    rst_n=0; start=0; param_write_en=0;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic prog_and_start();
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'h0100;
    @(posedge clk); #1; param_write_en=0;
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
endtask

task automatic wait_done(input int timeout_cycles, output bit timed_out);
    timed_out=0;
    for (int i=0; i<timeout_cycles; i++) begin
        @(posedge clk); #1;
        if (done) return;
    end
    timed_out=1;
endtask

int pass_count=0, fail_count=0;
task automatic chk(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-36s", name); pass_count++;
    end else begin
        $display("  [FAIL] %-36s got=%0h exp=%0h", name, got, exp); fail_count++;
    end
endtask

int kernel_count=0;
always @(posedge clk)
    if (kernel_valid && kernel_ready) kernel_count++;

initial begin
    bit timed_out;
    $display("=== tb_backpressure ===");

    // Sub-test A — kernel_ready=1 always
    $display("A: Baseline (kernel_ready=1 always)");
    do_reset(); kernel_ready=1; kernel_count=0;
    prog_and_start();
    wait_done(2000, timed_out);
    chk("A done fires",         !timed_out,   1);
    chk("A no error",           error,        0);
    chk("A kernel_count=N_SVS", kernel_count, N_SVS);

    // Sub-test B — same-cycle kernel_ready release
    $display("B: Same-cycle kernel_ready release");
    do_reset(); kernel_ready=0; kernel_count=0;
    fork
        prog_and_start();
        begin : releaser
            repeat(N_SVS) begin
                @(posedge kernel_valid);
                kernel_ready=1;
                @(posedge clk); #1;
                if (!done) kernel_ready=0;
            end
            kernel_ready=1;
        end
    join_none
    wait_done(2000, timed_out);
    disable releaser;
    chk("B done fires",   !timed_out,   1);
    chk("B no error",     error,        0);
    chk("B kernel_count", kernel_count, N_SVS);

    // Sub-test C — late release (3 cycles): kernel_valid must be held
    $display("C: Late release 3 cycles — kernel_valid must be held until accepted");
    do_reset(); kernel_ready=0; kernel_count=0;
    fork
        prog_and_start();
        begin : late_rel
            @(posedge kernel_valid);
            repeat(3) @(posedge clk); #1;
            kernel_ready=1;
        end
    join_none
    wait_done(2000, timed_out);
    disable late_rel;
    chk("C done fires after 3-cycle stall", !timed_out, 1);
    chk("C no error",                       error,      0);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_backpressure: FAIL");
    else $display("tb_backpressure: PASS");
    $finish;
end

initial begin #1_000_000; $fatal(1,"[FATAL] tb_backpressure: timeout"); end
endmodule
`default_nettype wire
