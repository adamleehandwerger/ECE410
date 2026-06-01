// ============================================================================
// tb_param_write.sv  —  ECE410_project_LUT  pre-netlist suite
// ============================================================================
// Verifies the gamma shadow-register fix for mid-compute param writes.
//
// Background:
//   gamma_int updates immediately on param_write_en.  Without a shadow
//   register a mid-pipeline write corrupts in-flight kernel values.
//   Fix applied: gamma_latched is captured from gamma_int at the start
//   pulse and used by the Horner engine throughout the batch.
//
// Test uses feature=0x0600, sv=0x0400 so distance is non-zero and gamma
// actually affects the output:
//   diff=512 per dim; drain flush accumulates all 16 dims: 16×256=4096
//   P = 256×4096 = 0x100000 → I=1 → lut_val=377, F_q=0 → Horner=1024
//   expected kernel_out = (377×1024)>>10 = 377
//
// Sequence:
//   1. Run baseline: gamma=256, no mid-write → expect kernel=377.
//   2. Run with mid-write: gamma=256 at start, write gamma=9000 mid-pipeline.
//      With fix: gamma_latched=256 throughout → kernel still 377.
//      Without fix: Horner would use gamma=9000 → P overflows → kernel=0.
//   3. Verify ERR_GAMMA_SAT fires after write (correct: illegal write detected).
//   4. Check both runs produce the same kernel output.
//
// Compile & run:
//   iverilog -g2012 -o tb_pw tb_param_write.sv svm_compute_core.sv
//   vvp tb_pw
// ============================================================================

`timescale 1ns/1ps
module tb_param_write;

localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 5;

logic clk = 0;
always #5 clk = ~clk;

logic        rst_n, param_write_en;
logic [2:0]  param_addr;
logic [DW-1:0] param_data, gamma_reg, c_reg;
logic [DW-1:0] bias_reg [5];
logic [7:0]  num_sv_per_class [5];
logic        qspi_valid;
logic [DW-1:0] qspi_data;
logic        qspi_ready;
logic [17:0] sv_ram_addr;
logic [DW-1:0] sv_ram_rdata;
logic        sv_ram_ren;
logic [17:0] work_ram_addr;
logic [DW-1:0] work_ram_wdata, work_ram_rdata;
logic        work_ram_wen, work_ram_ren;
logic        start;
logic [9:0]  num_samples;
logic        done, error;
logic [3:0]  error_code;
logic [DW-1:0] kernel_out;
logic        kernel_valid, kernel_ready;

svm_compute_core #(
    .DATA_WIDTH(DW), .FRAC_BITS(10), .FEATURE_DIM(FD),
    .NUM_SV(NSV), .FIFO_DEPTH(256), .ADDR_WIDTH(8),
    .DEFAULT_GAMMA(0.25)
) dut (
    .clk(clk), .rst_n(rst_n),
    .param_write_en(param_write_en), .param_addr(param_addr),
    .param_data(param_data), .gamma_reg(gamma_reg), .c_reg(c_reg),
    .bias_reg(bias_reg), .num_sv_per_class(num_sv_per_class),
    .qspi_valid(qspi_valid), .qspi_data(qspi_data), .qspi_ready(qspi_ready),
    .sv_ram_addr(sv_ram_addr), .sv_ram_rdata(sv_ram_rdata), .sv_ram_ren(sv_ram_ren),
    .work_ram_addr(work_ram_addr), .work_ram_wdata(work_ram_wdata),
    .work_ram_rdata(work_ram_rdata), .work_ram_wen(work_ram_wen),
    .work_ram_ren(work_ram_ren),
    .vbatt_warn(1'b0), .vbatt_ok(1'b1),
    .start(start), .num_samples(num_samples),
    .done(done), .error(error), .error_code(error_code),
    .kernel_out(kernel_out), .kernel_valid(kernel_valid),
    .kernel_ready(kernel_ready)
);
// SV features = 0x0400; heartbeat features = 0x0600 → non-zero distance
// so gamma actually affects kernel_out.
assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Kernel capture (declared before tasks that reference it) ──────────────────
int kernel_vals [NSV];
int kernel_count = 0;
always @(posedge clk)
    if (kernel_valid && kernel_ready && kernel_count < NSV) begin
        kernel_vals[kernel_count] = int'(kernel_out);
        kernel_count++;
    end

// ── Scoreboard ────────────────────────────────────────────────────────────────
int pass_count = 0, fail_count = 0;
task automatic chk(input string name, input int got, input int exp);
    if (got === exp) begin
        $display("  [PASS] %-40s got=%0d", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-40s got=%0d  exp=%0d", name, got, exp); fail_count++;
    end
endtask

// ── Tasks ─────────────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; param_addr = '0; param_data = '0;
    kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1; rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic run_batch(
    input  logic [15:0] feat_val,
    input  bit          do_midwrite,
    output int          kernel_sum
);
    int timeout;
    int kc_before = 0;

    // Program gamma = 0.25
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100;
    @(posedge clk); #1; param_write_en = 0;

    // Start
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;

    // Feed features
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = feat_val;
    end
    @(posedge clk); #1; qspi_valid = 0;

    if (do_midwrite) begin
        // Wait ~25 cycles to land inside COMPUTE_DIST
        repeat(25) @(posedge clk); #1;
        $display("    Injecting gamma=9000 mid-pipeline...");
        param_write_en = 1; param_addr = 3'b000; param_data = 16'd9000;
        @(posedge clk); #1; param_write_en = 0;
    end

    timeout = 2000;
    while (!done && timeout > 0) begin @(posedge clk); #1; timeout--; end
    if (timeout == 0) $fatal(1, "[FAIL] timeout in run_batch");

    kernel_sum = 0;
    for (int k = 0; k < NSV; k++) kernel_sum += int'(kernel_vals[k]);
endtask

// ── Main ──────────────────────────────────────────────────────────────────────
// Expected kernel: feature=0x0600, sv=0x0400, gamma=256
//   diff=512/dim, diff²>>10=256; drain flush captures all FEATURE_DIM=16 dims
//   accumulator=16×256=4096, P=256×4096=0x100000 → I=1, lut=377, F_q=0, x=0
//   Horner(x=0)=1024, kernel_out=(377×1024)>>10=377 per SV
localparam int EXPECTED_KERNEL = 377;

initial begin
    int ksum_baseline, ksum_midwrite;
    $display("=== tb_param_write ===");
    $display("  feature=0x0600, sv=0x0400 → expected kernel=%0d per SV", EXPECTED_KERNEL);

    // Baseline run: no mid-write
    $display("  Run 1: baseline (no gamma write)");
    do_reset();
    kernel_count = 0;
    run_batch(16'h0600, 0, ksum_baseline);
    $display("    kernel sum = %0d", ksum_baseline);

    // Mid-write run: gamma=256 at start, then gamma=9000 mid-pipeline
    $display("  Run 2: mid-pipeline gamma=9000 write");
    do_reset();
    kernel_count = 0;
    run_batch(16'h0600, 1, ksum_midwrite);
    $display("    kernel sum = %0d", ksum_midwrite);

    // ERR_GAMMA_SAT must have fired after the write
    repeat(3) @(posedge clk); #1;
    chk("ERR_GAMMA_SAT after mid-write",  error_code, 4'h4);
    chk("error flag set",                  error,      1);

    // With gamma shadow-register fix, both runs must produce the same result
    chk("kernel sums equal (shadow reg fix)", ksum_midwrite, ksum_baseline);
    chk("baseline kernel sum = 5×EXPECTED", ksum_baseline, NSV * EXPECTED_KERNEL);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count > 0) $fatal(1, "tb_param_write: FAIL");
    else $display("tb_param_write: PASS");
    $finish;
end

initial begin #2_000_000; $fatal(1, "[FATAL] tb_param_write: timeout"); end

endmodule
