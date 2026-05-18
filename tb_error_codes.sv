// ============================================================================
// tb_error_codes.sv  —  ECE410_project_LUT  pre-netlist suite
// ============================================================================
// Tests all five diagnostic error codes, sticky-latch behavior, and
// reset-clears guarantee.
//
//  Test 1  ERR_SV_ZERO     (0x1)  all sv_counts = 0
//  Test 2  ERR_SV_OVERFLOW (0x2)  sum(sv_counts) > NUM_SV
//  Test 3  ERR_GAMMA_SAT   (0x4)  gamma_int > GAMMA_SAT_THRESH (8192)
//  Test 4  ERR_FIFO_OVERFLOW(0x5) FIFO_DEPTH < FEATURE_DIM → FSM stuck + overflow
//  Test 5  Sticky           error_code holds first fault across 50 idle cycles
//  Test 6  Reset clears     rst_n pulse → error=0, error_code=0x0
//
// Two DUT instances:
//   dut     – FEATURE_DIM=16, NUM_SV=10, FIFO_DEPTH=256  (tests 1–3, 5–6)
//   dut_ovf – FEATURE_DIM=8,  NUM_SV=10, FIFO_DEPTH=4    (test 4)
//
// Compile & run:
//   iverilog -g2012 -o tb_errcodes tb_error_codes.sv svm_compute_core.sv
//   vvp tb_errcodes
// ============================================================================

`timescale 1ns/1ps
module tb_error_codes;

// ── Shared clock ──────────────────────────────────────────────────────────────
localparam int DW  = 16;
localparam int FD  = 16;   // FEATURE_DIM for main DUT
localparam int NSV = 10;   // NUM_SV for both DUTs

logic clk = 0;
always #5 clk = ~clk;   // 100 MHz

// ── Main DUT signals ─────────────────────────────────────────────────────────
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
assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Overflow DUT signals ──────────────────────────────────────────────────────
logic        o_rst_n, o_param_write_en;
logic [2:0]  o_param_addr;
logic [DW-1:0] o_param_data, o_gamma_reg, o_c_reg;
logic [DW-1:0] o_bias_reg [5];
logic [7:0]  o_num_sv_per_class [5];
logic        o_qspi_valid;
logic [DW-1:0] o_qspi_data;
logic        o_qspi_ready;
logic [17:0] o_sv_ram_addr;
logic [DW-1:0] o_sv_ram_rdata;
logic        o_sv_ram_ren;
logic [17:0] o_work_ram_addr;
logic [DW-1:0] o_work_ram_wdata, o_work_ram_rdata;
logic        o_work_ram_wen, o_work_ram_ren;
logic        o_start;
logic [9:0]  o_num_samples;
logic        o_done, o_error;
logic [3:0]  o_error_code;
logic [DW-1:0] o_kernel_out;
logic        o_kernel_valid, o_kernel_ready;

// FIFO_DEPTH=4 < FEATURE_DIM=8 → fifo_count can never reach 8 → FSM stuck in LOAD_FIFO
svm_compute_core #(
    .DATA_WIDTH(DW), .FRAC_BITS(10), .FEATURE_DIM(8),
    .NUM_SV(NSV), .FIFO_DEPTH(4), .ADDR_WIDTH(2),
    .DEFAULT_GAMMA(0.25)
) dut_ovf (
    .clk(clk), .rst_n(o_rst_n),
    .param_write_en(o_param_write_en), .param_addr(o_param_addr),
    .param_data(o_param_data), .gamma_reg(o_gamma_reg), .c_reg(o_c_reg),
    .bias_reg(o_bias_reg), .num_sv_per_class(o_num_sv_per_class),
    .qspi_valid(o_qspi_valid), .qspi_data(o_qspi_data), .qspi_ready(o_qspi_ready),
    .sv_ram_addr(o_sv_ram_addr), .sv_ram_rdata(o_sv_ram_rdata),
    .sv_ram_ren(o_sv_ram_ren),
    .work_ram_addr(o_work_ram_addr), .work_ram_wdata(o_work_ram_wdata),
    .work_ram_rdata(o_work_ram_rdata), .work_ram_wen(o_work_ram_wen),
    .work_ram_ren(o_work_ram_ren),
    .vbatt_warn(1'b0), .vbatt_ok(1'b1),
    .start(o_start), .num_samples(o_num_samples),
    .done(o_done), .error(o_error), .error_code(o_error_code),
    .kernel_out(o_kernel_out), .kernel_valid(o_kernel_valid),
    .kernel_ready(o_kernel_ready)
);
assign o_sv_ram_rdata   = 16'h0400;
assign o_work_ram_rdata = '0;

// ── Tasks ─────────────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; param_addr = '0; param_data = '0;
    kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 0;
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic do_reset_ovf();
    o_rst_n = 0; o_start = 0; o_qspi_valid = 0; o_qspi_data = '0;
    o_param_write_en = 0; o_param_addr = '0; o_param_data = '0;
    o_kernel_ready = 1; o_num_samples = 10'd1;
    for (int i = 0; i < 5; i++) o_num_sv_per_class[i] = 0;
    repeat(4) @(posedge clk); #1;
    o_rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic prog_gamma(input logic [15:0] g);
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = g;
    @(posedge clk); #1;
    param_write_en = 0;
endtask

task automatic set_svcounts(input logic [7:0] c0, c1, c2, c3, c4);
    num_sv_per_class[0] = c0; num_sv_per_class[1] = c1;
    num_sv_per_class[2] = c2; num_sv_per_class[3] = c3;
    num_sv_per_class[4] = c4;
endtask

task automatic pulse_start();
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;
endtask

// ── Scoreboard ────────────────────────────────────────────────────────────────
int pass_count = 0, fail_count = 0;
task automatic chk(input string name, input int got, input int exp);
    if (got === exp) begin
        $display("  [PASS] %-30s got=0x%0h", name, got);
        pass_count++;
    end else begin
        $display("  [FAIL] %-30s got=0x%0h  exp=0x%0h", name, got, exp);
        fail_count++;
    end
endtask

// ── Main sequence ─────────────────────────────────────────────────────────────
initial begin
    $display("=== tb_error_codes ===");

    // ------------------------------------------------------------------
    // Test 1: ERR_SV_ZERO — all sv_counts = 0
    // ------------------------------------------------------------------
    $display("T1: ERR_SV_ZERO");
    do_reset();
    prog_gamma(16'h0100);            // gamma=0.25, valid
    set_svcounts(0, 0, 0, 0, 0);
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T1 error",      error,      1);
    chk("T1 error_code", error_code, 4'h1);

    // ------------------------------------------------------------------
    // Test 2: ERR_SV_OVERFLOW — sum > NUM_SV=10
    // ------------------------------------------------------------------
    $display("T2: ERR_SV_OVERFLOW");
    do_reset();
    prog_gamma(16'h0100);
    set_svcounts(3, 2, 2, 2, 2);    // sum=11 > 10
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T2 error",      error,      1);
    chk("T2 error_code", error_code, 4'h2);

    // ------------------------------------------------------------------
    // Test 3: ERR_GAMMA_SAT — gamma_int > 8192 = GAMMA_SAT_THRESH
    // ------------------------------------------------------------------
    $display("T3: ERR_GAMMA_SAT");
    do_reset();
    prog_gamma(16'd9000);            // 9000 > 8192
    set_svcounts(2, 2, 2, 2, 2);    // sum=10=NUM_SV, valid
    pulse_start();
    repeat(4) @(posedge clk); #1;
    chk("T3 error",      error,      1);
    chk("T3 error_code", error_code, 4'h4);

    // ------------------------------------------------------------------
    // Test 4: ERR_FIFO_OVERFLOW — FIFO_DEPTH=4 < FEATURE_DIM=8
    // FSM stays in LOAD_FIFO; after 4 writes FIFO is full; 5th write
    // with qspi_valid=1 latches fifo_overflow_r → ERR_FIFO_OVERFLOW.
    // ------------------------------------------------------------------
    $display("T4: ERR_FIFO_OVERFLOW");
    do_reset_ovf();
    @(posedge clk); #1;
    o_param_write_en = 1; o_param_addr = 3'b000; o_param_data = 16'h0100;
    @(posedge clk); #1;
    o_param_write_en = 0;
    for (int i = 0; i < 5; i++) o_num_sv_per_class[i] = 8'd2; // sum=10=NUM_SV
    @(posedge clk); #1; o_start = 1;
    @(posedge clk); #1; o_start = 0;
    // Feed 16 words continuously; FIFO accepts 4, remainder trigger overflow
    o_qspi_data = 16'h0400;
    for (int i = 0; i < 16; i++) begin
        @(posedge clk); #1;
        o_qspi_valid = 1;
    end
    @(posedge clk); #1; o_qspi_valid = 0;
    repeat(5) @(posedge clk); #1;
    chk("T4 o_error",      o_error,      1);
    chk("T4 o_error_code", o_error_code, 4'h5);

    // ------------------------------------------------------------------
    // Test 5: Sticky — error_code holds first fault for 50 idle cycles
    // ------------------------------------------------------------------
    $display("T5: Sticky behavior");
    do_reset();
    prog_gamma(16'h0100);
    set_svcounts(0, 0, 0, 0, 0);   // ERR_SV_ZERO condition
    pulse_start();
    repeat(2) @(posedge clk); #1;
    chk("T5 initial latch", error_code, 4'h1);
    repeat(50) @(posedge clk); #1;  // 50 more cycles — must still be 0x1
    chk("T5 sticky @+50",  error_code, 4'h1);
    chk("T5 sticky error", error,      1);

    // ------------------------------------------------------------------
    // Test 6: Reset clears — rst_n pulse zeroes error and error_code
    // ------------------------------------------------------------------
    $display("T6: Reset clears");
    // Continue from the stuck state in T5
    rst_n = 0;
    repeat(4) @(posedge clk); #1;
    chk("T6 error_code during rst_n=0", error_code, 4'h0);
    chk("T6 error during rst_n=0",      error,      1'b0);
    rst_n = 1;
    repeat(2) @(posedge clk); #1;
    chk("T6 error_code after  rst_n=1", error_code, 4'h0);

    // ------------------------------------------------------------------
    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count > 0) $fatal(1, "tb_error_codes: FAIL");
    else $display("tb_error_codes: PASS");
    $finish;
end

// Watchdog
initial begin
    #500_000;
    $fatal(1, "[FATAL] tb_error_codes: simulation timeout at %0t", $time);
end

endmodule
