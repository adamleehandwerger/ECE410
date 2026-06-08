// ============================================================================
// tb_multi_heartbeat.sv  —  ECE410_project_tb_netlist  pre-netlist suite
// ============================================================================
// Verifies the num_samples > 1 loop-back path:
//   FSM must return to LOAD_FIFO after each non-final heartbeat and accept
//   a fresh batch of FEATURE_DIM feature words before continuing.
//
//  num_samples = 3 (three heartbeats processed in one start-to-done pass)
//  1 SV per class → 5 kernel outputs per heartbeat → 15 total
//  done must assert exactly once, after the third heartbeat
//
// Compile & run:
//   iverilog -g2012 -o tb_mhb tb_multi_heartbeat.sv svm_compute_core.sv
//   vvp tb_mhb
// ============================================================================

`timescale 1ns/1ps
module tb_multi_heartbeat;

localparam int DW         = 16;
localparam int FD         = 16;
localparam int NSV        = 5;
localparam int N_SAMPLES  = 3;
localparam int N_SVS_EACH = 5;   // 1 SV per class
localparam int N_KERNELS  = N_SAMPLES * N_SVS_EACH;  // 15 total

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
assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Kernel counter (concurrent) ───────────────────────────────────────────────
int  kernel_count = 0;
int  done_count   = 0;
always @(posedge clk) begin
    if (kernel_valid && kernel_ready) kernel_count++;
    if (done) done_count++;
end

// ── Feed one batch of FD features, waiting for qspi_ready each word ──────────
task automatic feed_batch();
    for (int i = 0; i < FD; i++) begin
        // Wait until LOAD_FIFO is active
        while (!qspi_ready) @(posedge clk);
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;
endtask

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    int timeout;
    $display("=== tb_multi_heartbeat ===");

    // Reset and program
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; param_addr = '0; param_data = '0;
    kernel_ready = 1;
    num_samples = N_SAMPLES;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(2) @(posedge clk); #1;

    // Program gamma
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100;
    @(posedge clk); #1; param_write_en = 0;

    // Start
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;

    // Feed all N_SAMPLES heartbeats via qspi_ready-gated task
    // The FSM returns to LOAD_FIFO between heartbeats; feed_batch() waits for it
    for (int hb = 0; hb < N_SAMPLES; hb++) begin
        $display("  feeding heartbeat %0d", hb);
        feed_batch();
    end

    // Wait for done
    timeout = 5000;
    while (!done && timeout > 0) begin
        @(posedge clk); #1; timeout--;
    end
    @(posedge clk); #1;  // capture done_count increment

    if (timeout == 0)
        $fatal(1, "[FAIL] timeout: done never asserted after %0d cycles", 5000);

    // Checks
    if (error && error_code < 4'h8)
        $fatal(1, "[FAIL] real fault asserted (error_code=0x%0h)", error_code);

    if (kernel_count === N_KERNELS)
        $display("  [PASS] kernel_count=%0d (expected %0d)", kernel_count, N_KERNELS);
    else
        $fatal(1, "[FAIL] kernel_count=%0d expected %0d", kernel_count, N_KERNELS);

    if (done_count === 1)
        $display("  [PASS] done fired exactly once");
    else
        $fatal(1, "[FAIL] done fired %0d times (expected 1)", done_count);

    $display("tb_multi_heartbeat: PASS");
    $finish;
end

initial begin #2_000_000; $fatal(1, "[FATAL] tb_multi_heartbeat: timeout"); end

endmodule
