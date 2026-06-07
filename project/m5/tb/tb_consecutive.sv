// ============================================================================
// tb_consecutive.sv  —  ECE410_project_LUT  pre-netlist suite
// ============================================================================
// Verifies that two complete classification batches can be run back-to-back
// without an intervening reset.  After the first batch completes (done=1,
// FSM returns to IDLE), a second start pulse must launch a fresh batch and
// produce the same kernel outputs.
//
//  Batch 1 → done fires → assert counters match → immediately pulse start
//  Batch 2 → done fires → assert counters match
//  No rst_n between batches.
//
// Compile & run:
//   iverilog -g2012 -o tb_consec tb_consecutive.sv svm_compute_core.sv
//   vvp tb_consec
// ============================================================================

`timescale 1ns/1ps
module tb_consecutive;

localparam int DW   = 16;
localparam int FD   = 16;
localparam int NSV  = 5;
localparam int N_SVS = 5;  // 1 SV per class

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

// ── Per-batch kernel capture ──────────────────────────────────────────────────
int batch_kernel_count = 0;
int batch_kernel_sum   = 0;
always @(posedge clk)
    if (kernel_valid && kernel_ready) begin
        batch_kernel_count++;
        batch_kernel_sum += int'(kernel_out);
    end

// ── Run one complete batch and return to IDLE ─────────────────────────────────
task automatic run_batch(input int batch_id, output int kcount, output int ksum);
    int timeout;

    // Reset per-batch counters (can't reset the always block, so snapshot instead)
    automatic int kc_before = batch_kernel_count;
    automatic int ks_before = batch_kernel_sum;

    // Pulse start
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;

    // Feed FD features
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;

    // Wait for done
    timeout = 2000;
    while (!done && timeout > 0) begin @(posedge clk); #1; timeout--; end
    if (timeout == 0)
        $fatal(1, "[FAIL] batch %0d: timeout waiting for done", batch_id);
    if (error && error_code < 4'h8)
        $fatal(1, "[FAIL] batch %0d: error_code=0x%0h", batch_id, error_code);

    @(posedge clk); #1;  // let done-triggered counters settle

    kcount = batch_kernel_count - kc_before;
    ksum   = batch_kernel_sum   - ks_before;
    $display("  Batch %0d: kernel_count=%0d  kernel_sum=%0d", batch_id, kcount, ksum);
endtask

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    int kcount1, ksum1, kcount2, ksum2;
    $display("=== tb_consecutive ===");

    // Single reset at the start
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1; rst_n = 1;
    repeat(2) @(posedge clk); #1;

    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100;
    @(posedge clk); #1; param_write_en = 0;

    // Batch 1
    run_batch(1, kcount1, ksum1);

    // No reset between batches — FSM must be in IDLE after done
    // Batch 2
    run_batch(2, kcount2, ksum2);

    // Checks
    if (kcount1 === N_SVS)
        $display("  [PASS] batch1 kernel_count=%0d", kcount1);
    else
        $fatal(1, "[FAIL] batch1 kernel_count=%0d expected %0d", kcount1, N_SVS);

    if (kcount2 === N_SVS)
        $display("  [PASS] batch2 kernel_count=%0d", kcount2);
    else
        $fatal(1, "[FAIL] batch2 kernel_count=%0d expected %0d", kcount2, N_SVS);

    // Kernel sums must be identical (same features, same SVs)
    if (ksum1 === ksum2)
        $display("  [PASS] kernel sums equal (%0d)", ksum1);
    else
        $fatal(1, "[FAIL] kernel sums differ: batch1=%0d  batch2=%0d", ksum1, ksum2);

    $display("tb_consecutive: PASS");
    $finish;
end

initial begin #2_000_000; $fatal(1, "[FATAL] tb_consecutive: timeout"); end

endmodule
