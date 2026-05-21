// ============================================================================
// tb_min_sv.sv  —  ECE410_project_tb_netlist  pre-netlist suite
// ============================================================================
// Verifies minimum-SV configuration: 1 SV per class = 5 total SVs.
//
//  sv_counts = [1, 1, 1, 1, 1]  →  sum = 5 (< NUM_SV=5, valid)
//  Expected:
//    - exactly 5 kernel outputs produced
//    - done fires once
//    - no error
//    - each kernel_out = 1024 (dist=0: feature = sv feature = 0x0400)
//
// This exercises the last_sv / last_class boundary logic with sv_counter
// never exceeding 0 for any class.
//
// Compile & run:
//   iverilog -g2012 -o tb_msv tb_min_sv.sv svm_compute_core.sv
//   vvp tb_msv
// ============================================================================

`timescale 1ns/1ps
module tb_min_sv;

localparam int DW              = 16;
localparam int FD              = 16;
localparam int NSV             = 5;   // NUM_SV = 5 (matches sum of sv_counts)
localparam int N_EXPECTED      = 5;   // 1 SV × 5 classes
localparam int EXPECTED_KERNEL = 1024;

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
assign sv_ram_rdata   = 16'h0400;  // SV = 0x0400 = heartbeat → dist = 0 → kernel = 1024
assign work_ram_rdata = '0;

// ── Kernel capture ────────────────────────────────────────────────────────────
int kernel_vals  [N_EXPECTED];
int kernel_count = 0;
always @(posedge clk)
    if (kernel_valid && kernel_ready && kernel_count < N_EXPECTED) begin
        kernel_vals[kernel_count] = int'(kernel_out);
        kernel_count++;
    end

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    int timeout;
    $display("=== tb_min_sv ===");
    $display("  sv_counts=[1,1,1,1,1] → 5 kernels expected, all=1024");

    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1; rst_n = 1;
    repeat(2) @(posedge clk); #1;

    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100;
    @(posedge clk); #1; param_write_en = 0;

    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;

    timeout = 2000;
    while (!done && timeout > 0) begin @(posedge clk); #1; timeout--; end
    if (timeout == 0) $fatal(1, "[FAIL] timeout waiting for done");
    if (error)        $fatal(1, "[FAIL] error asserted (code=0x%0h)", error_code);

    if (kernel_count === N_EXPECTED)
        $display("  [PASS] kernel_count = %0d", kernel_count);
    else
        $fatal(1, "[FAIL] kernel_count=%0d expected %0d", kernel_count, N_EXPECTED);

    for (int k = 0; k < N_EXPECTED; k++) begin
        if (kernel_vals[k] === EXPECTED_KERNEL)
            $display("  [PASS] kernel[%0d] = %0d", k, kernel_vals[k]);
        else
            $fatal(1, "[FAIL] kernel[%0d] = %0d  expected %0d",
                      k, kernel_vals[k], EXPECTED_KERNEL);
    end

    $display("tb_min_sv: PASS");
    $finish;
end

initial begin #1_000_000; $fatal(1, "[FATAL] tb_min_sv: timeout"); end

endmodule
