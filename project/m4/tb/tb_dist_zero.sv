// ============================================================================
// tb_dist_zero.sv  —  ECE410_project_tb_netlist  pre-netlist suite
// ============================================================================
// Verifies that a zero squared-distance produces kernel_out = 1024 (= 1.0
// in Q6.10), i.e. K(x, sv) = exp(0) = 1.
//
// Setup: feature[k] = sv[k] = 0x0400 for all k
//   diff        = 0 for every dimension
//   accumulator = 0  →  dist_out = 0
//   P           = gamma × 0 = 0
//   I           = 0  →  lut_val = EXP_INT_LUT[0] = 1024
//   F_q         = 0  →  x = 0  →  Horner(0) = COEFF_00 = 1024
//   result      = (1024 × 1024) >> 10 = 1024
//
// Compile & run:
//   iverilog -g2012 -o tb_dz tb_dist_zero.sv svm_compute_core.sv
//   vvp tb_dz
// ============================================================================

`timescale 1ns/1ps
module tb_dist_zero;

localparam int DW              = 16;
localparam int FD              = 16;
localparam int NSV             = 5;
localparam int EXPECTED_KERNEL = 1024;  // exp(0) = 1.0 in Q6.10

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
// SV features = 0x0400 = same as heartbeat features → dist = 0
assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Kernel capture ────────────────────────────────────────────────────────────
int kernel_values [NSV];
int kernel_count = 0;
always @(posedge clk)
    if (kernel_valid && kernel_ready) begin
        kernel_values[kernel_count] = int'(kernel_out);
        kernel_count++;
    end

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    int timeout;
    $display("=== tb_dist_zero ===");
    $display("  feature = sv = 0x0400 → dist=0 → expect kernel=%0d", EXPECTED_KERNEL);

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

    if (kernel_count !== NSV)
        $fatal(1, "[FAIL] expected %0d kernel values, got %0d", NSV, kernel_count);

    for (int k = 0; k < NSV; k++) begin
        if (kernel_values[k] === EXPECTED_KERNEL)
            $display("  [PASS] kernel[%0d] = %0d", k, kernel_values[k]);
        else
            $fatal(1, "[FAIL] kernel[%0d] = %0d  expected %0d",
                      k, kernel_values[k], EXPECTED_KERNEL);
    end

    $display("tb_dist_zero: PASS");
    $finish;
end

initial begin #1_000_000; $fatal(1, "[FATAL] tb_dist_zero: timeout"); end

endmodule
