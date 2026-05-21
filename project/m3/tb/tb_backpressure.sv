// ============================================================================
// tb_backpressure.sv  —  ECE410_project_tb_netlist  pre-netlist suite
// ============================================================================
// Verifies kernel_ready / kernel_valid handshake.
//
// FIX applied (svm_compute_core.sv):
//   kernel_valid is now held HIGH via a set/clear register until kernel_ready
//   acknowledges it.  Previously it followed horner_valid_out for exactly one
//   cycle, causing the FSM to stall permanently on any late kernel_ready.
//
//  Sub-test A  Baseline: kernel_ready=1 always → done fires after all SVs.
//  Sub-test B  Same-cycle release: kernel_ready=0, released on the posedge
//              that kernel_valid first asserts (via @(posedge kernel_valid)),
//              repeated for every SV output → done fires correctly.
//  Sub-test C  Late release: kernel_ready=0, released 3 cycles AFTER
//              kernel_valid rises → done must still fire (verifies the fix).
//
// Compile & run:
//   iverilog -g2012 -o tb_bp tb_backpressure.sv svm_compute_core.sv
//   vvp tb_bp
// ============================================================================

`timescale 1ns/1ps
module tb_backpressure;

localparam int DW  = 16;
localparam int FD  = 16;
localparam int NSV = 5;       // NUM_SV
localparam int N_CLASSES = 5;
localparam int N_SVS = 5;     // 1 SV per class → 5 total kernel outputs

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

// ── Tasks ─────────────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; param_addr = '0; param_data = '0;
    kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1; // 1 SV per class
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic prog_and_start();
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100;
    @(posedge clk); #1; param_write_en = 0;
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;
    // Feed FD features immediately (FIFO_DEPTH=256 >> FD, no backpressure)
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;
endtask

task automatic wait_done(input int timeout_cycles, output bit timed_out);
    timed_out = 0;
    for (int i = 0; i < timeout_cycles; i++) begin
        @(posedge clk); #1;
        if (done) return;
    end
    timed_out = 1;
endtask

// ── Scoreboard ────────────────────────────────────────────────────────────────
int pass_count = 0, fail_count = 0;
task automatic chk(input string name, input int got, input int exp);
    if (got === exp) begin
        $display("  [PASS] %-30s", name); pass_count++;
    end else begin
        $display("  [FAIL] %-30s got=%0h exp=%0h", name, got, exp); fail_count++;
    end
endtask

// Kernel output counter (concurrent)
int kernel_count = 0;
always @(posedge clk)
    if (kernel_valid && kernel_ready) kernel_count++;

// ── Sub-test A: Baseline ──────────────────────────────────────────────────────
initial begin
    bit timed_out;
    int kc_before;
    $display("=== tb_backpressure ===");

    // Sub-test A — kernel_ready=1 always
    $display("A: Baseline (kernel_ready=1 always)");
    do_reset();
    kernel_ready = 1;
    kernel_count = 0;
    prog_and_start();
    wait_done(2000, timed_out);
    chk("A done fires",         !timed_out, 1);
    chk("A no error",           error,      0);
    chk("A kernel_count=N_SVS", kernel_count, N_SVS);

    // Sub-test B — same-cycle release via @(posedge kernel_valid)
    $display("B: Same-cycle kernel_ready release");
    do_reset();
    kernel_ready = 0;
    kernel_count = 0;
    fork
        prog_and_start();
        // Release kernel_ready on each posedge of kernel_valid
        begin : releaser
            repeat(N_SVS) begin
                @(posedge kernel_valid);
                kernel_ready = 1;
                @(posedge clk); #1;
                if (!done) kernel_ready = 0;
            end
            kernel_ready = 1;
        end
    join_none
    wait_done(2000, timed_out);
    disable releaser;
    chk("B done fires",   !timed_out, 1);
    chk("B no error",     error,      0);
    chk("B kernel_count", kernel_count, N_SVS);

    // Sub-test C — late release (3 cycles): verifies kernel_valid hold fix
    $display("C: Late release 3 cycles — kernel_valid must be held until accepted");
    do_reset();
    kernel_ready = 0;
    kernel_count = 0;
    fork
        prog_and_start();
        begin : late_rel
            @(posedge kernel_valid);
            repeat(3) @(posedge clk); #1;   // 3-cycle delay — fixed RTL must survive
            kernel_ready = 1;
        end
    join_none
    wait_done(2000, timed_out);
    disable late_rel;
    chk("C done fires after 3-cycle stall", !timed_out, 1);
    chk("C no error",                       error,       0);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count > 0) $fatal(1, "tb_backpressure: FAIL");
    else $display("tb_backpressure: PASS");
    $finish;
end

initial begin #1_000_000; $fatal(1, "[FATAL] tb_backpressure: timeout"); end

endmodule
