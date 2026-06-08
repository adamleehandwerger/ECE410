// ============================================================================
// tb_warmup.sv  —  ECE410_project_LUT  pre-netlist suite
// ============================================================================
// Verifies ERR_WARMING_UP (0x8) and ERR_INTERRUPTED (0x9) advisory behavior.
//
// The core tracks completed heartbeats since the last rst_n.  Multi-scale
// feature slices (10-beat mean, 100-beat RR track) on the host MCU are
// unreliable until 100 continuous beats have accumulated.  Two non-sticky
// advisory codes distinguish the two situations:
//
//   ERR_WARMING_UP  (0x8) — counting up from a clean start or from a reset
//                           that fired when heartbeat_count was already 0 or
//                           had already reached 100 (normal restart).
//   ERR_INTERRUPTED (0x9) — rst_n fired while heartbeat_count was in [1,99];
//                           the previous warm-up was cut short; a full
//                           uninterrupted run of 100 beats is required.
//
// Both codes are advisory (non-sticky): they clear automatically once
// heartbeat_count reaches 100.  All real faults (codes 0x1–0x7) are still
// sticky and override any advisory that is currently showing.
//
//  T1  Fresh start  →  ERR_WARMING_UP fires after beat 1 (not ERR_INTERRUPTED)
//  T2  Advisory persists through beats 2–99 (spot-check at beat 50)
//  T3  Reset mid-warm-up (at beat 99)  →  ERR_INTERRUPTED fires on new beat 1;
//      real fault (ERR_GAMMA_SAT) then overrides and latches sticky
//  T4  Advisory clears at beat 100 after a reset with interrupted=1
//  T5  Reset after complete warm-up (count=100)  →  ERR_WARMING_UP re-fires
//      (NOT ERR_INTERRUPTED, because count was 100 when reset fired)
//
// Timing note:
//   heartbeat_count increments at the posedge where the last OUTPUT_RESULT
//   handshake fires (same edge as done).  err_detect is combinational and
//   reads the OLD counter value at that edge so error is still 0 at done+0.
//   The advisory latches one cycle later.  Checks wait two cycles after done.
//
// Compile & run:
//   iverilog -g2012 -o tb_wu tb_warmup.sv svm_compute_core.sv
//   vvp tb_wu
// ============================================================================

`timescale 1ns/1ps
module tb_warmup;

localparam int DW  = 16;
localparam int FD  = 8;   // small FEATURE_DIM → fast pipeline
localparam int NSV = 5;   // NUM_SV

localparam int ERR_WARMING_UP  = 4'h8;
localparam int ERR_INTERRUPTED = 4'h9;
localparam int ERR_GAMMA_SAT   = 4'h4;

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
logic [18:0] work_ram_addr;
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

// feature = sv = 0x0400 (1.0 in Q6.10)  →  dist=0  →  kernel_out=1024
assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Scoreboard ────────────────────────────────────────────────────────────────
int pass_count = 0, fail_count = 0;
task automatic chk(input string name, input int got, input int exp);
    if (got === exp) begin
        $display("  [PASS] %-44s got=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-44s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

// ── do_reset: full synchronous reset ─────────────────────────────────────────
// NOTE: the interrupted flag is captured from heartbeat_count at the negedge
// of rst_n (pre-reset value).  The caller controls what count is when reset fires.
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; param_write_en = 0;
    kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

// ── run_one_heartbeat: stream FD features and wait for done ───────────────────
task automatic run_one_heartbeat();
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;
    for (int t = 0; t < 5000; t++) begin
        @(posedge clk); #1;
        if (done) return;
    end
    $fatal(1, "[FATAL] run_one_heartbeat: timeout waiting for done");
endtask

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    $display("=== tb_warmup ===");

    // ── T1: clean start  →  ERR_WARMING_UP (not ERR_INTERRUPTED) ─────────────
    // Reset with heartbeat_count=0 (power-on) → interrupted=0.
    // After beat 1: error_code must be 0x8, NOT 0x9.
    $display("T1: clean start fires ERR_WARMING_UP (0x8), not ERR_INTERRUPTED");
    do_reset();   // heartbeat_count=0 → interrupted=0
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T1 error asserted",              error,      1);
    chk("T1 code = WARMING_UP (0x8)",     error_code, ERR_WARMING_UP);

    // ── T2: advisory persists through beats 2–99 ─────────────────────────────
    $display("T2: ERR_WARMING_UP persists through beats 2-99 (spot-check at 50)");
    for (int hb = 2; hb <= 99; hb++) begin
        run_one_heartbeat();
        if (hb == 50) begin
            repeat(2) @(posedge clk); #1;
            chk("T2 error asserted at beat 50",     error,      1);
            chk("T2 code = WARMING_UP at beat 50",  error_code, ERR_WARMING_UP);
        end
    end

    // ── T3: reset mid-warm-up  →  ERR_INTERRUPTED on restart ─────────────────
    // heartbeat_count is 99 here (after T2).  Resetting now sets interrupted=1.
    // The next beat's error_code must be 0x9, NOT 0x8.
    // Then a real fault (ERR_GAMMA_SAT) must override and latch sticky.
    $display("T3: reset at beat 99 fires ERR_INTERRUPTED (0x9) on restart");
    do_reset();   // heartbeat_count=99 → interrupted=1, count resets to 0
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T3 error asserted",              error,      1);
    chk("T3 code = INTERRUPTED (0x9)",    error_code, ERR_INTERRUPTED);

    // Inject gamma=9000 (>8192 = GAMMA_SAT_THRESH) → ERR_GAMMA_SAT (0x4)
    // must override the advisory and stick.
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'd9000;
    @(posedge clk); #1; param_write_en = 0;
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T3 real fault latched (0x4)",    error_code, ERR_GAMMA_SAT);
    chk("T3 real fault is sticky",        error,      1);
    repeat(20) @(posedge clk); #1;
    chk("T3 still GAMMA_SAT after 20cy",  error_code, ERR_GAMMA_SAT);

    // ── T4: advisory clears after beat 100 (with interrupted=1 still set) ────
    // Reset, restore gamma, run 100 beats.  interrupted is 1 here (T3 had
    // heartbeat_count=2 when its reset fires inside T4's do_reset) so
    // ERR_INTERRUPTED shows during the 100-beat run — but it must auto-clear
    // exactly like ERR_WARMING_UP does.
    $display("T4: advisory clears automatically after beat 100");
    do_reset();   // heartbeat_count=2 (after T3's 2 beats) → interrupted=1
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'h0100; // restore gamma=0.25
    @(posedge clk); #1; param_write_en = 0;
    for (int hb = 1; hb <= 100; hb++)
        run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T4 error cleared after beat 100", error,      0);
    chk("T4 error_code cleared",           error_code, 0);

    // ── T5: reset after completed warm-up  →  ERR_WARMING_UP (not INTERRUPTED)─
    // heartbeat_count=100 when reset fires → interrupted=0 (100 < 100 is false).
    // The new warm-up should show 0x8, not 0x9.
    $display("T5: reset after full warm-up shows WARMING_UP, not INTERRUPTED");
    do_reset();   // heartbeat_count=100 → interrupted=0
    repeat(2) @(posedge clk); #1;
    chk("T5 clean after reset",             error,      0);
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T5 code = WARMING_UP (0x8)",       error_code, ERR_WARMING_UP);
    chk("T5 NOT INTERRUPTED (0x9)",         error_code != ERR_INTERRUPTED, 1);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count > 0) $fatal(1, "tb_warmup: FAIL");
    else $display("tb_warmup: PASS");
    $finish;
end

initial begin #500_000_000; $fatal(1, "[FATAL] tb_warmup: watchdog timeout"); end

endmodule
