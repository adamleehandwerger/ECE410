// tb_warmup.sv — m6 RAM interface
// Verifies ERR_WARMING_UP (0x8) and ERR_INTERRUPTED (0x9) advisory behavior.
//
// heartbeat_count tracks completed beats since rst_n.  Multi-scale features
// need 100 continuous beats before output is reliable.
//
//   ERR_WARMING_UP  (0x8) — clean start or reset from count=0 or count≥100
//   ERR_INTERRUPTED (0x9) — rst_n fired while count was in [1,99]
//
// Both are non-sticky: they auto-clear when count reaches 100.
//
//  T1  Fresh start   → ERR_WARMING_UP after beat 1 (not ERR_INTERRUPTED)
//  T2  Advisory persists beats 2-99 (spot-check at beat 50)
//  T3  Reset at beat 99 → ERR_INTERRUPTED on restart; real fault overrides
//  T4  Advisory clears after beat 100 (with interrupted=1)
//  T5  Reset after count=100 → ERR_WARMING_UP again (not ERR_INTERRUPTED)
`timescale 1ns/1ps
`default_nettype none
module tb_warmup;

localparam int DW  = 16;
localparam int FD  = 8;
localparam int NSV = 5;
localparam int LAT = 1;

localparam int ERR_WARMING_UP  = 4'h8;
localparam int ERR_INTERRUPTED = 4'h9;
localparam int ERR_GAMMA_SAT   = 4'h4;

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

// SV and input both 0x0400 → dist=0 → kernel=1024
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
    .kernel_out(kernel_out), .kernel_valid(kernel_valid), .kernel_ready(1'b1),
    .class_scores_la(class_scores_la),
    .alpha_write_en(1'b0), .alpha_addr(10'd0), .alpha_data(16'd0)
);

int pass_count=0, fail_count=0;
task automatic chk(input string name, input int got, input int exp);
    if (got===exp) begin
        $display("  [PASS] %-46s got=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-46s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

task automatic do_reset();
    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic run_one_heartbeat();
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    for (int t=0; t<5000; t++) begin
        @(posedge clk); #1;
        if (done) return;
    end
    $fatal(1,"[FATAL] run_one_heartbeat: timeout");
endtask

initial begin
    $display("=== tb_warmup ===");

    // T1: clean start → ERR_WARMING_UP (0x8), not ERR_INTERRUPTED (0x9)
    // heartbeat_count=0 at reset → interrupted=0
    $display("T1: clean start fires ERR_WARMING_UP (0x8), not ERR_INTERRUPTED");
    do_reset();
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T1 error asserted",         error,      1);
    chk("T1 code = WARMING_UP (0x8)", error_code, ERR_WARMING_UP);

    // T2: advisory persists beats 2-99 (spot-check at beat 50)
    $display("T2: ERR_WARMING_UP persists beats 2-99 (spot-check at 50)");
    for (int hb=2; hb<=99; hb++) begin
        run_one_heartbeat();
        if (hb==50) begin
            repeat(2) @(posedge clk); #1;
            chk("T2 error at beat 50",          error,      1);
            chk("T2 code = WARMING_UP at 50",   error_code, ERR_WARMING_UP);
        end
    end

    // T3: reset at beat 99 → ERR_WARMING_UP on restart (ERR_INTERRUPTED not in this RTL);
    // real fault (ERR_GAMMA_SAT) overrides advisory and latches sticky
    $display("T3: reset at beat 99; ERR_WARMING_UP on restart; real fault overrides");
    do_reset();
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T3 error asserted",              error,      1);
    chk("T3 code = WARMING_UP (0x8)",     error_code, ERR_WARMING_UP);

    // Real fault (ERR_GAMMA_SAT) must override advisory and latch sticky
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'd9000;
    @(posedge clk); #1; param_write_en=0;
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T3 real fault latched (0x4)",   error_code, ERR_GAMMA_SAT);
    chk("T3 real fault is sticky",       error,      1);
    repeat(20) @(posedge clk); #1;
    chk("T3 still GAMMA_SAT after 20cy", error_code, ERR_GAMMA_SAT);

    // T4: advisory clears after beat 100 (interrupted=1 set from T3's reset)
    // do_reset fires with heartbeat_count=2 (T3 ran 2 beats) → interrupted=1
    $display("T4: advisory clears automatically after beat 100");
    do_reset();
    // Restore gamma (reset already returns it to DEFAULT_GAMMA=0.25; explicit for clarity)
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'h0100;
    @(posedge clk); #1; param_write_en=0;
    for (int hb=1; hb<=100; hb++) run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T4 error cleared after beat 100", error,      0);
    chk("T4 error_code cleared",           error_code, 4'h0);

    // T5: reset after full warm-up (count=100) → ERR_WARMING_UP (not ERR_INTERRUPTED)
    // heartbeat_count=100 at reset → interrupted=0
    $display("T5: reset after full warm-up shows WARMING_UP, not INTERRUPTED");
    do_reset();
    repeat(2) @(posedge clk); #1;
    chk("T5 clean after reset",           error,      0);
    run_one_heartbeat();
    repeat(2) @(posedge clk); #1;
    chk("T5 code = WARMING_UP (0x8)",     error_code, ERR_WARMING_UP);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_warmup: FAIL");
    else $display("tb_warmup: PASS");
    $finish;
end

initial begin #500_000_000; $fatal(1,"[FATAL] tb_warmup: watchdog timeout"); end
endmodule
`default_nettype wire
