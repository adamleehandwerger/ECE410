// tb_power.sv — m6 RAM interface
// Verifies ERR_LOW_BATTERY (0xA) and ERR_POWER_FAIL (0xB) advisory behavior.
//
//  T1  vbatt_warn=1, vbatt_ok=1  →  ERR_LOW_BATTERY; run completes
//  T2  vbatt_ok=0                →  ERR_POWER_FAIL; start blocked
//  T3  vbatt_ok restored         →  error clears; start proceeds
//  T4  vbatt_warn=1 + real fault →  real fault (ERR_GAMMA_SAT) overrides LOW_BATTERY
//  T5  vbatt_ok=0 mid-run        →  advisory; FSM finishes; done fires
//
// Advisory codes latch one cycle after triggering condition.
// 100-beat warm-up before advisory tests clears ERR_WARMING_UP.
`timescale 1ns/1ps
`default_nettype none
module tb_power;

localparam int DW  = 16;
localparam int FD  = 8;
localparam int NSV = 5;
localparam int LAT = 1;

localparam int ERR_NONE        = 4'h0;
localparam int ERR_LOW_BATTERY = 4'hA;
localparam int ERR_POWER_FAIL  = 4'hB;

logic clk = 0; always #5 clk = ~clk;

logic        rst_n=0, param_write_en=0, vbatt_warn=0, vbatt_ok=1;
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
    .vbatt_warn(vbatt_warn), .vbatt_ok(vbatt_ok),
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
        $display("  [PASS] %-50s got=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-50s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

task automatic do_reset();
    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    vbatt_warn=0; vbatt_ok=1;
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(4) @(posedge clk); #1;
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
    int timeout;
    $display("=== tb_power ===");

    // T1: vbatt_warn=1 → ERR_LOW_BATTERY; FSM runs normally
    $display("T1: vbatt_warn asserted → ERR_LOW_BATTERY (0xA); FSM runs normally");
    do_reset();
    for (int hb=1; hb<=100; hb++) run_one_heartbeat();  // clear warm-up advisory
    repeat(4) @(posedge clk); #1;
    vbatt_warn=1;
    repeat(4) @(posedge clk); #1;
    chk("T1 error asserted",           error,      1);
    chk("T1 code = LOW_BATTERY (0xA)", error_code, ERR_LOW_BATTERY);
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T1 code still LOW_BATTERY",   error_code, ERR_LOW_BATTERY);
    vbatt_warn=0;
    repeat(4) @(posedge clk); #1;
    chk("T1 error clears when warn gone", error, 0);

    // T2: vbatt_ok=0 → ERR_POWER_FAIL; start blocked
    $display("T2: vbatt_ok deasserted → ERR_POWER_FAIL (0xB); start blocked");
    vbatt_ok=0;
    repeat(4) @(posedge clk); #1;
    chk("T2 error asserted",           error,      1);
    chk("T2 code = POWER_FAIL (0xB)",  error_code, ERR_POWER_FAIL);
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    timeout=200;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    chk("T2 start blocked (no done)",  (timeout>0 ? 0:1), 1);

    // T3: vbatt_ok restored → error clears; run proceeds
    $display("T3: vbatt_ok restored → error clears; new run completes");
    vbatt_ok=1;
    repeat(4) @(posedge clk); #1;
    chk("T3 error cleared", error, 0);
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T3 run succeeded (no fault)", error, 0);

    // T4: vbatt_warn + real fault → real fault overrides LOW_BATTERY
    $display("T4: ERR_LOW_BATTERY overridden by real fault (ERR_GAMMA_SAT)");
    do_reset();
    for (int hb=1; hb<=100; hb++) run_one_heartbeat();
    @(posedge clk); #1;
    param_write_en=1; param_addr=3'h0; param_data=16'd9000;
    @(posedge clk); #1; param_write_en=0;
    vbatt_warn=1;
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T4 real fault overrides LOW_BATTERY", error_code, 4'h4);
    chk("T4 real fault is sticky",             error,      1);
    vbatt_warn=0;
    repeat(4) @(posedge clk); #1;
    chk("T4 fault persists after warn clears", error_code, 4'h4);

    // T5: vbatt_ok=0 mid-run; FSM finishes; ERR_POWER_FAIL advisory
    $display("T5: vbatt_ok drops mid-run; FSM finishes; done fires");
    do_reset();
    for (int hb=1; hb<=100; hb++) run_one_heartbeat();
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0; vbatt_ok=0;
    timeout=5000;
    while (!done && timeout>0) begin @(posedge clk); #1; timeout--; end
    chk("T5 FSM completes (done fires)",    (timeout>0 ? 1:0), 1);
    repeat(4) @(posedge clk); #1;
    chk("T5 code = POWER_FAIL (0xB)",       error_code, ERR_POWER_FAIL);
    vbatt_ok=1;
    repeat(4) @(posedge clk); #1;
    chk("T5 clears when vbatt_ok restored", error, 0);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count>0) $fatal(1,"tb_power: FAIL");
    else $display("tb_power: PASS");
    $finish;
end

initial begin #500_000_000; $fatal(1,"[FATAL] tb_power: watchdog timeout"); end
endmodule
`default_nettype wire
