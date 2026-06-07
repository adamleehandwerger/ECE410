// ============================================================================
// tb_power.sv  —  ECE410_project_LUT  pre-netlist suite
// ============================================================================
// Verifies ERR_LOW_BATTERY (0xA) and ERR_POWER_FAIL (0xB) advisory behavior.
//
//   vbatt_warn  (active-high): battery below soft threshold.
//               Reports ERR_LOW_BATTERY (0xA); FSM continues normally.
//
//   vbatt_ok    (active-high): battery above hard operational threshold.
//               When deasserted: reports ERR_POWER_FAIL (0xB) and blocks
//               start (FSM will not leave IDLE until vbatt_ok is restored).
//               A running classification is not aborted — the flag is advisory.
//
//  T1  vbatt_warn=1, vbatt_ok=1  →  ERR_LOW_BATTERY (0xA) shows; run completes
//  T2  vbatt_ok=0                →  ERR_POWER_FAIL (0xB) shows; start blocked
//  T3  vbatt_ok restored (=1)    →  error clears; start proceeds normally
//  T4  vbatt_warn=1 during run   →  advisory shows; FSM finishes; real faults
//                                    still sticky and override ERR_LOW_BATTERY
//  T5  vbatt_ok=0 during run     →  advisory shows; FSM finishes; done fires
//
// Timing note: advisory codes latch one cycle after the triggering condition
// is visible in err_detect.  Checks wait 2 cycles after a state change.
//
// Compile & run:
//   iverilog -g2012 -o tb_pwr tb_power.sv svm_compute_core.sv
//   vvp tb_pwr
// ============================================================================

`timescale 1ns/1ps
module tb_power;

localparam int DW  = 16;
localparam int FD  = 8;
localparam int NSV = 5;

localparam int ERR_NONE        = 4'h0;
localparam int ERR_LOW_BATTERY = 4'hA;
localparam int ERR_POWER_FAIL  = 4'hB;

logic clk = 0;
always #5 clk = ~clk;

logic        rst_n, param_write_en, vbatt_warn, vbatt_ok;
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
    .vbatt_warn(vbatt_warn), .vbatt_ok(vbatt_ok),
    .qspi_valid(qspi_valid), .qspi_data(qspi_data), .qspi_ready(qspi_ready),
    .sv_ram_addr(sv_ram_addr), .sv_ram_rdata(sv_ram_rdata), .sv_ram_ren(sv_ram_ren),
    .work_ram_addr(work_ram_addr), .work_ram_wdata(work_ram_wdata),
    .work_ram_rdata(work_ram_rdata), .work_ram_wen(work_ram_wen),
    .work_ram_ren(work_ram_ren),
    .start(start), .num_samples(num_samples),
    .done(done), .error(error), .error_code(error_code),
    .kernel_out(kernel_out), .kernel_valid(kernel_valid),
    .kernel_ready(kernel_ready)
);

assign sv_ram_rdata   = 16'h0400;
assign work_ram_rdata = '0;

// ── Scoreboard ────────────────────────────────────────────────────────────────
int pass_count = 0, fail_count = 0;
task automatic chk(input string name, input int got, input int exp);
    if (got === exp) begin
        $display("  [PASS] %-48s got=0x%0h", name, got); pass_count++;
    end else begin
        $display("  [FAIL] %-48s got=0x%0h  exp=0x%0h", name, got, exp); fail_count++;
    end
endtask

// ── do_reset ─────────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; param_write_en = 0;
    kernel_ready = 1; num_samples = 10'd1;
    vbatt_warn = 0; vbatt_ok = 1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(4) @(posedge clk); #1;
endtask

// ── run_one_heartbeat ─────────────────────────────────────────────────────────
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
    int timeout;
    $display("=== tb_power ===");

    // ── T1: vbatt_warn=1 → ERR_LOW_BATTERY fires; run completes ──────────────
    $display("T1: vbatt_warn asserted → ERR_LOW_BATTERY (0xA); FSM runs normally");
    do_reset();
    // Warm up 100 beats first so warm-up advisory doesn't mask the power code
    for (int hb = 1; hb <= 100; hb++) run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    // Power warning goes active
    vbatt_warn = 1;
    repeat(4) @(posedge clk); #1;
    chk("T1 error asserted",              error,      1);
    chk("T1 code = LOW_BATTERY (0xA)",    error_code, ERR_LOW_BATTERY);
    // Run a heartbeat — FSM must still accept start and complete
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T1 done after LOW_BATTERY",      done,       0); // done cleared after 2cy
    chk("T1 code still LOW_BATTERY",      error_code, ERR_LOW_BATTERY);
    // Clear warning
    vbatt_warn = 0;
    repeat(4) @(posedge clk); #1;
    chk("T1 error clears when warn gone", error,      0);

    // ── T2: vbatt_ok=0 → ERR_POWER_FAIL fires; start blocked ─────────────────
    $display("T2: vbatt_ok deasserted → ERR_POWER_FAIL (0xB); start blocked");
    vbatt_ok = 0;
    repeat(4) @(posedge clk); #1;
    chk("T2 error asserted",              error,      1);
    chk("T2 code = POWER_FAIL (0xB)",     error_code, ERR_POWER_FAIL);
    // Attempt start — FSM must stay in IDLE
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0;
    // Feed FD features; since FSM is in IDLE, qspi_ready stays low → no FIFO write
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;
    // Wait several cycles — done must never fire
    timeout = 200;
    while (!done && timeout > 0) begin @(posedge clk); #1; timeout--; end
    chk("T2 start blocked (no done)",     (timeout > 0 ? 0 : 1), 1);

    // ── T3: restore vbatt_ok → error clears; start proceeds ──────────────────
    $display("T3: vbatt_ok restored → error clears; new run completes");
    vbatt_ok = 1;
    repeat(4) @(posedge clk); #1;
    chk("T3 error cleared",               error,      0);
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T3 run succeeded (no fault)",    error, 0);

    // ── T4: vbatt_warn fires during run; real fault overrides ────────────────
    $display("T4: ERR_LOW_BATTERY overridden by real fault (ERR_GAMMA_SAT)");
    do_reset();
    // Warm up 100 beats
    for (int hb = 1; hb <= 100; hb++) run_one_heartbeat();
    // Inject bad gamma
    @(posedge clk); #1;
    param_write_en = 1; param_addr = 3'b000; param_data = 16'd9000;
    @(posedge clk); #1; param_write_en = 0;
    vbatt_warn = 1;  // warning active at same time
    run_one_heartbeat();
    repeat(4) @(posedge clk); #1;
    chk("T4 real fault overrides LOW_BATTERY", error_code, 4'h4); // ERR_GAMMA_SAT
    chk("T4 real fault is sticky",             error,      1);
    // Clear warning; real fault must persist
    vbatt_warn = 0;
    repeat(4) @(posedge clk); #1;
    chk("T4 fault persists after warn clears", error_code, 4'h4);

    // ── T5: vbatt_ok=0 mid-run; FSM still finishes; done fires ───────────────
    $display("T5: vbatt_ok drops mid-run; FSM finishes; ERR_POWER_FAIL advisory");
    do_reset();
    // Warm up 100 beats
    for (int hb = 1; hb <= 100; hb++) run_one_heartbeat();
    // Start a new run, then pull vbatt_ok low while features are streaming
    @(posedge clk); #1; start = 1;
    @(posedge clk); #1; start = 0; vbatt_ok = 0;  // drop power mid-stream
    for (int i = 0; i < FD; i++) begin
        @(posedge clk); #1;
        qspi_valid = 1; qspi_data = 16'h0400;
    end
    @(posedge clk); #1; qspi_valid = 0;
    // Wait for done — FSM must complete despite vbatt_ok=0
    timeout = 5000;
    while (!done && timeout > 0) begin @(posedge clk); #1; timeout--; end
    chk("T5 FSM completes (done fires)",   (timeout > 0 ? 1 : 0), 1);
    repeat(4) @(posedge clk); #1;
    chk("T5 code = POWER_FAIL (0xB)",      error_code, ERR_POWER_FAIL);
    vbatt_ok = 1;
    repeat(4) @(posedge clk); #1;
    chk("T5 clears when vbatt_ok restored", error,     0);

    $display("=== Results: %0d passed, %0d failed ===", pass_count, fail_count);
    if (fail_count > 0) $fatal(1, "tb_power: FAIL");
    else $display("tb_power: PASS");
    $finish;
end

initial begin #500_000_000; $fatal(1, "[FATAL] tb_power: watchdog timeout"); end

endmodule
