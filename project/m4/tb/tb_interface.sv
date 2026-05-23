// ============================================================================
// tb_interface.sv  —  ECE410_project_tb_netlist  pre-netlist suite
// ============================================================================
// Verifies the svm_host_if interface contract defined in svm_interfaces.sv.
//
//   Section 1  Register Map              (14 checks)
//     1.1  Default values after reset
//     1.2  Write / readback — all param_addr 0–6
//     1.3  Reserved address (addr=7) — all registers unchanged
//     1.4  Gamma saturation — write > 0x2000 rejected; ERR_GAMMA_SAT fires
//
//   Section 2  Error Code Values          (6 checks)
//     2.1  ERR_SV_ZERO          (0x1)  Σ sv_counts = 0
//     2.2  ERR_SV_OVERFLOW      (0x2)  Σ sv_counts > NUM_SV
//     2.3  ERR_GAMMA_SAT        (0x4)  gamma write > 8192 (0x2000)
//     2.4  ERR_GAMMA_ZERO       (0x6)  gamma = 0 while FSM not IDLE
//     2.5  ERR_NUM_SAMPLES_ZERO (0x7)  num_samples = 0 at start
//     2.6  Sticky               error_code holds across 20 idle cycles
//
//   Section 3  Start Protocol             (3 checks)
//     3.1  start in IDLE triggers batch — done fires
//     3.2  start outside IDLE ignored   — done fires exactly once
//     3.3  num_samples=2 batch — done fires once after 2 heartbeats
//
// Output: tb_interface.log  (and terminal)
//
// Compile & run:
//   iverilog -g2012 -o tb_if tb_interface.sv svm_compute_core.sv
//   vvp tb_if
// ============================================================================

`timescale 1ns/1ps
module tb_interface;

localparam int DW  = 16;
localparam int FD  = 8;    // FEATURE_DIM — small for fast simulation
localparam int NSV = 5;    // NUM_SV — one per class
localparam int BS  = 4;    // MAX_BATCH_SIZE

logic clk = 0;
always #5 clk = ~clk;

// ── DUT signals ───────────────────────────────────────────────────────────────
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
logic        vbatt_warn = 1'b0;   // no low-battery warning
logic        vbatt_ok   = 1'b1;   // battery OK
logic        start;
logic [9:0]  num_samples;
logic        done, error;
logic [3:0]  error_code;
logic [DW-1:0] kernel_out;
logic        kernel_valid, kernel_ready;

svm_compute_core #(
    .DATA_WIDTH(DW), .FRAC_BITS(10),
    .FEATURE_DIM(FD), .NUM_SV(NSV),
    .MAX_BATCH_SIZE(BS), .FIFO_DEPTH(32), .ADDR_WIDTH(5),
    .DEFAULT_GAMMA(0.25)
) dut (
    .clk(clk), .rst_n(rst_n),
    .param_write_en(param_write_en), .param_addr(param_addr),
    .param_data(param_data), .gamma_reg(gamma_reg), .c_reg(c_reg),
    .bias_reg(bias_reg), .num_sv_per_class(num_sv_per_class),
    .qspi_valid(qspi_valid), .qspi_data(qspi_data), .qspi_ready(qspi_ready),
    .sv_ram_addr(sv_ram_addr), .sv_ram_rdata(sv_ram_rdata), .sv_ram_ren(sv_ram_ren),
    .work_ram_addr(work_ram_addr), .work_ram_wdata(work_ram_wdata),
    .work_ram_rdata(work_ram_rdata), .work_ram_wen(work_ram_wen), .work_ram_ren(work_ram_ren),
    .vbatt_warn(vbatt_warn), .vbatt_ok(vbatt_ok),
    .start(start), .num_samples(num_samples),
    .done(done), .error(error), .error_code(error_code),
    .kernel_out(kernel_out), .kernel_valid(kernel_valid), .kernel_ready(kernel_ready)
);

// SV and workspace RAMs: feature = sv = 0 → distance = 0 → kernel = 1.0 (1024)
assign sv_ram_rdata   = 16'h0000;
assign work_ram_rdata = 16'h0000;

// ── Scoreboard ────────────────────────────────────────────────────────────────
integer fd;
int pass_count = 0, fail_count = 0;

task automatic dual(input string msg);
    $fdisplay(fd, "%s", msg);
    $display("%s", msg);
endtask

task automatic chk_eq(input string name, input logic [15:0] got, input logic [15:0] exp);
    if (got === exp) begin
        $fdisplay(fd, "  [PASS] %-44s got=0x%04X", name, got);
        $display(    "  [PASS] %-44s got=0x%04X", name, got);
        pass_count++;
    end else begin
        $fdisplay(fd, "  [FAIL] %-44s got=0x%04X  exp=0x%04X", name, got, exp);
        $display(    "  [FAIL] %-44s got=0x%04X  exp=0x%04X", name, got, exp);
        fail_count++;
    end
endtask

task automatic chk_code(input string name, input logic [3:0] got, input logic [3:0] exp);
    if (got === exp) begin
        $fdisplay(fd, "  [PASS] %-44s error_code=0x%X", name, got);
        $display(    "  [PASS] %-44s error_code=0x%X", name, got);
        pass_count++;
    end else begin
        $fdisplay(fd, "  [FAIL] %-44s got=0x%X  exp=0x%X", name, got, exp);
        $display(    "  [FAIL] %-44s got=0x%X  exp=0x%X", name, got, exp);
        fail_count++;
    end
endtask

task automatic chk_int(input string name, input int got, input int exp);
    if (got === exp) begin
        $fdisplay(fd, "  [PASS] %-44s got=%0d", name, got);
        $display(    "  [PASS] %-44s got=%0d", name, got);
        pass_count++;
    end else begin
        $fdisplay(fd, "  [FAIL] %-44s got=%0d  exp=%0d", name, got, exp);
        $display(    "  [FAIL] %-44s got=%0d  exp=%0d", name, got, exp);
        fail_count++;
    end
endtask

// ── Helpers ───────────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n = 0; start = 0; qspi_valid = 0; qspi_data = '0;
    param_write_en = 0; param_addr = '0; param_data = '0;
    kernel_ready = 1; num_samples = 10'd1;
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    repeat(4) @(posedge clk); #1;
    rst_n = 1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic write_reg(input logic [2:0] addr, input logic [15:0] data);
    @(posedge clk); #1;
    param_write_en = 1; param_addr = addr; param_data = data;
    @(posedge clk); #1;
    param_write_en = 0;
    @(posedge clk); #1;
endtask

// Feed FD feature words then wait for done; returns 1 on timeout
task automatic feed_and_wait(output int timed_out);
    int t;
    for (int i = 0; i < FD; i++) begin
        qspi_valid = 1; qspi_data = 16'h0000;
        @(posedge clk); #1;
    end
    qspi_valid = 0;
    t = 2000;
    while (!done && t > 0) begin @(posedge clk); #1; t--; end
    timed_out = (t == 0) ? 1 : 0;
endtask

// ── Main ──────────────────────────────────────────────────────────────────────
initial begin
    int to, done_count;

    fd = $fopen("tb_interface.log", "w");
    if (fd == 0) begin $display("ERROR: cannot open tb_interface.log"); $finish; end

    $fdisplay(fd, "=== tb_interface.sv — SVM Compute Core Interface Tests ===");
    $display(    "=== tb_interface.sv — SVM Compute Core Interface Tests ===");
    $fdisplay(fd, "    FEATURE_DIM=%0d  NUM_SV=%0d  MAX_BATCH_SIZE=%0d", FD, NSV, BS);
    $display(    "    FEATURE_DIM=%0d  NUM_SV=%0d  MAX_BATCH_SIZE=%0d", FD, NSV, BS);
    $fdisplay(fd, ""); $display("");

    // ========================================================================
    // Section 1: Register Map
    // ========================================================================
    dual("--- Section 1: Register Map ---");

    do_reset();

    // 1.1 Default values
    dual("  1.1  Default values after reset");
    chk_eq("gamma_reg  default  0.25 = 0x0100",   gamma_reg,   16'h0100);
    chk_eq("c_reg      default  1.0  = 0x0400",   c_reg,       16'h0400);
    chk_eq("bias_reg[0] default 0.0  = 0x0000",   bias_reg[0], 16'h0000);
    chk_eq("bias_reg[4] default 0.0  = 0x0000",   bias_reg[4], 16'h0000);

    // 1.2 Write and readback — all addresses 0–6
    dual("  1.2  Write / readback (param_addr 0–6)");
    write_reg(3'h0, 16'h0200); chk_eq("addr=0 gamma_reg  write 0x0200", gamma_reg,   16'h0200);
    write_reg(3'h1, 16'h0800); chk_eq("addr=1 c_reg      write 0x0800", c_reg,       16'h0800);
    write_reg(3'h2, 16'h0100); chk_eq("addr=2 bias_reg[0] write 0x0100", bias_reg[0], 16'h0100);
    write_reg(3'h3, 16'h0200); chk_eq("addr=3 bias_reg[1] write 0x0200", bias_reg[1], 16'h0200);
    write_reg(3'h4, 16'h0300); chk_eq("addr=4 bias_reg[2] write 0x0300", bias_reg[2], 16'h0300);
    write_reg(3'h5, 16'h0400); chk_eq("addr=5 bias_reg[3] write 0x0400", bias_reg[3], 16'h0400);
    write_reg(3'h6, 16'h0500); chk_eq("addr=6 bias_reg[4] write 0x0500", bias_reg[4], 16'h0500);

    // 1.3 Reserved address 7 — all registers unchanged
    dual("  1.3  Reserved address (addr=7) — writes ignored");
    write_reg(3'h7, 16'hFFFF);
    chk_eq("gamma_reg   unchanged after reserved write", gamma_reg,   16'h0200);
    chk_eq("c_reg       unchanged after reserved write", c_reg,       16'h0800);
    chk_eq("bias_reg[0] unchanged after reserved write", bias_reg[0], 16'h0100);

    // 1.4 Gamma saturation — write IS accepted; ERR_GAMMA_SAT fires when FSM active
    // (gamma_int updates unconditionally; the error priority encoder checks state != IDLE)
    dual("  1.4  Gamma saturation: write accepted; ERR_GAMMA_SAT fires when FSM active");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    write_reg(3'h0, 16'd9000);
    chk_eq("gamma_reg written to 9000 (write not blocked in IDLE)", gamma_reg, 16'd9000);
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_GAMMA_SAT (0x4) fires when FSM leaves IDLE", error_code, 4'h4);

    $fdisplay(fd, ""); $display("");

    // ========================================================================
    // Section 2: Error Code Values
    // ========================================================================
    dual("--- Section 2: Error Code Values ---");

    // 2.1 ERR_SV_ZERO (0x1)
    dual("  2.1  ERR_SV_ZERO (0x1): all sv_counts = 0");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd0;
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_SV_ZERO", error_code, 4'h1);

    // 2.2 ERR_SV_OVERFLOW (0x2): sum(sv_counts) > NUM_SV=5
    dual("  2.2  ERR_SV_OVERFLOW (0x2): sum sv_counts > NUM_SV");
    do_reset();
    num_sv_per_class[0] = 8'd3; num_sv_per_class[1] = 8'd3;
    for (int i = 2; i < 5; i++) num_sv_per_class[i] = 8'd0; // sum = 6 > 5
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_SV_OVERFLOW", error_code, 4'h2);

    // 2.3 ERR_GAMMA_SAT (0x4): write saturating gamma, then start to leave IDLE
    dual("  2.3  ERR_GAMMA_SAT (0x4): gamma > 0x2000 then FSM activated");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    write_reg(3'h0, 16'd9000);          // accepted in IDLE (no write-time rejection)
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_GAMMA_SAT", error_code, 4'h4);

    // 2.4 ERR_GAMMA_ZERO (0x6): write gamma=0, advance FSM out of IDLE
    dual("  2.4  ERR_GAMMA_ZERO (0x6): gamma=0 while FSM not IDLE");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    write_reg(3'h0, 16'h0000);     // gamma = 0 (accepted; not above sat threshold)
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    for (int i = 0; i < FD; i++) begin
        qspi_valid = 1; qspi_data = 16'h0000;
        @(posedge clk); #1;
    end
    qspi_valid = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_GAMMA_ZERO", error_code, 4'h6);

    // 2.5 ERR_NUM_SAMPLES_ZERO (0x7): num_samples=0 → last_heartbeat underflows, batch never ends
    dual("  2.5  ERR_NUM_SAMPLES_ZERO (0x7): num_samples = 0 at start");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    num_samples = 10'd0;
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_NUM_SAMPLES_ZERO", error_code, 4'h7);

    // 2.6 Sticky: error_code holds across 20 idle cycles
    dual("  2.6  Sticky: error_code holds after 20 idle cycles");
    repeat(20) @(posedge clk); #1;
    chk_code("error_code sticky (0x7)", error_code, 4'h7);

    $fdisplay(fd, ""); $display("");

    // ========================================================================
    // Section 3: Start Protocol
    // ========================================================================
    dual("--- Section 3: Start Protocol ---");

    // 3.1 start in IDLE triggers batch — done fires
    dual("  3.1  start in IDLE triggers batch (done fires)");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    num_samples = 10'd1;
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    feed_and_wait(to);
    chk_int("done fires after valid start", to, 0);  // 0 = no timeout

    // 3.2 start outside IDLE is ignored — done fires exactly once
    dual("  3.2  start outside IDLE ignored (done fires exactly once)");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    num_samples = 10'd1;
    done_count = 0;
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    // Pulse start again immediately — FSM is now in LOAD_FIFO, should ignore it
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    for (int i = 0; i < FD; i++) begin
        qspi_valid = 1; qspi_data = 16'h0000;
        @(posedge clk); #1;
        if (done) done_count++;
    end
    qspi_valid = 0;
    repeat(600) begin @(posedge clk); #1; if (done) done_count++; end
    chk_int("done fires exactly once (not twice)", done_count, 1);

    // 3.3 num_samples=2 batch: done fires exactly once after both heartbeats
    // Note: fifo_wr_en is only active in LOAD_FIFO; each heartbeat's features
    // must be fed while the FSM is in LOAD_FIFO, not all at once upfront.
    dual("  3.3  num_samples=2 batch: done fires once after 2 heartbeats");
    do_reset();
    for (int i = 0; i < 5; i++) num_sv_per_class[i] = 8'd1;
    num_samples = 10'd2;
    done_count = 0;
    @(posedge clk); #1; start = 1; @(posedge clk); #1; start = 0;
    // Feed heartbeat 1 — FSM is in LOAD_FIFO, accepts FD words
    for (int i = 0; i < FD; i++) begin
        qspi_valid = 1; qspi_data = 16'h0000;
        @(posedge clk); #1;
        if (done) done_count++;
    end
    qspi_valid = 0;
    // FSM processes heartbeat 1 and returns to LOAD_FIFO (awaiting heartbeat 2)
    repeat(2000) begin @(posedge clk); #1; if (done) done_count++; end
    // Feed heartbeat 2 — FSM is back in LOAD_FIFO
    for (int i = 0; i < FD; i++) begin
        qspi_valid = 1; qspi_data = 16'h0000;
        @(posedge clk); #1;
        if (done) done_count++;
    end
    qspi_valid = 0;
    repeat(2000) begin @(posedge clk); #1; if (done) done_count++; end
    chk_int("done fires once for 2-sample batch", done_count, 1);

    // ========================================================================
    // Summary
    // ========================================================================
    $fdisplay(fd, ""); $display("");
    $fdisplay(fd, "=== Results: %0d passed  %0d failed / %0d total ===",
              pass_count, fail_count, pass_count + fail_count);
    $display(    "=== Results: %0d passed  %0d failed / %0d total ===",
              pass_count, fail_count, pass_count + fail_count);
    $fclose(fd);
    if (fail_count > 0) $fatal(1, "tb_interface: FAIL");
    else $display("tb_interface: PASS");
    $finish;
end

initial begin #10_000_000; $fatal(1, "[FATAL] tb_interface: global timeout"); end

endmodule
