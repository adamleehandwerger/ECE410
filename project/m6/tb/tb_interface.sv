// tb_interface.sv — m6 RAM interface
// Verifies the svm_compute_core register map and control protocol.
//
//   Section 1  Register Map
//     1.1  Default values after reset (gamma_reg, c_reg)
//     1.2  Write / readback — addr 0 (gamma) and addr 1 (c_reg)
//     1.3  Reserved address (addr=7) — registers unchanged
//     1.4  Gamma saturation — write accepted; ERR_GAMMA_SAT fires on start
//
//   Section 2  Error Code Values
//     2.1  ERR_SV_ZERO          (0x1)
//     2.2  ERR_SV_OVERFLOW      (0x2)
//     2.3  ERR_GAMMA_SAT        (0x4)
//     2.4  ERR_GAMMA_ZERO       (0x6)
//     2.5  ERR_NUM_SAMPLES_ZERO (0x7)
//     2.6  Sticky               error_code holds across 20 idle cycles
//
//   Section 3  Start Protocol
//     3.1  start in IDLE triggers batch — done fires
//     3.2  start outside IDLE ignored  — done fires exactly once
//     3.3  num_samples=2 batch         — done fires once after 2 heartbeats
`timescale 1ns/1ps
`default_nettype none
module tb_interface;

localparam int DW  = 16;
localparam int FD  = 8;
localparam int NSV = 5;
localparam int BS  = 4;
localparam int LAT = 1;

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

logic [39:0] sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
assign num_sv_per_class_flat = sv_flat_r;

// All SRAM zeros → dist=0 → kernel=1024
logic [18:0] addr_r = '0;
always_ff @(posedge clk) addr_r <= ram_addr;
assign ram_rdata = 16'h0000;

svm_compute_core #(.DATA_WIDTH(DW), .FEATURE_DIM(FD), .NUM_SV(NSV),
                   .MAX_BATCH_SIZE(BS), .RAM_LATENCY(LAT)) dut (
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

// ── Scoreboard ────────────────────────────────────────────────────────────
integer fd;
int pass_count=0, fail_count=0;

task automatic dual(input string msg);
    $fdisplay(fd, "%s", msg); $display("%s", msg);
endtask

task automatic chk_eq(input string name, input logic [15:0] got, input logic [15:0] exp);
    if (got===exp) begin
        $fdisplay(fd,"  [PASS] %-44s got=0x%04X",name,got);
        $display(    "  [PASS] %-44s got=0x%04X",name,got); pass_count++;
    end else begin
        $fdisplay(fd,"  [FAIL] %-44s got=0x%04X  exp=0x%04X",name,got,exp);
        $display(    "  [FAIL] %-44s got=0x%04X  exp=0x%04X",name,got,exp); fail_count++;
    end
endtask

task automatic chk_code(input string name, input logic [3:0] got, input logic [3:0] exp);
    if (got===exp) begin
        $fdisplay(fd,"  [PASS] %-44s error_code=0x%X",name,got);
        $display(    "  [PASS] %-44s error_code=0x%X",name,got); pass_count++;
    end else begin
        $fdisplay(fd,"  [FAIL] %-44s got=0x%X  exp=0x%X",name,got,exp);
        $display(    "  [FAIL] %-44s got=0x%X  exp=0x%X",name,got,exp); fail_count++;
    end
endtask

task automatic chk_int(input string name, input int got, input int exp);
    if (got===exp) begin
        $fdisplay(fd,"  [PASS] %-44s got=%0d",name,got);
        $display(    "  [PASS] %-44s got=%0d",name,got); pass_count++;
    end else begin
        $fdisplay(fd,"  [FAIL] %-44s got=%0d  exp=%0d",name,got,exp);
        $display(    "  [FAIL] %-44s got=%0d  exp=%0d",name,got,exp); fail_count++;
    end
endtask

// ── Helpers ───────────────────────────────────────────────────────────────
task automatic do_reset();
    rst_n=0; start=0; param_write_en=0; num_samples=10'd1;
    sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
    repeat(4) @(posedge clk); #1; rst_n=1;
    repeat(2) @(posedge clk); #1;
endtask

task automatic write_reg(input logic [2:0] addr, input logic [15:0] data);
    @(posedge clk); #1;
    param_write_en=1; param_addr=addr; param_data=data;
    @(posedge clk); #1; param_write_en=0;
    @(posedge clk); #1;
endtask

// ── Main ──────────────────────────────────────────────────────────────────
initial begin
    int to, done_count, timeout;

    fd = $fopen("tb_interface.log","w");
    if (fd==0) begin $display("ERROR: cannot open tb_interface.log"); $finish; end

    $fdisplay(fd,"=== tb_interface.sv — SVM Compute Core Interface Tests ===");
    $display(    "=== tb_interface.sv — SVM Compute Core Interface Tests ===");
    $fdisplay(fd,"    FEATURE_DIM=%0d  NUM_SV=%0d  MAX_BATCH_SIZE=%0d",FD,NSV,BS);
    $display(    "    FEATURE_DIM=%0d  NUM_SV=%0d  MAX_BATCH_SIZE=%0d",FD,NSV,BS);

    // ======================================================================
    // Section 1: Register Map
    // ======================================================================
    dual("--- Section 1: Register Map ---");

    do_reset();

    // 1.1 Default values
    dual("  1.1  Default values after reset");
    chk_eq("gamma_reg default 0.25 = 0x0100", gamma_reg, 16'h0100);
    chk_eq("c_reg     default 1.0  = 0x0400", c_reg,     16'h0400);

    // 1.2 Write and readback — addr 0 (gamma) and 1 (c_reg)
    dual("  1.2  Write / readback (param_addr 0-1)");
    write_reg(3'h0, 16'h0200); chk_eq("addr=0 gamma_reg write 0x0200", gamma_reg, 16'h0200);
    write_reg(3'h1, 16'h0800); chk_eq("addr=1 c_reg     write 0x0800", c_reg,     16'h0800);
    // Write addr 2-6 (bias, internal — just verify no crash)
    write_reg(3'h2, 16'h0100); write_reg(3'h3, 16'h0200);
    write_reg(3'h4, 16'h0300); write_reg(3'h5, 16'h0400); write_reg(3'h6, 16'h0500);

    // 1.3 Reserved address 7 — gamma and c_reg unchanged
    dual("  1.3  Reserved address (addr=7) — writes ignored");
    write_reg(3'h7, 16'hFFFF);
    chk_eq("gamma_reg unchanged after reserved write", gamma_reg, 16'h0200);
    chk_eq("c_reg     unchanged after reserved write", c_reg,     16'h0800);

    // 1.4 Gamma saturation — write IS accepted; ERR_GAMMA_SAT fires when FSM active
    dual("  1.4  Gamma saturation: write accepted; ERR_GAMMA_SAT fires on start");
    do_reset();
    sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
    write_reg(3'h0, 16'd9000);
    chk_eq("gamma_reg written to 9000", gamma_reg, 16'd9000);
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_GAMMA_SAT (0x4) fires", error_code, 4'h4);

    $fdisplay(fd,""); $display("");

    // ======================================================================
    // Section 2: Error Code Values
    // ======================================================================
    dual("--- Section 2: Error Code Values ---");

    // 2.1 ERR_SV_ZERO (0x1)
    dual("  2.1  ERR_SV_ZERO (0x1)");
    do_reset(); sv_flat_r = 40'd0;
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_SV_ZERO", error_code, 4'h1);

    // 2.2 ERR_SV_OVERFLOW (0x2): sum=6 > NSV=5
    dual("  2.2  ERR_SV_OVERFLOW (0x2)");
    do_reset(); sv_flat_r = {8'd0,8'd0,8'd3,8'd3,8'd0};  // sum=6
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_SV_OVERFLOW", error_code, 4'h2);

    // 2.3 ERR_GAMMA_SAT (0x4)
    dual("  2.3  ERR_GAMMA_SAT (0x4)");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
    write_reg(3'h0, 16'd9000);
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_GAMMA_SAT", error_code, 4'h4);

    // 2.4 ERR_GAMMA_ZERO (0x6)
    dual("  2.4  ERR_GAMMA_ZERO (0x6): gamma=0 while FSM not IDLE");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
    write_reg(3'h0, 16'h0000);
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    timeout=2000;
    while (!done && !error && timeout>0) begin @(posedge clk); #1; timeout--; end
    chk_code("ERR_GAMMA_ZERO", error_code, 4'h6);

    // 2.5 ERR_NUM_SAMPLES_ZERO (0x7)
    dual("  2.5  ERR_NUM_SAMPLES_ZERO (0x7)");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1};
    num_samples = 10'd0;
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    repeat(4) @(posedge clk); #1;
    chk_code("ERR_NUM_SAMPLES_ZERO", error_code, 4'h7);

    // 2.6 Sticky
    dual("  2.6  Sticky: error_code holds after 20 idle cycles");
    repeat(20) @(posedge clk); #1;
    chk_code("error_code sticky (0x7)", error_code, 4'h7);

    $fdisplay(fd,""); $display("");

    // ======================================================================
    // Section 3: Start Protocol
    // ======================================================================
    dual("--- Section 3: Start Protocol ---");

    // 3.1 start in IDLE → done fires
    dual("  3.1  start in IDLE triggers batch (done fires)");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1}; num_samples=10'd1;
    write_reg(3'h0, 16'h0100);
    @(posedge clk); #1; start=1; @(posedge clk); #1; start=0;
    to=2000;
    while (!done && to>0) begin @(posedge clk); #1; to--; end
    chk_int("done fires after valid start", (to>0 ? 1:0), 1);

    // 3.2 start outside IDLE ignored — done fires exactly once
    dual("  3.2  start outside IDLE ignored (done fires exactly once)");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1}; num_samples=10'd1;
    write_reg(3'h0, 16'h0100);
    done_count=0;
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    @(posedge clk); #1; start=1;  // FSM busy, should ignore
    @(posedge clk); #1; start=0;
    for (int t=0; t<2000; t++) begin
        @(posedge clk); #1;
        if (done) done_count++;
    end
    chk_int("done fires exactly once", done_count, 1);

    // 3.3 num_samples=2 batch — done fires once after both heartbeats
    dual("  3.3  num_samples=2: done fires once after 2 heartbeats");
    do_reset(); sv_flat_r = {8'd1,8'd1,8'd1,8'd1,8'd1}; num_samples=10'd2;
    write_reg(3'h0, 16'h0100);
    done_count=0;
    @(posedge clk); #1; start=1;
    @(posedge clk); #1; start=0;
    for (int t=0; t<5000; t++) begin
        @(posedge clk); #1;
        if (done) done_count++;
    end
    chk_int("done fires once for 2-sample batch", done_count, 1);

    // ======================================================================
    $fdisplay(fd,""); $display("");
    $fdisplay(fd,"=== Results: %0d passed  %0d failed / %0d total ===",
              pass_count, fail_count, pass_count+fail_count);
    $display(    "=== Results: %0d passed  %0d failed / %0d total ===",
              pass_count, fail_count, pass_count+fail_count);
    $fclose(fd);
    if (fail_count>0) $fatal(1,"tb_interface: FAIL");
    else $display("tb_interface: PASS");
    $finish;
end

initial begin #10_000_000; $fatal(1,"[FATAL] tb_interface: global timeout"); end
endmodule
`default_nettype wire
