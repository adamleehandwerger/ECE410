// cosim_tb.sv — native (non-cocotb) full-dataset RTL cosim of svm_compute_core.
// Streams the 300-sample PhysioNet test set through the signed-off 600-SV core:
// loads the unified off-chip RAM ($readmemh), loads the Q6.10 model (gamma, 5
// OVR biases, 600 alphas), pulses start, and writes class_out at each sample_rdy
// to /tmp/cosim_preds.txt. Runs at native iverilog speed (no per-cycle Python).
`timescale 1ns/1ps
`default_nettype none
module cosim_tb;
  localparam integer NUM_SV = 600;
  localparam integer NTEST  = 300;

  reg clk = 1'b0;
  always #5 clk = ~clk;                 // 100 MHz sim clock

  reg          rst_n, param_write_en, vbatt_warn, vbatt_ok, start, kernel_ready;
  reg  [2:0]   param_addr;
  reg  [15:0]  param_data;
  reg  [39:0]  num_sv_per_class_flat;
  reg  [9:0]   num_samples;
  reg          alpha_write_en;
  reg  [9:0]   alpha_addr;
  reg  [15:0]  alpha_data;
  reg  [15:0]  ram_rdata;

  wire [18:0]  ram_addr;
  wire         ram_ren, sample_rdy, done, error, kernel_valid;
  wire [2:0]   class_out;
  wire [3:0]   error_code;
  wire [15:0]  gamma_reg, c_reg, kernel_out;
  wire [127:0] class_scores_la;

  // Unified off-chip RAM (rows 0..599 = SVs, 600..899 = features) + alpha image.
  reg [15:0] ram_mem   [0:230399];
  reg [15:0] alpha_mem [0:599];
  always @(posedge clk) ram_rdata <= ram_mem[ram_addr];   // 1-cycle registered read

  svm_compute_core #(.DATA_WIDTH(16), .FEATURE_DIM(256), .NUM_SV(NUM_SV)) dut (
    .clk(clk), .rst_n(rst_n),
    .param_write_en(param_write_en), .param_addr(param_addr), .param_data(param_data),
    .gamma_reg(gamma_reg), .c_reg(c_reg),
    .num_sv_per_class_flat(num_sv_per_class_flat),
    .ram_addr(ram_addr), .ram_rdata(ram_rdata), .ram_ren(ram_ren),
    .vbatt_warn(vbatt_warn), .vbatt_ok(vbatt_ok),
    .start(start), .num_samples(num_samples),
    .sample_rdy(sample_rdy), .class_out(class_out),
    .done(done), .error(error), .error_code(error_code),
    .kernel_out(kernel_out), .kernel_valid(kernel_valid), .kernel_ready(kernel_ready),
    .class_scores_la(class_scores_la),
    .alpha_write_en(alpha_write_en), .alpha_addr(alpha_addr), .alpha_data(alpha_data)
  );

  integer fd, np, a;
  reg sample_rdy_d;
  reg [63:0] cyc;

  // capture class_out on each sample_rdy rising edge
  always @(posedge clk) begin
    cyc <= cyc + 64'd1;
    sample_rdy_d <= sample_rdy;
    if (rst_n && sample_rdy && !sample_rdy_d) begin
      $fwrite(fd, "%0d\n", class_out);
      $fflush(fd);
      np = np + 1;
      if (np % 25 == 0) $display("  [%0t] classified %0d/%0d", $time, np, NTEST);
    end
    if (cyc > 64'd2_000_000_000) begin
      $display("TIMEOUT at np=%0d", np); $fclose(fd); $finish;
    end
  end

  task wr_param(input [2:0] addr, input [15:0] data);
    begin
      @(posedge clk); param_write_en = 1'b1; param_addr = addr; param_data = data;
      @(posedge clk); param_write_en = 1'b0;
    end
  endtask

  initial begin
    $readmemh("/tmp/cosim_ram.hex",   ram_mem);
    $readmemh("/tmp/cosim_alpha.hex", alpha_mem);
    fd = $fopen("/tmp/cosim_preds.txt", "w");
    np = 0; cyc = 0; sample_rdy_d = 0;
    rst_n=0; param_write_en=0; param_addr=0; param_data=0;
    num_sv_per_class_flat = 40'h7878787878;   // [120,120,120,120,120]
    vbatt_warn=0; vbatt_ok=1; start=0; num_samples=10'd300; kernel_ready=1;
    alpha_write_en=0; alpha_addr=0; alpha_data=0;
    repeat(6) @(posedge clk);
    rst_n = 1;
    repeat(3) @(posedge clk);

    wr_param(3'd0, 16'h0100);   // gamma = 0.25
    wr_param(3'd2, 16'hfc2a);   // bias Normal
    wr_param(3'd3, 16'h006a);   // bias PVC
    wr_param(3'd4, 16'hfc2f);   // bias AFib
    wr_param(3'd5, 16'hfd9e);   // bias VT
    wr_param(3'd6, 16'hfd3c);   // bias SVT

    for (a = 0; a < NUM_SV; a = a + 1) begin
      @(posedge clk); alpha_write_en = 1'b1; alpha_addr = a[9:0]; alpha_data = alpha_mem[a];
    end
    @(posedge clk); alpha_write_en = 1'b0;
    repeat(4) @(posedge clk);

    $display("[%0t] model loaded (NUM_SV=%0d gamma=%h) streaming %0d samples",
             $time, NUM_SV, gamma_reg, NTEST);
    @(posedge clk); start = 1'b1;
    @(posedge clk); start = 1'b0;

    wait (done && np >= NTEST);
    repeat(2) @(posedge clk);
    $display("[%0t] COSIM DONE: %0d/%0d predictions written", $time, np, NTEST);
    $fclose(fd);
    $finish;
  end
endmodule
`default_nettype wire
