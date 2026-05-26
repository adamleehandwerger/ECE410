// svm_ram_latency_tb.sv
// Verifies RAM_LATENCY wait-state logic: 10 beats, FEATURE_DIM=4,
// NUM_SV=5 (1 SV per class), RAM_LATENCY=3 (3-cycle async SRAM model).
//
// Pass criteria:
//   - sample_rdy fires exactly 10 times
//   - done fires exactly once
//   - no sticky error (error_code < 4'h8)

`timescale 1ns/1ps
`default_nettype none

module svm_ram_latency_tb;

    localparam int FEAT  = 4;
    localparam int NSV   = 5;    // 1 SV per class × 5 classes
    localparam int BEATS = 10;
    localparam int LAT   = 3;
    localparam int DW    = 16;

    // SRAM depth: address = {row[10:0], col[7:0]}, row stride = 256
    localparam int SRAM_DEPTH = (NSV + BEATS) * 256;  // 3840 words

    // ── clock ──────────────────────────────────────────────────────
    logic clk = 0;
    always #12.5 clk = ~clk;   // 40 MHz

    // ── DUT ports ──────────────────────────────────────────────────
    logic                rst_n          = 0;
    logic                param_write_en = 0;
    logic [2:0]          param_addr     = 0;
    logic [DW-1:0]       param_data     = 0;
    logic [DW-1:0]       gamma_reg, c_reg;
    logic [39:0]         num_sv_per_class_flat;
    logic [18:0]         ram_addr;
    logic [DW-1:0]       ram_rdata;
    logic                ram_ren;
    logic                vbatt_warn     = 0;
    logic                vbatt_ok       = 1;
    logic                start          = 0;
    logic [9:0]          num_samples;
    logic                sample_rdy;
    logic [2:0]          class_out;
    logic                done, error;
    logic [3:0]          error_code;
    logic [DW-1:0]       kernel_out;
    logic                kernel_valid;
    logic [127:0]        class_scores_la;
    logic                alpha_write_en = 0;
    logic [8:0]          alpha_addr_in  = 0;
    logic [DW-1:0]       alpha_data_in  = 0;

    assign num_samples            = 10'd10;
    assign num_sv_per_class_flat  = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1}; // 1 SV/class

    // ── SRAM model: combinational read from LAT-cycle delayed address ──
    logic [DW-1:0]  sram [0:SRAM_DEPTH-1];
    logic [18:0]    addr_pipe [0:LAT-1];

    always_ff @(posedge clk) begin
        addr_pipe[0] <= ram_addr;
        for (int i = 1; i < LAT; i++)
            addr_pipe[i] <= addr_pipe[i-1];
    end
    // Data valid exactly LAT cycles after address presented
    assign ram_rdata = sram[addr_pipe[LAT-1] % SRAM_DEPTH];

    // ── DUT ────────────────────────────────────────────────────────
    svm_compute_core #(
        .DATA_WIDTH    (DW),
        .FEATURE_DIM   (FEAT),
        .NUM_SV        (NSV),
        .MAX_BATCH_SIZE(BEATS),
        .RAM_LATENCY   (LAT)
    ) dut (
        .clk                  (clk),
        .rst_n                (rst_n),
        .param_write_en       (param_write_en),
        .param_addr           (param_addr),
        .param_data           (param_data),
        .gamma_reg            (gamma_reg),
        .c_reg                (c_reg),
        .num_sv_per_class_flat(num_sv_per_class_flat),
        .ram_addr             (ram_addr),
        .ram_rdata            (ram_rdata),
        .ram_ren              (ram_ren),
        .vbatt_warn           (vbatt_warn),
        .vbatt_ok             (vbatt_ok),
        .start                (start),
        .num_samples          (num_samples),
        .sample_rdy           (sample_rdy),
        .class_out            (class_out),
        .done                 (done),
        .error                (error),
        .error_code           (error_code),
        .kernel_out           (kernel_out),
        .kernel_valid         (kernel_valid),
        .kernel_ready         (1'b1),
        .class_scores_la      (class_scores_la),
        .alpha_write_en       (alpha_write_en),
        .alpha_addr           (alpha_addr_in),
        .alpha_data           (alpha_data_in)
    );

    // ── scoreboard ─────────────────────────────────────────────────
    int sample_count  = 0;
    int done_count    = 0;
    int cycle_count   = 0;
    int prev_beat_cyc = 0;

    always @(posedge clk) cycle_count++;

    always @(posedge clk) begin
        if (sample_rdy) begin
            sample_count++;
            $display("[cyc %0d]  sample_rdy beat %0d  class=%0d  error_code=0x%h  (delta=%0d cyc)",
                     cycle_count, sample_count, class_out, error_code,
                     cycle_count - prev_beat_cyc);
            prev_beat_cyc = cycle_count;
        end
        if (done) begin
            done_count++;
            $display("[cyc %0d]  DONE (batch complete)", cycle_count);
        end
        // Only display sticky errors (< 4'h8); suppress advisory ERR_WARMING_UP (0x8)
        if (error && error_code < 4'h8)
            $display("[cyc %0d]  STICKY ERROR code=0x%h", cycle_count, error_code);
    end

    // ── periodic FSM state dump (first 50 000 cycles, every 5000 cycles) ──
    always @(posedge clk) begin
        if (cycle_count > 0 && cycle_count <= 50_000 && (cycle_count % 5_000 == 0))
            $display("[cyc %0d]  FSM state=%0d  feat_wr=%0d  feat_rd=%0d  samp=%0d  cls=%0d  sv=%0d",
                     cycle_count,
                     dut.state,
                     dut.feat_wr_count,
                     dut.feat_rd_addr,
                     dut.sample_counter,
                     dut.class_counter,
                     dut.sv_counter);
    end

    // ── stimulus ───────────────────────────────────────────────────
    initial begin
        // Fill SRAM: SV rows 0..4 and input rows 5..14
        // All features = 0x0100 (= 1.0 in Q6.10)
        for (int r = 0; r < NSV + BEATS; r++)
            for (int c = 0; c < FEAT; c++)
                sram[r * 256 + c] = 16'h0100;

        // Reset
        rst_n = 0;
        repeat(8) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(4) @(posedge clk);

        // Fire start
        @(negedge clk); start = 1;
        @(posedge clk);
        @(negedge clk); start = 0;

        // Wait for done with timeout
        fork
            begin : wait_done
                wait(done_count >= 1);
            end
            begin : timeout
                repeat(2_000_000) @(posedge clk);
                $display("TIMEOUT: done never fired after 2M cycles");
                $finish;
            end
        join_any
        disable fork;

        repeat(8) @(posedge clk);

        // Verdict — advisory errors (code >= 0x8) are expected for <100-beat batches
        if (sample_count == BEATS && done_count == 1 &&
                (error_code == 4'h0 || error_code >= 4'h8))
            $display("PASS: %0d/%0d beats classified, done=1, no sticky errors (error_code=0x%h)",
                     sample_count, BEATS, error_code);
        else
            $display("FAIL: samples=%0d done=%0d error_code=0x%h",
                     sample_count, done_count, error_code);

        $finish;
    end

endmodule
`default_nettype wire
