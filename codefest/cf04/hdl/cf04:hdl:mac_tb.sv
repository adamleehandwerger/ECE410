module mac_tb;

    logic        clk;
    logic        rst;
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    logic signed [31:0] out;

    mac dut (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .out(out)
    );

    // 10ns clock period
    initial clk = 0;
    always #5 clk = ~clk;

    task apply(input logic signed [7:0] ta, tb);
        a = ta; b = tb;
        @(posedge clk); #1;
    endtask

    initial begin
        // Reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        rst = 0;

        // Cycle 1: 3 * 4 = 12  -> out = 12
        apply(3, 4);
        assert (out == 12) else $error("FAIL cycle 1: out=%0d", out);

        // Cycle 2: 2 * -5 = -10 -> out = 2
        apply(2, -5);
        assert (out == 2) else $error("FAIL cycle 2: out=%0d", out);

        // Cycle 3: -1 * -1 = 1  -> out = 3
        apply(-1, -1);
        assert (out == 3) else $error("FAIL cycle 3: out=%0d", out);

        // Cycle 4: 127 * 127 = 16129 -> out = 16132
        apply(127, 127);
        assert (out == 16132) else $error("FAIL cycle 4: out=%0d", out);

        // Synchronous reset mid-stream
        rst = 1;
        @(posedge clk); #1;
        rst = 0;
        assert (out == 0) else $error("FAIL reset: out=%0d", out);

        $display("ALL TESTS PASSED");
        $finish;
    end

endmodule
