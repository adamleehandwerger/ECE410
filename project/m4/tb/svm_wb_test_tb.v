// SPDX-FileCopyrightText: 2024 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// Testbench: svm_wb_test — Wishbone register read/write for user_project_wrapper
// Monitors mprj_io[31:16] (checkbits) for firmware test-completion codes.

`default_nettype none
`timescale 1 ns / 1 ps

module svm_wb_test_tb;
    reg  clock;
    reg  RSTB;
    reg  CSB;
    reg  power1, power2;
    reg  power3, power4;

    wire        gpio;
    wire [37:0] mprj_io;
    wire [15:0] checkbits;

    assign checkbits  = mprj_io[31:16];
    assign mprj_io[3] = 1'b1;  // flash IO

    always #12.5 clock <= (clock === 1'b0);
    initial clock = 0;

`ifdef ENABLE_SDF
    initial begin
        $sdf_annotate("../../../sdf/user_project_wrapper.sdf", uut.mprj.mprj);
    end
`endif

    initial begin
        $dumpfile("svm_wb_test.vcd");
        $dumpvars(0, svm_wb_test_tb);

        repeat (100) begin
            repeat (1000) @(posedge clock);
        end
        $display("%c[1;31m", 27);
`ifdef GL
        $display("Monitor: Timeout, SVM WB Test (GL) Failed");
`else
        $display("Monitor: Timeout, SVM WB Test (RTL) Failed");
`endif
        $display("%c[0m", 27);
        $finish;
    end

    initial begin
        wait (checkbits == 16'hBB90);
        $display("Monitor: SVM WB test started");

        wait (checkbits == 16'hBB91 || checkbits == 16'hBB00);
        if (checkbits == 16'hBB91) begin
`ifdef GL
            $display("Monitor: SVM WB Test (GL) Passed");
`else
            $display("Monitor: SVM WB Test (RTL) Passed");
`endif
        end else begin
            $display("%c[1;31m", 27);
            $display("Monitor: SVM WB Test FAILED — register readback error");
            $display("%c[0m", 27);
        end
        $finish;
    end

    initial begin
        RSTB <= 1'b0;
        CSB  <= 1'b1;
        #2000;
        RSTB <= 1'b1;
        #100000;
        CSB = 1'b0;
    end

    initial begin
        power1 <= 1'b0;
        power2 <= 1'b0;
        #200;
        power1 <= 1'b1;
        #200;
        power2 <= 1'b1;
    end

    wire flash_csb;
    wire flash_clk;
    wire flash_io0;
    wire flash_io1;

    wire VDD3V3 = power1;
    wire VDD1V8 = power2;
    wire VSS    = 1'b0;

    caravel uut (
        .vddio    (VDD3V3),
        .vddio_2  (VDD3V3),
        .vssio    (VSS),
        .vssio_2  (VSS),
        .vdda     (VDD3V3),
        .vssa     (VSS),
        .vccd     (VDD1V8),
        .vssd     (VSS),
        .vdda1    (VDD3V3),
        .vdda1_2  (VDD3V3),
        .vdda2    (VDD3V3),
        .vssa1    (VSS),
        .vssa1_2  (VSS),
        .vssa2    (VSS),
        .vccd1    (VDD1V8),
        .vccd2    (VDD1V8),
        .vssd1    (VSS),
        .vssd2    (VSS),
        .clock    (clock),
        .gpio     (gpio),
        .mprj_io  (mprj_io),
        .flash_csb(flash_csb),
        .flash_clk(flash_clk),
        .flash_io0(flash_io0),
        .flash_io1(flash_io1),
        .resetb   (RSTB)
    );

    spiflash #(
        .FILENAME("svm_wb_test.hex")
    ) spiflash (
        .csb(flash_csb),
        .clk(flash_clk),
        .io0(flash_io0),
        .io1(flash_io1),
        .io2(),
        .io3()
    );

endmodule
`default_nettype wire
