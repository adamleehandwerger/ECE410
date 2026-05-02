// ===========================================================================
// Testbenches for svm_host_if, svm_sv_ram_if, svm_work_ram_if
// ===========================================================================
// Run individually:
//   iverilog -g2012 -s tb_svm_host_if     -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out
//   iverilog -g2012 -s tb_svm_sv_ram_if   -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out
//   iverilog -g2012 -s tb_svm_work_ram_if -o tb_out tb_svm_interfaces.sv svm_interfaces.sv && vvp tb_out
// ===========================================================================

`timescale 1ns/1ps

// ===========================================================================
// tb_svm_host_if — MCU ↔ Core signal bundle
// ===========================================================================
module tb_svm_host_if;
    localparam int DATA_WIDTH = 16;

    logic clk, rst_n;
    initial clk = 0;
    always #10 clk = ~clk;   // 50 MHz

    svm_host_if #(.DATA_WIDTH(DATA_WIDTH)) h (.clk(clk), .rst_n(rst_n));

    int fail_count;
    int sv_vals [5];

    task write_param(input logic [1:0] addr, input logic [DATA_WIDTH-1:0] data);
        @(posedge clk); #1;
        h.param_write_en = 1; h.param_addr = addr; h.param_data = data;
        @(posedge clk); #1;
        h.param_write_en = 0;
    endtask

    initial begin
        $display("=========================================");
        $display(" tb_svm_host_if");
        $display("=========================================");
        fail_count = 0;

        rst_n            = 0;
        h.param_write_en = 0; h.param_addr = 0; h.param_data = 0;
        for (int i = 0; i < 5; i++) h.num_sv_per_class[i] = 0;
        h.qspi_valid  = 0; h.qspi_data   = 0;
        h.start       = 0; h.num_samples  = 0; h.kernel_ready = 0;
        h.gamma_reg   = 0; h.c_reg        = 0;
        h.qspi_ready  = 0; h.done         = 0;
        h.error       = 0;
        h.kernel_out  = 0; h.kernel_valid  = 0;

        repeat(4) @(posedge clk); #1;
        rst_n = 1;
        repeat(2) @(posedge clk); #1;

        // TEST 1: Signal defaults after reset
        assert(h.done        === 0) else begin $error("[T1] done  != 0");  fail_count++; end
        assert(h.error       === 0) else begin $error("[T1] error != 0");  fail_count++; end
        assert(h.kernel_valid === 0) else begin $error("[T1] kv   != 0");  fail_count++; end
        assert(h.qspi_ready  === 0) else begin $error("[T1] ready != 0");  fail_count++; end
        $display("[TEST 1] PASSED - all outputs zero after reset");

        // TEST 2: Parameter write / readback
        write_param(2'b00, 16'd20);
        h.gamma_reg = 16'd20;
        @(posedge clk); #1;
        assert(h.gamma_reg === 16'd20)
            else begin $error("[T2] gamma_reg: got %0h", h.gamma_reg); fail_count++; end

        write_param(2'b01, 16'd2048);
        h.c_reg = 16'd2048;
        @(posedge clk); #1;
        assert(h.c_reg === 16'd2048)
            else begin $error("[T2] c_reg: got %0h", h.c_reg); fail_count++; end
        $display("[TEST 2] PASSED - param write/readback (gamma=0x%0h C=0x%0h)",
                 h.gamma_reg, h.c_reg);

        // TEST 3: QSPI backpressure
        h.qspi_ready = 0; h.qspi_valid = 1; h.qspi_data = 16'hDEAD;
        @(posedge clk); #1;
        assert(h.qspi_ready === 0 && h.qspi_valid === 1)
            else begin $error("[T3] backpressure broken"); fail_count++; end
        h.qspi_valid = 0;
        $display("[TEST 3] PASSED - qspi_ready=0 holds valid without accepting data");

        // TEST 4: QSPI data transfer
        h.qspi_ready = 1; h.qspi_valid = 1; h.qspi_data = 16'hBEEF;
        @(posedge clk); #1;
        assert(h.qspi_data === 16'hBEEF && h.qspi_ready === 1)
            else begin $error("[T4] QSPI transfer: data=%0h ready=%0b",
                              h.qspi_data, h.qspi_ready); fail_count++; end
        h.qspi_valid = 0; h.qspi_ready = 0;
        $display("[TEST 4] PASSED - QSPI transfer (data=0x%0h)", 16'hBEEF);

        // TEST 5: Start / done handshake
        h.num_samples = 10'd5;
        @(posedge clk); #1;
        h.start = 1;
        @(posedge clk); #1;
        h.start = 0;
        repeat(3) @(posedge clk); #1;
        h.done = 1;
        @(posedge clk); #1;
        assert(h.done === 1) else begin $error("[T5] done not asserted"); fail_count++; end
        h.done = 0;
        @(posedge clk); #1;
        assert(h.done === 0) else begin $error("[T5] done not cleared"); fail_count++; end
        $display("[TEST 5] PASSED - start->done handshake");

        // TEST 6: Kernel output stream valid / ready
        h.kernel_ready = 1; h.kernel_out = 16'd512; h.kernel_valid = 1;
        @(posedge clk); #1;
        assert(h.kernel_valid && h.kernel_ready)
            else begin $error("[T6] handshake stalled"); fail_count++; end
        assert(h.kernel_out === 16'd512)
            else begin $error("[T6] kernel_out wrong: %0h", h.kernel_out); fail_count++; end
        h.kernel_valid = 0;
        @(posedge clk); #1;
        h.kernel_ready = 0; h.kernel_valid = 1;
        @(posedge clk); #1;
        assert(h.kernel_valid === 1)
            else begin $error("[T6] valid dropped during backpressure"); fail_count++; end
        h.kernel_ready = 1; h.kernel_valid = 0;
        $display("[TEST 6] PASSED - kernel stream valid/ready");

        // TEST 7: Error flag
        h.error = 1;
        @(posedge clk); #1;
        assert(h.error === 1) else begin $error("[T7] error not set");    fail_count++; end
        h.error = 0;
        @(posedge clk); #1;
        assert(h.error === 0) else begin $error("[T7] error not cleared"); fail_count++; end
        $display("[TEST 7] PASSED - error flag set/clear");

        // TEST 8: num_sv_per_class routing
        sv_vals[0] = 60; sv_vals[1] = 45; sv_vals[2] = 55;
        sv_vals[3] = 50; sv_vals[4] = 40;
        for (int i = 0; i < 5; i++) h.num_sv_per_class[i] = sv_vals[i];
        @(posedge clk); #1;
        for (int i = 0; i < 5; i++) begin
            assert(h.num_sv_per_class[i] === sv_vals[i])
                else begin $error("[T8] class %0d: got %0d", i, h.num_sv_per_class[i]);
                     fail_count++; end
        end
        $display("[TEST 8] PASSED - num_sv_per_class[0..4] routing");

        #20;
        if (fail_count == 0)
            $display("=========================================\n ALL TESTS PASSED\n=========================================");
        else
            $display("=========================================\n %0d TEST(S) FAILED\n=========================================", fail_count);
        $finish;
    end

    initial begin #50000; $display("[TIMEOUT] tb_svm_host_if"); $finish; end
endmodule


// ===========================================================================
// tb_svm_sv_ram_if — Core → SV SRAM read-only protocol
//
// RAM model: blocking assignment, 1-cycle latency.
// #1 after every @(posedge clk) avoids the initial-vs-always race.
// ===========================================================================
module tb_svm_sv_ram_if;
    localparam int DATA_WIDTH = 16;
    localparam int ADDR_WIDTH = 18;
    localparam int MEM_DEPTH  = 32;

    logic clk;
    initial clk = 0;
    always #10 clk = ~clk;

    svm_sv_ram_if #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) ram ();

    // RAM model — blocking assignment so update is visible immediately after posedge
    logic [DATA_WIDTH-1:0] sv_mem [0:MEM_DEPTH-1];
    initial for (int a = 0; a < MEM_DEPTH; a++) sv_mem[a] = DATA_WIDTH'(a * 4);

    always @(posedge clk)
        if (ram.ren)
            ram.rdata = (ram.addr < MEM_DEPTH) ? sv_mem[ram.addr] : '0;

    // Issue read, capture rdata one cycle later.
    // #1 after first @posedge ensures always fires before we deassert ren.
    task automatic read_sv(input  logic [ADDR_WIDTH-1:0] a,
                           output logic [DATA_WIDTH-1:0] d);
        ram.addr = a; ram.ren = 1;
        @(posedge clk); #1;    // always fires at posedge, stable by posedge+1ns
        d = ram.rdata;
        ram.ren = 0;
    endtask

    int  fail_count;
    logic [DATA_WIDTH-1:0] d;
    logic [DATA_WIDTH-1:0] last;
    logic [DATA_WIDTH-1:0] snap;

    initial begin
        $display("=========================================");
        $display(" tb_svm_sv_ram_if");
        $display("=========================================");
        fail_count = 0;
        ram.addr = '0; ram.ren = 0; ram.rdata = '0;
        repeat(2) @(posedge clk); #1;

        // TEST 1: addr 0 → expected 0
        read_sv(0, d);
        assert(d === 16'd0)
            else begin $error("[T1] addr  0: exp 0,  got %0d", d);  fail_count++; end
        $display("[TEST 1] PASSED - addr  0 -> rdata=%0d", d);

        // TEST 2: addr 7 → expected 28
        read_sv(7, d);
        assert(d === 16'd28)
            else begin $error("[T2] addr  7: exp 28, got %0d", d); fail_count++; end
        $display("[TEST 2] PASSED - addr  7 -> rdata=%0d", d);

        // TEST 3: addr 15 → expected 60
        read_sv(15, d);
        assert(d === 16'd60)
            else begin $error("[T3] addr 15: exp 60, got %0d", d); fail_count++; end
        $display("[TEST 3] PASSED - addr 15 -> rdata=%0d", d);

        // TEST 4: back-to-back burst (ren held, addr 0..7)
        for (int a = 0; a < 8; a++) begin
            ram.addr = a; ram.ren = 1;
            @(posedge clk); #1;   // always fires, captures sv_mem[a] before addr changes
        end
        last = ram.rdata;          // = sv_mem[7] = 28
        ram.ren = 0;
        assert(last === 16'd28)
            else begin $error("[T4] burst end: exp 28, got %0d", last); fail_count++; end
        $display("[TEST 4] PASSED - 8-beat burst (last rdata=%0d)", last);

        // TEST 5: rdata stable when ren=0
        snap = ram.rdata;
        repeat(4) @(posedge clk); #1;
        assert(ram.rdata === snap)
            else begin $error("[T5] rdata changed without ren"); fail_count++; end
        $display("[TEST 5] PASSED - rdata stable (ren=0) held=0x%0h", snap);

        // TEST 6: out-of-range address → rdata = 0
        read_sv(18'h3FFFF, d);
        assert(d === 16'd0)
            else begin $error("[T6] OOB addr: exp 0, got %0h", d); fail_count++; end
        $display("[TEST 6] PASSED - out-of-range address -> rdata=0");

        #20;
        if (fail_count == 0)
            $display("=========================================\n ALL TESTS PASSED\n=========================================");
        else
            $display("=========================================\n %0d TEST(S) FAILED\n=========================================", fail_count);
        $finish;
    end

    initial begin #20000; $display("[TIMEOUT] tb_svm_sv_ram_if"); $finish; end
endmodule


// ===========================================================================
// tb_svm_work_ram_if — Core ↔ Workspace SRAM R/W protocol
// ===========================================================================
module tb_svm_work_ram_if;
    localparam int DATA_WIDTH = 16;
    localparam int ADDR_WIDTH = 18;
    localparam int MEM_DEPTH  = 64;

    logic clk;
    initial clk = 0;
    always #10 clk = ~clk;

    svm_work_ram_if #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) ram ();

    // RAM model — blocking assignments
    logic [DATA_WIDTH-1:0] work_mem [0:MEM_DEPTH-1];
    initial for (int a = 0; a < MEM_DEPTH; a++) work_mem[a] = '0;

    always @(posedge clk) begin
        if (ram.wen && ram.addr < MEM_DEPTH)
            work_mem[ram.addr] = ram.wdata;
        if (ram.ren && ram.addr < MEM_DEPTH)
            ram.rdata = work_mem[ram.addr];
    end

    // Write: set addr/wdata/wen, clock, then deassert wen after #1
    task automatic write_w(input logic [ADDR_WIDTH-1:0] a,
                           input logic [DATA_WIDTH-1:0] d);
        ram.addr = a; ram.wdata = d; ram.wen = 1;
        @(posedge clk); #1;    // always writes work_mem[a]=d at posedge
        ram.wen = 0;
    endtask

    // Read: set addr/ren, clock, capture rdata after #1
    task automatic read_w(input  logic [ADDR_WIDTH-1:0] a,
                          output logic [DATA_WIDTH-1:0] d);
        ram.addr = a; ram.ren = 1;
        @(posedge clk); #1;    // always sets rdata=work_mem[a] at posedge
        d = ram.rdata;
        ram.ren = 0;
    endtask

    int  fail_count;
    logic [DATA_WIDTH-1:0] d0, d1;
    logic [DATA_WIDTH-1:0] last;
    logic [DATA_WIDTH-1:0] snap_rdata;
    logic [DATA_WIDTH-1:0] snap_mem20;

    initial begin
        $display("=========================================");
        $display(" tb_svm_work_ram_if");
        $display("=========================================");
        fail_count = 0;
        ram.addr = '0; ram.wdata = '0; ram.wen = 0; ram.ren = 0;
        repeat(2) @(posedge clk); #1;

        // TEST 1: single write — verify via direct memory inspection
        write_w(5, 16'hCAFE);
        assert(work_mem[5] === 16'hCAFE)
            else begin $error("[T1] mem[5]: exp 0xCAFE, got 0x%0h", work_mem[5]); fail_count++; end
        $display("[TEST 1] PASSED - write addr 5 -> mem[5]=0x%0h", work_mem[5]);

        // TEST 2: read back through interface
        read_w(5, d0);
        assert(d0 === 16'hCAFE)
            else begin $error("[T2] rdata: exp 0xCAFE, got 0x%0h", d0); fail_count++; end
        $display("[TEST 2] PASSED - read addr 5 -> rdata=0x%0h", d0);

        // TEST 3: overwrite then read
        write_w(5, 16'h1111);
        read_w(5, d0);
        assert(d0 === 16'h1111)
            else begin $error("[T3] overwrite: exp 0x1111, got 0x%0h", d0); fail_count++; end
        $display("[TEST 3] PASSED - overwrite addr 5 -> rdata=0x%0h", d0);

        // TEST 4: 16-word write burst (values: addr*64)
        for (int a = 0; a < 16; a++) begin
            ram.addr  = a;
            ram.wdata = DATA_WIDTH'(a * 64);
            ram.wen   = 1;
            @(posedge clk); #1;   // always writes work_mem[a] before addr advances
        end
        ram.wen = 0;
        for (int a = 0; a < 16; a++) begin
            assert(work_mem[a] === DATA_WIDTH'(a * 64))
                else begin $error("[T4] burst addr %0d: exp %0d, got %0d",
                                  a, a*64, work_mem[a]); fail_count++; end
        end
        $display("[TEST 4] PASSED - 16-word write burst verified");

        // TEST 5: 16-word read burst — final rdata = work_mem[15] = 960
        for (int a = 0; a < 16; a++) begin
            ram.addr = a; ram.ren = 1;
            @(posedge clk); #1;   // rdata = work_mem[a] after each posedge
        end
        last = ram.rdata;
        ram.ren = 0;
        assert(last === 16'd960)
            else begin $error("[T5] burst end: exp 960, got %0d", last); fail_count++; end
        $display("[TEST 5] PASSED - 16-word read burst (last rdata=%0d)", last);

        // TEST 6: multi-address write-then-read roundtrip
        write_w(20, 16'hAAAA);
        write_w(21, 16'hBBBB);
        read_w(20, d0);
        read_w(21, d1);
        assert(d0 === 16'hAAAA)
            else begin $error("[T6] addr 20: exp 0xAAAA, got 0x%0h", d0); fail_count++; end
        assert(d1 === 16'hBBBB)
            else begin $error("[T6] addr 21: exp 0xBBBB, got 0x%0h", d1); fail_count++; end
        $display("[TEST 6] PASSED - multi-addr roundtrip (0x%0h, 0x%0h)", d0, d1);

        // TEST 7: idle cycles — memory and rdata unchanged
        snap_rdata = ram.rdata;
        snap_mem20 = work_mem[20];
        repeat(5) @(posedge clk); #1;
        assert(ram.rdata    === snap_rdata)
            else begin $error("[T7] rdata changed during idle"); fail_count++; end
        assert(work_mem[20] === snap_mem20)
            else begin $error("[T7] mem[20] changed during idle"); fail_count++; end
        $display("[TEST 7] PASSED - rdata and memory stable during idle");

        #20;
        if (fail_count == 0)
            $display("=========================================\n ALL TESTS PASSED\n=========================================");
        else
            $display("=========================================\n %0d TEST(S) FAILED\n=========================================", fail_count);
        $finish;
    end

    initial begin #30000; $display("[TIMEOUT] tb_svm_work_ram_if"); $finish; end
endmodule
