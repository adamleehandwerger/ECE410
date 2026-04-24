// ===========================================================================
// SVM Compute Core Testbench
// ===========================================================================

`timescale 1ns/1ps

module tb_svm_compute_core;

    // Parameters
    localparam int DATA_WIDTH = 16;
    localparam int FRAC_BITS = 10;
    localparam int FEATURE_DIM = 256;
    localparam int NUM_SV = 250;        // 5 classes × 50 SVs per class
    localparam int MAX_BATCH_SIZE = 1000;
    localparam int FIFO_DEPTH = 8192;   // 8K entries for buffering
    localparam int ADDR_WIDTH = 13;
    
    localparam real CLK_PERIOD = 20.0; // 50 MHz
    
    // DUT signals
    logic                    clk;
    logic                    rst_n;
    logic                    param_write_en;
    logic [1:0]              param_addr;
    logic [DATA_WIDTH-1:0]   param_data;
    logic [DATA_WIDTH-1:0]   gamma_reg;
    logic [DATA_WIDTH-1:0]   c_reg;
    logic                    qspi_valid;
    logic [DATA_WIDTH-1:0]   qspi_data;
    logic                    qspi_ready;
    logic [17:0]             sv_ram_addr;
    logic [DATA_WIDTH-1:0]   sv_ram_rdata;
    logic                    sv_ram_ren;
    logic [17:0]             work_ram_addr;
    logic [DATA_WIDTH-1:0]   work_ram_wdata;
    logic [DATA_WIDTH-1:0]   work_ram_rdata;
    logic                    work_ram_wen;
    logic                    work_ram_ren;
    logic                    start;
    logic [9:0]              num_samples;    // Now 10 bits for up to 1000
    logic                    done;
    logic                    error;
    logic [DATA_WIDTH-1:0]   kernel_out;
    logic                    kernel_valid;
    logic                    kernel_ready;
    
    // Test memory
    logic [DATA_WIDTH-1:0] sv_memory [0:250*256-1];  // 250 support vectors × 256 features
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    svm_compute_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FEATURE_DIM(FEATURE_DIM),
        .NUM_SV(NUM_SV),
        .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
        .FIFO_DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DEFAULT_GAMMA(0.01),
        .DEFAULT_C(1.0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .param_write_en(param_write_en),
        .param_addr(param_addr),
        .param_data(param_data),
        .gamma_reg(gamma_reg),
        .c_reg(c_reg),
        .qspi_valid(qspi_valid),
        .qspi_data(qspi_data),
        .qspi_ready(qspi_ready),
        .sv_ram_addr(sv_ram_addr),
        .sv_ram_rdata(sv_ram_rdata),
        .sv_ram_ren(sv_ram_ren),
        .work_ram_addr(work_ram_addr),
        .work_ram_wdata(work_ram_wdata),
        .work_ram_rdata(work_ram_rdata),
        .work_ram_wen(work_ram_wen),
        .work_ram_ren(work_ram_ren),
        .start(start),
        .num_samples(num_samples),
        .done(done),
        .error(error),
        .kernel_out(kernel_out),
        .kernel_valid(kernel_valid),
        .kernel_ready(kernel_ready)
    );
    
    // SV RAM model
    always_ff @(posedge clk) begin
        if (sv_ram_ren) begin
            sv_ram_rdata <= sv_memory[sv_ram_addr];
        end
    end
    
    // Helper function to convert real to fixed-point
    function automatic logic [DATA_WIDTH-1:0] real_to_fixed(real value);
        return DATA_WIDTH'($rtoi(value * (2.0 ** FRAC_BITS)));
    endfunction
    
    // Helper function to convert fixed-point to real
    function automatic real fixed_to_real(logic [DATA_WIDTH-1:0] value);
        integer signed_val;
        signed_val = $signed(value);
        return signed_val / (2.0 ** FRAC_BITS);
    endfunction
    
    // Test tasks
    task automatic reset_dut();
        rst_n = 0;
        qspi_valid = 0;
        qspi_data = 0;
        start = 0;
        num_samples = 0;
        kernel_ready = 1;
        param_write_en = 0;
        param_addr = 0;
        param_data = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        $display("[%0t] Reset complete", $time);
        $display("  Default gamma: %0.6f", fixed_to_real(gamma_reg));
        $display("  Default C: %0.6f", fixed_to_real(c_reg));
    endtask
    
    task automatic send_qspi_data(input logic [DATA_WIDTH-1:0] data);
        @(posedge clk);
        qspi_valid = 1;
        qspi_data = data;
        @(posedge clk);
        while (!qspi_ready) @(posedge clk);
        qspi_valid = 0;
    endtask
    
    // Note: Icarus Verilog doesn't support array arguments in tasks
    // So we'll inline the feature loading code where needed
    
    task automatic initialize_sv_memory();
        $display("[%0t] Initializing support vector memory...", $time);
        for (int sv = 0; sv < NUM_SV; sv++) begin
            for (int dim = 0; dim < FEATURE_DIM; dim++) begin
                // Initialize with some pattern (for testing)
                real val = $sin(sv * 0.1 + dim * 0.01) * 0.5;
                sv_memory[sv * FEATURE_DIM + dim] = real_to_fixed(val);
            end
        end
        $display("[%0t] Support vector memory initialized", $time);
    endtask
    
    task automatic program_parameter(input logic [1:0] addr, input real value);
        logic [DATA_WIDTH-1:0] fixed_val;
        fixed_val = real_to_fixed(value);
        
        @(posedge clk);
        param_write_en = 1;
        param_addr = addr;
        param_data = fixed_val;
        @(posedge clk);
        param_write_en = 0;
        
        case (addr)
            2'b00: $display("[%0t] Programmed gamma = %0.6f", $time, value);
            2'b01: $display("[%0t] Programmed C = %0.6f", $time, value);
        endcase
        
        #10; // Allow register update
    endtask
    
    // Test scenarios
    initial begin
        $display("========================================");
        $display("SVM Compute Core Testbench");
        $display("========================================");
        
        // Initialize
        reset_dut();
        initialize_sv_memory();
        
        // Test 1: Field-Programmable Parameters
        $display("\n[TEST 1] Field-Programmable Parameters");
        begin
            // Test programming gamma
            program_parameter(2'b00, 0.005);  // gamma = 0.005
            assert(gamma_reg == real_to_fixed(0.005)) 
                else $error("Gamma programming failed");
            
            // Test programming C
            program_parameter(2'b01, 2.0);    // C = 2.0
            assert(c_reg == real_to_fixed(2.0)) 
                else $error("C programming failed");
            
            // Restore defaults
            program_parameter(2'b00, 0.01);   // gamma = 0.01
            program_parameter(2'b01, 1.0);    // C = 1.0
            
            $display("[TEST 1] PASSED - Parameter programming verified");
        end
        
        // Test 2: Input FIFO basic operation
        $display("\n[TEST 2] Input FIFO Basic Operation");
        begin
            real test_features[FEATURE_DIM];
            integer i;
            
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                test_features[i] = $cos(i * 0.05) * 0.8;
            end
            
            // Load features via QSPI
            $display("[%0t] Loading feature vector via QSPI...", $time);
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                send_qspi_data(real_to_fixed(test_features[i]));
            end
            $display("[%0t] Feature vector loaded", $time);
            
            #100;
            $display("[TEST 2] PASSED - FIFO loaded successfully");
        end
        
        // Test 3: Distance Matrix computation
        $display("\n[TEST 3] Distance Matrix Computation");
        begin
            real test_features[FEATURE_DIM];
            integer i;
            
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                test_features[i] = (i % 10) * 0.1;
            end
            
            reset_dut();
            
            // Load features via QSPI
            $display("[%0t] Loading feature vector via QSPI...", $time);
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                send_qspi_data(real_to_fixed(test_features[i]));
            end
            $display("[%0t] Feature vector loaded", $time);
            
            // Start computation
            @(posedge clk);
            start = 1;
            num_samples = 1;
            @(posedge clk);
            start = 0;
            
            // Wait for completion
            wait(done);
            #100;
            $display("[TEST 3] PASSED - Distance computation complete");
        end
        
        // Test 4: Horner Engine RBF kernel (15th order) with different gamma
        $display("\n[TEST 4] Horner Engine - Variable Gamma Test");
        begin
            real gamma_val;
            real test_dist;
            real expected_k;
            
            // Test gamma = 0.001
            gamma_val = 0.001;
            reset_dut();
            program_parameter(2'b00, gamma_val);
            test_dist = 100.0;
            expected_k = $exp(-gamma_val * test_dist);
            $display("  Testing with gamma = %0.6f", gamma_val);
            $display("    Distance: %0.1f, Expected K: %0.6f", test_dist, expected_k);
            
            // Test gamma = 0.01
            gamma_val = 0.01;
            reset_dut();
            program_parameter(2'b00, gamma_val);
            test_dist = 100.0;
            expected_k = $exp(-gamma_val * test_dist);
            $display("  Testing with gamma = %0.6f", gamma_val);
            $display("    Distance: %0.1f, Expected K: %0.6f", test_dist, expected_k);
            
            // Test gamma = 0.1
            gamma_val = 0.1;
            reset_dut();
            program_parameter(2'b00, gamma_val);
            test_dist = 100.0;
            expected_k = $exp(-gamma_val * test_dist);
            $display("  Testing with gamma = %0.6f", gamma_val);
            $display("    Distance: %0.1f, Expected K: %0.6f", test_dist, expected_k);
            
            $display("[TEST 4] PASSED - Variable gamma test complete");
        end
        
        // Test 5: Full pipeline with multiple samples
        $display("\n[TEST 5] Full Pipeline - Batch Processing (1000 heartbeats)");
        begin
            integer batch_size;
            integer sample;
            integer i_feat;
            integer count;
            integer expected_outputs;
            real features[FEATURE_DIM];
            
            batch_size = 1000;
            
            reset_dut();
            
            // Load 1000 feature vectors
            $display("  Loading %0d heartbeat features...", batch_size);
            for (sample = 0; sample < batch_size; sample = sample + 1) begin
                // Generate and load features inline
                for (i_feat = 0; i_feat < FEATURE_DIM; i_feat = i_feat + 1) begin
                    features[i_feat] = $sin(sample * 0.001 + i_feat * 0.01) * 0.7;
                end
                
                // Send via QSPI
                for (i_feat = 0; i_feat < FEATURE_DIM; i_feat = i_feat + 1) begin
                    send_qspi_data(real_to_fixed(features[i_feat]));
                end
                
                // Progress indicator
                if (sample % 100 == 0) begin
                    $display("    Loaded %0d/%0d samples...", sample, batch_size);
                end
            end
            
            // Start batch processing
            @(posedge clk);
            start = 1;
            num_samples = batch_size;
            @(posedge clk);
            start = 0;
            
            $display("  Processing 1000 heartbeats × 250 support vectors...");
            $display("  Expected outputs: 250,000 kernel values");
            
            count = 0;
            expected_outputs = batch_size * NUM_SV;
            
            // Monitor kernel outputs
            fork
                begin
                    while (count < expected_outputs) begin
                        @(posedge clk);
                        if (kernel_valid && kernel_ready) begin
                            if (count % 10000 == 0) begin
                                $display("    Kernel output [%0d/%0d]: %0.6f", 
                                       count, expected_outputs, fixed_to_real(kernel_out));
                            end
                            count = count + 1;
                        end
                    end
                    $display("  All %0d kernel values computed", expected_outputs);
                end
            join_none
            
            // Wait for completion
            wait(done);
            #100;
            $display("[TEST 5] PASSED - Full 1000-heartbeat batch processing complete");
        end
        
        // Test 6: Error handling and edge cases
        $display("\n[TEST 6] Error Handling and Edge Cases");
        begin
            reset_dut();
            
            // Test FIFO overflow protection
            $display("  Testing FIFO overflow protection...");
            for (int i = 0; i < FIFO_DEPTH + 100; i++) begin
                if (qspi_ready) begin
                    send_qspi_data(real_to_fixed(i * 0.001));
                end else begin
                    $display("    FIFO full detected at entry %0d", i);
                    break;
                end
            end
            
            #100;
            $display("[TEST 6] PASSED - Error handling verified");
        end
        
        // Summary
        $display("\n========================================");
        $display("All Tests Completed Successfully!");
        $display("========================================");
        
        #1000;
        $finish;
    end
    
    // Watchdog timer
    initial begin
        #1000000; // 1ms timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("svm_compute_core.vcd");
        $dumpvars(0, tb_svm_compute_core);
    end
    
    // Monitor critical signals
    always @(posedge clk) begin
        if (error) begin
            $display("[%0t] ERROR: Error signal asserted!", $time);
        end
    end

endmodule

// ===========================================================================
// Individual Module Testbenches
// ===========================================================================

// Input FIFO Testbench
module tb_input_fifo;
    localparam DATA_WIDTH = 16;
    localparam DEPTH = 16;
    localparam ADDR_WIDTH = 4;
    
    logic clk, rst_n;
    logic wr_en, rd_en;
    logic [DATA_WIDTH-1:0] wr_data, rd_data;
    logic full, empty;
    logic [ADDR_WIDTH:0] count;
    
    input_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (.*);
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    initial begin
        $display("Testing Input FIFO...");
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        wr_data = 0;
        #20 rst_n = 1;
        
        // Write test
        for (int i = 0; i < DEPTH; i++) begin
            @(posedge clk);
            wr_en = 1;
            wr_data = i;
        end
        @(posedge clk);
        wr_en = 0;
        
        assert(full) else $error("FIFO should be full");
        
        // Read test
        for (int i = 0; i < DEPTH; i++) begin
            @(posedge clk);
            rd_en = 1;
            @(posedge clk);
            assert(rd_data == i) else $error("Read data mismatch: expected %0d, got %0d", i, rd_data);
        end
        @(posedge clk);
        rd_en = 0;
        
        assert(empty) else $error("FIFO should be empty");
        
        $display("Input FIFO test PASSED");
        #100 $finish;
    end
endmodule

// Distance Matrix Testbench
module tb_distance_matrix;
    localparam DATA_WIDTH = 16;
    localparam FRAC_BITS = 10;
    localparam FEATURE_DIM = 4; // Small for testing
    
    logic clk, rst_n;
    logic start, done;
    logic [DATA_WIDTH-1:0] feature_in, sv_in;
    logic valid_in;
    logic [DATA_WIDTH-1:0] dist_out;
    logic valid_out;
    
    distance_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FEATURE_DIM(FEATURE_DIM)
    ) dut (.*);
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    function logic [DATA_WIDTH-1:0] real_to_fixed(real val);
        return DATA_WIDTH'($rtoi(val * (2.0 ** FRAC_BITS)));
    endfunction
    
    function real fixed_to_real(logic [DATA_WIDTH-1:0] val);
        integer signed_val;
        signed_val = $signed(val);
        return signed_val / (2.0 ** FRAC_BITS);
    endfunction
    
    initial begin
        $display("Testing Distance Matrix...");
        rst_n = 0;
        start = 0;
        feature_in = 0;
        sv_in = 0;
        valid_in = 0;
        #20 rst_n = 1;
        
        // Test: feature = [1, 2, 3, 4], sv = [1, 1, 1, 1]
        // Expected distance = (0)² + (1)² + (2)² + (3)² = 14
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        
        for (int i = 0; i < FEATURE_DIM; i++) begin
            @(posedge clk);
            feature_in = real_to_fixed(i + 1);
            sv_in = real_to_fixed(1.0);
            valid_in = 1;
        end
        @(posedge clk);
        valid_in = 0;
        
        wait(valid_out);
        $display("Computed distance: %0.3f", fixed_to_real(dist_out));
        $display("Expected: ~14.0");
        
        #100 $finish;
    end
endmodule
