// ===========================================================================
// Multi-Class Cardiac Arrhythmia Detection - SVM Compute Core
// ===========================================================================
// Top-level module integrating:
//   - Input FIFO (16 KB SRAM)
//   - Distance Matrix Engine (D = ||X[i] - SV[j]||²)
//   - Horner Engine (RBF kernel approximation)
// ===========================================================================

module svm_compute_core #(
    parameter int DATA_WIDTH = 16,          // Fixed-point data width
    parameter int FRAC_BITS = 10,           // Fractional bits for Q format
    parameter int FEATURE_DIM = 256,        // Features per heartbeat
    parameter int NUM_SV = 250,             // Number of support vectors (5 classes × 50 SVs/class)
    parameter int MAX_BATCH_SIZE = 1000,    // Maximum heartbeats per batch
    parameter int FIFO_DEPTH = 8192,        // Input FIFO depth (1000 beats × 256 features = 256K entries, use 8K for buffering)
    parameter int ADDR_WIDTH = 13,          // Address width for FIFO (2^13 = 8192)
    // Default SVM hyperparameters (from training)
    parameter real DEFAULT_GAMMA = 0.01,    // RBF kernel gamma (default from MIT-BIH training)
    parameter real DEFAULT_C = 1.0          // SVM regularization parameter C
) (
    // Clock and Reset
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Field-Programmable Parameters (can be updated via QSPI)
    input  logic                    param_write_en,
    input  logic [1:0]              param_addr,     // 0=gamma, 1=C, 2-3=reserved
    input  logic [DATA_WIDTH-1:0]   param_data,
    output logic [DATA_WIDTH-1:0]   gamma_reg,      // Current gamma value
    output logic [DATA_WIDTH-1:0]   c_reg,          // Current C value
    
    // QSPI Interface (simplified - from host MCU)
    input  logic                    qspi_valid,
    input  logic [DATA_WIDTH-1:0]   qspi_data,
    output logic                    qspi_ready,
    
    // Memory Interface - Support Vector RAM
    output logic [17:0]             sv_ram_addr,    // 250 SVs × 256 features × 2 bytes = 128 KB
    input  logic [DATA_WIDTH-1:0]   sv_ram_rdata,
    output logic                    sv_ram_ren,
    
    // Memory Interface - Distance/Kernel Workspace
    output logic [17:0]             work_ram_addr,  // 1000 × 250 kernel matrix = 500 KB workspace
    output logic [DATA_WIDTH-1:0]   work_ram_wdata,
    input  logic [DATA_WIDTH-1:0]   work_ram_rdata,
    output logic                    work_ram_wen,
    output logic                    work_ram_ren,
    
    // Control and Status
    input  logic                    start,
    input  logic [9:0]              num_samples,    // Batch size (up to 1000 heartbeats)
    output logic                    done,
    output logic                    error,
    
    // Kernel Output (to software processing)
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    kernel_valid,
    input  logic                    kernel_ready
);

    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Parameter registers (field-programmable)
    logic [DATA_WIDTH-1:0] gamma_int;       // Internal gamma register
    logic [DATA_WIDTH-1:0] c_int;           // Internal C register
    
    // Convert default parameters to fixed-point at compile time
    localparam logic [DATA_WIDTH-1:0] GAMMA_DEFAULT = 
        DATA_WIDTH'($rtoi(DEFAULT_GAMMA * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] C_DEFAULT = 
        DATA_WIDTH'($rtoi(DEFAULT_C * (2.0 ** FRAC_BITS)));
    
    // FIFO signals
    logic                    fifo_wr_en;
    logic                    fifo_rd_en;
    logic [DATA_WIDTH-1:0]   fifo_wr_data;
    logic [DATA_WIDTH-1:0]   fifo_rd_data;
    logic                    fifo_full;
    logic                    fifo_empty;
    logic [ADDR_WIDTH:0]     fifo_count;
    
    // Distance Matrix signals
    logic                    dist_start;
    logic                    dist_done;
    logic [DATA_WIDTH-1:0]   dist_feature_in;
    logic [DATA_WIDTH-1:0]   dist_sv_in;
    logic                    dist_valid_in;
    logic [DATA_WIDTH-1:0]   dist_out;
    logic                    dist_valid_out;
    
    // Horner Engine signals
    logic                    horner_start;
    logic                    horner_done;
    logic [DATA_WIDTH-1:0]   horner_dist_in;
    logic                    horner_valid_in;
    logic [DATA_WIDTH-1:0]   horner_kernel_out;
    logic                    horner_valid_out;
    
    // State machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_FIFO,
        COMPUTE_DIST,
        COMPUTE_KERNEL,
        OUTPUT_RESULT,
        ERROR_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Counters
    logic [9:0]  sample_counter;    // Up to 1000 samples
    logic [8:0]  feature_counter;   // Up to 256 features
    logic [7:0]  sv_counter;        // Up to 250 support vectors
    
    // =========================================================================
    // Input FIFO Instance
    // =========================================================================
    
    input_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_input_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(fifo_wr_en),
        .wr_data(fifo_wr_data),
        .rd_en(fifo_rd_en),
        .rd_data(fifo_rd_data),
        .full(fifo_full),
        .empty(fifo_empty),
        .count(fifo_count)
    );
    
    // =========================================================================
    // Distance Matrix Engine Instance
    // =========================================================================
    
    distance_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FEATURE_DIM(FEATURE_DIM)
    ) u_distance_matrix (
        .clk(clk),
        .rst_n(rst_n),
        .start(dist_start),
        .feature_in(dist_feature_in),
        .sv_in(dist_sv_in),
        .valid_in(dist_valid_in),
        .dist_out(dist_out),
        .valid_out(dist_valid_out),
        .done(dist_done)
    );
    
    // =========================================================================
    // Horner Engine Instance (RBF Kernel Approximation)
    // =========================================================================
    
    horner_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) u_horner_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(horner_start),
        .dist_in(horner_dist_in),
        .valid_in(horner_valid_in),
        .gamma(gamma_int),              // Use field-programmable gamma
        .kernel_out(horner_kernel_out),
        .valid_out(horner_valid_out),
        .done(horner_done)
    );
    
    // =========================================================================
    // Field-Programmable Parameter Registers
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Load defaults on reset
            gamma_int <= GAMMA_DEFAULT;
            c_int <= C_DEFAULT;
        end else if (param_write_en) begin
            case (param_addr)
                2'b00: gamma_int <= param_data;  // Update gamma
                2'b01: c_int <= param_data;      // Update C
                default: begin
                    // Reserved addresses - no action
                end
            endcase
        end
    end
    
    // Output current parameter values
    assign gamma_reg = gamma_int;
    assign c_reg = c_int;
    
    // =========================================================================
    // Control Logic - State Machine
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = LOAD_FIFO;
                end
            end
            
            LOAD_FIFO: begin
                if (fifo_count >= (num_samples * FEATURE_DIM)) begin
                    next_state = COMPUTE_DIST;
                end
            end
            
            COMPUTE_DIST: begin
                if (dist_done) begin
                    next_state = COMPUTE_KERNEL;
                end
            end
            
            COMPUTE_KERNEL: begin
                if (horner_done) begin
                    next_state = OUTPUT_RESULT;
                end
            end
            
            OUTPUT_RESULT: begin
                if (kernel_ready && kernel_valid) begin
                    if (sample_counter >= num_samples - 1) begin
                        next_state = IDLE;
                    end else begin
                        next_state = COMPUTE_DIST;
                    end
                end
            end
            
            ERROR_STATE: begin
                next_state = IDLE;
            end
            
            default: begin
                next_state = ERROR_STATE;
            end
        endcase
    end
    
    // =========================================================================
    // FIFO Control
    // =========================================================================
    
    always_comb begin
        fifo_wr_en = qspi_valid && !fifo_full && (state == LOAD_FIFO);
        fifo_wr_data = qspi_data;
        qspi_ready = !fifo_full && (state == LOAD_FIFO);
    end
    
    // =========================================================================
    // Counter Management
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter <= '0;
            feature_counter <= '0;
            sv_counter <= '0;
        end else begin
            case (state)
                IDLE: begin
                    sample_counter <= '0;
                    feature_counter <= '0;
                    sv_counter <= '0;
                end
                
                COMPUTE_DIST: begin
                    if (dist_valid_out) begin
                        if (sv_counter >= NUM_SV - 1) begin
                            sv_counter <= '0;
                            if (feature_counter >= FEATURE_DIM - 1) begin
                                feature_counter <= '0;
                            end else begin
                                feature_counter <= feature_counter + 1;
                            end
                        end else begin
                            sv_counter <= sv_counter + 1;
                        end
                    end
                end
                
                OUTPUT_RESULT: begin
                    if (kernel_ready && kernel_valid) begin
                        sample_counter <= sample_counter + 1;
                    end
                end
            endcase
        end
    end
    
    // =========================================================================
    // Output Assignments
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            error <= 1'b0;
            kernel_out <= '0;
            kernel_valid <= 1'b0;
        end else begin
            done <= (state == OUTPUT_RESULT) && (sample_counter >= num_samples - 1);
            error <= (state == ERROR_STATE);
            kernel_out <= horner_kernel_out;
            kernel_valid <= horner_valid_out;
        end
    end
    
    // Connect internal engines
    assign dist_start = (state == COMPUTE_DIST);
    assign horner_start = (state == COMPUTE_KERNEL);
    assign fifo_rd_en = (state == COMPUTE_DIST) && !fifo_empty;
    assign dist_feature_in = fifo_rd_data;
    assign dist_valid_in = fifo_rd_en;
    assign horner_dist_in = dist_out;
    assign horner_valid_in = dist_valid_out;

endmodule

// ===========================================================================
// Input FIFO Module (16 KB SRAM)
// ===========================================================================

module input_fifo #(
    parameter int DATA_WIDTH = 16,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = 10
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Write interface
    input  logic                    wr_en,
    input  logic [DATA_WIDTH-1:0]   wr_data,
    
    // Read interface
    input  logic                    rd_en,
    output logic [DATA_WIDTH-1:0]   rd_data,
    
    // Status
    output logic                    full,
    output logic                    empty,
    output logic [ADDR_WIDTH:0]     count
);

    // FIFO memory
    logic [DATA_WIDTH-1:0] mem [DEPTH];
    
    // Pointers
    logic [ADDR_WIDTH:0] wr_ptr;
    logic [ADDR_WIDTH:0] rd_ptr;
    
    // Status flags
    assign full = (count == DEPTH);
    assign empty = (count == 0);
    
    // Write pointer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (wr_en && !full) begin
            wr_ptr <= wr_ptr + 1;
        end
    end
    
    // Read pointer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
        end else if (rd_en && !empty) begin
            rd_ptr <= rd_ptr + 1;
        end
    end
    
    // Count
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end
    
    // Memory write
    always_ff @(posedge clk) begin
        if (wr_en && !full) begin
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
        end
    end
    
    // Memory read
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else if (rd_en && !empty) begin
            rd_data <= mem[rd_ptr[ADDR_WIDTH-1:0]];
        end
    end

endmodule

// ===========================================================================
// Distance Matrix Engine
// Computes D = ||X[i] - SV[j]||² (squared Euclidean distance)
// ===========================================================================

module distance_matrix #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int FEATURE_DIM = 256
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control
    input  logic                    start,
    output logic                    done,
    
    // Data inputs
    input  logic [DATA_WIDTH-1:0]   feature_in,
    input  logic [DATA_WIDTH-1:0]   sv_in,
    input  logic                    valid_in,
    
    // Distance output
    output logic [DATA_WIDTH-1:0]   dist_out,
    output logic                    valid_out
);

    // Internal state
    typedef enum logic [1:0] {
        IDLE,
        ACCUMULATE,
        OUTPUT,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // Accumulators and temporary values
    logic [2*DATA_WIDTH-1:0] diff;
    logic [2*DATA_WIDTH-1:0] diff_squared;
    logic [2*DATA_WIDTH+8-1:0] accumulator;  // Extra bits for accumulation
    logic [8:0] dim_counter;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = ACCUMULATE;
                end
            end
            
            ACCUMULATE: begin
                if (dim_counter >= FEATURE_DIM - 1 && valid_in) begin
                    next_state = OUTPUT;
                end
            end
            
            OUTPUT: begin
                next_state = DONE_STATE;
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Dimension counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dim_counter <= '0;
        end else begin
            case (state)
                IDLE: dim_counter <= '0;
                ACCUMULATE: begin
                    if (valid_in) begin
                        dim_counter <= dim_counter + 1;
                    end
                end
                default: dim_counter <= dim_counter;
            endcase
        end
    end
    
    // Compute difference and square
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff <= '0;
            diff_squared <= '0;
        end else if (valid_in && state == ACCUMULATE) begin
            // Compute (feature_in - sv_in)
            diff <= $signed(feature_in) - $signed(sv_in);
            // Square the difference
            diff_squared <= $signed(diff) * $signed(diff);
        end
    end
    
    // Accumulator
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else begin
            case (state)
                IDLE: accumulator <= '0;
                ACCUMULATE: begin
                    if (valid_in) begin
                        // Accumulate squared differences
                        // Shift right by FRAC_BITS to account for fixed-point multiplication
                        accumulator <= accumulator + (diff_squared >>> FRAC_BITS);
                    end
                end
                default: accumulator <= accumulator;
            endcase
        end
    end
    
    // Output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out <= '0;
            valid_out <= 1'b0;
            done <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    // Saturate and output
                    if (accumulator > {DATA_WIDTH{1'b1}}) begin
                        dist_out <= {DATA_WIDTH{1'b1}};  // Saturate to max
                    end else begin
                        dist_out <= accumulator[DATA_WIDTH-1:0];
                    end
                    valid_out <= 1'b1;
                    done <= 1'b0;
                end
                
                DONE_STATE: begin
                    valid_out <= 1'b0;
                    done <= 1'b1;
                end
                
                default: begin
                    valid_out <= 1'b0;
                    done <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ===========================================================================
// Horner Engine (RBF Kernel Approximation)
// Computes K = exp(-gamma * D) using Horner's method for polynomial approx
// ===========================================================================

module horner_engine #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control
    input  logic                    start,
    output logic                    done,
    
    // Field-programmable gamma parameter
    input  logic [DATA_WIDTH-1:0]   gamma,
    
    // Distance input
    input  logic [DATA_WIDTH-1:0]   dist_in,
    input  logic                    valid_in,
    
    // Kernel output
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    valid_out
);

    // RBF kernel: K(x, x') = exp(-gamma * ||x - x'||²)
    // Using 15th order Taylor series for high accuracy:
    // exp(x) ≈ Σ(x^n / n!) for n = 0 to 15
    // Horner form: exp(x) ≈ 1 + x(1 + x(1/2 + x(1/6 + ... + x/15!)))
    
    // Polynomial coefficients (fixed-point representation)
    // For 15th order approximation: 1/n! for n = 0 to 15
    localparam logic [DATA_WIDTH-1:0] COEFF_00 = (1 << FRAC_BITS);                    // 1/0! = 1.0
    localparam logic [DATA_WIDTH-1:0] COEFF_01 = (1 << FRAC_BITS);                    // 1/1! = 1.0
    localparam logic [DATA_WIDTH-1:0] COEFF_02 = (1 << (FRAC_BITS-1));                // 1/2! = 0.5
    localparam logic [DATA_WIDTH-1:0] COEFF_03 = ((1 << FRAC_BITS) / 6);              // 1/3! = 0.166667
    localparam logic [DATA_WIDTH-1:0] COEFF_04 = ((1 << FRAC_BITS) / 24);             // 1/4! = 0.041667
    localparam logic [DATA_WIDTH-1:0] COEFF_05 = ((1 << FRAC_BITS) / 120);            // 1/5! = 0.008333
    localparam logic [DATA_WIDTH-1:0] COEFF_06 = ((1 << FRAC_BITS) / 720);            // 1/6! = 0.001389
    localparam logic [DATA_WIDTH-1:0] COEFF_07 = ((1 << FRAC_BITS) / 5040);           // 1/7! = 0.000198
    localparam logic [DATA_WIDTH-1:0] COEFF_08 = ((1 << FRAC_BITS) / 40320);          // 1/8! = 0.000025
    localparam logic [DATA_WIDTH-1:0] COEFF_09 = ((1 << FRAC_BITS) / 362880);         // 1/9! = 0.0000028
    localparam logic [DATA_WIDTH-1:0] COEFF_10 = ((1 << FRAC_BITS) / 3628800);        // 1/10! ≈ 0.00000028
    localparam logic [DATA_WIDTH-1:0] COEFF_11 = ((1 << FRAC_BITS) / 39916800);       // 1/11! ≈ 0.000000025
    localparam logic [DATA_WIDTH-1:0] COEFF_12 = ((1 << FRAC_BITS) / 479001600);      // 1/12! ≈ 0.0000000021
    localparam logic [DATA_WIDTH-1:0] COEFF_13 = 1;  // 1/13! ≈ 1.6e-10 (rounds to ~1 in fixed-point)
    localparam logic [DATA_WIDTH-1:0] COEFF_14 = 1;  // 1/14! ≈ 1.1e-11
    localparam logic [DATA_WIDTH-1:0] COEFF_15 = 1;  // 1/15! ≈ 7.6e-13
    
    // Note: gamma is now a field-programmable input port
    // Default gamma = 0.01 is set at the top level
    
    // State machine (16 Horner iterations + scale + output)
    typedef enum logic [4:0] {
        IDLE,
        SCALE,
        HORNER_14,  // Start from highest order
        HORNER_13,
        HORNER_12,
        HORNER_11,
        HORNER_10,
        HORNER_9,
        HORNER_8,
        HORNER_7,
        HORNER_6,
        HORNER_5,
        HORNER_4,
        HORNER_3,
        HORNER_2,
        HORNER_1,
        HORNER_0,
        OUTPUT
    } state_t;
    
    state_t state, next_state;
    
    // Internal registers
    logic signed [DATA_WIDTH-1:0] x;           // -gamma * dist_in
    logic signed [2*DATA_WIDTH-1:0] temp;
    logic signed [DATA_WIDTH-1:0] result;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE:      if (start && valid_in) next_state = SCALE;
            SCALE:     next_state = HORNER_14;
            HORNER_14: next_state = HORNER_13;
            HORNER_13: next_state = HORNER_12;
            HORNER_12: next_state = HORNER_11;
            HORNER_11: next_state = HORNER_10;
            HORNER_10: next_state = HORNER_9;
            HORNER_9:  next_state = HORNER_8;
            HORNER_8:  next_state = HORNER_7;
            HORNER_7:  next_state = HORNER_6;
            HORNER_6:  next_state = HORNER_5;
            HORNER_5:  next_state = HORNER_4;
            HORNER_4:  next_state = HORNER_3;
            HORNER_3:  next_state = HORNER_2;
            HORNER_2:  next_state = HORNER_1;
            HORNER_1:  next_state = HORNER_0;
            HORNER_0:  next_state = OUTPUT;
            OUTPUT:    next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end
    
    // Computation using Horner's method
    // Builds polynomial from innermost term outward
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x <= '0;
            temp <= '0;
            result <= '0;
        end else begin
            case (state)
                IDLE: begin
                    result <= COEFF_00;  // Initialize to 1.0
                end
                
                SCALE: begin
                    // x = -gamma * dist_in (using field-programmable gamma)
                    temp <= -($signed(gamma) * $signed(dist_in));
                    x <= temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];  // Scale back
                end
                
                HORNER_14: begin
                    // Start: result = COEFF_15
                    result <= COEFF_15;
                end
                
                HORNER_13: begin
                    // result = COEFF_14 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_14) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_12: begin
                    // result = COEFF_13 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_13) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_11: begin
                    // result = COEFF_12 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_12) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_10: begin
                    // result = COEFF_11 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_11) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_9: begin
                    // result = COEFF_10 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_10) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_8: begin
                    // result = COEFF_09 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_09) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_7: begin
                    // result = COEFF_08 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_08) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_6: begin
                    // result = COEFF_07 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_07) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_5: begin
                    // result = COEFF_06 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_06) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_4: begin
                    // result = COEFF_05 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_05) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_3: begin
                    // result = COEFF_04 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_04) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_2: begin
                    // result = COEFF_03 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_03) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_1: begin
                    // result = COEFF_02 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_02) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                HORNER_0: begin
                    // result = COEFF_01 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_01) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                
                OUTPUT: begin
                    // Final multiplication: result = COEFF_00 + x * result
                    temp <= $signed(x) * $signed(result);
                    result <= $signed(COEFF_00) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
            endcase
        end
    end
    
    // Output control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_out <= '0;
            valid_out <= 1'b0;
            done <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    // Clamp to [0, 1] range
                    if (result < 0) begin
                        kernel_out <= '0;
                    end else if (result > COEFF_00) begin
                        kernel_out <= COEFF_00;  // Max = 1.0
                    end else begin
                        kernel_out <= result;
                    end
                    valid_out <= 1'b1;
                    done <= 1'b1;
                end
                
                default: begin
                    valid_out <= 1'b0;
                    done <= 1'b0;
                end
            endcase
        end
    end

endmodule
