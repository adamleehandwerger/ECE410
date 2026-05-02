// ============================================================================
// Multi-Class Cardiac Arrhythmia Detection — SVM Compute Core
// ECE 410 Project
// ============================================================================
//
// OVERVIEW
// --------
// Four modules implement a duty-cycling RBF-SVM classifier for five cardiac
// arrhythmia classes (Normal, PVC, AFib, VT, SVT).  The system collects up
// to 1000 heartbeats at low power, then classifies them in a burst before
// entering a long sleep (7-day battery target).
//
//   svm_compute_core  — top-level FSM; integrates the three submodules below
//   input_fifo        — 8192-word (16 KB) synchronous FIFO; buffers QSPI data
//   distance_matrix   — squared Euclidean distance Σ(x[k]-sv[k])²
//   horner_engine     — 15th-order Horner polynomial for exp(-γD)
//
// FIXED-POINT FORMAT
// ------------------
// All data words use Q6.10 (16-bit, 10 fractional bits).
// To convert: real_value = raw_integer / 1024.0
//
// ============================================================================
// CLOCK CHARACTERISTICS
// ============================================================================
//   Domain     : Single clock — all modules share clk.
//   Target     : 50 MHz (20 ns period).
//   Edge        : Rising edge; all flip-flops sample on posedge clk.
//   CDC         : None.  qspi_valid/qspi_data must be synchronous to clk
//                 (no internal synchronisers; host QSPI controller must
//                 re-synchronise if it operates in a different domain).
//
// ============================================================================
// RESET BEHAVIOR
// ============================================================================
//   Signal : rst_n  (active-low)
//   Type   : Asynchronous assert, synchronous deassert.
//            Assertion (rst_n=0) takes effect immediately, independent of clk.
//            Deassert is sampled on the next posedge clk.
//   On reset:
//     FSM states        → IDLE (all four modules)
//     gamma_int         → DEFAULT_GAMMA in Q6.10  (≈ 0.0098 at default 0.01)
//     c_int             → DEFAULT_C in Q6.10      (= 1.0)
//     All counters      → 0  (sample, sv, class, feat_wr/rd, kernel_out)
//     feat_wr_en_r      → 0
//     feat_rd_en_r      → 0
//     feat_rd_data      → 0
//     dist_valid_latch  → 0
//     done, error       → 0
//     kernel_out/valid  → 0
//     FIFO ptrs/count   → 0 ; rd_data → 0
//     distance_matrix   → accumulator, dim_counter, dist_out, valid_out → 0
//     horner_engine     → x, temp, result, kernel_out, valid_out → 0
//   Note: feature_bank[] contents are NOT reset (async reset omitted from
//         that always block).  Contents are invalid until LOAD_FEATURES
//         completes and must not be read before that state.
//
// ============================================================================
// svm_compute_core — PORT REFERENCE
// ============================================================================
//
// --- CLOCK / RESET ---
//   clk            in   1    System clock (50 MHz target).
//   rst_n          in   1    Active-low asynchronous reset.
//
// --- PARAMETER PROGRAMMING (host → core) ---
//   param_write_en in   1    Assert one cycle to write a parameter register.
//   param_addr     in   2    Register select: 2'b00=gamma  2'b01=C.
//   param_data     in  16    Value to write (Q6.10).
//   gamma_reg      out 16    Readback of gamma register (Q6.10, comb alias).
//   c_reg          out 16    Readback of C register (Q6.10, comb alias).
//
// --- PER-CLASS SV COUNTS (host → core) ---
//   num_sv_per_class in 8×5  SV count for each of the 5 classes.  Latched on
//                            start; held for the entire batch.  Sum must
//                            satisfy 0 < sum ≤ NUM_SV=250; violation sets error.
//
// --- QSPI FEATURE STREAM (host → core, ready/valid) ---
//   qspi_valid     in   1    Feature word valid strobe.
//   qspi_data      in  16    Feature word (Q6.10).  Captured when both
//                            qspi_valid and qspi_ready are high.
//   qspi_ready     out  1    Core can accept data.  High only in LOAD_FIFO
//                            when FIFO is not full (combinational from fifo_full).
//
// --- SUPPORT VECTOR SRAM (core → external read-only SRAM) ---
//   sv_ram_addr    out 18    {sv_base[9:0], feat_rd_addr[7:0]}  where
//                            sv_base = cumulative SV offset for the current
//                            (class_counter, sv_counter) pair.
//                            Exploits FEATURE_DIM=2^8 for exact bit-concat.
//                            Max address: 250×256-1 = 63 999.
//   sv_ram_rdata   in  16    SV feature word (Q6.10).  1-cycle registered
//                            latency assumed; must align with feat_rd_data.
//   sv_ram_ren     out  1    Read enable (alias of feat_rd_en, combinational).
//
// --- WORKSPACE SRAM (core → external read/write SRAM) ---
//   work_ram_addr  out 18    Linear write address = kernel_out_counter.
//                            Layout: row = heartbeat index, col = SV index.
//                            Max address: 1000×250-1 = 249 999 (fits 18 bits).
//   work_ram_wdata out 16    Kernel value K(x,sv) to store (Q6.10).
//   work_ram_rdata in  16    Read data (currently unused; reserved).
//   work_ram_wen   out  1    Write enable; high for one cycle per accepted
//                            kernel (kernel_valid & kernel_ready in OUTPUT_RESULT).
//   work_ram_ren   out  1    Read enable (permanently 0; reserved).
//
// --- BATCH CONTROL (host → core) ---
//   start          in   1    One-cycle pulse to launch a batch.  Sampled only
//                            in IDLE.  A second pulse while processing is
//                            ignored.
//   num_samples    in  10    Heartbeats to classify this batch (1–1000).
//                            Latched when start is sampled.
//
// --- STATUS (core → host) ---
//   done           out  1    One-cycle registered pulse.  Fires the cycle
//                            after the last kernel of the batch is accepted
//                            (kernel_valid & kernel_ready & last_sv &
//                            last_class & last_heartbeat in OUTPUT_RESULT).
//   error          out  1    Persistent registered flag.  Set on:
//                              (a) illegal FSM state encoding → ERROR_STATE
//                              (b) total_sv_check == 0 while not IDLE
//                              (c) total_sv_check > NUM_SV while not IDLE
//                            Cleared by reset only.
//
// --- KERNEL OUTPUT STREAM (core → host, ready/valid) ---
//   kernel_out     out 16    exp(-γ·‖x-sv‖²) in Q6.10, clamped [0, 1.0].
//   kernel_valid   out  1    Registered; high one cycle per result.
//   kernel_ready   in   1    Consumer ready.  Transfer on posedge clk when
//                            kernel_valid & kernel_ready are both high.
//
// ============================================================================
// FSM STATE SUMMARY (svm_compute_core)
// ============================================================================
//   IDLE           Awaiting start pulse; all counters held at 0.
//   LOAD_FIFO      Filling FIFO from QSPI; exits when count ≥ FEATURE_DIM.
//   LOAD_FEATURES  Transfers FEATURE_DIM words from FIFO into the on-chip
//                  feature_bank[256] register array (512 B); exits when
//                  feat_wr_count reaches FEATURE_DIM.
//   COMPUTE_DIST   Streams feature_bank and SV RAM in parallel into the
//                  distance_matrix engine; exits on dist_done.
//   COMPUTE_KERNEL Runs Horner engine on the accumulated distance; exits on
//                  horner_done.  18-cycle latency.
//   OUTPUT_RESULT  Presents kernel_out/valid; waits for kernel_ready handshake.
//                  Increments sv/class/sample counters; transitions to
//                  COMPUTE_DIST (next SV), LOAD_FIFO (next heartbeat), or
//                  IDLE (batch complete).
//   ERROR_STATE    Entered on illegal state encoding; returns to IDLE next
//                  cycle.
//
// ============================================================================
// input_fifo — PORT REFERENCE
// ============================================================================
// RTL note — wr_ptr_idx / rd_ptr_idx continuous assign wires:
//   `wr_ptr[ADDR_WIDTH-1:0]` and `rd_ptr[ADDR_WIDTH-1:0]` are parameterised
//   bit-selects.  Icarus does not fully support constant selects with parameter
//   indices inside always_ff blocks (same issue as temp_shifted in horner_engine).
//   Both are extracted to `wire` continuous assigns (`wr_ptr_idx`, `rd_ptr_idx`)
//   and those wires are used for the mem array index in the write and read paths.
//
//   clk      in   1           System clock.
//   rst_n    in   1           Active-low async reset; clears ptrs, count, rd_data.
//   wr_en    in   1           Write request; ignored when full.
//   wr_data  in  DATA_WIDTH   Word to enqueue; captured on posedge when !full.
//   rd_en    in   1           Read request; ignored when empty.
//   rd_data  out DATA_WIDTH   Dequeued word; 1-cycle registered latency after
//                             rd_en & !empty.
//   full     out  1           Combinational; high when count == DEPTH.
//   empty    out  1           Combinational; high when count == 0.
//   count    out ADDR_WIDTH+1 Current occupancy in words (0–DEPTH).
//
// ============================================================================
// distance_matrix — PORT REFERENCE
// ============================================================================
// Computes D = Σ(x[k]-sv[k])² over FEATURE_DIM dimensions in Q6.10.
// Internal pipeline: diff (registered) → diff² (registered) →
//                    accumulate (registered) → clamp & output.
//
//   clk        in   1           System clock.
//   rst_n      in   1           Active-low async reset; all regs → 0.
//   start      in   1           Held high by top-level in COMPUTE_DIST.
//   done       out  1           One-cycle registered pulse when D is ready.
//   feature_in in  DATA_WIDTH   Sample component x[k] (Q6.10); valid when
//                               valid_in is high.
//   sv_in      in  DATA_WIDTH   SV component sv[k] (Q6.10); time-aligned with
//                               feature_in (both have 1-cycle read latency).
//   valid_in   in   1           feature_in/sv_in pair is valid.  Driven by
//                               feat_rd_en_r (1-cycle-delayed read enable) to
//                               align registered data from feature_bank and SRAM.
//   dist_out   out DATA_WIDTH   Final D (Q6.10); clamped to 0xFFFF on overflow.
//   valid_out  out  1           Registered; high one cycle when dist_out valid.
//
// ============================================================================
// horner_engine — PORT REFERENCE
// ============================================================================
// Evaluates K = exp(-γD) via 15th-order Horner polynomial.
// Latency: 18 cycles (SCALE + SCALE2 + HORNER_14..0 + OUTPUT =
//          1 + 1 + 15 + 1 = 18).
//
// RTL note — result_next forwarding (added to fix 1-cycle Horner lag):
//   The naive Horner loop `temp <= x*result; result <= COEFF+temp>>FRAC` has a
//   1-cycle lag: `result` used in `temp` is one cycle stale because both
//   assignments are NBA.  Fix: a combinational `result_next` mux computes the
//   updated result from the already-committed `temp` (via the `temp_shifted`
//   continuous assign), and `temp <= x * result_next` is used instead.  This
//   eliminates the lag and makes the polynomial evaluation exact.
//
// RTL note — temp_shifted continuous assign:
//   `temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]` is a parameterised bit-select.
//   Icarus does not fully support constant selects with parameter indices inside
//   always_comb blocks, so this is extracted to a continuous assign wire
//   `temp_shifted` used throughout the result_next always_comb.
//
//   clk        in   1           System clock.
//   rst_n      in   1           Active-low async reset; all regs → 0.
//   start      in   1           Held high by top-level in COMPUTE_KERNEL.
//                               Evaluation begins when start & valid_in in IDLE.
//   done       out  1           One-cycle registered pulse coincident with
//                               valid_out (asserted in OUTPUT state).
//   gamma      in  DATA_WIDTH   RBF bandwidth γ (Q6.10); field-programmable
//                               via top-level gamma_int register.
//   dist_in    in  DATA_WIDTH   Squared distance D from distance_matrix (Q6.10);
//                               captured in SCALE state.
//   valid_in   in   1           dist_in valid.  Driven by dist_valid_latch, a
//                               1-bit SR that holds dist_valid_out until
//                               COMPUTE_KERNEL consumes it.
//   kernel_out out DATA_WIDTH   exp(-γD) in Q6.10; clamped to [0, 1.0].
//   valid_out  out  1           Registered; high one cycle when kernel_out valid.
//
// ============================================================================

module svm_compute_core #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20,
    parameter int FEATURE_DIM = 256,
    parameter int NUM_SV = 250,
    parameter int MAX_BATCH_SIZE = 1000,
    parameter int FIFO_DEPTH = 8192,
    parameter int ADDR_WIDTH = 13,
    parameter real DEFAULT_GAMMA = 0.01,
    parameter real DEFAULT_C = 1.0
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    param_write_en,
    input  logic [1:0]              param_addr,
    input  logic [DATA_WIDTH-1:0]   param_data,
    output logic [DATA_WIDTH-1:0]   gamma_reg,
    output logic [DATA_WIDTH-1:0]   c_reg,

    input  logic [7:0]              num_sv_per_class [5],

    input  logic                    qspi_valid,
    input  logic [DATA_WIDTH-1:0]   qspi_data,
    output logic                    qspi_ready,

    output logic [17:0]             sv_ram_addr,
    input  logic [DATA_WIDTH-1:0]   sv_ram_rdata,
    output logic                    sv_ram_ren,

    output logic [17:0]             work_ram_addr,
    output logic [DATA_WIDTH-1:0]   work_ram_wdata,
    input  logic [DATA_WIDTH-1:0]   work_ram_rdata,
    output logic                    work_ram_wen,
    output logic                    work_ram_ren,

    input  logic                    start,
    input  logic [9:0]              num_samples,
    output logic                    done,
    output logic                    error,

    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    kernel_valid,
    input  logic                    kernel_ready
);

    // =========================================================================
    // Constants
    // =========================================================================

    localparam logic [DATA_WIDTH-1:0] GAMMA_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_GAMMA * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] C_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_C * (2.0 ** FRAC_BITS)));

    // =========================================================================
    // Internal Signals
    // =========================================================================

    logic [DATA_WIDTH-1:0] gamma_int;
    logic [DATA_WIDTH-1:0] c_int;

    // FIFO
    logic                    fifo_wr_en;
    logic                    fifo_rd_en;
    logic [DATA_WIDTH-1:0]   fifo_wr_data;
    logic [DATA_WIDTH-1:0]   fifo_rd_data;
    logic                    fifo_full;
    logic                    fifo_empty;
    logic [ADDR_WIDTH:0]     fifo_count;

    // Distance Matrix
    logic                    dist_start;
    logic                    dist_done;
    logic [DATA_WIDTH-1:0]   dist_feature_in;
    logic [DATA_WIDTH-1:0]   dist_sv_in;
    logic                    dist_valid_in;
    logic [DIST_WIDTH-1:0]   dist_out;
    logic                    dist_valid_out;

    // Horner Engine
    logic                    horner_start;
    logic                    horner_done;
    logic [DIST_WIDTH-1:0]   horner_dist_in;
    logic                    horner_valid_in;
    logic [DATA_WIDTH-1:0]   horner_kernel_out;
    logic                    horner_valid_out;

    // Holds dist_valid_out until COMPUTE_KERNEL consumes it on its first cycle.
    logic                    dist_valid_latch;

    // Feature Register Bank — 256 × 16-bit = 512 B, one heartbeat on-chip.
    // Written once per heartbeat in LOAD_FEATURES; re-read for every SV in
    // COMPUTE_DIST so the FIFO is never drained more than once per heartbeat.
    logic [DATA_WIDTH-1:0]   feature_bank [FEATURE_DIM];

    // Feature bank write path (LOAD_FEATURES).
    // fifo_rd_data has 1-cycle registered latency, so feat_wr_en_r /
    // feat_wr_addr_r (1-cycle delayed versions of fifo_rd_en / feat_wr_addr)
    // align the bank write with the data arriving from the FIFO.
    logic [7:0]              feat_wr_addr;
    logic                    feat_wr_en_r;
    logic [7:0]              feat_wr_addr_r;
    logic [8:0]              feat_wr_count;   // completed writes; triggers LOAD_FEATURES→COMPUTE_DIST

    // Feature bank read path (COMPUTE_DIST).
    // 1-cycle registered read: feat_rd_en_r aligns dist_valid_in with
    // feat_rd_data and sv_ram_rdata (both have identical 1-cycle latency).
    logic [7:0]              feat_rd_addr;
    logic                    feat_rd_en;
    logic                    feat_rd_en_r;
    logic [DATA_WIDTH-1:0]   feat_rd_data;

    // Global SV index: sv_counter + cumulative SV count of earlier classes.
    // sv_ram_addr = {sv_base[9:0], feat_rd_addr[7:0]} exploits FEATURE_DIM=2^8.
    logic [9:0]              sv_base;

    // Linear counter for work_ram write addressing (max 1000×250 = 250 000)
    logic [17:0]             kernel_out_counter;

    // FSM — 7 states fit in 3 bits
    typedef enum logic [2:0] {
        IDLE,
        LOAD_FIFO,
        LOAD_FEATURES,
        COMPUTE_DIST,
        COMPUTE_KERNEL,
        OUTPUT_RESULT,
        ERROR_STATE
    } state_t;

    state_t state, next_state;

    // Counters
    logic [9:0]  sample_counter;
    logic [7:0]  sv_counter;
    logic [2:0]  class_counter;

    // Per-class SV counts latched from num_sv_per_class on start.
    logic [7:0] sv_count_reg [5];

    // Runtime validation: sum must be > 0 and ≤ NUM_SV
    logic [10:0] total_sv_check;
    assign total_sv_check = sv_count_reg[0] + sv_count_reg[1]
                          + sv_count_reg[2] + sv_count_reg[3]
                          + sv_count_reg[4];

    // Combinational boundary flags
    wire last_sv        = (sv_counter >= sv_count_reg[class_counter] - 1);
    wire last_class     = (class_counter >= 3'd4);
    wire last_heartbeat = (sample_counter >= num_samples - 1);

    // =========================================================================
    // Sub-module Instances
    // =========================================================================

    input_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_input_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr_en), .wr_data(fifo_wr_data),
        .rd_en(fifo_rd_en), .rd_data(fifo_rd_data),
        .full(fifo_full), .empty(fifo_empty), .count(fifo_count)
    );

    distance_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .DIST_WIDTH(DIST_WIDTH),
        .FEATURE_DIM(FEATURE_DIM)
    ) u_distance_matrix (
        .clk(clk), .rst_n(rst_n),
        .start(dist_start),
        .feature_in(dist_feature_in), .sv_in(dist_sv_in), .valid_in(dist_valid_in),
        .dist_out(dist_out), .valid_out(dist_valid_out), .done(dist_done)
    );

    horner_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .DIST_WIDTH(DIST_WIDTH)
    ) u_horner_engine (
        .clk(clk), .rst_n(rst_n),
        .start(horner_start), .dist_in(horner_dist_in),
        .valid_in(horner_valid_in), .gamma(gamma_int),
        .kernel_out(horner_kernel_out), .valid_out(horner_valid_out), .done(horner_done)
    );

    // =========================================================================
    // Parameter Registers
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gamma_int <= GAMMA_DEFAULT;
            c_int     <= C_DEFAULT;
        end else if (param_write_en) begin
            case (param_addr)
                2'b00: gamma_int <= param_data;
                2'b01: c_int     <= param_data;
                default: begin end
            endcase
        end
    end

    assign gamma_reg = gamma_int;
    assign c_reg     = c_int;

    // =========================================================================
    // FSM
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:          if (start) next_state = LOAD_FIFO;

            // Wait for one full heartbeat (FEATURE_DIM words) in the FIFO
            LOAD_FIFO:     if (fifo_count >= FEATURE_DIM) next_state = LOAD_FEATURES;

            // Drain exactly FEATURE_DIM words from FIFO into feature_bank
            LOAD_FEATURES: if (feat_wr_count == FEATURE_DIM) next_state = COMPUTE_DIST;

            COMPUTE_DIST:  if (dist_done) next_state = COMPUTE_KERNEL;

            COMPUTE_KERNEL: if (horner_done) next_state = OUTPUT_RESULT;

            OUTPUT_RESULT: begin
                if (kernel_ready && kernel_valid) begin
                    if (last_sv && last_class) begin
                        // All SVs for this heartbeat done
                        if (last_heartbeat) next_state = IDLE;
                        else                next_state = LOAD_FIFO;
                    end else begin
                        next_state = COMPUTE_DIST;  // next SV
                    end
                end
            end

            ERROR_STATE: next_state = IDLE;
            default:     next_state = ERROR_STATE;
        endcase
    end

    // =========================================================================
    // FIFO Control
    // =========================================================================

    always_comb begin
        fifo_wr_en   = qspi_valid && !fifo_full && (state == LOAD_FIFO);
        fifo_wr_data = qspi_data;
        qspi_ready   = !fifo_full && (state == LOAD_FIFO);
        fifo_rd_en   = (state == LOAD_FEATURES) && !fifo_empty
                       && (feat_wr_addr < FEATURE_DIM);
    end

    // =========================================================================
    // Feature Register Bank — Write Path (LOAD_FEATURES)
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_wr_en_r   <= 1'b0;
            feat_wr_addr_r <= '0;
        end else begin
            feat_wr_en_r   <= fifo_rd_en;
            feat_wr_addr_r <= feat_wr_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_addr <= '0;
        else begin
            case (state)
                LOAD_FEATURES: if (fifo_rd_en) feat_wr_addr <= feat_wr_addr + 1;
                default:       feat_wr_addr <= '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_count <= '0;
        else begin
            case (state)
                LOAD_FEATURES: if (feat_wr_en_r) feat_wr_count <= feat_wr_count + 1;
                default:       feat_wr_count <= '0;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (feat_wr_en_r)
            feature_bank[feat_wr_addr_r] <= fifo_rd_data;
    end

    // =========================================================================
    // Feature Register Bank — Read Path (COMPUTE_DIST)
    // =========================================================================

    assign feat_rd_en = (state == COMPUTE_DIST) && (feat_rd_addr < FEATURE_DIM);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_en_r <= 1'b0;
        else        feat_rd_en_r <= feat_rd_en;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_addr <= '0;
        else begin
            case (state)
                COMPUTE_DIST: if (feat_rd_en) feat_rd_addr <= feat_rd_addr + 1;
                default:      feat_rd_addr <= '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_data <= '0;
        else if (feat_rd_en)
            feat_rd_data <= feature_bank[feat_rd_addr];
    end

    // =========================================================================
    // SV RAM Addressing
    // =========================================================================

    always_comb begin
        sv_base = {2'b00, sv_counter};
        if (class_counter >= 1) sv_base = sv_base + {2'b00, sv_count_reg[0]};
        if (class_counter >= 2) sv_base = sv_base + {2'b00, sv_count_reg[1]};
        if (class_counter >= 3) sv_base = sv_base + {2'b00, sv_count_reg[2]};
        if (class_counter >= 4) sv_base = sv_base + {2'b00, sv_count_reg[3]};
    end

    assign sv_ram_addr = {sv_base, feat_rd_addr};   // 10 + 8 = 18 bits
    assign sv_ram_ren  = feat_rd_en;
    assign dist_sv_in  = sv_ram_rdata;

    // =========================================================================
    // Pipeline Connections
    // =========================================================================

    assign dist_start      = (state == COMPUTE_DIST);
    assign dist_feature_in = feat_rd_data;
    assign dist_valid_in   = feat_rd_en_r;

    assign horner_start    = (state == COMPUTE_KERNEL);
    assign horner_dist_in  = dist_out;
    assign horner_valid_in = dist_valid_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dist_valid_latch <= 1'b0;
        else if (dist_valid_out)
            dist_valid_latch <= 1'b1;
        else if (state == COMPUTE_KERNEL)
            dist_valid_latch <= 1'b0;
    end

    // =========================================================================
    // Counter Management
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter <= '0;
            sv_counter     <= '0;
            class_counter  <= '0;
            for (int i = 0; i < 5; i++) sv_count_reg[i] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    sample_counter <= '0;
                    sv_counter     <= '0;
                    class_counter  <= '0;
                    if (start)
                        for (int i = 0; i < 5; i++)
                            sv_count_reg[i] <= num_sv_per_class[i];
                end

                LOAD_FIFO: begin
                    sv_counter    <= '0;
                    class_counter <= '0;
                end

                OUTPUT_RESULT: begin
                    if (kernel_ready && kernel_valid) begin
                        if (last_sv && last_class) begin
                            sv_counter     <= '0;
                            class_counter  <= '0;
                            sample_counter <= sample_counter + 1;
                        end else if (last_sv) begin
                            sv_counter    <= '0;
                            class_counter <= class_counter + 1;
                        end else begin
                            sv_counter <= sv_counter + 1;
                        end
                    end
                end

                default: begin end
            endcase
        end
    end

    // =========================================================================
    // Work RAM Output
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) kernel_out_counter <= '0;
        else begin
            case (state)
                IDLE:          kernel_out_counter <= '0;
                OUTPUT_RESULT: if (kernel_ready && kernel_valid)
                                   kernel_out_counter <= kernel_out_counter + 1;
                default: ;
            endcase
        end
    end

    assign work_ram_addr  = kernel_out_counter;
    assign work_ram_wdata = kernel_out;
    assign work_ram_wen   = (state == OUTPUT_RESULT) && kernel_valid && kernel_ready;
    assign work_ram_ren   = 1'b0;

    // =========================================================================
    // Output Registers
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done         <= 1'b0;
            error        <= 1'b0;
            kernel_out   <= '0;
            kernel_valid <= 1'b0;
        end else begin
            done  <= (state == OUTPUT_RESULT) && kernel_valid && kernel_ready
                      && last_sv && last_class && last_heartbeat;
            error <= (state == ERROR_STATE)
                  || ((state != IDLE) && (total_sv_check == 0))
                  || ((state != IDLE) && (total_sv_check > NUM_SV));
            kernel_out   <= horner_kernel_out;
            kernel_valid <= horner_valid_out;
        end
    end

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

    logic [DATA_WIDTH-1:0] mem [DEPTH];
    logic [ADDR_WIDTH:0]   wr_ptr;
    logic [ADDR_WIDTH:0]   rd_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // Wires for parameterized bit-selects; required because Icarus does not fully
    // support constant selects with parameter indices inside always_ff blocks
    // (same workaround applied to Horner temp_shifted).
    wire [ADDR_WIDTH-1:0] wr_ptr_idx = wr_ptr[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] rd_ptr_idx = rd_ptr[ADDR_WIDTH-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wr_ptr <= '0;
        else if (wr_en && !full) wr_ptr <= wr_ptr + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_ptr <= '0;
        else if (rd_en && !empty) rd_ptr <= rd_ptr + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1;
                2'b01:   count <= count - 1;
                default: count <= count;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (wr_en && !full)
            mem[wr_ptr_idx] <= wr_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data <= '0;
        else if (rd_en && !empty) rd_data <= mem[rd_ptr_idx];
    end

endmodule

// ===========================================================================
// Distance Matrix Engine
// Computes D = ||X[i] - SV[j]||² (squared Euclidean distance)
// ===========================================================================

module distance_matrix #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20,
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

    // Distance output — DIST_WIDTH-bit Q6.10; max = (1<<DIST_WIDTH)-1
    output logic [DIST_WIDTH-1:0]   dist_out,
    output logic                    valid_out
);

    typedef enum logic [1:0] {
        IDLE,
        ACCUMULATE,
        OUTPUT,
        DONE_STATE
    } state_t;

    state_t state, next_state;

    logic [2*DATA_WIDTH-1:0]   diff;
    logic [2*DATA_WIDTH-1:0]   diff_squared;
    logic [2*DATA_WIDTH+8-1:0] accumulator;
    logic [8:0]                dim_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:       if (start) next_state = ACCUMULATE;
            ACCUMULATE: if (dim_counter >= FEATURE_DIM - 1 && valid_in) next_state = OUTPUT;
            OUTPUT:     next_state = DONE_STATE;
            DONE_STATE: next_state = IDLE;
            default:    next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dim_counter <= '0;
        end else begin
            case (state)
                IDLE:       dim_counter <= '0;
                ACCUMULATE: if (valid_in) dim_counter <= dim_counter + 1;
                default:    dim_counter <= dim_counter;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff         <= '0;
            diff_squared <= '0;
        end else if (valid_in && state == ACCUMULATE) begin
            diff         <= $signed(feature_in) - $signed(sv_in);
            diff_squared <= $signed(diff) * $signed(diff);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else begin
            case (state)
                IDLE:       accumulator <= '0;
                ACCUMULATE: if (valid_in)
                    accumulator <= accumulator + (diff_squared >>> FRAC_BITS);
                default:    accumulator <= accumulator;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out  <= {DIST_WIDTH{1'b0}};
            valid_out <= 1'b0;
            done      <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    // Saturate at DIST_WIDTH all-ones (= max representable Q6.10 value)
                    dist_out  <= (|accumulator[2*DATA_WIDTH+8-1:DIST_WIDTH])
                                  ? {DIST_WIDTH{1'b1}}
                                  : accumulator[DIST_WIDTH-1:0];
                    valid_out <= 1'b1;
                    done      <= 1'b0;
                end
                DONE_STATE: begin
                    valid_out <= 1'b0;
                    done      <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
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
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // Control
    input  logic                    start,
    output logic                    done,

    // Field-programmable gamma parameter
    input  logic [DATA_WIDTH-1:0]   gamma,

    // Distance input — DIST_WIDTH-bit Q6.10 (matches distance_matrix.dist_out)
    input  logic [DIST_WIDTH-1:0]   dist_in,
    input  logic                    valid_in,

    // Kernel output
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    valid_out
);

    // 15th-order Taylor series via Horner's method: exp(x) ≈ Σ(xⁿ/n!) for n=0..15
    localparam logic [DATA_WIDTH-1:0] COEFF_00 = (1 << FRAC_BITS);
    localparam logic [DATA_WIDTH-1:0] COEFF_01 = (1 << FRAC_BITS);
    localparam logic [DATA_WIDTH-1:0] COEFF_02 = (1 << (FRAC_BITS-1));
    localparam logic [DATA_WIDTH-1:0] COEFF_03 = ((1 << FRAC_BITS) / 6);
    localparam logic [DATA_WIDTH-1:0] COEFF_04 = ((1 << FRAC_BITS) / 24);
    localparam logic [DATA_WIDTH-1:0] COEFF_05 = ((1 << FRAC_BITS) / 120);
    localparam logic [DATA_WIDTH-1:0] COEFF_06 = ((1 << FRAC_BITS) / 720);
    localparam logic [DATA_WIDTH-1:0] COEFF_07 = ((1 << FRAC_BITS) / 5040);
    localparam logic [DATA_WIDTH-1:0] COEFF_08 = ((1 << FRAC_BITS) / 40320);
    localparam logic [DATA_WIDTH-1:0] COEFF_09 = ((1 << FRAC_BITS) / 362880);
    localparam logic [DATA_WIDTH-1:0] COEFF_10 = ((1 << FRAC_BITS) / 3628800);
    localparam logic [DATA_WIDTH-1:0] COEFF_11 = ((1 << FRAC_BITS) / 39916800);
    localparam logic [DATA_WIDTH-1:0] COEFF_12 = ((1 << FRAC_BITS) / 479001600);
    localparam logic [DATA_WIDTH-1:0] COEFF_13 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_14 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_15 = 1;

    typedef enum logic [4:0] {
        IDLE,
        SCALE,
        SCALE2,
        HORNER_14,
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

    logic signed [DATA_WIDTH-1:0]         x;
    logic signed [DATA_WIDTH+DIST_WIDTH-1:0] temp;  // wide enough for gamma×dist_in
    logic signed [DATA_WIDTH-1:0]         result;
    logic signed [DATA_WIDTH-1:0]         result_next;
    logic signed [DATA_WIDTH-1:0]         temp_shifted;
    assign temp_shifted = temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start && valid_in) next_state = SCALE;
            SCALE:     next_state = SCALE2;
            SCALE2:    next_state = HORNER_14;
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

    // Forward the NEW result combinationally so temp can multiply by it in the
    // same cycle, eliminating the one-cycle Horner multiply lag.
    always_comb begin
        case (state)
            HORNER_14: result_next = COEFF_15;
            HORNER_13: result_next = $signed(COEFF_14) + $signed(temp_shifted);
            HORNER_12: result_next = $signed(COEFF_13) + $signed(temp_shifted);
            HORNER_11: result_next = $signed(COEFF_12) + $signed(temp_shifted);
            HORNER_10: result_next = $signed(COEFF_11) + $signed(temp_shifted);
            HORNER_9:  result_next = $signed(COEFF_10) + $signed(temp_shifted);
            HORNER_8:  result_next = $signed(COEFF_09) + $signed(temp_shifted);
            HORNER_7:  result_next = $signed(COEFF_08) + $signed(temp_shifted);
            HORNER_6:  result_next = $signed(COEFF_07) + $signed(temp_shifted);
            HORNER_5:  result_next = $signed(COEFF_06) + $signed(temp_shifted);
            HORNER_4:  result_next = $signed(COEFF_05) + $signed(temp_shifted);
            HORNER_3:  result_next = $signed(COEFF_04) + $signed(temp_shifted);
            HORNER_2:  result_next = $signed(COEFF_03) + $signed(temp_shifted);
            HORNER_1:  result_next = $signed(COEFF_02) + $signed(temp_shifted);
            HORNER_0:  result_next = $signed(COEFF_01) + $signed(temp_shifted);
            OUTPUT:    result_next = $signed(COEFF_00) + $signed(temp_shifted);
            default:   result_next = result;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x      <= '0;
            temp   <= '0;
            result <= '0;
        end else begin
            case (state)
                IDLE: result <= COEFF_00;

                // SCALE1: compute -gamma*dist into temp (NBA commits next cycle)
                SCALE: begin
                    temp <= -(  $signed({{DIST_WIDTH{1'b0}}, gamma})
                              * $signed({{DATA_WIDTH{1'b0}}, dist_in}));
                end
                // SCALE2: read committed temp into x (fixes NBA race)
                SCALE2: begin
                    x <= temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
                end

                HORNER_14: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= result_next;
                end

                HORNER_13: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_14) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_12: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_13) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_11: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_12) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_10: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_11) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_9: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_10) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_8: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_09) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_7: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_08) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_6: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_07) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_5: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_06) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_4: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_05) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_3: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_04) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_2: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_03) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_1: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_02) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_0: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_01) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                OUTPUT: begin
                    temp   <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_00) + $signed(temp[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_out <= '0;
            valid_out  <= 1'b0;
            done       <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    kernel_out <= (result_next < 0)        ? '0       :
                                  (result_next > COEFF_00) ? COEFF_00 :
                                                             result_next;
                    valid_out  <= 1'b1;
                    done       <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
                end
            endcase
        end
    end

endmodule
