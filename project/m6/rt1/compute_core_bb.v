/// sta-blackbox
// Black-box stub for svm_compute_core — used by top-level synthesis.
// Ports match compute_core.sv exactly; widths are concrete (DATA_WIDTH=16).
module svm_compute_core #(
    parameter DATA_WIDTH     = 16,
    parameter FRAC_BITS      = 10,
    parameter DIST_WIDTH     = 20,
    parameter FEATURE_DIM    = 256,
    parameter NUM_SV         = 500,
    parameter MAX_BATCH_SIZE = 1000,
    parameter RAM_LATENCY    = 3
) (
    input  wire                clk,
    input  wire                rst_n,

    input  wire                param_write_en,
    input  wire [2:0]          param_addr,
    input  wire [15:0]         param_data,
    output wire [15:0]         gamma_reg,
    output wire [15:0]         c_reg,

    input  wire [39:0]         num_sv_per_class_flat,

    output wire [18:0]         ram_addr,
    input  wire [15:0]         ram_rdata,
    output wire                ram_ren,

    input  wire                vbatt_warn,
    input  wire                vbatt_ok,

    input  wire                start,
    input  wire [9:0]          num_samples,

    output wire                sample_rdy,
    output wire [2:0]          class_out,

    output wire                done,
    output wire                error,
    output wire [3:0]          error_code,

    output wire [15:0]         kernel_out,
    output wire                kernel_valid,
    input  wire                kernel_ready,
    output wire [127:0]        class_scores_la,

    input  wire                alpha_write_en,
    input  wire [9:0]          alpha_addr,
    input  wire [15:0]         alpha_data
);
endmodule
