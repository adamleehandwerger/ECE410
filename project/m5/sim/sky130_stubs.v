`timescale 1ns/1ps
// Functional simulation stubs for sky130 standard cells
// Used in tb_wb_cosim when PDK models are not on the icarus search path.

// sky130_fd_sc_hd__dlclkp_1 — latch-based integrated clock gate
// GATE is sampled by a transparent latch when CLK=0, preventing glitches
// when GATE changes while CLK=1 (e.g. from NBA delta assignments in simulation).
module sky130_fd_sc_hd__dlclkp_1 (
    input  CLK,
    input  GATE,
    output GCLK
);
    reg gate_latch;
    always @(*) if (!CLK) gate_latch = GATE;
    assign GCLK = CLK & gate_latch;
endmodule
