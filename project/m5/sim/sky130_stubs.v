// Functional simulation stubs for sky130 standard cells
// Used in tb_wb_cosim when PDK models are not on the icarus search path.

// sky130_fd_sc_hd__dlclkp_1 — latch-based integrated clock gate
// Behavioral approximation: simple AND (glitch-free in sim because clock is clean).
module sky130_fd_sc_hd__dlclkp_1 (
    input  CLK,
    input  GATE,
    output GCLK
);
    assign GCLK = CLK & GATE;
endmodule
