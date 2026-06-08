/// sta-blackbox
// sim_sram_models.sv — behavioral models for sky130 SRAM macros used in simulation.
// Replace the real hard macros which require the PDK GDS for LVS/DRC but have
// no synthesizable behavioral model in OpenLane.
//
// Only compiled when SIMULATION is defined:
//   iverilog -DSIMULATION ...
//
// Two macros modeled:
//   sky130_sram_8kbyte_1rw1r_32x2048_8   — used by svm_fifo_sram
//   sky130_sram_16kbyte_1rw1r_32x4096_8  — used by svm_sv_ram

`ifdef SIMULATION

// ── 8 KB SRAM: 2048 words × 32-bit (1RW + 1RO port) ─────────────────────────
module sky130_sram_8kbyte_1rw1r_32x2048_8 (
    // Port 0: read-write
    input  wire        clk0,
    input  wire        csb0,    // active-low chip select
    input  wire        web0,    // active-low write enable
    input  wire [3:0]  wmask0,  // byte write mask
    input  wire [10:0] addr0,
    input  wire [31:0] din0,
    output reg  [31:0] dout0,
    // Port 1: read-only
    input  wire        clk1,
    input  wire        csb1,
    input  wire [10:0] addr1,
    output reg  [31:0] dout1
);
    reg [31:0] mem [0:2047];
    integer j;

    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                // byte-masked write
                if (wmask0[0]) mem[addr0][ 7: 0] <= din0[ 7: 0];
                if (wmask0[1]) mem[addr0][15: 8] <= din0[15: 8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    always @(posedge clk1) begin
        if (!csb1)
            dout1 <= mem[addr1];
    end

    initial begin
        for (j = 0; j < 2048; j = j+1) mem[j] = 32'h0;
    end
endmodule

// ── 16 KB SRAM: 4096 words × 32-bit (1RW + 1RO port) ────────────────────────
module sky130_sram_16kbyte_1rw1r_32x4096_8 (
    // Port 0: read-write (Wishbone loader)
    input  wire        clk0,
    input  wire        csb0,
    input  wire        web0,
    input  wire [3:0]  wmask0,
    input  wire [11:0] addr0,
    input  wire [31:0] din0,
    output reg  [31:0] dout0,
    // Port 1: read-only (svm_compute_core)
    input  wire        clk1,
    input  wire        csb1,
    input  wire [11:0] addr1,
    output reg  [31:0] dout1
);
    reg [31:0] mem [0:4095];
    integer j;

    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                if (wmask0[0]) mem[addr0][ 7: 0] <= din0[ 7: 0];
                if (wmask0[1]) mem[addr0][15: 8] <= din0[15: 8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    always @(posedge clk1) begin
        if (!csb1)
            dout1 <= mem[addr1];
    end

    initial begin
        for (j = 0; j < 4096; j = j+1) mem[j] = 32'h0;
    end
endmodule

`endif // SIMULATION
