// LLM Gemeni
module mac_module (
    input  logic              clk,
    input  logic              rst,
    input  signed [7:0]       a,
    input  signed [7:0]       b,
    output logic signed [31:0] out
);

    // Accumulator logic
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            // Sign extension is handled automatically by the 'signed' types
            out <= out + (a * b);
        end
    end

endmodule