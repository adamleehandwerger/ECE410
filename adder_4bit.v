// 4-bit Adder with Carry-out
// Inputs: A (4-bit), B (4-bit), Cin (carry-in)
// Outputs: Sum (4-bit), Cout (carry-out)

module adder_4bit (
    input [3:0] A,
    input [3:0] B,
    input Cin,
    output [3:0] Sum,
    output Cout
);

    wire [4:0] result;
    
    // Perform 5-bit addition (includes carry-out)
    assign result = A + B + Cin;
    
    // Assign outputs
    assign Sum = result[3:0];      // Lower 4 bits for sum
    assign Cout = result[4];        // MSB is carry-out

endmodule
