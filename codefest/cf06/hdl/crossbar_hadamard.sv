`timescale 1ns/1ps
module crossbar_hadamard (
    input  wire signed [7:0] v0, v1, v2, v3,
    output wire signed [9:0] i0, i1, i2, i3
);
    // Hadamard weight matrix (full rank, trivial null space):
    // W = [[ 1,  1,  1,  1],
    //      [ 1, -1,  1, -1],
    //      [ 1,  1, -1, -1],
    //      [ 1, -1, -1,  1]]
    //
    // W.T @ v:
    // i0 =  v0 + v1 + v2 + v3
    // i1 =  v0 - v1 + v2 - v3
    // i2 =  v0 + v1 - v2 - v3
    // i3 =  v0 - v1 - v2 + v3

    wire signed [9:0] s0 = v0;
    wire signed [9:0] s1 = v1;
    wire signed [9:0] s2 = v2;
    wire signed [9:0] s3 = v3;

    assign i0 =  s0 + s1 + s2 + s3;
    assign i1 =  s0 - s1 + s2 - s3;
    assign i2 =  s0 + s1 - s2 - s3;
    assign i3 =  s0 - s1 - s2 + s3;

endmodule
