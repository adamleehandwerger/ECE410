`timescale 1ns/1ps
module crossbar_sds (
    input  wire signed [7:0] v0, v1, v2, v3,
    output wire signed [9:0] i0, i1, i2, i3
);
    wire signed [9:0] s0 = v0;
    wire signed [9:0] s1 = v1;
    wire signed [9:0] s2 = v2;
    wire signed [9:0] s3 = v3;

    // col 0: W[:,0]=[+1,+1,-1,-1]  ->  i0 = (v0+v1) - (v2+v3)
    wire signed [9:0] i0_pos = s0 + s1;
    wire signed [9:0] i0_neg = s2 + s3;
    assign i0 = i0_pos - i0_neg;

    // col 1: W[:,1]=[-1,+1,+1,-1]  ->  i1 = (v1+v2) - (v0+v3)
    wire signed [9:0] i1_pos = s1 + s2;
    wire signed [9:0] i1_neg = s0 + s3;
    assign i1 = i1_pos - i1_neg;

    // col 2: W[:,2]=[+1,-1,+1,-1]  ->  i2 = (v0+v2) - (v1+v3)
    wire signed [9:0] i2_pos = s0 + s2;
    wire signed [9:0] i2_neg = s1 + s3;
    assign i2 = i2_pos - i2_neg;

    // col 3: W[:,3]=[-1,-1,-1,+1]  ->  i3 = v3 - (v0+v1+v2)
    wire signed [9:0] i3_pos = s3;
    wire signed [9:0] i3_neg = s0 + s1 + s2;
    assign i3 = i3_pos - i3_neg;

endmodule
