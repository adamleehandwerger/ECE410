`timescale 1ns/1ps
module crossbar (
    input  wire signed [7:0] v0, v1, v2, v3,
    output reg  signed [9:0] i0, i1, i2, i3
);
    wire signed [9:0] s0 = v0;
    wire signed [9:0] s1 = v1;
    wire signed [9:0] s2 = v2;
    wire signed [9:0] s3 = v3;

    always @(*) begin
        i0 =  s0 + s1 - s2 - s3;
        i1 = -s0 + s1 + s2 - s3;
        i2 =  s0 - s1 + s2 - s3;
        i3 = -s0 - s1 - s2 + s3;
    end
endmodule

