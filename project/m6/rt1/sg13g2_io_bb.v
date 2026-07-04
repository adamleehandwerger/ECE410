// sg13g2_io_bb.v — Black-box stubs for sg13g2 IO pad cells.
// (* blackbox *) tells Yosys to preserve instances rather than synthesize through.

(* blackbox *)
module sg13g2_IOPadIn (iovdd, iovss, vdd, vss, pad, p2c);
    inout  iovdd, iovss, vdd, vss, pad;
    output p2c;
endmodule

(* blackbox *)
module sg13g2_IOPadOut4mA (iovdd, iovss, vdd, vss, pad, c2p);
    inout  iovdd, iovss, vdd, vss, pad;
    input  c2p;
endmodule

(* blackbox *)
module sg13g2_IOPadOut16mA (iovdd, iovss, vdd, vss, pad, c2p);
    inout  iovdd, iovss, vdd, vss, pad;
    input  c2p;
endmodule

(* blackbox *)
module sg13g2_IOPadOut30mA (iovdd, iovss, vdd, vss, pad, c2p);
    inout  iovdd, iovss, vdd, vss, pad;
    input  c2p;
endmodule

(* blackbox *)
module sg13g2_IOPadIOVdd (iovdd, iovss, vdd, vss);
    inout  iovdd, iovss, vdd, vss;
endmodule

(* blackbox *)
module sg13g2_IOPadIOVss (iovdd, iovss, vdd, vss);
    inout  iovdd, iovss, vdd, vss;
endmodule

(* blackbox *)
module sg13g2_IOPadVdd (iovdd, iovss, vdd, vss);
    inout  iovdd, iovss, vdd, vss;
endmodule

(* blackbox *)
module sg13g2_IOPadVss (iovdd, iovss, vdd, vss);
    inout  iovdd, iovss, vdd, vss;
endmodule
