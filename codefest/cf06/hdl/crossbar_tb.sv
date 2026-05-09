`timescale 1ns/1ps

module tb_crossbar_mac;

    reg  signed [7:0] v0, v1, v2, v3;
    wire signed [9:0] i0, i1, i2, i3;

    crossbar dut (
        .v0(v0), .v1(v1), .v2(v2), .v3(v3),
        .i0(i0), .i1(i1), .i2(i2), .i3(i3)
    );

    integer fd;
    integer pass;
    integer fail;

    task check;
        input signed [7:0]  a, b, c, d;
        input signed [9:0]  exp0, exp1, exp2, exp3;
        input [255:0] label;
        begin
            v0 = a; v1 = b; v2 = c; v3 = d;
            #1;
            if (i0===exp0 && i1===exp1 && i2===exp2 && i3===exp3) begin
                $fdisplay(fd, "PASS  %s  v=[%0d,%0d,%0d,%0d]  i=[%0d,%0d,%0d,%0d]",
                          label, a, b, c, d, i0, i1, i2, i3);
                $display("PASS  %s  v=[%0d,%0d,%0d,%0d]  i=[%0d,%0d,%0d,%0d]",
                         label, a, b, c, d, i0, i1, i2, i3);
                pass = pass + 1;
            end else begin
                $fdisplay(fd, "FAIL  %s  v=[%0d,%0d,%0d,%0d]  got=[%0d,%0d,%0d,%0d]  exp=[%0d,%0d,%0d,%0d]",
                          label, a, b, c, d, i0, i1, i2, i3, exp0, exp1, exp2, exp3);
                $display("FAIL  %s  v=[%0d,%0d,%0d,%0d]  got=[%0d,%0d,%0d,%0d]  exp=[%0d,%0d,%0d,%0d]",
                         label, a, b, c, d, i0, i1, i2, i3, exp0, exp1, exp2, exp3);
                fail = fail + 1;
            end
        end
    endtask

    `define LOG(msg) $fdisplay(fd, msg); $display(msg);

    reg signed [9:0] ia0, ia1, ia2, ia3;
    reg signed [9:0] ib0, ib1, ib2, ib3;

    initial begin
        fd = $fopen("crossbar_mac_sim.log", "w");
        if (fd == 0) begin $display("ERROR: could not open log"); $finish; end

        pass = 0; fail = 0;

        `LOG("=== crossbar_mac simulation log ===")
        `LOG("DUT: crossbar (crossbar_mac.sv)  --  W.T @ v  (4x4, entries +/-1)")
        `LOG("W.T:  i0=[+1,+1,-1,-1]  i1=[-1,+1,+1,-1]  i2=[+1,-1,+1,-1]  i3=[-1,-1,-1,+1]")
        `LOG("det(W.T) = -8  =>  full rank  =>  trivial null space")
        `LOG("")

        // ── Section 1: Basis vectors ───────────────────────────────────────
        `LOG("--- 1. Basis vector tests ---")
        check( 1, 0, 0, 0,  1,-1, 1,-1, "e0");
        check( 0, 1, 0, 0,  1, 1,-1,-1, "e1");
        check( 0, 0, 1, 0, -1, 1, 1,-1, "e2");
        check( 0, 0, 0, 1, -1,-1,-1, 1, "e3");
        `LOG("")

        // ── Section 2: Superposition ───────────────────────────────────────
        `LOG("--- 2. Superposition test  (a=[10,0,0,0], b=[0,20,0,0]) ---")
        v0=10; v1=0; v2=0; v3=0; #1;
        ia0=i0; ia1=i1; ia2=i2; ia3=i3;
        v0=0; v1=20; v2=0; v3=0; #1;
        ib0=i0; ib1=i1; ib2=i2; ib3=i3;
        check(10, 20, 0, 0,
              ia0+ib0, ia1+ib1, ia2+ib2, ia3+ib3,
              "superposition");
        `LOG("")

        // ── Section 3: Overflow boundary ──────────────────────────────────
        // Signed 8-bit asymmetry (max=127, min=-128) means coupled
        // outputs are not always zero even at peak inputs.
        `LOG("--- 3. Overflow boundary tests ---")
        check( 127, 127,-128,-128,  510,   0,   0,-254, "i0 peak");
        check(-128, 127, 127,-128,    0, 510,   0,-254, "i1 peak");
        check( 127,-128, 127,-128,    0,   0, 510,-254, "i2 peak");
        check(-128,-128,-128, 127, -255,-255,-255, 511, "i3 peak");
        `LOG("")

        $fdisplay(fd, "Result: %0d passed, %0d failed", pass, fail);
        $display("Result: %0d passed, %0d failed", pass, fail);
        `LOG("")
        `LOG("=== simulation complete ===")
        $fclose(fd);
        $finish;
    end

endmodule
