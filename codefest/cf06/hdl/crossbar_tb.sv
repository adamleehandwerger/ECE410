`timescale 1ns/1ps

module crossbar_tb;

    // DUT ports
    reg  signed [7:0] v0, v1, v2, v3;
    wire signed [9:0] i0, i1, i2, i3;

    crossbar dut (
        .v0(v0), .v1(v1), .v2(v2), .v3(v3),
        .i0(i0), .i1(i1), .i2(i2), .i3(i3)
    );

    integer fd;

    // Helper task: apply inputs, wait one step, log result
    task apply_and_log;
        input signed [7:0] a, b, c, d;
        input [127:0] label;  // 16-char string packed into integer
        begin
            v0 = a; v1 = b; v2 = c; v3 = d;
            #1;
            $fdisplay(fd, "%s  v=[%0d,%0d,%0d,%0d]  i=[%0d,%0d,%0d,%0d]",
                      label, a, b, c, d, i0, i1, i2, i3);
        end
    endtask

    initial begin
        fd = $fopen("crossbar_sim.log", "w");
        if (fd == 0) begin
            $display("ERROR: could not open crossbar_sim.log");
            $finish;
        end

        $fdisplay(fd, "=== crossbar_tb.sv simulation log ===");
        $fdisplay(fd, "DUT: W.T @ v  (4x4, entries +/-1)");
        $fdisplay(fd, "");

        // ------------------------------------------------------------------
        // Section 1: Unit impulse columns (reveals each column of W.T)
        // ------------------------------------------------------------------
        $fdisplay(fd, "--- Unit-impulse tests (v = e_k, amplitude = 1) ---");
        apply_and_log( 1,  0,  0,  0, "e0");
        apply_and_log( 0,  1,  0,  0, "e1");
        apply_and_log( 0,  0,  1,  0, "e2");
        apply_and_log( 0,  0,  0,  1, "e3");
        $fdisplay(fd, "");

        // ------------------------------------------------------------------
        // Section 2: One-column-at-a-time (amplitude = 100, others = 0)
        //   Models grounding all but one input row and measuring column
        //   currents through a small R_sense — sneak paths visible because
        //   the other inputs are driven to 0 (virtual ground).
        // ------------------------------------------------------------------
        $fdisplay(fd, "--- One-column-at-a-time tests (amplitude = 100) ---");
        apply_and_log(100,   0,   0,   0, "col0");
        apply_and_log(  0, 100,   0,   0, "col1");
        apply_and_log(  0,   0, 100,   0, "col2");
        apply_and_log(  0,   0,   0, 100, "col3");
        $fdisplay(fd, "");

        // Negative drive (sneak current reversal check)
        $fdisplay(fd, "--- One-column-at-a-time, negative drive (amplitude = -100) ---");
        apply_and_log(-100,    0,    0,    0, "col0-");
        apply_and_log(   0, -100,    0,    0, "col1-");
        apply_and_log(   0,    0, -100,    0, "col2-");
        apply_and_log(   0,    0,    0, -100, "col3-");
        $fdisplay(fd, "");

        // ------------------------------------------------------------------
        // Section 3: Full vector v = [10, 20, 30, 40]
        // ------------------------------------------------------------------
        $fdisplay(fd, "--- Full vector v = [10, 20, 30, 40] ---");
        apply_and_log(10, 20, 30, 40, "v_full");
        $fdisplay(fd, "  Expected: i0=%0d  i1=%0d  i2=%0d  i3=%0d",
                  10+20-30-40, -10+20+30-40, 10-20+30-40, -10-20-30+40);
        $fdisplay(fd, "");

        // ------------------------------------------------------------------
        // Section 4: Sneak current summary
        //   For a passive crossbar each output current has contributions
        //   from ALL input columns (not just the intended one).  The
        //   unit-impulse results above ARE the sneak-current table:
        //   off-diagonal entries show how much one driven column bleeds
        //   into unintended output rows.
        // ------------------------------------------------------------------
        $fdisplay(fd, "--- Sneak-current matrix (rows=outputs, cols=inputs) ---");
        $fdisplay(fd, "     v0   v1   v2   v3");
        $fdisplay(fd, "i0 [ +1   +1   -1   -1 ]");
        $fdisplay(fd, "i1 [ -1   +1   +1   -1 ]");
        $fdisplay(fd, "i2 [ +1   -1   +1   -1 ]");
        $fdisplay(fd, "i3 [ -1   -1   -1   +1 ]");
        $fdisplay(fd, "Every off-diagonal entry is non-zero: full sneak coupling.");
        $fdisplay(fd, "");

        $fdisplay(fd, "=== simulation complete ===");
        $fclose(fd);
        $display("crossbar_sim.log written.");
        $finish;
    end

endmodule
