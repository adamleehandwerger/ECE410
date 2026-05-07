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
    integer cancel_count;

    // Sweep grid values: -40, -20, 0, 20, 40  (5^4 = 625 combinations)
    reg signed [7:0] sweep [0:4];

    // Helper task: apply inputs, wait one step, log to file and terminal
    task apply_and_log;
        input signed [7:0] a, b, c, d;
        input [127:0] label;
        begin
            v0 = a; v1 = b; v2 = c; v3 = d;
            #1;
            $fdisplay(fd, "%s  v=[%0d,%0d,%0d,%0d]  i=[%0d,%0d,%0d,%0d]",
                      label, a, b, c, d, i0, i1, i2, i3);
            $display("%s  v=[%0d,%0d,%0d,%0d]  i=[%0d,%0d,%0d,%0d]",
                     label, a, b, c, d, i0, i1, i2, i3);
        end
    endtask

    // Cancellation check task: flag any zero output with non-zero input
    task check_cancel;
        input signed [7:0] a, b, c, d;
        begin
            v0 = a; v1 = b; v2 = c; v3 = d;
            #1;
            if (a != 0 || b != 0 || c != 0 || d != 0) begin
                if (i0 == 0) begin
                    $fdisplay(fd, "CANCEL i0=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                              a, b, c, d, a, b, -c, -d);
                    $display("CANCEL i0=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                             a, b, c, d, a, b, -c, -d);
                    cancel_count = cancel_count + 1;
                end
                if (i1 == 0) begin
                    $fdisplay(fd, "CANCEL i1=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                              a, b, c, d, -a, b, c, -d);
                    $display("CANCEL i1=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                             a, b, c, d, -a, b, c, -d);
                    cancel_count = cancel_count + 1;
                end
                if (i2 == 0) begin
                    $fdisplay(fd, "CANCEL i2=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                              a, b, c, d, a, -b, c, -d);
                    $display("CANCEL i2=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                             a, b, c, d, a, -b, c, -d);
                    cancel_count = cancel_count + 1;
                end
                if (i3 == 0) begin
                    $fdisplay(fd, "CANCEL i3=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                              a, b, c, d, -a, -b, -c, d);
                    $display("CANCEL i3=0  v=[%0d,%0d,%0d,%0d]  cells: (%0d)+(%0d)+(%0d)+(%0d)=0",
                             a, b, c, d, -a, -b, -c, d);
                    cancel_count = cancel_count + 1;
                end
            end
        end
    endtask

    // Write a line to both file and terminal
    `define LOG(msg) $fdisplay(fd, msg); $display(msg);

    integer ia, ib, ic, id;

    initial begin
        fd = $fopen("crossbar_sim.log", "w");
        if (fd == 0) begin
            $display("ERROR: could not open crossbar_sim.log");
            $finish;
        end

        sweep[0] = -40; sweep[1] = -20; sweep[2] = 0; sweep[3] = 20; sweep[4] = 40;

        `LOG("=== crossbar_tb.sv simulation log ===")
        `LOG("DUT: W.T @ v  (4x4, entries +/-1)")
        `LOG("")

        // ------------------------------------------------------------------
        // Section 1: Unit impulse columns
        // ------------------------------------------------------------------
        `LOG("--- Unit-impulse tests (v = e_k, amplitude = 1) ---")
        apply_and_log( 1,  0,  0,  0, "e0");
        apply_and_log( 0,  1,  0,  0, "e1");
        apply_and_log( 0,  0,  1,  0, "e2");
        apply_and_log( 0,  0,  0,  1, "e3");
        `LOG("")

        // ------------------------------------------------------------------
        // Section 2: One-column-at-a-time
        // ------------------------------------------------------------------
        `LOG("--- One-column-at-a-time tests (amplitude = 100) ---")
        apply_and_log(100,   0,   0,   0, "col0");
        apply_and_log(  0, 100,   0,   0, "col1");
        apply_and_log(  0,   0, 100,   0, "col2");
        apply_and_log(  0,   0,   0, 100, "col3");
        `LOG("")

        `LOG("--- One-column-at-a-time, negative drive (amplitude = -100) ---")
        apply_and_log(-100,    0,    0,    0, "col0-");
        apply_and_log(   0, -100,    0,    0, "col1-");
        apply_and_log(   0,    0, -100,    0, "col2-");
        apply_and_log(   0,    0,    0, -100, "col3-");
        `LOG("")

        // ------------------------------------------------------------------
        // Section 3: Full vector v = [10, 20, 30, 40]
        // ------------------------------------------------------------------
        `LOG("--- Full vector v = [10, 20, 30, 40] ---")
        apply_and_log(10, 20, 30, 40, "v_full");
        $fdisplay(fd, "  Expected: i0=%0d  i1=%0d  i2=%0d  i3=%0d",
                  10+20-30-40, -10+20+30-40, 10-20+30-40, -10-20-30+40);
        $display("  Expected: i0=%0d  i1=%0d  i2=%0d  i3=%0d",
                 10+20-30-40, -10+20+30-40, 10-20+30-40, -10-20-30+40);
        `LOG("")

        // ------------------------------------------------------------------
        // Section 4: Sneak current matrix
        // ------------------------------------------------------------------
        `LOG("--- Sneak-current matrix (rows=outputs, cols=inputs) ---")
        `LOG("     v0   v1   v2   v3")
        `LOG("i0 [ +1   +1   -1   -1 ]")
        `LOG("i1 [ -1   +1   +1   -1 ]")
        `LOG("i2 [ +1   -1   +1   -1 ]")
        `LOG("i3 [ -1   -1   -1   +1 ]")
        `LOG("Every off-diagonal entry is non-zero: full sneak coupling.")
        `LOG("")

        // ------------------------------------------------------------------
        // Section 5: Cancellation sweep
        // ------------------------------------------------------------------
        `LOG("--- Cancellation sweep over {-40,-20,0,20,40}^4 ---")
        `LOG("    (flags outputs where sneak currents cancel to zero)")
        cancel_count = 0;
        for (ia = 0; ia < 5; ia = ia + 1)
        for (ib = 0; ib < 5; ib = ib + 1)
        for (ic = 0; ic < 5; ic = ic + 1)
        for (id = 0; id < 5; id = id + 1)
            check_cancel(sweep[ia], sweep[ib], sweep[ic], sweep[id]);
        $fdisplay(fd, "Total cancellation events found: %0d / 625 input combos", cancel_count);
        $display("Total cancellation events found: %0d / 625 input combos", cancel_count);
        `LOG("")

        `LOG("=== simulation complete ===")
        $fclose(fd);
        $finish;
    end

endmodule
