// SPDX-FileCopyrightText: © 2026 Adam Handwerger
// SPDX-License-Identifier: Apache-2.0
//
// chip_top.sv — Full-chip wrapper for svm_top_ihp with IHP SG13G2 pad ring.
// Pad ring: sg13g2_IOPadIn / sg13g2_IOPadOut30mA / sg13g2_IOPadVdd/Vss/IOVdd/IOVss.
// Bondpads: sg13g2_ip__bondpad_70x70 (added by LibreLane Chip flow).
// Sealring and fill: added automatically by LibreLane Chip flow.

`default_nettype none

module chip_top (
    `ifdef USE_POWER_PINS
    inout wire IOVDD,
    inout wire IOVSS,
    inout wire VDD,
    inout wire VSS,
    `endif

    // ── Signal pads (inout per IHP pad-cell convention) ───────────
    inout wire        clk_PAD,
    inout wire        rst_n_PAD,

    // SPI
    inout wire        spi_csn_PAD,
    inout wire        spi_sclk_PAD,
    inout wire        spi_mosi_PAD,
    inout wire        spi_miso_PAD,

    // Off-chip SRAM read data (16 input pads)
    inout wire [15:0] ram_rdata_PAD,

    // Off-chip SRAM address + read-enable (20 output pads)
    inout wire [18:0] ram_addr_PAD,
    inout wire        ram_ren_PAD,

    // Classification result (3 output pads)
    inout wire [2:0]  class_out_PAD,

    // Status outputs
    inout wire        sample_rdy_PAD,
    inout wire        done_PAD,
    inout wire        error_PAD,
    inout wire [3:0]  error_code_PAD,

    // Interrupt outputs
    inout wire        irq_sample_rdy_PAD,
    inout wire        irq_done_PAD
);

    // ── Internal core wires ───────────────────────────────────────
    wire        clk_core, rst_n_core;
    wire        spi_csn_core, spi_sclk_core, spi_mosi_core;
    wire        spi_miso_core;
    wire [15:0] ram_rdata_core;
    wire [18:0] ram_addr_core;
    wire        ram_ren_core;
    wire [2:0]  class_out_core;
    wire        sample_rdy_core, done_core, error_core;
    wire [3:0]  error_code_core;
    wire        irq_sample_rdy_core, irq_done_core;

    // ── SOUTH pads: clk, rst_n, SPI, power ───────────────────────
    sg13g2_IOPadIn clk_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(clk_PAD), .p2c(clk_core)
    );
    sg13g2_IOPadIn rst_n_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(rst_n_PAD), .p2c(rst_n_core)
    );
    sg13g2_IOPadIn spi_csn_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(spi_csn_PAD), .p2c(spi_csn_core)
    );
    sg13g2_IOPadIn spi_sclk_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(spi_sclk_PAD), .p2c(spi_sclk_core)
    );
    sg13g2_IOPadIn spi_mosi_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(spi_mosi_PAD), .p2c(spi_mosi_core)
    );
    sg13g2_IOPadOut30mA spi_miso_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(spi_miso_PAD), .c2p(spi_miso_core)
    );

    (* keep *) sg13g2_IOPadVdd south_vdd_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadVss south_vss_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadIOVdd south_iovdd_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadIOVss south_iovss_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );

    // ── EAST pads: ram_rdata_in[15:0] + power ────────────────────
    generate
        genvar ri;
        for (ri = 0; ri < 16; ri++) begin : g_rdata
            sg13g2_IOPadIn rdata_pad (
                .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
                .pad(ram_rdata_PAD[ri]), .p2c(ram_rdata_core[ri])
            );
        end
    endgenerate
    (* keep *) sg13g2_IOPadVdd east_vdd_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadVss east_vss_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );

    // ── NORTH pads: ram_addr_out[18:0] + ram_ren_out + power ─────
    generate
        genvar ai;
        for (ai = 0; ai < 19; ai++) begin : g_addr
            sg13g2_IOPadOut30mA addr_pad (
                .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
                .pad(ram_addr_PAD[ai]), .c2p(ram_addr_core[ai])
            );
        end
    endgenerate
    sg13g2_IOPadOut30mA ram_ren_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(ram_ren_PAD), .c2p(ram_ren_core)
    );
    (* keep *) sg13g2_IOPadVdd north_vdd_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadVss north_vss_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );

    // ── WEST pads: status + IRQ + power ──────────────────────────
    generate
        genvar ci;
        for (ci = 0; ci < 3; ci++) begin : g_class
            sg13g2_IOPadOut30mA class_pad (
                .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
                .pad(class_out_PAD[ci]), .c2p(class_out_core[ci])
            );
        end
    endgenerate
    sg13g2_IOPadOut30mA sample_rdy_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(sample_rdy_PAD), .c2p(sample_rdy_core)
    );
    sg13g2_IOPadOut30mA done_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(done_PAD), .c2p(done_core)
    );
    sg13g2_IOPadOut30mA error_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(error_PAD), .c2p(error_core)
    );
    generate
        genvar ei;
        for (ei = 0; ei < 4; ei++) begin : g_ecode
            sg13g2_IOPadOut30mA ecode_pad (
                .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
                .pad(error_code_PAD[ei]), .c2p(error_code_core[ei])
            );
        end
    endgenerate
    sg13g2_IOPadOut30mA irq_srdy_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(irq_sample_rdy_PAD), .c2p(irq_sample_rdy_core)
    );
    sg13g2_IOPadOut30mA irq_done_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
        .pad(irq_done_PAD), .c2p(irq_done_core)
    );
    (* keep *) sg13g2_IOPadIOVdd west_iovdd_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );
    (* keep *) sg13g2_IOPadIOVss west_iovss_pad (
        .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
    );

    // ── Core design ───────────────────────────────────────────────
    svm_top_ihp u_svm_top (
        .clk            (clk_core),
        .rst_n          (rst_n_core),
        .spi_csn        (spi_csn_core),
        .spi_sclk       (spi_sclk_core),
        .spi_mosi       (spi_mosi_core),
        .spi_miso       (spi_miso_core),
        .ram_rdata_in   (ram_rdata_core),
        .ram_addr_out   (ram_addr_core),
        .ram_ren_out    (ram_ren_core),
        .class_out      (class_out_core),
        .sample_rdy     (sample_rdy_core),
        .done           (done_core),
        .error          (error_core),
        .error_code     (error_code_core),
        .irq_sample_rdy (irq_sample_rdy_core),
        .irq_done       (irq_done_core)
    );

endmodule

`default_nettype wire
