# chip_top.sdc — Timing constraints for full-chip harden (Chip flow)
# Clock arrives through sg13g2_IOPadIn (clk_pad); CLOCK_NET = clk_pad/p2c

set clk_period 25.0

# Primary clock on the core-side net after the input pad
create_clock [get_nets clk_pad/p2c] -name clk -period $clk_period

# SPI clock: treated as asynchronous to clk; false path between clock domains
set_clock_groups -asynchronous -group {clk} -group {spi_sclk_pad/p2c}

# Input delays (relative to clk, worst-case combo path through pad)
set_input_delay  -clock clk -max [expr {$clk_period * 0.4}] [get_ports {rst_n_PAD spi_csn_PAD spi_mosi_PAD ram_rdata_PAD}]
set_input_delay  -clock clk -min 0.5                         [get_ports {rst_n_PAD spi_csn_PAD spi_mosi_PAD ram_rdata_PAD}]

# Output delays
set_output_delay -clock clk -max [expr {$clk_period * 0.4}] [get_ports {spi_miso_PAD ram_addr_PAD ram_ren_PAD class_out_PAD sample_rdy_PAD done_PAD error_PAD error_code_PAD irq_sample_rdy_PAD irq_done_PAD}]
set_output_delay -clock clk -min 0.0                         [get_ports {spi_miso_PAD ram_addr_PAD ram_ren_PAD class_out_PAD sample_rdy_PAD done_PAD error_PAD error_code_PAD irq_sample_rdy_PAD irq_done_PAD}]

# False paths from asynchronous resets
set_false_path -from [get_ports rst_n_PAD]

# Pad timing — no timing through the pad cells themselves
set_false_path -through [get_cells -hierarchical -filter {IS_SEQUENTIAL == 0 && REF_NAME =~ sg13g2_IOPad*}]
