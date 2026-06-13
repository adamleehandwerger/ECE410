# svm_top_ihp.sdc — IHP SG13G2 top-level timing constraints
# System clock: 40 MHz (25 ns). SPI slave is asynchronous (false path).

create_clock -name clk -period 25.0 [get_ports clk]
set_propagated_clock [get_clocks clk]
set_clock_uncertainty -setup 0.5  [get_clocks clk]
set_clock_uncertainty -hold  0.25 [get_clocks clk]

# SPI inputs are asynchronous — 2-FF synchronisers inside the design
# handle metastability. Mark SPI pins as false paths from timing analysis.
set_false_path -from [get_ports spi_csn]
set_false_path -from [get_ports spi_sclk]
set_false_path -from [get_ports spi_mosi]
set_false_path -from [get_ports ram_rdata_in]

# SRAM outputs: combinational paths, constrained to half a clock period
set_output_delay -clock clk -max 5.0 [get_ports ram_addr_out]
set_output_delay -clock clk -max 5.0 [get_ports ram_ren_out]

# Result GPIOs: relaxed (not timing-critical relative to CLK)
set_output_delay -clock clk -max 10.0 [get_ports class_out]
set_output_delay -clock clk -max 10.0 [get_ports sample_rdy]
set_output_delay -clock clk -max 10.0 [get_ports done]
set_output_delay -clock clk -max 10.0 [get_ports error]
set_output_delay -clock clk -max 10.0 [get_ports error_code]
set_output_delay -clock clk -max 10.0 [get_ports irq_sample_rdy]
set_output_delay -clock clk -max 10.0 [get_ports irq_done]
set_output_delay -clock clk -max 10.0 [get_ports spi_miso]
