#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vadder_4bit.h"
#include <iostream>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    
    // Instantiate the module
    Vadder_4bit* dut = new Vadder_4bit;
    
    // Setup VCD tracing
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("waveform.vcd");
    
    vluint64_t sim_time = 0;
    
    std::cout << "Running simulation with waveform generation...\n";
    
    // Test case 1: 3 + 5 = 8
    dut->A = 0b0011;
    dut->B = 0b0101;
    dut->Cin = 0;
    dut->eval();
    tfp->dump(sim_time++);
    
    // Test case 2: 15 + 1 = 16 (overflow)
    dut->A = 0b1111;
    dut->B = 0b0001;
    dut->Cin = 0;
    dut->eval();
    tfp->dump(sim_time++);
    
    // Test case 3: 10 + 6 + 1 = 17 (overflow)
    dut->A = 0b1010;
    dut->B = 0b0110;
    dut->Cin = 1;
    dut->eval();
    tfp->dump(sim_time++);
    
    // Test case 4: 7 + 8 = 15
    dut->A = 0b0111;
    dut->B = 0b1000;
    dut->Cin = 0;
    dut->eval();
    tfp->dump(sim_time++);
    
    // More test cases for better waveform
    for (int i = 0; i < 10; i++) {
        dut->A = i % 16;
        dut->B = (i * 3) % 16;
        dut->Cin = i % 2;
        dut->eval();
        tfp->dump(sim_time++);
    }
    
    std::cout << "Simulation complete! Waveform saved to waveform.vcd\n";
    
    // Cleanup
    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}