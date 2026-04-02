// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vadder_4bit.h for the primary calling header

#include "Vadder_4bit__pch.h"

void Vadder_4bit___024root___ctor_var_reset(Vadder_4bit___024root* vlSelf);

Vadder_4bit___024root::Vadder_4bit___024root(Vadder_4bit__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vadder_4bit___024root___ctor_var_reset(this);
}

void Vadder_4bit___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vadder_4bit___024root::~Vadder_4bit___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
