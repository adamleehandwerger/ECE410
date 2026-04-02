// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vadder_4bit.h for the primary calling header

#ifndef VERILATED_VADDER_4BIT___024ROOT_H_
#define VERILATED_VADDER_4BIT___024ROOT_H_  // guard

#include "verilated.h"


class Vadder_4bit__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vadder_4bit___024root final {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(A,3,0);
    VL_IN8(B,3,0);
    VL_IN8(Cin,0,0);
    VL_OUT8(Sum,3,0);
    VL_OUT8(Cout,0,0);
    CData/*4:0*/ adder_4bit__DOT__result;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VstlPhaseResult;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __VicoPhaseResult;
    VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VicoTriggered;

    // INTERNAL VARIABLES
    Vadder_4bit__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vadder_4bit___024root(Vadder_4bit__Syms* symsp, const char* namep);
    ~Vadder_4bit___024root();
    VL_UNCOPYABLE(Vadder_4bit___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
