// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals

#include "verilated_vcd_c.h"
#include "Vadder_4bit__Syms.h"


void Vadder_4bit___024root__trace_chg_0_sub_0(Vadder_4bit___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vadder_4bit___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vadder_4bit___024root__trace_chg_0\n"); );
    // Body
    Vadder_4bit___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vadder_4bit___024root*>(voidSelf);
    Vadder_4bit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    Vadder_4bit___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vadder_4bit___024root__trace_chg_0_sub_0(Vadder_4bit___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vadder_4bit___024root__trace_chg_0_sub_0\n"); );
    Vadder_4bit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 0);
    bufp->chgCData(oldp+0,(vlSelfRef.A),4);
    bufp->chgCData(oldp+1,(vlSelfRef.B),4);
    bufp->chgBit(oldp+2,(vlSelfRef.Cin));
    bufp->chgCData(oldp+3,(vlSelfRef.Sum),4);
    bufp->chgBit(oldp+4,(vlSelfRef.Cout));
    bufp->chgCData(oldp+5,(vlSelfRef.adder_4bit__DOT__result),5);
}

void Vadder_4bit___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vadder_4bit___024root__trace_cleanup\n"); );
    // Locals
    VlUnpacked<CData/*0:0*/, 1> __Vm_traceActivity;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vm_traceActivity[__Vi0] = 0;
    }
    // Body
    Vadder_4bit___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vadder_4bit___024root*>(voidSelf);
    Vadder_4bit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    vlSymsp->__Vm_activity = false;
    __Vm_traceActivity[0U] = 0U;
}
