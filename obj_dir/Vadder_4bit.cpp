// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vadder_4bit__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vadder_4bit::Vadder_4bit(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vadder_4bit__Syms(contextp(), _vcname__, this)}
    , A{vlSymsp->TOP.A}
    , B{vlSymsp->TOP.B}
    , Cin{vlSymsp->TOP.Cin}
    , Sum{vlSymsp->TOP.Sum}
    , Cout{vlSymsp->TOP.Cout}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
    contextp()->traceBaseModelCbAdd(
        [this](VerilatedTraceBaseC* tfp, int levels, int options) { traceBaseModel(tfp, levels, options); });
}

Vadder_4bit::Vadder_4bit(const char* _vcname__)
    : Vadder_4bit(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vadder_4bit::~Vadder_4bit() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vadder_4bit___024root___eval_debug_assertions(Vadder_4bit___024root* vlSelf);
#endif  // VL_DEBUG
void Vadder_4bit___024root___eval_static(Vadder_4bit___024root* vlSelf);
void Vadder_4bit___024root___eval_initial(Vadder_4bit___024root* vlSelf);
void Vadder_4bit___024root___eval_settle(Vadder_4bit___024root* vlSelf);
void Vadder_4bit___024root___eval(Vadder_4bit___024root* vlSelf);

void Vadder_4bit::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vadder_4bit::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vadder_4bit___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vadder_4bit___024root___eval_static(&(vlSymsp->TOP));
        Vadder_4bit___024root___eval_initial(&(vlSymsp->TOP));
        Vadder_4bit___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vadder_4bit___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vadder_4bit::eventsPending() { return false; }

uint64_t Vadder_4bit::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vadder_4bit::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vadder_4bit___024root___eval_final(Vadder_4bit___024root* vlSelf);

VL_ATTR_COLD void Vadder_4bit::final() {
    Vadder_4bit___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vadder_4bit::hierName() const { return vlSymsp->name(); }
const char* Vadder_4bit::modelName() const { return "Vadder_4bit"; }
unsigned Vadder_4bit::threads() const { return 1; }
void Vadder_4bit::prepareClone() const { contextp()->prepareClone(); }
void Vadder_4bit::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vadder_4bit::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vadder_4bit___024root__trace_decl_types(VerilatedVcd* tracep);

void Vadder_4bit___024root__trace_init_top(Vadder_4bit___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vadder_4bit___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vadder_4bit___024root*>(voidSelf);
    Vadder_4bit__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(vlSymsp->name(), VerilatedTracePrefixType::SCOPE_MODULE);
    Vadder_4bit___024root__trace_decl_types(tracep);
    Vadder_4bit___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vadder_4bit___024root__trace_register(Vadder_4bit___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vadder_4bit::traceBaseModel(VerilatedTraceBaseC* tfp, int levels, int options) {
    (void)levels; (void)options;
    VerilatedVcdC* const stfp = dynamic_cast<VerilatedVcdC*>(tfp);
    if (VL_UNLIKELY(!stfp)) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vadder_4bit::trace()' called on non-VerilatedVcdC object;"
            " use --trace-fst with VerilatedFst object, and --trace-vcd with VerilatedVcd object");
    }
    stfp->spTrace()->addModel(this);
    stfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP), name(), false, 6);
    Vadder_4bit___024root__trace_register(&(vlSymsp->TOP), stfp->spTrace());
}
