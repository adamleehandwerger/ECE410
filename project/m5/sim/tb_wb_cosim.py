"""
tb_wb_cosim.py — ECE410 m5 Wishbone Interface cocotb Simulation
================================================================
Exercises user_project_wrapper through its Wishbone register interface.
ALL feature writes, configuration, and result reads go through WB registers.
The off-chip SV RAM is modeled in Python, responding to GPIO/LA signals.

Protocol per sample:
  1. Write NUM_SAMPLES = 1 (one-time)
  2. Write NUM_SV[0..4] matching sklearn SV counts (one-time)
  3. Write PARAM_WR to set gamma (one-time)
  4. Write CONTROL = 0x0A  → set vbatt_ok=1, kern_ready=1, wait 3 clocks
  5. Per sample: write CONTROL = 0x0B (start pulse)
  6. Write 256 × FIFO_DATA (feature words, Q6.10, 16-bit signed)
  7. Wait for GPIO[3] (io_out[3] = svm_done)
  8. Capture class from GPIO[2:0] (io_out[2:0] = class_out_r)

Outputs:
  asic_preds.csv  — one integer class (0-4) per test sample
  (consumed by confusion_comparison_m5.py for the ASIC column)

Run: make cosim   (from project/m5/sim/)
"""

import os, sys, warnings, math, ctypes, csv
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match svm_compute_core.sv
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS        = 10
SCALE            = 1 << FRAC_BITS   # 1024
FEATURE_DIM      = 256
FEAT_SINGLE      = 128
FEAT_10BEAT      = 64
FEAT_100RR       = 64
NUM_CLASSES      = 5
MAX_SV_PER_CLASS = 50
CLASS_NAMES      = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA    = 0.25
NORMAL_RR        = 308

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wishbone register addresses (base 0x30000000)
WB_FIFO_DATA   = 0x30000000
WB_CONTROL     = 0x30000004
WB_STATUS      = 0x30000008
WB_NUM_SAMPLES = 0x3000000C
WB_NUM_SV      = [0x30000010, 0x30000014, 0x30000018, 0x3000001C, 0x30000020]
WB_PARAM_WR    = 0x30000024
WB_WORK_RD     = 0x30000038
WB_STATUS2     = 0x3000003C

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to confusion_comparison_m5.py — same random_state → same split)
# ─────────────────────────────────────────────────────────────────────────────
def _gauss(t, mu, sig, a): return a * np.exp(-0.5 * ((t - mu) / sig) ** 2)

def _make_ms(beat_fn, rr_fn, n, noise_e, noise_b, rng):
    beats, rr_hist = [], [NORMAL_RR] * 100
    for _ in range(n):
        rr_vals = rr_fn(10, rng).astype(int)
        t = np.linspace(-0.2, 0.6, FEAT_SINGLE)
        raw = np.array([beat_fn(t, rng) for _ in rr_vals])
        sig = raw.mean(0) + rng.normal(0, noise_b, FEAT_SINGLE)
        feat_single = sig + rng.normal(0, noise_e, FEAT_SINGLE)
        rr_std  = np.std(rr_vals[:FEAT_10BEAT // 2])
        ctx10 = np.concatenate([rr_vals[:FEAT_10BEAT//2] / NORMAL_RR,
                                 np.full(FEAT_10BEAT//2, rr_std / NORMAL_RR)])
        rh = np.array(rr_hist[-100:], dtype=float) / NORMAL_RR
        ctx100 = np.concatenate([np.resize(rh, (FEAT_100RR//2,)),
                                  np.full(FEAT_100RR//2, np.std(rh))])
        beats.append(np.concatenate([feat_single, ctx10, ctx100]).astype(np.float32))
        rr_hist.extend(list(rr_vals)); rr_hist = rr_hist[-100:]
    return beats

def synth_normal(n, r):
    def beat(t, r): return (_gauss(t,-0.03,0.030,-0.15)+_gauss(t,0.00,0.035,1.10)
                             +_gauss(t,0.04,0.028,-0.12)+_gauss(t,0.25,0.08,0.30))
    def rr(nb, r): return r.normal(NORMAL_RR, 10, nb)
    return _make_ms(beat, rr, n, 0.02, 0.01, r)

def synth_pvc(n, r):
    def beat(t, r):
        w=r.uniform(0.06,0.09); s=r.choice([-1,1])
        return (_gauss(t,-0.03,w,s*1.20)+_gauss(t,0.06,w*0.85,-s*0.55)+_gauss(t,0.28,0.10,s*0.18))
    def rr(nb, r): rrs=r.normal(NORMAL_RR,12,nb); rrs[0]=r.uniform(0.6,0.8)*NORMAL_RR; return rrs
    return _make_ms(beat, rr, n, 0.03, 0.02, r)

def synth_afib(n, r):
    def beat(t, r):
        bl=sum(r.uniform(0.03,0.08)*np.sin(2*np.pi*ff*t+r.uniform(0,6.28)) for ff in r.uniform(4,10,6))
        o=r.uniform(-0.05,0.05)
        return (_gauss(t,o-0.04,0.025,-0.12)+_gauss(t,o,0.035,0.90)
                +_gauss(t,o+0.04,0.025,-0.08)+_gauss(t,o+0.30,0.09,0.25)+bl)
    def rr(nb, r): return r.uniform(0.50*NORMAL_RR,1.50*NORMAL_RR,nb)
    return _make_ms(beat, rr, n, 0.06, 0.05, r)

def synth_vt(n, r):
    def beat(t, r):
        w=r.uniform(0.10,0.16); p=r.choice([-1,1])
        return (_gauss(t,0.00,w,p*0.95)+_gauss(t,0.08,w*0.9,p*0.30)+_gauss(t,0.35,0.14,-p*0.35))
    def rr(nb, r): return r.normal(144, 4, nb)
    return _make_ms(beat, rr, n, 0.04, 0.03, r)

def synth_svt(n, r):
    def beat(t, r): return (_gauss(t,-0.02,0.030,-0.10)+_gauss(t,0.00,0.035,1.00)
                             +_gauss(t,0.04,0.025,-0.08)+_gauss(t,0.22,0.070,0.25)
                             +_gauss(t,0.18,0.040,-0.10))
    def rr(nb, r): return r.normal(120, 3, nb)
    return _make_ms(beat, rr, n, 0.03, 0.02, r)

SYNTH_FNS = [synth_normal, synth_pvc, synth_afib, synth_vt, synth_svt]

def load_mitbih_beats(max_per_class=300):
    try:
        import wfdb
    except ImportError:
        return {}
    BEAT_MAP = {'N':0,'L':0,'R':0,'e':0,'j':0,'V':1,'E':1,
                'A':2,'a':2,'J':2,'S':2,'F':3,'/':4,'f':4,'Q':4}
    records = ['100','101','102','103','104','105','106','107','108','109',
               '111','112','113','114','115','116','117','118','119','121',
               '122','123','124','200','201','202','203','205','207','208',
               '209','210','212','213','214','215','217','219','220','221',
               '222','223','228','230','231','232','233','234']
    beats = {i: [] for i in range(NUM_CLASSES)}
    for rec in records:
        if all(len(v) >= max_per_class for v in beats.values()):
            break
        try:
            import wfdb
            sig, _ = wfdb.rdsamp(rec, pn_dir='mitdb')
            ann    = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        except Exception:
            continue
        ecg = sig[:, 0]
        rr_hist, prev_idx = [NORMAL_RR]*100, 0
        for idx, sym in zip(ann.sample, ann.symbol):
            cls = BEAT_MAP.get(sym)
            if cls is None or len(beats[cls]) >= max_per_class:
                continue
            win = ecg[max(0, idx-50): idx+78]
            if len(win) < 128:
                continue
            rr = idx - prev_idx
            ctx10  = np.concatenate([np.full(FEAT_10BEAT//2, rr/NORMAL_RR),
                                     np.full(FEAT_10BEAT//2, abs(rr-NORMAL_RR)/NORMAL_RR)])
            rh = np.array(rr_hist[-100:], dtype=float)/NORMAL_RR
            ctx100 = np.concatenate([np.resize(rh,(FEAT_100RR//2,)),
                                     np.full(FEAT_100RR//2, np.std(rh))])
            beats[cls].append(np.concatenate([win[:FEAT_SINGLE].astype(np.float32), ctx10, ctx100]).astype(np.float32))
            rr_hist.append(rr); rr_hist = rr_hist[-100:]
            prev_idx = idx
    return beats

def build_dataset(n_per_class=300):
    print("\n[cosim] Loading dataset (256-dim multi-scale features)...")
    real = load_mitbih_beats(max_per_class=n_per_class)
    rng  = np.random.default_rng(42)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        rb=real.get(cls,[]); n_real=len(rb); n_syn=max(0, n_per_class-n_real)
        for b in rb: X.append(b); y.append(cls)
        for b in SYNTH_FNS[cls](n_syn, rng): X.append(b); y.append(cls)
    return np.array(X, np.float32), np.array(y, np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# Wishbone Bus Functional Model
# ─────────────────────────────────────────────────────────────────────────────
async def wb_write(dut, addr, data):
    """Single Wishbone write: assert cyc+stb+we, wait for ack, deassert."""
    dut.wbs_stb_i.value = 1
    dut.wbs_cyc_i.value = 1
    dut.wbs_we_i.value  = 1
    dut.wbs_sel_i.value = 0xF
    dut.wbs_adr_i.value = addr
    dut.wbs_dat_i.value = data & 0xFFFF_FFFF
    while not int(dut.wbs_ack_o.value):
        await RisingEdge(dut.wb_clk_i)
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_stb_i.value = 0
    dut.wbs_cyc_i.value = 0
    dut.wbs_we_i.value  = 0

async def wb_read(dut, addr):
    """Single Wishbone read: assert cyc+stb, wait for ack, return data."""
    dut.wbs_stb_i.value = 1
    dut.wbs_cyc_i.value = 1
    dut.wbs_we_i.value  = 0
    dut.wbs_sel_i.value = 0xF
    dut.wbs_adr_i.value = addr
    while not int(dut.wbs_ack_o.value):
        await RisingEdge(dut.wb_clk_i)
    val = int(dut.wbs_dat_o.value)
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_stb_i.value = 0
    dut.wbs_cyc_i.value = 0
    return val

# ─────────────────────────────────────────────────────────────────────────────
# SV RAM model — responds to GPIO sv_ram_ren / sv_ram_addr with LA sv_rdata
# ─────────────────────────────────────────────────────────────────────────────
async def sv_ram_model(dut, sv_data_q):
    """
    Emulates off-chip SV RAM (1-cycle read latency).

    The hardware drives:
      io_out[25]      = sv_ram_ren  (combinational from feat_rd_en)
      sv_ram_addr_w   = {sv_base[9:0], feat_rd_addr[7:0]}  (18 bits, internal wire)

    We respond by writing la_data_in[15:0] = sv_data_q[addr] after each clock
    where ren is high. This models the 1-cycle SRAM read latency expected by the RTL.
    """
    while True:
        await RisingEdge(dut.wb_clk_i)
        try:
            ren = int(dut.sv_ram_ren_w.value)
        except Exception:
            ren = (int(dut.io_out.value) >> 25) & 0x1
        if ren:
            try:
                addr = int(dut.sv_ram_addr_w.value)
            except Exception:
                addr = (int(dut.io_out.value) >> 10) & 0x7FFF
            word = int(sv_data_q[addr]) if addr < len(sv_data_q) else 0
            la_val = int(dut.la_data_in.value)
            dut.la_data_in.value = (la_val & ~0xFFFF) | (word & 0xFFFF)

# ─────────────────────────────────────────────────────────────────────────────
# Quantization helpers
# ─────────────────────────────────────────────────────────────────────────────
def q10_u16(x):
    return ctypes.c_uint16(int(round(x * SCALE))).value

def feats_to_q16(X):
    return (np.clip(np.round(X * SCALE).astype(np.int64), -(1<<15), (1<<15)-1)
            .astype(np.int16))

# ─────────────────────────────────────────────────────────────────────────────
# Main cocotb test
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def run_wb_cosim(dut):
    """
    Full Wishbone interface classification sweep.
    Writes all feature vectors via FIFO_DATA register, reads class labels
    from GPIO[2:0] (io_out[2:0]) after the done pulse on GPIO[3].
    Outputs: asic_preds.csv
    """
    # ── Clock & reset ────────────────────────────────────────────────────────
    cocotb.start_soon(Clock(dut.wb_clk_i, 25, unit="ns").start())

    dut.wb_rst_i.value = 1
    dut.wbs_stb_i.value = 0
    dut.wbs_cyc_i.value = 0
    dut.wbs_we_i.value  = 0
    dut.wbs_sel_i.value = 0xF
    dut.wbs_dat_i.value = 0
    dut.wbs_adr_i.value = 0x30000000
    dut.la_data_in.value = 0
    dut.io_in.value = 0

    for _ in range(12): await RisingEdge(dut.wb_clk_i)
    dut.wb_rst_i.value = 0
    for _ in range(5):  await RisingEdge(dut.wb_clk_i)

    # ── Train sklearn SVM (same params as confusion_comparison_m5.py) ────────
    print("\n[cosim] Training sklearn RBF SVM...")
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    clf = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0,
              decision_function_shape="ovr", random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"[cosim] SVs per class: {clf.n_support_}  total: {clf.n_support_.sum()}")

    # ── Build SV RAM flat array (Q6.10, 16-bit unsigned) ─────────────────────
    sv_counts = np.minimum(clf.n_support_, MAX_SV_PER_CLASS).astype(int)
    sv_total  = int(sv_counts.sum())
    sv_data_q = np.zeros(sv_total * FEATURE_DIM, dtype=np.uint16)
    sv_start_in_clf = 0
    sv_offset = 0
    for c in range(NUM_CLASSES):
        n = sv_counts[c]
        svs = clf.support_vectors_[sv_start_in_clf: sv_start_in_clf + n]
        for sv in svs:
            for fi, f in enumerate(sv):
                sv_data_q[sv_offset * FEATURE_DIM + fi] = q10_u16(f)
            sv_offset += 1
        sv_start_in_clf += clf.n_support_[c]

    print(f"[cosim] SV RAM: {sv_total} SVs × {FEATURE_DIM} features = "
          f"{len(sv_data_q)} words ({len(sv_data_q)*2/1024:.1f} KB)")

    # ── Start SV RAM model background coroutine ───────────────────────────────
    cocotb.start_soon(sv_ram_model(dut, sv_data_q))

    # ── Configure via WB registers (one-time setup) ───────────────────────────
    print("[cosim] Configuring DUT via Wishbone...")
    await wb_write(dut, WB_NUM_SAMPLES, 1)
    for c in range(NUM_CLASSES):
        await wb_write(dut, WB_NUM_SV[c], sv_counts[c])
    # Write gamma via PARAM_WR: [19]=en [18:16]=addr(0=gamma) [15:0]=Q6.10 value
    gamma_q = q10_u16(DEFAULT_GAMMA)
    await wb_write(dut, WB_PARAM_WR, (1 << 19) | (0 << 16) | gamma_q)

    # ── Set vbatt_ok=1, kern_ready=1 persistently (CONTROL = 0x0A) ───────────
    # vbatt_ok passes through a 2-FF synchronizer; must be stable before start.
    await wb_write(dut, WB_CONTROL, 0x0A)
    for _ in range(6):  # 6 cycles > 2-FF sync delay
        await RisingEdge(dut.wb_clk_i)

    # ── Classify all test samples through the WB interface ───────────────────
    X_te_q = feats_to_q16(X_te)
    N_eval  = len(X_te_q)
    preds   = []

    # Upper bound on compute cycles per sample (generous): SVs × dims + FIFO + overhead
    MAX_WAIT = sv_total * FEATURE_DIM + FEATURE_DIM * 4 + 2048

    print(f"[cosim] Classifying {N_eval} samples (max {MAX_WAIT} cycles each)...")

    for i, feat_q in enumerate(X_te_q):
        # Step 1: Pulse start=1 (auto-clears after 1 clock); keep vbatt_ok=1, kern_ready=1
        await wb_write(dut, WB_CONTROL, 0x0B)

        # Step 2: Write 256 feature words to FIFO_DATA
        for f in feat_q:
            await wb_write(dut, WB_FIFO_DATA, int(f) & 0xFFFF)

        # Step 3: Wait for done via GPIO[3] = io_out[3] = svm_done
        class_out = None
        for _ in range(MAX_WAIT):
            await RisingEdge(dut.wb_clk_i)
            io_val = int(dut.io_out.value)
            if io_val & 0x8:          # bit 3 = svm_done (1-cycle pulse)
                class_out = io_val & 0x7   # bits [2:0] = class_out_r
                break

        if class_out is None:
            # Timeout fallback: read work_ram[0] via WORK_RD → STATUS2
            await wb_write(dut, WB_WORK_RD, 0)
            s2 = await wb_read(dut, WB_STATUS2)
            class_out = s2 & 0x7
            print(f"[cosim] WARNING: sample {i} timed out — fallback class={class_out}")

        preds.append(class_out)

        if (i + 1) % 60 == 0:
            acc_so_far = sum(p == t for p, t in zip(preds, y_te[:len(preds)])) / len(preds)
            print(f"[cosim]   {i+1}/{N_eval} classified  running acc={acc_so_far:.2%}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_csv = os.path.join(SCRIPT_DIR, "asic_preds.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for p in preds:
            writer.writerow([p])

    # Final accuracy
    acc = sum(p == t for p, t in zip(preds, y_te)) / N_eval
    print(f"\n[cosim] ASIC accuracy: {acc:.4f}  ({int(acc*N_eval)}/{N_eval})")
    print(f"[cosim] Saved {len(preds)} predictions → {out_csv}")
    print(f"[cosim] Run: python3 confusion_comparison_m5.py  to generate the plot")
