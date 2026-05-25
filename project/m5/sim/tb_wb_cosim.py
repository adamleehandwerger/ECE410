"""
tb_wb_cosim.py — ECE410 m5 Batch Architecture cocotb Simulation  (v8)
======================================================================
Exercises user_project_wrapper with the batch RBF-SVM architecture.

Host pre-loads both the SV matrix and the input matrix into a Python
SRAM model, fires start once, then the ASIC autonomously classifies
all samples.  Results are captured per-sample via GPIO.

Off-chip RAM layout  (row × FEATURE_DIM, FEATURE_DIM = 256):
  Rows  0..249         SV matrix      (250 × 256 × 2 B = 128 KB)
  Rows  250..1249      input batch    (1000 × 256 × 2 B = 512 KB max)

GPIO signals (io_out):
  [2:0]   class_out   — stable when sample_rdy fires
  [3]     sample_rdy  — pulses one cycle per classified heartbeat
  [4]     svm_done    — pulses once at end of batch
  [28:10] ram_addr    — 19-bit off-chip address (output from ASIC)
  [29]    ram_ren     — off-chip read enable   (output from ASIC)

la_data_in[15:0] = ram_rdata  (host drives, 1-cycle latency after ren)

Outputs:
  asic_preds.csv — one integer class (0-4) per test sample
"""

import os, warnings, ctypes, csv
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match svm_compute_core.sv parameters
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS        = 10
SCALE            = 1 << FRAC_BITS        # 1024
FEATURE_DIM      = 256
FEAT_SINGLE      = 128
FEAT_10BEAT      = 64
FEAT_100RR       = 64
NUM_CLASSES      = 5
MAX_SV_PER_CLASS = 50
NUM_SV_ROWS      = 250   # RTL NUM_SV — input matrix starts at this row index
MAX_BATCH_SIZE   = 1000  # RTL MAX_BATCH_SIZE
CLASS_NAMES      = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA    = 0.25
NORMAL_RR        = 308

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wishbone addresses (base 0x3000_0000) — FIFO_DATA removed in v8
WB_CONTROL     = 0x30000004
WB_NUM_SAMPLES = 0x3000000C
WB_NUM_SV      = [0x30000010, 0x30000014, 0x30000018, 0x3000001C, 0x30000020]
WB_PARAM_WR    = 0x30000024

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset  (identical generator to confusion_comparison_m5.py)
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
            beats[cls].append(np.concatenate(
                [win[:FEAT_SINGLE].astype(np.float32), ctx10, ctx100]).astype(np.float32))
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
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def q10_u16(x):
    """Float → unsigned Q6.10 bit pattern (for SV features and gamma)."""
    return ctypes.c_uint16(int(round(x * SCALE))).value

def feats_to_q16(X):
    """Float array → int16 Q6.10 (for input feature matrix)."""
    return (np.clip(np.round(X * SCALE).astype(np.int64), -(1<<15), (1<<15)-1)
            .astype(np.int16))

# ─────────────────────────────────────────────────────────────────────────────
# Wishbone BFM
# ─────────────────────────────────────────────────────────────────────────────
async def wb_write(dut, addr, data):
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
# Unified off-chip RAM model
# ─────────────────────────────────────────────────────────────────────────────
async def ram_model(dut, ram):
    """
    Emulates off-chip SRAM serving both SVs and input vectors (1-cycle latency).

    Address map (row × FEATURE_DIM):
      Rows 0..NUM_SV_ROWS-1      SV matrix
      Rows NUM_SV_ROWS..1249     input matrix

    Monitors io_out[29]=ram_ren, io_out[28:10]=ram_addr.
    Responds on la_data_in[15:0] = ram[addr] at the next delta after each ren edge.
    Falls back to reading internal signals directly if the hierarchy is accessible.
    """
    while True:
        await RisingEdge(dut.wb_clk_i)
        try:
            ren  = int(dut.u_svm.ram_ren.value)
            addr = int(dut.u_svm.ram_addr.value)
        except Exception:
            io_val = int(dut.io_out.value)
            ren    = (io_val >> 29) & 0x1
            addr   = (io_val >> 10) & 0x7FFFF   # bits [28:10] = 19 bits
        if ren:
            word   = ram.get(addr, 0)
            la_val = int(dut.la_data_in.value)
            dut.la_data_in.value = (la_val & ~0xFFFF) | (word & 0xFFFF)

# ─────────────────────────────────────────────────────────────────────────────
# RAM builder helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_sv_ram(ram, clf, sv_counts):
    """Populate SV rows (0..NUM_SV_ROWS-1) in the shared RAM dict."""
    sv_row = 0
    sv_offset_in_clf = 0
    for c in range(NUM_CLASSES):
        n   = sv_counts[c]
        svs = clf.support_vectors_[sv_offset_in_clf: sv_offset_in_clf + n]
        for sv in svs:
            base = sv_row * FEATURE_DIM
            for fi, f in enumerate(sv):
                ram[base + fi] = q10_u16(f)
            sv_row += 1
        sv_offset_in_clf += clf.n_support_[c]

def load_input_ram(ram, X_batch_q):
    """Populate input rows (NUM_SV_ROWS..) in the shared RAM dict."""
    for sample_idx, feat_q in enumerate(X_batch_q):
        base = (NUM_SV_ROWS + sample_idx) * FEATURE_DIM
        for fi, f in enumerate(feat_q):
            ram[base + fi] = int(f) & 0xFFFF

def clear_input_ram(ram, n_samples):
    """Remove input rows from the shared RAM dict between batches."""
    for sample_idx in range(n_samples):
        base = (NUM_SV_ROWS + sample_idx) * FEATURE_DIM
        for fi in range(FEATURE_DIM):
            ram.pop(base + fi, None)

# ─────────────────────────────────────────────────────────────────────────────
# Main cocotb test
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def run_batch_cosim(dut):
    """
    Batch RBF-SVM classification sweep.

    Protocol:
      1. Train sklearn SVM, build sv_counts.
      2. One-time WB config: NUM_SV, PARAM_WR, CONTROL=vbatt_ok.
      3. For each batch of up to MAX_BATCH_SIZE samples:
         a. Populate shared RAM (SVs + input vectors).
         b. Write NUM_SAMPLES, fire CONTROL=start.
         c. Capture class_out on every sample_rdy pulse (GPIO[3]).
         d. Stop after svm_done (GPIO[4]) or N_batch captures.
    """
    # ── Clock & reset ─────────────────────────────────────────────────────────
    cocotb.start_soon(Clock(dut.wb_clk_i, 25, unit="ns").start())
    dut.wb_rst_i.value  = 1
    dut.wbs_stb_i.value = 0
    dut.wbs_cyc_i.value = 0
    dut.wbs_we_i.value  = 0
    dut.wbs_sel_i.value = 0xF
    dut.wbs_dat_i.value = 0
    dut.wbs_adr_i.value = WB_CONTROL
    dut.la_data_in.value = 0
    dut.io_in.value      = 0
    for _ in range(12): await RisingEdge(dut.wb_clk_i)
    dut.wb_rst_i.value = 0
    for _ in range(5):  await RisingEdge(dut.wb_clk_i)

    # ── Train sklearn SVM ─────────────────────────────────────────────────────
    print("\n[cosim] Training sklearn RBF SVM...")
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    clf = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0,
              decision_function_shape="ovr", random_state=42)
    clf.fit(X_tr, y_tr)
    sv_counts = np.minimum(clf.n_support_, MAX_SV_PER_CLASS).astype(int)
    sv_total  = int(sv_counts.sum())
    print(f"[cosim] SVs per class: {sv_counts}  total: {sv_total}")

    X_te_q = feats_to_q16(X_te)
    N_eval  = len(X_te_q)

    # ── One-time WB config ────────────────────────────────────────────────────
    print("[cosim] Configuring DUT...")
    for c in range(NUM_CLASSES):
        await wb_write(dut, WB_NUM_SV[c], int(sv_counts[c]))
    gamma_q = q10_u16(DEFAULT_GAMMA)
    await wb_write(dut, WB_PARAM_WR, (1 << 19) | (0 << 16) | gamma_q)
    # Assert vbatt_ok=1 (bit 1) and wait for 2-FF synchroniser
    await wb_write(dut, WB_CONTROL, 0x02)
    for _ in range(6): await RisingEdge(dut.wb_clk_i)

    # ── Shared RAM dict + persistent background RAM model ─────────────────────
    ram = {}
    load_sv_ram(ram, clf, sv_counts)
    cocotb.start_soon(ram_model(dut, ram))
    print(f"[cosim] SV RAM loaded: {sv_total} SVs × {FEATURE_DIM} features")

    # ── Process in batches of MAX_BATCH_SIZE ──────────────────────────────────
    all_preds = []
    batch_start = 0

    # Generous per-sample cycle budget: LOAD_INPUT + sv_total × (dist+kernel) + overhead
    cycles_per_sample = FEATURE_DIM + 2 + sv_total * (FEATURE_DIM + 22) + 10

    while batch_start < N_eval:
        batch_end = min(batch_start + MAX_BATCH_SIZE, N_eval)
        X_batch   = X_te_q[batch_start:batch_end]
        N_batch   = batch_end - batch_start

        # Load this batch's input vectors into RAM
        load_input_ram(ram, X_batch)

        # Write batch size and fire start (vbatt_ok=1 + start=1 → CONTROL = 0x03)
        await wb_write(dut, WB_NUM_SAMPLES, N_batch)
        await wb_write(dut, WB_CONTROL, 0x03)

        print(f"[cosim] Batch {batch_start}..{batch_end-1} started "
              f"({N_batch} samples, ~{N_batch * cycles_per_sample:,} cycles)")

        # ── Capture per-sample results ─────────────────────────────────────────
        # sample_rdy = io_out[3]  (1-cycle pulse per heartbeat)
        # svm_done   = io_out[4]  (1-cycle pulse at end of batch)
        # class_out  = io_out[2:0] (stable on same cycle as sample_rdy)
        batch_preds = []
        MAX_CYCLES  = N_batch * cycles_per_sample + 10_000

        for cyc in range(MAX_CYCLES):
            await RisingEdge(dut.wb_clk_i)
            io_val = int(dut.io_out.value)

            if (io_val >> 3) & 0x1:               # sample_rdy
                batch_preds.append(io_val & 0x7)
                if len(batch_preds) == N_batch:
                    break

            if (io_val >> 4) & 0x1:               # svm_done (batch done)
                if len(batch_preds) < N_batch:
                    print(f"[cosim] WARNING: svm_done but only "
                          f"{len(batch_preds)}/{N_batch} captures")
                break
        else:
            print(f"[cosim] ERROR: timeout after {MAX_CYCLES} cycles "
                  f"({len(batch_preds)}/{N_batch} captured)")

        all_preds.extend(batch_preds)
        clear_input_ram(ram, N_batch)
        batch_start = batch_end

        acc = sum(p == t for p, t in zip(all_preds, y_te[:len(all_preds)])) / len(all_preds)
        print(f"[cosim] {len(all_preds)}/{N_eval} classified   running acc = {acc:.2%}")

    # ── Write predictions CSV ─────────────────────────────────────────────────
    out_csv = os.path.join(SCRIPT_DIR, "asic_preds.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for p in all_preds:
            writer.writerow([p])

    acc = sum(p == t for p, t in zip(all_preds, y_te)) / N_eval
    print(f"\n[cosim] ASIC accuracy : {acc:.4f}  ({int(acc*N_eval)}/{N_eval})")
    print(f"[cosim] Saved {len(all_preds)} predictions → {out_csv}")
    print(f"[cosim] Run: python3 confusion_comparison_m5.py  to generate the plot")
