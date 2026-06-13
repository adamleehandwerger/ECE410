"""
tb_spi_cosim.py — ECE410 Milestone 6 SPI cosimulation testbench
================================================================
Cocotb testbench for svm_top_ihp (IHP SG13G2 standalone target).

Tests the full SPI configuration path + SRAM model + batch classification
accuracy against PhysioNet ground truth.

SPI: CPOL=0 CPHA=0, 40-bit frames (8-bit addr + 32-bit data), MSB first.
  Write: CS# low | 0b0_addr[6:0] (8-bit) | data[31:0] | CS# high
  Read:  CS# low | 0b1_addr[6:0] (8-bit) | data[31:0] clocked out MISO | CS# high

SPI register map:
  0x01 RW  CONTROL      [0]=start(auto-clear) [1]=vbatt_ok [2]=vbatt_warn
  0x02 RO  STATUS       [0]=done [1]=error [5:2]=error_code [8:6]=class [9]=sample_rdy
  0x03 RW  NUM_SAMPLES  [9:0]
  0x04-08 RW NUM_SV[0-4] [7:0] default=120
  0x09 WO  PARAM_WR     [19]=en [18:16]=addr [15:0]=data
  0x0A WO  ALPHA_WR     [25:16]=sv_idx(10-bit) [15:0]=alpha Q6.10

SRAM address map (word-addressed, 16 bits per word):
  Words 0 .. 600*256-1    SV matrix     (600 SVs x 256 features)
  Words 600*256 .. end    Input matrix  (N_batch x 256 features)

Outputs:
  ../sim/asic_preds.csv — one integer class (0-4) per test sample

Real ECG data only — raises RuntimeError if wfdb / PhysioNet unavailable.
"""

import os, sys, warnings, ctypes, csv
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match svm_top_ihp / svm_compute_core parameters
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS    = 10
SCALE        = 1 << FRAC_BITS       # 1024
FEATURE_DIM  = 256
FEAT_SINGLE  = 128
FEAT_10BEAT  = 64
FEAT_100RR   = 64
NUM_CLASSES  = 5
NUM_SV_ROWS  = 600                  # RTL NUM_SV — input matrix starts at this row
MAX_BATCH    = 1000
CLASS_NAMES  = ["Normal", "PVC", "AFib", "VT", "SVT"]
SV_ALLOC     = [120, 120, 120, 120, 120]   # uniform optimal at 600 SVs
DEFAULT_GAMMA = float(os.environ.get("COSIM_GAMMA", "0.25"))
N_EVAL_OVERRIDE = int(os.environ.get("COSIM_N_EVAL", "0"))   # 0 = full test set
NORMAL_RR    = 308
HALF_SINGLE  = FEAT_SINGLE // 2
HALF_10BEAT  = FEAT_10BEAT // 2
_BEAT_SYMS   = set("NLReEjJAaSVF/fQ")

# SPI register addresses
SPI_CONTROL     = 0x01
SPI_STATUS      = 0x02
SPI_NUM_SAMPLES = 0x03
SPI_NUM_SV      = [0x04, 0x05, 0x06, 0x07, 0x08]
SPI_PARAM_WR    = 0x09
SPI_ALPHA_WR    = 0x0A

# SPI timing: 10 system-clock cycles per SPI half-period → safe for 2-FF sync
SPI_HALF_PER_CYCLES = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR    = os.path.join(SCRIPT_DIR, "..", "sim")

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (256-dim multi-scale: 128 beat + 64 mean10 + 64 RR)
# ─────────────────────────────────────────────────────────────────────────────
def extract_multiscale(sig, all_beat_samples, beat_idx):
    s = int(all_beat_samples[beat_idx]); n = len(sig)
    if s < HALF_SINGLE or s + HALF_SINGLE > n:
        return None
    seg1 = sig[s - HALF_SINGLE : s + HALF_SINGLE].copy()
    pk1  = np.max(np.abs(seg1))
    if pk1 < 1e-6: return None
    seg1 = (seg1 / pk1).astype(np.float32)

    half10 = 10 // 2
    i0 = max(0, beat_idx - half10); i1 = min(len(all_beat_samples), beat_idx + half10)
    segs2 = []
    for bi in range(i0, i1):
        bs = int(all_beat_samples[bi])
        if bs >= HALF_10BEAT and bs + HALF_10BEAT <= n:
            seg = sig[bs - HALF_10BEAT : bs + HALF_10BEAT].astype(np.float32)
            pk2 = np.max(np.abs(seg))
            if pk2 > 1e-6: segs2.append(seg / pk2)
    if not segs2: return None
    seg2 = np.mean(segs2, axis=0).astype(np.float32)

    j0 = max(0, beat_idx - 100)
    rr_raw = np.diff(all_beat_samples[j0 : beat_idx + 1]).astype(np.float32)
    if len(rr_raw) < 2: return None
    rr_norm = np.clip(rr_raw / NORMAL_RR, 0.0, 2.0)
    seg3 = np.interp(np.linspace(0, 1, FEAT_100RR),
                     np.linspace(0, 1, len(rr_norm)), rr_norm).astype(np.float32)
    return np.concatenate([seg1, seg2, seg3])


PHYSIONET_CACHE = os.path.expanduser("~/.physionet_cache")

def _rdrecord_cached(rec, pn_dir):
    import wfdb, wfdb.io.download as wdl
    local_dir = os.path.join(PHYSIONET_CACHE, pn_dir)
    hea_path  = os.path.join(local_dir, rec + ".hea")
    if not os.path.exists(hea_path):
        os.makedirs(local_dir, exist_ok=True)
        wdl.dl_files(pn_dir, local_dir, [rec + ".hea"], keep_subdirs=False)
        hdr = wfdb.rdheader(os.path.join(local_dir, rec))
        dat_files = list({seg + ".dat" for seg in (
                          hdr.seg_name if hasattr(hdr, 'seg_name') and hdr.seg_name
                          else [rec])})
        wdl.dl_files(pn_dir, local_dir,
                     dat_files + [rec + ".atr"],
                     keep_subdirs=False)
    r   = wfdb.rdrecord(os.path.join(local_dir, rec))
    ann = wfdb.rdann(os.path.join(local_dir, rec), 'atr')
    return r, ann


def load_mitbih_beats(max_per_class=300):
    try:
        import wfdb
    except ImportError:
        raise RuntimeError("wfdb not installed: pip install wfdb")
    BMAP = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "V": 1, "E": 1,
            "F": 3,
            "A": 4, "a": 4, "J": 4, "S": 4, "/": 4, "f": 4, "Q": 4}
    sources = [
        (['100','101','102','103','104','105','106','107','108','109',
          '111','112','113','114','115','116','117','118','119','121',
          '122','123','124','200','201','202','203','205','207','208',
          '209','210','212','213','214','215','217','219','220','221',
          '222','223','228','230','231','232','233','234'], 'mitdb'),
        ([f'e{i:04d}' for i in range(1, 79)], 'svdb'),
        ([f'I{i:02d}' for i in range(1, 76)], 'incartdb'),
    ]
    beats = {i: [] for i in range(NUM_CLASSES)}
    for rec_list, pn_dir in sources:
        if all(len(v) >= max_per_class for v in beats.values()): break
        print(f"[cosim]   Scanning {pn_dir} ({len(rec_list)} records)...")
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in beats.values()): break
            try:
                r, ann = _rdrecord_cached(rec, pn_dir)
            except Exception: continue
            sig = r.p_signal[:, 0].astype(np.float32)
            beat_samp_list, beat_sym_list = [], []
            for s, sym in zip(ann.sample, ann.symbol):
                if sym in _BEAT_SYMS:
                    beat_samp_list.append(s); beat_sym_list.append(sym)
            if len(beat_samp_list) < 3: continue
            all_beat_samples = np.array(beat_samp_list, dtype=np.int32)
            afib_regions = []
            if hasattr(ann, 'aux_note'):
                in_afib = False; afib_start = None
                for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                    if sym == '+' and aux:
                        if '(AFIB' in aux: in_afib = True; afib_start = s
                        elif in_afib and '(' in aux:
                            afib_regions.append((afib_start, s)); in_afib = False
                if in_afib and afib_start is not None:
                    afib_regions.append((afib_start, len(sig)))
            for beat_idx, (s_idx, sym) in enumerate(
                    zip(all_beat_samples.tolist(), beat_sym_list)):
                in_afib_flag = any(a0 <= s_idx <= a1 for a0, a1 in afib_regions)
                if in_afib_flag and sym in ('N','L','R','e','j','V','A','a','J','S'):
                    cls = 2
                elif sym in BMAP: cls = BMAP[sym]
                else: continue
                if len(beats[cls]) >= max_per_class: continue
                feat = extract_multiscale(sig, all_beat_samples, beat_idx)
                if feat is not None: beats[cls].append(feat)
    return beats


def build_dataset(n_per_class=300):
    import tempfile
    cache_path = os.path.join(tempfile.gettempdir(),
                              f"cosim_cache_ecg_n{n_per_class}_d256.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
        print(f"[cosim] Dataset from cache: {counts}  total={sum(counts)}")
        return X.astype(np.float32), y.astype(np.int32)
    print("[cosim] Loading ECG (MIT-BIH + SVDB + INCART) — real data only...")
    real = load_mitbih_beats(max_per_class=n_per_class)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for b in real.get(cls, []):
            X.append(b); y.append(cls)
    if not X:
        raise RuntimeError("No real beats found — install wfdb and check network access.")
    counts = [len(real.get(c, [])) for c in range(NUM_CLASSES)]
    print(f"[cosim] Real beats per class: {counts}  total={sum(counts)}")
    Xa, ya = np.array(X, np.float32), np.array(y, np.int32)
    np.savez_compressed(cache_path, X=Xa, y=ya)
    print(f"[cosim] Cached → {cache_path}")
    return Xa, ya

# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def q10_u16(x):
    return ctypes.c_uint16(int(round(x * SCALE))).value

def q10_s16(x):
    v = int(round(x * SCALE))
    v = max(-32768, min(32767, v))
    return v & 0xFFFF

def feats_to_q16(X):
    return (np.clip(np.round(X * SCALE).astype(np.int64), -(1 << 15), (1 << 15) - 1)
            .astype(np.int16))

# ─────────────────────────────────────────────────────────────────────────────
# SPI BFM — CPOL=0 CPHA=0, 40-bit frames, MSB first
# Each SPI half-period = SPI_HALF_PER_CYCLES system-clock cycles.
# ─────────────────────────────────────────────────────────────────────────────
async def _spi_half(dut):
    """Wait one SPI half-period (in system clock cycles)."""
    for _ in range(SPI_HALF_PER_CYCLES):
        await RisingEdge(dut.clk)


async def spi_write(dut, addr, data):
    """Drive a 40-bit SPI write frame: addr[7]=0, addr[6:0], data[31:0]."""
    frame = ((addr & 0x7F) << 32) | (data & 0xFFFFFFFF)   # addr bit7=0 for write

    dut.spi_csn.value  = 0
    dut.spi_sclk.value = 0
    await _spi_half(dut)

    for bit in range(39, -1, -1):
        dut.spi_mosi.value = (frame >> bit) & 1
        dut.spi_sclk.value = 1
        await _spi_half(dut)
        dut.spi_sclk.value = 0
        await _spi_half(dut)

    dut.spi_csn.value = 1
    await _spi_half(dut)


async def spi_read(dut, addr):
    """Drive a 40-bit SPI read frame: addr[7]=1, addr[6:0]; capture MISO[31:0]."""
    addr_byte = 0x80 | (addr & 0x7F)   # bit7=1 for read
    frame = (addr_byte << 32)          # data field is don't-care for reads

    dut.spi_csn.value  = 0
    dut.spi_sclk.value = 0
    await _spi_half(dut)

    rx = 0
    for bit in range(39, -1, -1):
        dut.spi_mosi.value = (frame >> bit) & 1
        dut.spi_sclk.value = 1
        await _spi_half(dut)
        if bit < 32:   # data phase — capture MISO
            rx = (rx << 1) | (int(dut.spi_miso.value) & 1)
        dut.spi_sclk.value = 0
        await _spi_half(dut)

    dut.spi_csn.value = 1
    await _spi_half(dut)
    return rx & 0xFFFFFFFF

# ─────────────────────────────────────────────────────────────────────────────
# Off-chip SRAM model
# Monitors ram_ren_out + ram_addr_out; drives ram_rdata_in.
# LAT=3 in RTL: core waits 3 cycles after ren before sampling rdata.
# We respond on the same cycle ren is asserted — 3 cycles of margin.
# ─────────────────────────────────────────────────────────────────────────────
async def ram_model(dut, ram):
    while True:
        await RisingEdge(dut.clk)
        if int(dut.ram_ren_out.value):
            addr = int(dut.ram_addr_out.value)
            dut.ram_rdata_in.value = ram.get(addr, 0) & 0xFFFF


def build_sv_ram(ram, sv_vecs_per_class):
    """Populate SV words (rows 0..NUM_SV_ROWS-1 × FEATURE_DIM) in RAM dict."""
    row = 0
    for c in range(NUM_CLASSES):
        for sv in sv_vecs_per_class[c]:
            base = row * FEATURE_DIM
            for fi, f in enumerate(sv):
                ram[base + fi] = q10_u16(float(f))
            row += 1

def load_input_ram(ram, X_batch_q):
    """Populate input rows starting at NUM_SV_ROWS."""
    for s_idx, feat_q in enumerate(X_batch_q):
        base = (NUM_SV_ROWS + s_idx) * FEATURE_DIM
        for fi, f in enumerate(feat_q):
            ram[base + fi] = int(f) & 0xFFFF

def clear_input_ram(ram, n_samples):
    for s_idx in range(n_samples):
        base = (NUM_SV_ROWS + s_idx) * FEATURE_DIM
        for fi in range(FEATURE_DIM):
            ram.pop(base + fi, None)

# ─────────────────────────────────────────────────────────────────────────────
# Main cocotb test
# ─────────────────────────────────────────────────────────────────────────────
@cocotb.test()
async def run_spi_cosim(dut):
    """
    Full SPI cosimulation:
      1. Load ECG dataset, train 5 binary OVR SVMs (600 SVs uniform).
      2. One-time SPI config: NUM_SV, PARAM_WR (gamma + biases), ALPHA_WR (600).
      3. For each batch: populate SRAM, fire start, capture class_out on sample_rdy.
      4. Report accuracy; write asic_preds.csv.
    """
    # ── Clock & reset ─────────────────────────────────────────────────────────
    cocotb.start_soon(Clock(dut.clk, 25, unit="ns").start())  # 40 MHz

    dut.rst_n.value       = 0
    dut.spi_csn.value     = 1
    dut.spi_sclk.value    = 0
    dut.spi_mosi.value    = 0
    dut.ram_rdata_in.value = 0

    for _ in range(16):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(8):
        await RisingEdge(dut.clk)

    # ── Dataset + model ───────────────────────────────────────────────────────
    print("\n[cosim] Loading dataset (256-dim multi-scale features, real data only)...")
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("[cosim] Training 5 binary OVR SVMs with SV_ALLOC =", SV_ALLOC, "...")
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    sv_vecs_per_class   = []
    sv_alphas_per_class = []
    sv_biases           = []
    sv_counts_actual    = []
    for c in range(NUM_CLASSES):
        y_bin  = np.where(y_tr == c, 1, -1)
        svm_c  = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        alphas = svm_c.dual_coef_[0]
        svs    = svm_c.support_vectors_
        budget = SV_ALLOC[c]
        n      = min(len(alphas), budget)
        idx    = (np.argsort(-np.abs(alphas))[:budget]
                  if len(alphas) > budget else np.arange(len(alphas)))
        sel_sv = svs[idx]; sel_a = alphas[idx]
        K_tr      = sk_rbf(X_tr, sel_sv, gamma=DEFAULT_GAMMA)
        partial   = (K_tr * sel_a).sum(axis=1)
        full_tr   = svm_c.decision_function(X_tr)
        bias_hw   = float(np.mean(full_tr - partial))
        sv_vecs_per_class.append(sel_sv)
        sv_alphas_per_class.append(sel_a)
        sv_biases.append(bias_hw)
        sv_counts_actual.append(n)
        print(f"[cosim]   Class {c} ({CLASS_NAMES[c]:7s}): "
              f"{len(alphas):3d} natural SVs, budget={budget}, using={n}")

    sv_counts_actual = np.array(sv_counts_actual, dtype=int)
    sv_total = int(sv_counts_actual.sum())
    print(f"[cosim] Total SVs loaded: {sv_total}")

    # sklearn OVR reference accuracy
    binary_svms = []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        svm_c = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        binary_svms.append(svm_c)
    scores_sk = np.column_stack([s.decision_function(X_te) for s in binary_svms])
    sk_acc = np.mean(np.argmax(scores_sk, axis=1) == y_te)
    print(f"[cosim] sklearn OVR reference accuracy: {sk_acc:.4f} ({int(sk_acc*len(y_te))}/{len(y_te)})")

    X_te_q = feats_to_q16(X_te)
    N_eval  = N_EVAL_OVERRIDE if N_EVAL_OVERRIDE > 0 else len(X_te_q)
    X_te_q  = X_te_q[:N_eval]
    y_te    = y_te[:N_eval]

    # ── One-time SPI configuration ────────────────────────────────────────────
    print("[cosim] SPI config: vbatt_ok=1 ...")
    # Assert vbatt_ok first so the clock gate enables for all subsequent writes
    await spi_write(dut, SPI_CONTROL, 0x02)

    print("[cosim] SPI config: NUM_SV[0-4] ...")
    for c in range(NUM_CLASSES):
        await spi_write(dut, SPI_NUM_SV[c], int(sv_counts_actual[c]))

    print("[cosim] SPI config: PARAM_WR — gamma ...")
    gamma_q = q10_u16(DEFAULT_GAMMA)
    await spi_write(dut, SPI_PARAM_WR, (1 << 19) | (0 << 16) | gamma_q)

    print("[cosim] SPI config: PARAM_WR — per-class biases ...")
    for c in range(NUM_CLASSES):
        bias_q = q10_s16(sv_biases[c])
        await spi_write(dut, SPI_PARAM_WR, (1 << 19) | ((c + 2) << 16) | bias_q)

    print(f"[cosim] SPI config: ALPHA_WR — loading {sv_total} alphas ...")
    global_idx = 0
    for c in range(NUM_CLASSES):
        for alpha_val in sv_alphas_per_class[c]:
            alpha_q = q10_s16(float(alpha_val))
            await spi_write(dut, SPI_ALPHA_WR,
                            (global_idx << 16) | alpha_q)
            global_idx += 1
    print(f"[cosim] Alpha table loaded: {global_idx} entries")

    # ── Build SV RAM, start background SRAM model ─────────────────────────────
    ram = {}
    build_sv_ram(ram, sv_vecs_per_class)
    cocotb.start_soon(ram_model(dut, ram))
    print(f"[cosim] SV RAM built: {sv_total} SVs × {FEATURE_DIM} features")

    # ── Batch classification loop ─────────────────────────────────────────────
    all_preds  = []
    batch_start = 0
    cycles_per_sample = (FEATURE_DIM + 5 +
                         sv_total * (FEATURE_DIM + 30) + 200)

    while batch_start < N_eval:
        batch_end = min(batch_start + MAX_BATCH, N_eval)
        X_batch   = X_te_q[batch_start:batch_end]
        N_batch   = batch_end - batch_start

        load_input_ram(ram, X_batch)

        # Write NUM_SAMPLES then fire start (start + vbatt_ok = 0x03)
        await spi_write(dut, SPI_NUM_SAMPLES, N_batch)
        await spi_write(dut, SPI_CONTROL, 0x03)

        print(f"[cosim] Batch {batch_start}..{batch_end-1} started "
              f"({N_batch} samples, ~{N_batch * cycles_per_sample:,} cycles)")

        # ── Capture results via direct ports ──────────────────────────────────
        # sample_rdy pulses one cycle per classified beat; class_out stable same cycle
        batch_preds = []
        MAX_CYCLES  = N_batch * cycles_per_sample * 2 + 100_000

        for _ in range(MAX_CYCLES):
            await RisingEdge(dut.clk)

            if int(dut.sample_rdy.value):
                pred = int(dut.class_out.value) & 0x7
                batch_preds.append(pred)
                n         = len(batch_preds)
                true_lbl  = int(y_te[batch_start + n - 1])
                running   = sum(p == int(y_te[batch_start + i])
                                for i, p in enumerate(batch_preds)) / n
                print(f"[cosim] sample {n:3d}/{N_batch}  "
                      f"pred={CLASS_NAMES[pred]:<7s}  "
                      f"true={CLASS_NAMES[true_lbl]:<7s}  "
                      f"acc={running:.1%}", flush=True)
                if n == N_batch:
                    break

            if int(dut.done.value):
                if len(batch_preds) < N_batch:
                    print(f"[cosim] WARNING: done before "
                          f"{len(batch_preds)}/{N_batch} captures")
                break
        else:
            print(f"[cosim] ERROR: timeout — "
                  f"{len(batch_preds)}/{N_batch} captured")

        all_preds.extend(batch_preds)
        clear_input_ram(ram, N_batch)
        batch_start = batch_end

        acc = (sum(p == int(y_te[i]) for i, p in enumerate(all_preds))
               / len(all_preds))
        print(f"[cosim] {len(all_preds)}/{N_eval} classified   acc={acc:.2%}")

    # ── Write CSV + final summary ─────────────────────────────────────────────
    os.makedirs(SIM_DIR, exist_ok=True)
    out_csv = os.path.join(SIM_DIR, "asic_preds.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for p in all_preds:
            writer.writerow([p])

    acc = sum(p == int(y_te[i]) for i, p in enumerate(all_preds)) / N_eval
    print(f"\n[cosim] =========================================")
    print(f"[cosim] ASIC accuracy : {acc:.4f}  ({int(acc*N_eval)}/{N_eval})")
    print(f"[cosim] sklearn ref   : {sk_acc:.4f}  ({int(sk_acc*len(y_te))}/{len(y_te)})")
    print(f"[cosim] Saved {len(all_preds)} predictions → {out_csv}")
    print(f"[cosim] Run: python3 ../sim/confusion_comparison_m6.py")
    print(f"[cosim] =========================================")

    assert acc >= 0.95, f"Accuracy {acc:.4f} below 95% minimum"
