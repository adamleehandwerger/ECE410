"""
tb_wb_cosim.py — ECE410 m5 Batch Architecture cocotb Simulation  (v8)
======================================================================
Exercises user_project_wrapper with the batch RBF-SVM architecture.

Host pre-loads both the SV matrix and the input matrix into a Python
SRAM model, fires start once, then the ASIC autonomously classifies
all samples.  Results are captured per-sample via GPIO.

Off-chip RAM layout  (row × FEATURE_DIM, FEATURE_DIM = 256):
  Rows  0..499         SV matrix      (500 × 256 × 2 B = 256 KB)
  Rows  500..1499      input batch    (1000 × 256 × 2 B = 512 KB max)

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

import os, sys, warnings, ctypes, csv
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True)

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
MAX_SV_PER_CLASS = 100
NUM_SV_ROWS      = 500   # RTL NUM_SV — input matrix starts at this row index
MAX_BATCH_SIZE   = 1000  # RTL MAX_BATCH_SIZE
CLASS_NAMES      = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA    = float(os.environ.get("COSIM_GAMMA",  "0.25"))
N_EVAL_OVERRIDE  = int(os.environ.get("COSIM_N_EVAL",  "0"))    # 0 = use full dataset
NORMAL_RR        = 308
HALF_SINGLE      = FEAT_SINGLE // 2   # 64
HALF_10BEAT      = FEAT_10BEAT // 2   # 32
N_BEATS_10       = 10
N_BEATS_100      = 100
_BEAT_SYMS       = set("NLReEjJAaSVF/fQ")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wishbone addresses (base 0x3000_0000)
WB_CONTROL     = 0x30000004
WB_NUM_SAMPLES = 0x3000000C
WB_NUM_SV      = [0x30000010, 0x30000014, 0x30000018, 0x3000001C, 0x30000020]
WB_PARAM_WR    = 0x30000024
WB_ALPHA_WR    = 0x30000028  # [24:16]=sv_global_idx (9-bit) [15:0]=alpha Q6.10 signed

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction + ECG loader (real data only)
# ─────────────────────────────────────────────────────────────────────────────
def extract_multiscale(sig, all_beat_samples, beat_idx):
    s = int(all_beat_samples[beat_idx]); n = len(sig)
    if s < HALF_SINGLE or s + HALF_SINGLE > n:
        return None
    seg1 = sig[s - HALF_SINGLE : s + HALF_SINGLE].copy()
    pk1  = np.max(np.abs(seg1))
    if pk1 < 1e-6: return None
    seg1 = (seg1 / pk1).astype(np.float32)

    half10 = N_BEATS_10 // 2
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

    j0 = max(0, beat_idx - N_BEATS_100)
    rr_raw = np.diff(all_beat_samples[j0 : beat_idx + 1]).astype(np.float32)
    if len(rr_raw) < 2: return None
    rr_norm = np.clip(rr_raw / NORMAL_RR, 0.0, 2.0)
    seg3 = np.interp(np.linspace(0, 1, FEAT_100RR),
                     np.linspace(0, 1, len(rr_norm)), rr_norm).astype(np.float32)
    return np.concatenate([seg1, seg2, seg3])

# Persistent local cache for PhysioNet records (~500 MB for the records we need).
PHYSIONET_CACHE = os.path.expanduser("~/.physionet_cache")

def _rdrecord_cached(rec, pn_dir):
    """Read a WFDB record, caching .hea/.dat/.atr under PHYSIONET_CACHE."""
    import wfdb, wfdb.io.download as wdl
    local_dir = os.path.join(PHYSIONET_CACHE, pn_dir)
    hea_path  = os.path.join(local_dir, rec + ".hea")
    if not os.path.exists(hea_path):
        os.makedirs(local_dir, exist_ok=True)
        # Download header first to discover the exact .dat segment files
        wdl.dl_files(pn_dir, local_dir, [rec + ".hea"], keep_subdirs=False)
        # Parse header to find data file names, then grab them + annotation
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
        return {}
    BMAP = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "V": 1, "E": 1,
            "F": 3,
            "A": 4, "a": 4, "J": 4, "S": 4,
            "/": 4, "f": 4, "Q": 4}
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
    recs_loaded = 0
    for rec_list, pn_dir in sources:
        if all(len(v) >= max_per_class for v in beats.values()):
            break
        print(f"[cosim]   Scanning {pn_dir} ({len(rec_list)} records)...")
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in beats.values()):
                break
            try:
                r, ann = _rdrecord_cached(rec, pn_dir)
            except Exception:
                continue
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
                elif sym in BMAP:
                    cls = BMAP[sym]
                else:
                    continue
                if len(beats[cls]) >= max_per_class: continue
                feat = extract_multiscale(sig, all_beat_samples, beat_idx)
                if feat is not None: beats[cls].append(feat)
    return beats

def build_dataset(n_per_class=300):
    import os, hashlib, tempfile
    cache_key = f"ecg_n{n_per_class}_d256"
    cache_path = os.path.join(tempfile.gettempdir(), f"cosim_cache_{cache_key}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
        print(f"\n[cosim] Loaded from cache ({cache_path})")
        print(f"[cosim] Real beats per class: {counts}  total: {sum(counts)}")
        return X.astype(np.float32), y.astype(np.int32)
    print("\n[cosim] Loading ECG databases: MIT-BIH + SVDB + INCART (256-dim multi-scale features, real data only)...")
    real = load_mitbih_beats(max_per_class=n_per_class)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for b in real.get(cls, []):
            X.append(b)
            y.append(cls)
    if not X:
        raise RuntimeError("No real MIT-BIH beats found. Install wfdb: pip install wfdb")
    counts = [len(real.get(c, [])) for c in range(NUM_CLASSES)]
    print(f"[cosim] Real beats per class: {counts}  total: {sum(counts)}")
    Xa, ya = np.array(X, np.float32), np.array(y, np.int32)
    np.savez_compressed(cache_path, X=Xa, y=ya)
    print(f"[cosim] Dataset cached → {cache_path}")
    return Xa, ya

# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def q10_u16(x):
    """Float → unsigned Q6.10 bit pattern (for SV features and gamma)."""
    return ctypes.c_uint16(int(round(x * SCALE))).value

def q10_s16(x):
    """Float → signed Q6.10 as uint16 bit pattern (for alpha coefficients and biases)."""
    v = int(round(x * SCALE))
    v = max(-32768, min(32767, v))
    return v & 0xFFFF

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
def load_sv_ram(ram, sv_vecs_per_class):
    """Populate SV rows (0..NUM_SV_ROWS-1) in the shared RAM dict."""
    sv_row = 0
    for c in range(NUM_CLASSES):
        for sv in sv_vecs_per_class[c]:
            base = sv_row * FEATURE_DIM
            for fi, f in enumerate(sv):
                ram[base + fi] = q10_u16(float(f))
            sv_row += 1

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

    # ── Train 5 binary OVR SVMs ───────────────────────────────────────────────
    print("\n[cosim] Loading dataset (256-dim multi-scale features)...")
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("[cosim] Training 5 binary OVR SVMs...")
    binary_svms = []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        svm_c = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        binary_svms.append(svm_c)

    # Select top-MAX_SV_PER_CLASS SVs per class by |alpha| (signed, mixed-sign).
    # The sklearn bias is calibrated for ALL SVs; after truncation the partial sum
    # differs from the full score by a roughly-constant offset across samples.
    # Bias is recalibrated as mean(full_sklearn_score - partial_hw_score) over
    # the training set so the hardware decision boundary matches sklearn's.
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    sv_vecs_per_class   = []
    sv_alphas_per_class = []
    sv_biases           = []   # recalibrated per-class bias
    sv_counts = []
    for c in range(NUM_CLASSES):
        alphas = binary_svms[c].dual_coef_[0]   # signed: α_i * y_i
        svs    = binary_svms[c].support_vectors_
        n = min(len(alphas), MAX_SV_PER_CLASS)
        top_idx = np.argsort(-np.abs(alphas))[:n] if len(alphas) > MAX_SV_PER_CLASS else np.arange(len(alphas))
        sel_sv  = svs[top_idx]
        sel_a   = alphas[top_idx]
        # Recalibrate bias: bias_hw = mean(sklearn_decision(X_tr) - partial(X_tr))
        K_tr       = sk_rbf(X_tr, sel_sv, gamma=DEFAULT_GAMMA)
        partial_tr = (K_tr * sel_a).sum(axis=1)
        full_tr    = binary_svms[c].decision_function(X_tr)
        bias_hw    = float(np.mean(full_tr - partial_tr))
        sv_vecs_per_class.append(sel_sv)
        sv_alphas_per_class.append(sel_a)
        sv_biases.append(bias_hw)
        sv_counts.append(n)

    sv_counts = np.array(sv_counts, dtype=int)
    sv_total  = int(sv_counts.sum())
    print(f"[cosim] SVs per class: {sv_counts}  total: {sv_total}")

    # sklearn OVR accuracy reference
    scores_sklearn = np.column_stack(
        [svm.decision_function(X_te) for svm in binary_svms])
    sklearn_acc = np.mean(np.argmax(scores_sklearn, axis=1) == y_te)
    print(f"[cosim] sklearn OVR accuracy: {sklearn_acc:.4f} ({int(sklearn_acc*len(y_te))}/{len(y_te)})")

    X_te_q = feats_to_q16(X_te)
    N_eval  = N_EVAL_OVERRIDE if N_EVAL_OVERRIDE > 0 else len(X_te_q)
    X_te_q  = X_te_q[:N_eval]
    y_te    = y_te[:N_eval]

    # ── One-time WB config ────────────────────────────────────────────────────
    print("[cosim] Configuring DUT...")
    # Assert vbatt_ok=1 FIRST so the 2-FF synchroniser inside the gated clock
    # domain sees d=1 during all subsequent param/alpha writes.  Without this,
    # the svm_gclk edges generated by those writes flush vbatt_ok_s to 0 and
    # the FSM won't leave IDLE when start fires later.
    await wb_write(dut, WB_CONTROL, 0x02)
    for _ in range(6): await RisingEdge(dut.wb_clk_i)

    for c in range(NUM_CLASSES):
        await wb_write(dut, WB_NUM_SV[c], int(sv_counts[c]))
    gamma_q = q10_u16(DEFAULT_GAMMA)
    await wb_write(dut, WB_PARAM_WR, (1 << 19) | (0 << 16) | gamma_q)

    # Load per-class biases (recalibrated for truncated top-50 SV set)
    for c in range(NUM_CLASSES):
        bias_val = sv_biases[c]
        bias_q   = q10_s16(bias_val)
        await wb_write(dut, WB_PARAM_WR, (1 << 19) | ((c + 2) << 16) | bias_q)

    # Load alpha table: one WB write per SV (global index order)
    global_sv_idx = 0
    for c in range(NUM_CLASSES):
        for sv_local_idx in range(len(sv_alphas_per_class[c])):
            alpha_val = float(sv_alphas_per_class[c][sv_local_idx])
            alpha_q   = q10_s16(alpha_val)
            await wb_write(dut, WB_ALPHA_WR,
                           (global_sv_idx << 16) | alpha_q)
            global_sv_idx += 1
    print(f"[cosim] Alpha table loaded: {global_sv_idx} coefficients")

    # ── Shared RAM dict + persistent background RAM model ─────────────────────
    ram = {}
    load_sv_ram(ram, sv_vecs_per_class)
    cocotb.start_soon(ram_model(dut, ram))
    print(f"[cosim] SV RAM loaded: {sv_total} SVs × {FEATURE_DIM} features")

    # ── Process in batches of MAX_BATCH_SIZE ──────────────────────────────────
    all_preds = []
    batch_start = 0

    # Per-sample cycle budget with margin:
    #   LOAD_INPUT: FEATURE_DIM+3 cycles, COMPUTE_DIST: FEATURE_DIM+4 per SV,
    #   COMPUTE_KERNEL: 22 cycles, OUTPUT_RESULT: 1, WRITE_CLASS: 1 + FSM transitions
    cycles_per_sample = FEATURE_DIM + 5 + sv_total * (FEATURE_DIM + 30) + 200

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
        MAX_CYCLES  = N_batch * cycles_per_sample * 2 + 100_000

        for cyc in range(MAX_CYCLES):
            await RisingEdge(dut.wb_clk_i)
            io_val = int(dut.io_out.value)

            if (io_val >> 3) & 0x1:               # sample_rdy
                pred = io_val & 0x7
                batch_preds.append(pred)
                n = len(batch_preds)
                true_label = int(y_te[batch_start + n - 1])
                running_acc = sum(p == t for p, t in zip(batch_preds, [int(y_te[batch_start+i]) for i in range(n)])) / n
                print(f"[cosim] sample {n:3d}/{N_batch}  pred={CLASS_NAMES[pred]:<6s}  true={CLASS_NAMES[true_label]:<6s}  running_acc={running_acc:.1%}", flush=True)
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
