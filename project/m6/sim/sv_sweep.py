"""
sv_sweep.py — ECE410 Milestone 5
=================================
Uniform SV-count sweep: all 5 classes get the same N SVs.
Train once at full natural-SV budget, then slice top-N by |alpha|
for each config — no retraining per step.

Sweep: N = [20, 40, 60, 80, 90, 100, 120, 150, 200] per class
       (total SVs = 5N; hardware ceiling = 500 → max N=100 for equal split,
        but hardware allows up to 255/class so N>100 is also valid)

Outputs:
  sv_sweep.png        accuracy vs SV count (float64 + Q6.10)
  sv_sweep_report.txt per-config table

Real data only — raises RuntimeError if wfdb unavailable.
"""

import os, math, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS     = 10
SCALE         = 1 << FRAC_BITS
FEATURE_DIM   = 256
FEAT_SINGLE   = 128
FEAT_10BEAT   = 64
FEAT_100RR    = 64
NUM_CLASSES   = 5
CLASS_NAMES   = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA = 0.25
NORMAL_RR     = 308

SWEEP_N = [20, 40, 60, 80, 90, 100, 120, 150, 200]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
HALF_SINGLE = FEAT_SINGLE // 2
HALF_10BEAT = FEAT_10BEAT // 2
N_BEATS_10  = 10
N_BEATS_100 = 100
_BEAT_SYMS  = set("NLReEjJAaSVF/fQ")

def extract_multiscale(sig, all_beat_samples, beat_idx):
    s = int(all_beat_samples[beat_idx]); n = len(sig)
    if s < HALF_SINGLE or s + HALF_SINGLE > n: return None
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

def load_mitbih_beats(max_per_class=300):
    try:
        import wfdb
    except ImportError:
        raise RuntimeError("wfdb not installed.")
    BMAP = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "V": 1, "E": 1, "F": 3,
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
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in beats.values()): break
            try:
                r   = wfdb.rdrecord(rec, pn_dir=pn_dir)
                ann = wfdb.rdann(rec, 'atr', pn_dir=pn_dir)
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
            for beat_idx, (s_idx, sym) in enumerate(zip(all_beat_samples.tolist(), beat_sym_list)):
                in_afib_flag = any(a0 <= s_idx <= a1 for a0, a1 in afib_regions)
                if in_afib_flag and sym in ('N','L','R','e','j','V','A','a','J','S'): cls = 2
                elif sym in BMAP: cls = BMAP[sym]
                else: continue
                if len(beats[cls]) >= max_per_class: continue
                feat = extract_multiscale(sig, all_beat_samples, beat_idx)
                if feat is not None: beats[cls].append(feat)
    return beats

def build_dataset(n_per_class=300):
    print("\n=== Loading ECG databases: MIT-BIH + SVDB + INCART (real data only) ===")
    real = load_mitbih_beats(max_per_class=n_per_class)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for b in real.get(cls, []): X.append(b); y.append(cls)
    if not X:
        raise RuntimeError("No real beats found — install wfdb.")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls} ({CLASS_NAMES[cls]:7s}): {len(real.get(cls,[]))} real beats")
    return np.array(X, np.float32), np.array(y, np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# Train once at full natural-SV budget
# ─────────────────────────────────────────────────────────────────────────────
def train_full(X_tr, y_tr):
    """Train 5 binary OVR SVMs, keep ALL natural SVs sorted by |alpha| desc."""
    sv_vecs_full, sv_alphas_full, biases = [], [], []
    natural_counts = []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        svm_c = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        alphas = svm_c.dual_coef_[0]
        svs    = svm_c.support_vectors_
        # sort by |alpha| descending so slicing top-N is trivial
        order  = np.argsort(-np.abs(alphas))
        sv_vecs_full.append(svs[order])
        sv_alphas_full.append(alphas[order])
        biases.append(svm_c.intercept_[0])
        natural_counts.append(len(alphas))
        print(f"  Class {c} ({CLASS_NAMES[c]:7s}): {len(alphas)} natural SVs")
    return sv_vecs_full, sv_alphas_full, np.array(biases), natural_counts

# ─────────────────────────────────────────────────────────────────────────────
# Float + Q6.10 inference
# ─────────────────────────────────────────────────────────────────────────────
def ovr_predict_float(X_te, sv_vecs, sv_alphas, biases, gamma=DEFAULT_GAMMA):
    n_test = len(X_te)
    scores = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        sv = sv_vecs[c]; alp = sv_alphas[c]
        sq_x  = np.sum(X_te**2, axis=1, keepdims=True)
        sq_sv = np.sum(sv**2,   axis=1, keepdims=True).T
        dist2 = sq_x - 2*(X_te @ sv.T) + sq_sv
        scores[:, c] = np.exp(-gamma * dist2) @ alp + biases[c]
    return np.argmax(scores, axis=1)

HORNER_COEFFS = [1024, 1024, 512, 170, 42, 8, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
EXP_INT_LUT   = [round(math.exp(-i) * SCALE) for i in range(16)]

def _to_i16(v):
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v

def hw_kernel(gamma_q, x_q, sv_q):
    acc = 0; prev_d = 0; prev_dsq = 0
    for k in range(FEATURE_DIM):
        xi = _to_i16(x_q[k]); si = _to_i16(sv_q[k])
        diff = _to_i16(xi - si)
        acc     += prev_dsq >> FRAC_BITS
        prev_dsq = prev_d * prev_d
        prev_d   = diff
    acc += prev_dsq >> FRAC_BITS
    prev_dsq = prev_d * prev_d
    acc += prev_dsq >> FRAC_BITS
    acc = min(acc, 0xFFFFF)
    P = _to_i16(gamma_q) * acc
    I = P >> 20; F_q = (P >> 10) & 1023
    if I >= 16: return 0
    lut_val = EXP_INT_LUT[I]
    x = _to_i16(-F_q)
    result = _to_i16(HORNER_COEFFS[15])
    for n in range(14, -1, -1):
        result = _to_i16(HORNER_COEFFS[n] + (_to_i16(x) * _to_i16(result) >> FRAC_BITS))
    return max(0, min(1024, (lut_val * max(0, min(1024, result))) >> FRAC_BITS))

def vecs_to_q10(X):
    return np.clip(np.round(X * SCALE), -32768, 32767).astype(np.int32)

def ovr_predict_q10(X_te, sv_vecs, sv_alphas, biases, gamma=DEFAULT_GAMMA):
    gamma_q = int(round(gamma * SCALE)) & 0xFFFF
    n_test  = len(X_te); X_q = vecs_to_q10(X_te)
    scores  = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        SV_q = vecs_to_q10(sv_vecs[c]); alp = sv_alphas[c]
        for i in range(n_test):
            k_sum = sum(alp[j] * (hw_kernel(gamma_q, X_q[i], SV_q[j]) / 1024.0)
                        for j in range(len(alp)))
            scores[i, c] = k_sum + biases[c]
    return np.argmax(scores, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    N_eval = len(X_te)
    print(f"\n  Train: {len(X_tr)}  Test: {N_eval}")

    print(f"\n=== Training 5 binary OVR SVMs at full natural-SV budget ===")
    sv_vecs_full, sv_alphas_full, biases, natural_counts = train_full(X_tr, y_tr)
    print(f"  Natural SV counts: {natural_counts}")

    results = []

    print(f"\n{'='*65}")
    print(f"{'N/class':>8}  {'Total':>6}  {'Float':>8}  {'Q6.10':>8}  "
          f"{'Gap':>7}  {'Flips':>6}  {'Time':>6}")
    print(f"{'-'*65}")

    for N in SWEEP_N:
        # Slice to top-N SVs per class (capped at natural count)
        sv_vecs  = [sv_vecs_full[c][:N]   for c in range(NUM_CLASSES)]
        sv_alphas = [sv_alphas_full[c][:N] for c in range(NUM_CLASSES)]
        actual_n  = [len(sv_vecs[c]) for c in range(NUM_CLASSES)]
        total_sv  = sum(actual_n)

        y_float   = ovr_predict_float(X_te, sv_vecs, sv_alphas, biases)
        acc_float = accuracy_score(y_te, y_float)

        t0 = time.perf_counter()
        y_q10     = ovr_predict_q10(X_te, sv_vecs, sv_alphas, biases)
        elapsed   = time.perf_counter() - t0
        acc_q10   = accuracy_score(y_te, y_q10)
        n_flips   = int(np.sum(y_float != y_q10))
        gap       = acc_float - acc_q10

        results.append(dict(N=N, total=total_sv, actual_n=actual_n,
                            acc_float=acc_float, acc_q10=acc_q10,
                            gap=gap, n_flips=n_flips, elapsed=elapsed,
                            y_float=y_float, y_q10=y_q10))

        print(f"{N:>8}  {total_sv:>6}  {acc_float:>8.4f}  {acc_q10:>8.4f}  "
              f"{gap:>+7.4f}  {n_flips:>6}  {elapsed:>5.1f}s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    Ns         = [r['N']         for r in results]
    acc_floats = [r['acc_float'] for r in results]
    acc_q10s   = [r['acc_q10']   for r in results]
    flips      = [r['n_flips']   for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(Ns, [a*100 for a in acc_floats], 'o-', color='steelblue',
             linewidth=2, markersize=7, label='Float64 (sklearn)')
    ax1.plot(Ns, [a*100 for a in acc_q10s],  's--', color='tomato',
             linewidth=2, markersize=7, label='Q6.10 hardware model')

    # Shade the gap
    ax1.fill_between(Ns, [a*100 for a in acc_q10s],
                         [a*100 for a in acc_floats],
                     alpha=0.15, color='orange', label='Quantization gap')

    # Mark hardware ceiling (N=100 per class = 500 total)
    ax1.axvline(x=100, color='gray', linestyle=':', linewidth=1.5,
                label='Hardware ceiling (100/class = 500 total)')

    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('SV Count Sweep — Float64 vs Q6.10 Accuracy\n'
                  '5 binary OVR SVMs, uniform N SVs/class, gamma=0.25, '
                  'MIT-BIH+SVDB+INCART',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(min(acc_q10s)*100 - 1, 95), 100])
    ax1.set_xticks(Ns)

    ax2.bar(Ns, flips, color=['tomato' if f > 0 else 'steelblue' for f in flips],
            width=8, alpha=0.8)
    ax2.set_xlabel('SVs per class (N)', fontsize=11)
    ax2.set_ylabel('Quant. flips', fontsize=10)
    ax2.set_xticks(Ns)
    ax2.set_yticks(range(max(flips) + 2))
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(x=100, color='gray', linestyle=':', linewidth=1.5)

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "sv_sweep.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved -> {out_png}")

    # ── Text report ───────────────────────────────────────────────────────────
    lines = [
        "SV Count Sweep Report — ECE410 m5",
        "=" * 60,
        f"Dataset: MIT-BIH+SVDB+INCART  Train={len(X_tr)}  Test={N_eval}",
        f"Natural SV counts: {natural_counts}",
        f"gamma=0.25  C=1.0  random_state=42",
        "",
        f"{'N/class':>8}  {'Total':>6}  {'Float':>8}  {'Q6.10':>8}  "
        f"{'Gap':>7}  {'Flips':>6}",
        "-" * 60,
    ]
    for r in results:
        lines.append(f"{r['N']:>8}  {r['total']:>6}  {r['acc_float']:>8.4f}  "
                     f"{r['acc_q10']:>8.4f}  {r['gap']:>+7.4f}  {r['n_flips']:>6}")
    lines += [
        "",
        "Note: SVs sorted by |alpha| descending. Each row uses top-N SVs",
        "from the same trained model — no retraining per step.",
        "Q6.10 inference uses pure-Python hardware model (svm_compute_core algorithm).",
    ]
    rpt_path = os.path.join(SCRIPT_DIR, "sv_sweep_report.txt")
    with open(rpt_path, 'w') as f: f.write('\n'.join(lines))
    print(f"Saved -> {rpt_path}")


if __name__ == "__main__":
    main()
