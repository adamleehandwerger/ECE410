"""
confusion_3way.py — ECE410 Milestone 5
=======================================
Three-column confusion matrix:

  Col 1: Optimal sklearn OVR   (float, all SVs, joint multiclass training)
  Col 2: ASIC model — float    (5 binary OVR SVMs, top-100 SVs/class, float64)
  Col 3: ASIC model — Q6.10    (same 5 binary SVMs, cocotb RTL predictions)

Col 1 vs Col 2  →  model architecture difference
                    (joint vs. separate binary, SV count constraint)
Col 2 vs Col 3  →  Q6.10 quantization error only
                    (identical model, float vs. fixed-point)

Real data only — raises RuntimeError if wfdb / PhysioNet data unavailable.

Outputs:
  confusion_3way.png  in the same directory as this script
"""

import os, sys, math, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS        = 10
SCALE            = 1 << FRAC_BITS          # 1024
DIST_WIDTH       = 20

FEATURE_DIM      = 256
FEAT_SINGLE      = 128
FEAT_10BEAT      = 64
FEAT_100RR       = 64
NUM_CLASSES      = 5
MAX_SV_PER_CLASS = 100
CLASS_NAMES      = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA    = 0.25
NORMAL_RR        = 308

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (256-dim multi-scale)
# ─────────────────────────────────────────────────────────────────────────────
HALF_SINGLE = FEAT_SINGLE // 2
HALF_10BEAT = FEAT_10BEAT // 2
N_BEATS_10  = 10
N_BEATS_100 = 100
_BEAT_SYMS  = set("NLReEjJAaSVF/fQ")

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

def load_mitbih_beats(max_per_class=300):
    try:
        import wfdb
    except ImportError:
        raise RuntimeError("wfdb not installed. Run: pip install wfdb")
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
    for rec_list, pn_dir in sources:
        if all(len(v) >= max_per_class for v in beats.values()):
            break
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in beats.values()):
                break
            try:
                r   = wfdb.rdrecord(rec, pn_dir=pn_dir)
                ann = wfdb.rdann(rec, 'atr', pn_dir=pn_dir)
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
    print("\n=== Loading ECG databases: MIT-BIH + SVDB + INCART (real data only) ===")
    real = load_mitbih_beats(max_per_class=n_per_class)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for b in real.get(cls, []):
            X.append(b); y.append(cls)
    if not X:
        raise RuntimeError("No real MIT-BIH beats found — install wfdb and check network.")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls} ({CLASS_NAMES[cls]:7s}): {len(real.get(cls,[]))} real beats")
    return np.array(X, np.float32), np.array(y, np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# Train 5 binary OVR SVMs — same model as ASIC
# ─────────────────────────────────────────────────────────────────────────────
def train_binary_ovr(X_tr, y_tr):
    """5 binary SVMs (class c vs. rest), top-100 SVs/class by |alpha|."""
    sv_vecs, sv_alphas, biases, sv_counts = [], [], [], []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        svm_c = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0, random_state=42)
        svm_c.fit(X_tr, y_bin)
        alphas = svm_c.dual_coef_[0]
        svs    = svm_c.support_vectors_
        n      = min(len(alphas), MAX_SV_PER_CLASS)
        idx    = (np.argsort(-np.abs(alphas))[:MAX_SV_PER_CLASS]
                  if len(alphas) > MAX_SV_PER_CLASS else np.arange(len(alphas)))
        sv_vecs.append(svs[idx])
        sv_alphas.append(alphas[idx])
        biases.append(svm_c.intercept_[0])
        sv_counts.append(n)
        print(f"  Class {c} ({CLASS_NAMES[c]:7s}): {len(alphas):3d} total SVs → "
              f"top {n} selected  bias={svm_c.intercept_[0]:+.4f}")
    return sv_vecs, sv_alphas, np.array(biases), np.array(sv_counts)

# ─────────────────────────────────────────────────────────────────────────────
# Float OVR inference with the binary-SVM model
# ─────────────────────────────────────────────────────────────────────────────
def ovr_predict_float(X_te, sv_vecs, sv_alphas, biases, gamma=DEFAULT_GAMMA):
    """Float64 RBF kernel OVR: score_c = sum_j alpha_j * K(x, sv_j) + bias_c."""
    n_test  = len(X_te)
    scores  = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        sv  = sv_vecs[c]                          # (n_sv, 256)
        alp = sv_alphas[c]                        # (n_sv,)
        # ||x - sv||^2 for each (test, sv) pair
        sq_norms_x  = np.sum(X_te**2, axis=1, keepdims=True)   # (N, 1)
        sq_norms_sv = np.sum(sv**2,   axis=1, keepdims=True).T  # (1, M)
        dot         = X_te @ sv.T                               # (N, M)
        dist2       = sq_norms_x - 2 * dot + sq_norms_sv       # (N, M)
        K           = np.exp(-gamma * dist2)                    # (N, M)
        scores[:, c] = K @ alp + biases[c]
    return np.argmax(scores, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Load ASIC RTL predictions
# ─────────────────────────────────────────────────────────────────────────────
def load_asic_preds(n_expected):
    csv_path = os.path.join(SCRIPT_DIR, "asic_preds.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ASIC predictions not found: {csv_path}\n"
                                "Run cocotb simulation first.")
    data = np.loadtxt(csv_path, delimiter=",", dtype=int).flatten()
    if len(data) < n_expected:
        raise ValueError(f"asic_preds.csv has {len(data)} predictions, expected {n_expected}")
    print(f"  Loaded {len(data)} ASIC predictions from {csv_path}")
    return data[:n_expected]

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_cm(ax, cm, title, acc, fig):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True")
    ax.set_title(f"{title}\nAccuracy: {acc:.2%}", fontsize=10, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Dataset
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    N_eval = len(X_te)
    print(f"\n  Train: {len(X_tr)}  Test: {N_eval}")

    # Col 1 — Optimal sklearn OVR (joint, float, all SVs)
    print(f"\n=== Col 1: Optimal sklearn OVR (joint, float, all SVs) ===")
    clf_opt = SVC(kernel="rbf", gamma=DEFAULT_GAMMA, C=1.0,
                  decision_function_shape="ovr", random_state=42)
    clf_opt.fit(X_tr, y_tr)
    y_pred_opt = clf_opt.predict(X_te)
    acc_opt    = accuracy_score(y_te, y_pred_opt)
    print(f"  SVs: {clf_opt.n_support_}  total={clf_opt.n_support_.sum()}  "
          f"accuracy={acc_opt:.4f}")

    # Train binary OVR model (same as ASIC)
    print(f"\n=== Training 5 binary OVR SVMs (ASIC model, <=100 SVs/class) ===")
    sv_vecs, sv_alphas, biases, sv_counts = train_binary_ovr(X_tr, y_tr)
    print(f"  SV counts: {sv_counts}  total: {sv_counts.sum()}")

    # Col 2 — Same binary OVR model, float64
    print(f"\n=== Col 2: ASIC binary OVR model, float64 reference ===")
    y_pred_float = ovr_predict_float(X_te, sv_vecs, sv_alphas, biases)
    acc_float    = accuracy_score(y_te, y_pred_float)
    print(f"  Accuracy={acc_float:.4f}")

    # Col 3 — ASIC RTL predictions (Q6.10, cocotb)
    print(f"\n=== Col 3: ASIC RTL predictions (Q6.10, cocotb simulation) ===")
    y_pred_asic = load_asic_preds(N_eval)
    acc_asic    = accuracy_score(y_te, y_pred_asic)
    print(f"  Accuracy={acc_asic:.4f}")

    # Confusion matrices
    cm_opt   = confusion_matrix(y_te, y_pred_opt,   labels=list(range(NUM_CLASSES)))
    cm_float = confusion_matrix(y_te, y_pred_float, labels=list(range(NUM_CLASSES)))
    cm_asic  = confusion_matrix(y_te, y_pred_asic,  labels=list(range(NUM_CLASSES)))

    # Differences
    diff_model = int(np.sum(y_pred_opt != y_pred_float))
    diff_quant = int(np.sum(y_pred_float != y_pred_asic))
    print(f"\n=== Comparison ===")
    print(f"  Col1 vs Col2 (model architecture): {diff_model} samples differ")
    print(f"  Col2 vs Col3 (Q6.10 quantization): {diff_quant} samples differ")

    if diff_quant > 0:
        idx = np.where(y_pred_float != y_pred_asic)[0]
        for i in idx:
            print(f"    Sample {i}: true={CLASS_NAMES[y_te[i]]}  "
                  f"float={CLASS_NAMES[y_pred_float[i]]}  "
                  f"Q6.10={CLASS_NAMES[y_pred_asic[i]]}")
    else:
        print("  No quantization-induced prediction changes — zero Q6.10 gap.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    plot_cm(axes[0], cm_opt,
            "Optimal sklearn OVR\n(float, all SVs, joint training)", acc_opt, fig)
    plot_cm(axes[1], cm_float,
            f"ASIC binary OVR model\n(float64, {sv_counts.sum()} SVs, 5 binary SVMs)",
            acc_float, fig)
    plot_cm(axes[2], cm_asic,
            "ASIC RTL\n(Q6.10, cocotb simulation)", acc_asic, fig)

    fig.suptitle(
        "RBF-SVM — 3-Way Comparison: Optimal sklearn  /  ASIC model (float)  /  ASIC RTL (Q6.10)\n"
        "Col1 vs Col2 = model architecture difference  |  "
        "Col2 vs Col3 = Q6.10 quantization error only\n"
        "gamma=0.25, 256-dim features (128+64+64), MIT-BIH+SVDB+INCART, ECE410 PSU",
        fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "confusion_3way.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {out_path}")

if __name__ == "__main__":
    main()
