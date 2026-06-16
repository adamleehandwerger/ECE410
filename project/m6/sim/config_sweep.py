"""
config_sweep.py — compare feature splits for m6 hardening decision
===================================================================
Evaluates two candidate configurations and prints confusion matrices.

  A: [128,64,64] / [95,95,95,120,95]  — original m5 256-dim split
  B: [64,64,64]  / [120,120,120,120,120] — 192-dim equal split

Real ECG data only (mitdb + svdb + incartdb from ~/.physionet_cache).
"""

import os, sys, warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

NUM_CLASSES = 5
CLASS_NAMES = ["Normal", "PVC", "AFib", "VT", "SVT"]
FRAC_BITS   = 10
SCALE       = 1 << FRAC_BITS   # 1024
NORMAL_RR   = 308
_BEAT_SYMS  = set("NLReEjJAaSVF/fQ")
PHYSIONET_CACHE = os.path.expanduser("~/.physionet_cache")
BMAP = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
        "V": 1, "E": 1,
        "F": 3,
        "A": 4, "a": 4, "J": 4, "S": 4, "/": 4, "f": 4, "Q": 4}
SOURCES = [
    (['100','101','102','103','104','105','106','107','108','109',
      '111','112','113','114','115','116','117','118','119','121',
      '122','123','124','200','201','202','203','205','207','208',
      '209','210','212','213','214','215','217','219','220','221',
      '222','223','228','230','231','232','233','234'], 'mitdb'),
    ([f'e{i:04d}' for i in range(1, 79)], 'svdb'),
    ([f'I{i:02d}' for i in range(1, 76)], 'incartdb'),
]


def rdrecord_cached(rec, pn_dir):
    import wfdb
    import wfdb.io.download as wdl
    local_dir = os.path.join(PHYSIONET_CACHE, pn_dir)
    hea_path  = os.path.join(local_dir, rec + ".hea")
    if not os.path.exists(hea_path):
        os.makedirs(local_dir, exist_ok=True)
        wdl.dl_files(pn_dir, local_dir, [rec + ".hea"], keep_subdirs=False)
        hdr = wfdb.rdheader(os.path.join(local_dir, rec))
        dat_files = list({seg + ".dat" for seg in (
            hdr.seg_name if hasattr(hdr, 'seg_name') and hdr.seg_name else [rec])})
        wdl.dl_files(pn_dir, local_dir, dat_files + [rec + ".atr"], keep_subdirs=False)
    r   = wfdb.rdrecord(os.path.join(local_dir, rec))
    ann = wfdb.rdann(os.path.join(local_dir, rec), 'atr')
    return r, ann


def load_raw_beats(max_per_class=300):
    """
    Scan databases and return raw beat tuples per class.
    Each entry: (sig, all_beat_samples, beat_idx)
    The raw signal is retained so features can be re-extracted at any window size.
    """
    raw = {i: [] for i in range(NUM_CLASSES)}
    for rec_list, pn_dir in SOURCES:
        if all(len(v) >= max_per_class for v in raw.values()):
            break
        print(f"[sweep] Scanning {pn_dir} ({len(rec_list)} records)...")
        for rec in rec_list:
            if all(len(v) >= max_per_class for v in raw.values()):
                break
            try:
                r, ann = rdrecord_cached(rec, pn_dir)
            except Exception:
                continue
            sig = r.p_signal[:, 0].astype(np.float32)
            beat_samp, beat_sym = [], []
            for s, sym in zip(ann.sample, ann.symbol):
                if sym in _BEAT_SYMS:
                    beat_samp.append(s); beat_sym.append(sym)
            if len(beat_samp) < 3:
                continue
            all_samp = np.array(beat_samp, dtype=np.int32)

            afib_regions = []
            if hasattr(ann, 'aux_note'):
                in_afib = False; afib_start = None
                for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                    if sym == '+' and aux:
                        if '(AFIB' in aux:
                            in_afib = True; afib_start = s
                        elif in_afib and '(' in aux:
                            afib_regions.append((afib_start, s)); in_afib = False
                if in_afib and afib_start is not None:
                    afib_regions.append((afib_start, len(sig)))

            for beat_idx, (s_idx, sym) in enumerate(zip(all_samp.tolist(), beat_sym)):
                in_afib = any(a0 <= s_idx <= a1 for a0, a1 in afib_regions)
                if in_afib and sym in ('N','L','R','e','j','V','A','a','J','S'):
                    cls = 2
                elif sym in BMAP:
                    cls = BMAP[sym]
                else:
                    continue
                if len(raw[cls]) >= max_per_class:
                    continue
                raw[cls].append((sig, all_samp, beat_idx))
    return raw


def extract_features(raw, feat_single, feat_10beat, feat_100rr, pad_to=None):
    """Re-extract features with configurable window sizes.
    pad_to: zero-pad to this length for non-power-of-2 dims (RTL requires 2^N stride).
    """
    half_s  = feat_single  // 2
    half_10 = feat_10beat  // 2
    nat_dim = feat_single + feat_10beat + feat_100rr
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for (sig, all_samp, beat_idx) in raw[cls]:
            n = len(sig); s = int(all_samp[beat_idx])
            if s < half_s or s + half_s > n:
                continue
            seg1 = sig[s - half_s : s + half_s].copy()
            pk1  = np.max(np.abs(seg1))
            if pk1 < 1e-6:
                continue
            seg1 = (seg1 / pk1).astype(np.float32)

            i0 = max(0, beat_idx - 5); i1 = min(len(all_samp), beat_idx + 5)
            segs2 = []
            for bi in range(i0, i1):
                bs = int(all_samp[bi])
                if bs >= half_10 and bs + half_10 <= n:
                    seg = sig[bs - half_10 : bs + half_10].astype(np.float32)
                    pk2 = np.max(np.abs(seg))
                    if pk2 > 1e-6:
                        segs2.append(seg / pk2)
            if not segs2:
                continue
            seg2 = np.mean(segs2, axis=0).astype(np.float32)

            j0   = max(0, beat_idx - 100)
            rr   = np.diff(all_samp[j0 : beat_idx + 1]).astype(np.float32)
            if len(rr) < 2:
                continue
            rr_n = np.clip(rr / NORMAL_RR, 0.0, 2.0)
            seg3 = np.interp(np.linspace(0, 1, feat_100rr),
                             np.linspace(0, 1, len(rr_n)), rr_n).astype(np.float32)

            feat = np.concatenate([seg1, seg2, seg3])
            if pad_to and pad_to > nat_dim:
                feat = np.concatenate([feat, np.zeros(pad_to - nat_dim, np.float32)])
            X.append(feat); y.append(cls)
    return np.array(X, np.float32), np.array(y, np.int32)


def print_confusion(cm, acc_sk, acc_q):
    header = f"{'':12}" + "".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(cm):
        cells = ""
        for j, v in enumerate(row):
            mark = "*" if i != j and v > 0 else " "
            cells += f"  {v:5d}{mark}"
        rec = row[i] / row.sum() if row.sum() > 0 else 0.0
        print(f"  {CLASS_NAMES[i]:10}{cells}   recall={rec:.1%}")
    prec = [cm[j, j] / cm[:, j].sum() if cm[:, j].sum() > 0 else 0.0
            for j in range(NUM_CLASSES)]
    print("  " + "".join(f"  prec={p:.0%} " for p in prec))
    print(f"\n  sklearn accuracy : {acc_sk:.4f}")
    print(f"  Q6.10  accuracy  : {acc_q:.4f}")


def eval_config(name, raw, feat_single, feat_10beat, feat_100rr, sv_alloc,
                gamma=0.25, pad_to=None):
    nat_dim = feat_single + feat_10beat + feat_100rr
    rtl_dim = pad_to if pad_to else nat_dim
    pad_note = f" (zero-padded to {rtl_dim} for RTL)" if pad_to else ""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  split [{feat_single},{feat_10beat},{feat_100rr}] = {nat_dim}-dim{pad_note}")
    print(f"  SV alloc {sv_alloc}  total={sum(sv_alloc)}")
    print(f"{'='*65}")

    X, y = extract_features(raw, feat_single, feat_10beat, feat_100rr, pad_to=pad_to)
    counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
    print(f"  Dataset {X.shape}  classes: {counts}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Train OVR SVMs with SV budgets
    sv_vecs, sv_als, sv_bs = [], [], []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        clf   = SVC(kernel="rbf", gamma=gamma, C=1.0, random_state=42)
        clf.fit(X_tr, y_bin)
        als = clf.dual_coef_[0]; svs = clf.support_vectors_
        budget = sv_alloc[c]
        idx = (np.argsort(-np.abs(als))[:budget]
               if len(als) > budget else np.arange(len(als)))
        sv_vecs.append(svs[idx]); sv_als.append(als[idx])
        sv_bs.append(float(clf.intercept_[0]))
        print(f"  Class {c} ({CLASS_NAMES[c]:7s}): {len(als):3d} SVs → budget {budget}, using {len(idx)}")

    # sklearn OVR reference (full SVs, no budget truncation)
    binary = []
    for c in range(NUM_CLASSES):
        y_bin = np.where(y_tr == c, 1, -1)
        clf   = SVC(kernel="rbf", gamma=gamma, C=1.0, random_state=42)
        clf.fit(X_tr, y_bin)
        binary.append(clf)
    scores_sk = np.column_stack([clf.decision_function(X_te) for clf in binary])
    preds_sk  = np.argmax(scores_sk, axis=1)
    acc_sk    = float(np.mean(preds_sk == y_te))

    # Q6.10: quantize features + SVs, reconstruct distances in fixed-point
    def q16(X_f):
        return np.clip(np.round(X_f * SCALE), -(1 << 15), (1 << 15) - 1).astype(np.int16)

    def q_s16(v):
        return max(-32768, min(32767, int(round(v * SCALE))))

    X_te_qi  = q16(X_te).astype(np.int32)
    sv_qi    = [q16(v).astype(np.int32) for v in sv_vecs]
    sv_al_q  = [[q_s16(a) for a in als] for als in sv_als]
    sv_bs_q  = [q_s16(b) for b in sv_bs]

    scores_q = np.zeros((len(X_te_qi), NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        for sv_row, alpha_q in zip(sv_qi[c], sv_al_q[c]):
            diff     = X_te_qi - sv_row                    # Q6.10 integer diff
            sq       = (diff.astype(np.int64) ** 2)        # squared
            dist_q   = (sq >> FRAC_BITS).sum(axis=1)       # shift + accumulate (RTL match)
            k_float  = np.exp(-gamma * dist_q / SCALE)     # gamma is Q6.10, dist is Q6.10
            scores_q[:, c] += (alpha_q / SCALE) * k_float
        scores_q[:, c] += sv_bs_q[c] / SCALE

    preds_q = np.argmax(scores_q, axis=1)
    acc_q   = float(np.mean(preds_q == y_te))

    cm = confusion_matrix(y_te, preds_q)
    print(f"\n  Confusion matrix (Q6.10):")
    print_confusion(cm, acc_sk, acc_q)
    return acc_sk, acc_q


if __name__ == "__main__":
    print("Loading raw ECG records from PhysioNet cache...")
    raw = load_raw_beats(max_per_class=300)
    counts = [len(raw[c]) for c in range(NUM_CLASSES)]
    print(f"Raw beats per class: {counts}  total={sum(counts)}")

    # (name, feat_single, feat_10beat, feat_100rr, sv_alloc, pad_to)
    CONFIGS = [
        ("A: [128,64,64]  / [95,95,95,120,95]",
         128, 64, 64, [95, 95, 95, 120, 95], None),
        ("B: [64,64,64]   / [120×5]",
         64, 64, 64, [120, 120, 120, 120, 120], None),
        ("C: [96,64,96]   / [95,95,95,120,95]",
         96, 64, 96, [95, 95, 95, 120, 95], None),
        ("D: [160,32,64]  / [95,95,95,120,95]",
         160, 32, 64, [95, 95, 95, 120, 95], None),
        ("E: [128,32,32]  / [120×5] (pad→256)",
         128, 32, 32, [120, 120, 120, 120, 120], 256),
        ("F: [128,32,96]  / [95,95,95,120,95]",
         128, 32, 96, [95, 95, 95, 120, 95], None),
    ]

    summary = []
    for (name, fs, f10, frr, alloc, pad) in CONFIGS:
        acc_sk, acc_q = eval_config(name, raw, fs, f10, frr, alloc, pad_to=pad)
        summary.append((name, acc_sk, acc_q))

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Config':<45}  {'sklearn':>8}  {'Q6.10':>8}")
    for name, acc_sk, acc_q in summary:
        print(f"{name:<45}  {acc_sk:>8.4f}  {acc_q:>8.4f}")
    print()
