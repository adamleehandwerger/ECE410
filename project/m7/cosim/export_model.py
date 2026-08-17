#!/usr/bin/env python3
"""
export_model.py — build the 300-sample PhysioNet test set + train the 600-SV
[120x5] OVR RBF-SVM, and export everything in the exact Q6.10 form the
svm_compute_core RTL consumes, so the cocotb cosim can drive the real design.

Reuses confusion_comparison_m6.py (the m6 tool that realizes 98.67%) verbatim
for data + model, so the split (random_state=42) and thus the 296/300 result
are reproduced exactly.

Emits /tmp/svm_cosim_model.npz with:
  X_te  (Ntest,256) int   Q6.10 test features
  y_te  (Ntest,)    int   true labels
  SV    (600,256)   int   Q6.10 support vectors, ordered by class (120 each)
  alpha (600,)      int   Q6.10 dual coefficients (round(alpha*1024))
  bias  (5,)        int   Q6.10 per-class OVR intercepts
  gamma int               Q6.10 gamma (=256, i.e. 0.25)
  counts(5,)        int   SVs per class [120,120,120,120,120]
  y_q10 (Ntest,)    int   pure-Python Q6.10 reference predictions (should be 296/300)
"""
import sys, os, numpy as np
sys.path.insert(0, "/Users/user/Desktop/Indently/project/m6/sim")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import confusion_comparison_m6 as m6

SCALE = m6.SCALE  # 1024

def q10(x):
    return np.clip(np.round(np.asarray(x) * SCALE), -32768, 32767).astype(np.int32)

def main():
    X, y = m6.build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    sv_vecs, sv_alphas, biases, counts = m6.train_binary_ovr(X_tr, y_tr, m6.SV_ALLOC)

    # Concatenate SVs + alphas in class order (class 0's 120, then class 1's, ...)
    SV_all    = np.vstack([m6.vecs_to_q10(v) for v in sv_vecs]).astype(np.int32)  # (600,256)
    alpha_all = np.concatenate(sv_alphas).astype(np.float64)                       # (600,)
    alpha_q   = q10(alpha_all)                                                     # Q6.10
    bias_q    = q10(biases)                                                        # Q6.10
    gamma_q   = int(round(m6.DEFAULT_GAMMA * SCALE))
    X_te_q    = m6.vecs_to_q10(X_te)

    # Reference: pure-Python Q6.10 model (the 98.67% path)
    y_q10 = m6.ovr_predict_q10(X_te, sv_vecs, sv_alphas, biases)
    acc = accuracy_score(y_te, y_q10)

    print(f"\n  test samples          : {len(y_te)}")
    print(f"  SV total / per class  : {SV_all.shape[0]}  {list(counts)}")
    print(f"  alpha float range     : [{alpha_all.min():+.4f}, {alpha_all.max():+.4f}]  -> Q6.10 [{alpha_q.min()},{alpha_q.max()}]")
    print(f"  bias  float           : {np.round(biases,4)}")
    print(f"  bias  Q6.10 (must fit +-32768) : {list(bias_q)}")
    print(f"  gamma Q6.10           : {gamma_q}")
    print(f"  Python Q6.10 accuracy : {acc:.4f}  ({int(round(acc*len(y_te)))}/{len(y_te)})")

    out = "/tmp/svm_cosim_model.npz"
    np.savez(out, X_te=X_te_q, y_te=np.asarray(y_te, dtype=np.int32),
             SV=SV_all, alpha=alpha_q, bias=bias_q, gamma=np.int32(gamma_q),
             counts=np.asarray(counts, dtype=np.int32), y_q10=np.asarray(y_q10, dtype=np.int32))
    print(f"  saved -> {out}")

if __name__ == "__main__":
    main()
