#!/usr/bin/env python3
"""
score_cosim.py — score the RTL cosim predictions against ground truth and
render the final confusion matrix + accuracy.

Inputs:
  /tmp/svm_cosim_model.npz   (y_te ground truth, y_q10 python reference)
  /tmp/svm_cosim_preds.npy   (y_hw  RTL cosim predictions)

Outputs:
  <cosim>/confusion_matrix_cosim.png   final hardware confusion matrix
  prints accuracy, per-class breakdown, and RTL-vs-Q6.10 agreement.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

CLASS_NAMES = ["Normal", "PVC", "AFib", "VT", "SVT"]
HERE = os.path.dirname(os.path.abspath(__file__))

M     = np.load("/tmp/svm_cosim_model.npz")
y_te  = M["y_te"].astype(int)
y_q10 = M["y_q10"].astype(int)
y_hw  = np.load("/tmp/svm_cosim_preds.npy").astype(int)

n = len(y_hw)
y_true = y_te[:n]; y_ref = y_q10[:n]
acc     = accuracy_score(y_true, y_hw)
acc_ref = accuracy_score(y_true, y_ref)
agree   = int((y_hw == y_ref).sum())

print(f"=== RTL cosim results  ({n} samples) ===")
print(f"  HARDWARE (RTL) accuracy : {acc:.4f}   ({int(round(acc*n))}/{n})")
print(f"  Python Q6.10 reference  : {acc_ref:.4f}   ({int(round(acc_ref*n))}/{n})")
print(f"  RTL == Q6.10 agreement  : {agree}/{n}  ({100*agree/n:.2f}%)")

cm = confusion_matrix(y_true, y_hw, labels=list(range(5)))
print("\n  Confusion matrix (rows=true, cols=predicted):")
print("           " + " ".join(f"{c:>7s}" for c in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:>7s}  " + " ".join(f"{v:7d}" for v in row))
print("\n  Per-class recall:")
for i in range(5):
    tot = cm[i].sum()
    print(f"    {CLASS_NAMES[i]:>7s}: {cm[i,i]}/{tot} = {100*cm[i,i]/tot:.1f}%" if tot else f"    {CLASS_NAMES[i]}: n/a")

# ---- plot ----
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted (RTL class_out)"); ax.set_ylabel("True label")
ax.set_title(f"svm_compute_core RTL cosim — GF180MCU 600-SV\n"
             f"Accuracy {acc*100:.2f}%  ({int(round(acc*n))}/{n})   "
             f"RTL≡Q6.10: {agree}/{n}")
thr = cm.max()/2.0
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thr else "black", fontsize=11)
plt.tight_layout()
out = os.path.join(HERE, "confusion_matrix_cosim.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  saved -> {out}")
