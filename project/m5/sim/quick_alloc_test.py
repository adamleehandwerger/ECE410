"""Quick allocation test — imports from confusion_comparison_m5."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from confusion_comparison_m5 import (
    build_dataset, train_binary_ovr,
    ovr_predict_float, ovr_predict_q10,
    CLASS_NAMES
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

ALLOC = [115, 115, 115, 140, 115]   # total = 600

X, y = build_dataset(n_per_class=300)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

sv_vecs, sv_alphas, biases, sv_counts = train_binary_ovr(X_tr, y_tr, ALLOC)
print(f"SV counts: {sv_counts}  total: {sv_counts.sum()}")

y_fl  = ovr_predict_float(X_te, sv_vecs, sv_alphas, biases)
y_q10 = ovr_predict_q10 (X_te, sv_vecs, sv_alphas, biases)

acc_fl  = accuracy_score(y_te, y_fl)
acc_q10 = accuracy_score(y_te, y_q10)
n_flips = int(np.sum(y_fl != y_q10))

print(f"\nAlloc {ALLOC}  total={sum(ALLOC)}")
print(f"  Float : {acc_fl:.4f}  ({int(acc_fl*len(y_te))}/{len(y_te)})")
print(f"  Q6.10 : {acc_q10:.4f}  ({int(acc_q10*len(y_te))}/{len(y_te)})")
print(f"  Flips : {n_flips}")
if n_flips:
    for i in np.where(y_fl != y_q10)[0]:
        print(f"    [{i}] true={CLASS_NAMES[y_te[i]]} "
              f"float={CLASS_NAMES[y_fl[i]]} Q6.10={CLASS_NAMES[y_q10[i]]}")
