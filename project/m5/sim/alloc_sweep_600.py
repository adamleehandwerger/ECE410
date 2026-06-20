"""Sweep non-uniform 600-SV allocations to find Q6.10 optimum."""
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

TOTAL = 600

# Build allocations: uniform + each class getting a boost (others share remainder)
def make_allocs(total=600):
    allocs = []
    # Uniform
    base = total // 5
    allocs.append([base]*5)
    # Each class gets a boost, others share remainder equally
    for boost in [20, 40, 60, 80]:
        for k in range(5):
            extra = boost
            remainder = (total - (base + extra)) // 4
            alloc = [remainder]*5
            alloc[k] = base + extra
            if sum(alloc) == total:
                allocs.append(alloc)
    return allocs

X, y = build_dataset(n_per_class=300)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

allocs = make_allocs(TOTAL)
# Deduplicate
seen = set()
unique = []
for a in allocs:
    key = tuple(a)
    if key not in seen:
        seen.add(key)
        unique.append(a)

print(f"\n{'Allocation':<30} {'Total':>6} {'Float':>8} {'Q6.10':>8} {'Flips':>6}")
print("-"*62)

best_q10 = 0
best_alloc = None

for alloc in unique:
    sv_vecs, sv_alphas, biases, sv_counts = train_binary_ovr(X_tr, y_tr, alloc)
    y_fl  = ovr_predict_float(X_te, sv_vecs, sv_alphas, biases)
    y_q10 = ovr_predict_q10 (X_te, sv_vecs, sv_alphas, biases)
    acc_fl  = accuracy_score(y_te, y_fl)
    acc_q10 = accuracy_score(y_te, y_q10)
    n_flips = int(np.sum(y_fl != y_q10))
    marker = " ←" if acc_q10 > best_q10 else ""
    print(f"{str(alloc):<30} {sum(alloc):>6} {acc_fl:>8.4f} {acc_q10:>8.4f} {n_flips:>6}{marker}")
    if acc_q10 > best_q10:
        best_q10 = acc_q10
        best_alloc = alloc

print(f"\nBest Q6.10: {best_alloc}  →  {best_q10:.4f}")
