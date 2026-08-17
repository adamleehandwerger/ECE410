#!/usr/bin/env python3
"""
export_hex.py — turn /tmp/svm_cosim_model.npz into $readmemh images for the
native Verilog cosim testbench.

  /tmp/cosim_ram.hex   : unified off-chip RAM, flattened mem[row*256+col] as
                         4-hex-digit Q6.10 words. rows 0..599 = SV matrix,
                         rows 600..899 = the 300 test-feature vectors.
                         (ram_addr = {row[10:0],col[7:0]} = row*256+col, so
                          the core's ram_addr indexes this array directly.)
  /tmp/cosim_alpha.hex : 600 Q6.10 alpha coefficients (one per line).

Also prints gamma / biases / counts / #samples for the TB header, and saves
y_te to /tmp/svm_cosim_yte.npy for scoring.
"""
import numpy as np

M      = np.load("/tmp/svm_cosim_model.npz")
X_te   = M["X_te"].astype(np.int64)     # (Ntest,256)
SV     = M["SV"].astype(np.int64)       # (600,256)
ALPHA  = M["alpha"].astype(np.int64)    # (600,)
BIAS   = M["bias"].astype(np.int64)     # (5,)
GAMMA  = int(M["gamma"])
COUNTS = [int(c) for c in M["counts"]]
y_te   = M["y_te"].astype(np.int64)
NUM_SV = SV.shape[0]
NTEST  = X_te.shape[0]

def h16(v):
    return f"{int(v) & 0xFFFF:04x}"

# Unified RAM: SV rows first (0..599), then feature rows (600..899).
with open("/tmp/cosim_ram.hex", "w") as f:
    for r in range(NUM_SV):
        for c in range(256):
            f.write(h16(SV[r][c]) + "\n")
    for s in range(NTEST):
        for c in range(256):
            f.write(h16(X_te[s][c]) + "\n")

with open("/tmp/cosim_alpha.hex", "w") as f:
    for a in range(NUM_SV):
        f.write(h16(ALPHA[a]) + "\n")

np.save("/tmp/svm_cosim_yte.npy", y_te)

print(f"wrote /tmp/cosim_ram.hex   ({NUM_SV*256 + NTEST*256} words: {NUM_SV} SVs + {NTEST} samples x 256)")
print(f"wrote /tmp/cosim_alpha.hex ({NUM_SV} words)")
print(f"--- TB header values ---")
print(f"  NUM_SV = {NUM_SV}   NTEST = {NTEST}")
print(f"  GAMMA  = 16'h{h16(GAMMA)}  ({GAMMA})")
print(f"  COUNTS = {COUNTS}   num_sv_per_class_flat = 40'h" +
      "".join(f"{c:02x}" for c in reversed(COUNTS)))
print(f"  BIAS   (param 2..6): " + "  ".join(f"16'h{h16(b)}" for b in BIAS))
