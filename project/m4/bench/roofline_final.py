"""
roofline_final.py — ECE410 SVM ASIC (m4)
Dual-panel figure:
  Left:  Traditional roofline (Orca Xeon DRAM/L3/compute ceilings)
  Right: Throughput vs active power with iso-efficiency lines
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ────────────────────────────────────────────────────────────────────

implementations = {
    "sklearn\n(1 core)":      dict(ops_b=2.0, gops=2.15,    mw=15_000,  inf_s=4_200,  color="#4878d0", marker="s"),
    "Numba\n(8 cores)":       dict(ops_b=2.0, gops=48.6,    mw=80_000,  inf_s=95_000, color="#ee854a", marker="^"),
    "ASIC\nsky130A":          dict(ops_b=2.0, gops=0.1582,  mw=66,      inf_s=309,    color="#6acc65", marker="o"),
}

# Orca Xeon platform ceilings
DRAM_BW_GBs   = 100    # GB/s  (Orca dual-channel DDR4)
L3_BW_GBs     = 400    # GB/s  (L3 effective streaming BW)
PEAK_GOPS     = 1200   # GOPS  (Orca 8-core AVX-512 peak, ~150 GOPS/core)

# Wearable classification requirement
WEARABLE_INF_S = 1.34  # inf/s at 80 bpm

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("ECE410 SVM ASIC — Roofline & Power Efficiency (m4)\n"
             "MIT-BIH 5-class Arrhythmia (256-dim, 500 SVs, Q6.10, RAM_LATENCY)",
             fontsize=12, fontweight="bold", y=1.01)

# ╔══════════════════════════════════╗
# ║  LEFT PANEL — Roofline           ║
# ╚══════════════════════════════════╝
ax = axes[0]

oi = np.logspace(-2, 4, 400)   # operational intensity (ops/byte)

# Memory ceiling lines
dram_ridge  = PEAK_GOPS / DRAM_BW_GBs
l3_ridge    = PEAK_GOPS / L3_BW_GBs

dram_roof = np.minimum(DRAM_BW_GBs * oi, PEAK_GOPS)
l3_roof   = np.minimum(L3_BW_GBs   * oi, PEAK_GOPS)

ax.loglog(oi, dram_roof, color="steelblue",   lw=1.8, ls="--",  label=f"Orca DRAM  ({DRAM_BW_GBs} GB/s)")
ax.loglog(oi, l3_roof,   color="darkorange",  lw=1.8, ls="-.",  label=f"Orca L3    ({L3_BW_GBs} GB/s)")
ax.axhline(PEAK_GOPS,    color="firebrick",   lw=1.8, ls=":",   label=f"Orca peak  ({PEAK_GOPS} GOPS)")

# Ridge points (vertical dotted lines)
for ridge, bw_label, col in [(dram_ridge, "DRAM ridge", "steelblue"),
                               (l3_ridge,  "L3 ridge",   "darkorange")]:
    ax.axvline(ridge, color=col, lw=0.8, ls=":", alpha=0.5)

# Implementation points
for label, d in implementations.items():
    ax.scatter(d["ops_b"], d["gops"],
               color=d["color"], marker=d["marker"],
               s=120, zorder=5, label=label.replace("\n", " "))
    ax.annotate(label,
                xy=(d["ops_b"], d["gops"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=d["color"], fontweight="bold")

ax.set_xlabel("Operational Intensity (ops / byte)", fontsize=10)
ax.set_ylabel("Performance (GOPS)", fontsize=10)
ax.set_title("Roofline Model — Orca Xeon vs ASIC", fontsize=10)
ax.set_xlim(0.1, 1000)
ax.set_ylim(1e-4, 1e4)
ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.set_facecolor("#f9f9f9")

# ╔══════════════════════════════════╗
# ║  RIGHT PANEL — Power Efficiency  ║
# ╚══════════════════════════════════╝
ax2 = axes[1]

# Iso-efficiency lines (inf/J = throughput / power)
power_range_mw = np.logspace(1, 8, 400)   # 10 mW → 100 W
for eff, ls_style in [(1e6, ":"), (1e4, "-."), (1e2, "--"), (1e0, "-")]:
    inf_per_J = eff
    y_vals = inf_per_J * (power_range_mw * 1e-3)    # power in W
    label = f"{eff:,.0f} inf/J" if eff >= 1 else f"{eff:.1f} inf/J"
    ax2.loglog(power_range_mw, y_vals, color="gray", lw=1.0, ls=ls_style,
               alpha=0.6, label=label)

# Wearable requirement line
ax2.axhline(WEARABLE_INF_S, color="crimson", lw=1.5, ls="--",
            label=f"Wearable min ({WEARABLE_INF_S} inf/s @ 80 bpm)")

# Duty-cycled ASIC average power point
asic_avg_mw  = 0.284
asic_inf_s   = 309
ax2.scatter(asic_avg_mw, asic_inf_s, color="#6acc65", marker="o", s=150,
            zorder=6, label="ASIC duty-cycled (0.284 mW avg)")
ax2.annotate("ASIC\nduty-cycled\n0.284 mW",
             xy=(asic_avg_mw, asic_inf_s),
             xytext=(8, -30), textcoords="offset points",
             fontsize=7.5, color="#3a8c36", fontweight="bold")

# Active power implementation points
for label, d in implementations.items():
    ax2.scatter(d["mw"], d["inf_s"],
                color=d["color"], marker=d["marker"],
                s=120, zorder=5, label=label.replace("\n", " ") + " (active)")
    ax2.annotate(label,
                 xy=(d["mw"], d["inf_s"]),
                 xytext=(5, 4), textcoords="offset points",
                 fontsize=8, color=d["color"], fontweight="bold")

ax2.set_xlabel("Active Power (mW)", fontsize=10)
ax2.set_ylabel("Throughput (inf / s)", fontsize=10)
ax2.set_title("Throughput vs Power — Active & Duty-Cycled", fontsize=10)
ax2.set_xlim(0.1, 2e5)
ax2.set_ylim(0.1, 5e5)
ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.85, ncol=1)
ax2.grid(True, which="both", ls=":", alpha=0.4)
ax2.set_facecolor("#f9f9f9")

# ── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout()
out = "roofline_final.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")
