"""Main-result figure for the report: what the fraud rates for the eight wide fields add, and where.

Usage (project root): .venv/bin/python -I src/figure_gain.py [--out report_tmp/figs]
Reads results/bins_compare.json (src/bins_compare.py) and results/eval_boost_narrow_s42.json (src/xsource_eval.py with the
narrow-only run as the second arm). Every point is a change in AUC against the narrow-only model, which uses no fraud
rates for the wide fields: boosted main model, seed 42. Competition intervals are paired DeLong 95% intervals on the
pooled out-of-fold predictions; the source-table interval is the paired, label-stratified bootstrap.
Writes <out>/gain.pdf and gain.png in the style of src/figures_report.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"  # the boosted model keeps the colour it has in the transfer figure
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
W = 3.25  # CVPR column width in inches
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
ap = argparse.ArgumentParser()
ap.add_argument("--bins", default="results/bins_compare.json")
ap.add_argument("--source", default="results/eval_boost_narrow_s42.json")
ap.add_argument("--out", default="report_tmp/figs")
a = ap.parse_args()

d = json.load(open(a.bins))["differences"]
src = {r["set"]: r for r in json.load(open(a.source))["rows"]}["original, all rows"]
rows = [  # (y, label, estimate, low, high)
    (5.0, "rates for 32 bins", *[d["bins32 - narrow-only"][k] for k in ("delta", "low", "high")]),
    (4.0, "rates for 256 bins", *[d["bins256 - narrow-only"][k] for k in ("delta", "low", "high")]),
    (3.0, "rates for 2,048 bins", *[d["bins2048 - narrow-only"][k] for k in ("delta", "low", "high")]),
    (2.0, "exact values and digits", *[d["real - narrow-only"][k] for k in ("delta", "low", "high")]),
    (0.2, "exact values and digits", src["delta"], *src["delta_boot_ci"]),
]
fig, ax = plt.subplots(figsize=(W, 2.15), layout="constrained")
ax.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
for y, lab, est, lo, hi in rows:
    ax.plot([lo * 1e3, hi * 1e3], [y, y], color=BLUE, lw=1.1, solid_capstyle="butt", zorder=2)
    ax.plot(est * 1e3, y, marker="o", ms=4, color=BLUE, mec="white", mew=0.6, ls="none", zorder=3)
    txt = f"{est * 1e3:+.1f}".replace("-", "−")  # the label sits on the side away from zero
    if est >= 0:
        ax.text(hi * 1e3 + 0.25, y, txt, ha="left", va="center", fontsize=6.5, color=INK)
    else:
        ax.text(lo * 1e3 - 0.25, y, txt, ha="right", va="center", fontsize=6.5, color=INK)
    print(f"{lab:26s} y={y:.1f}  {est:+.5f} [{lo:+.5f}, {hi:+.5f}]")
for y, txt in ((5.75, "Competition data"), (0.95, "Source table")):
    ax.text(-9.4, y, txt, ha="left", va="center", fontsize=6.5, color=INK2, style="italic")
ax.text(0, 6.35, "no change", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows], fontsize=6.5)
ax.set_ylim(-0.5, 6.3)
ax.set_xlim(-9.6, 3.6)
ax.set_xlabel("change in AUC ($\\times10^{-3}$)\nagainst the model without these rates")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"gain.{ext}")
print(f"wrote {out / 'gain'}.pdf/.png")
