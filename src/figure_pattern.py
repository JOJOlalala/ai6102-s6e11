"""What the value-level pattern looks like: raw fraud rates on the labelled competition rows.

Usage (project root, on the machine that holds data/):
  .venv/bin/python -I src/figure_pattern.py --out report_tmp/figs > results/figure_pattern.log
Left panel: fraud rate at each customer age (64 ages) in the competition data, and in 16-year age bands in the source
table (about 300 source rows per age are too few to plot single ages). Right panel: fraud rate by first decimal digit
of three two-decimal fields in the competition data. Nothing here is used for modelling; the labels are the training
labels of the competition and the labels of the source table.
Prints every plotted number and the statistics quoted in the report. Writes <out>/pattern.pdf and pattern.png in the
style of src/figures_report.py.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 of the reference palette (validated, light mode)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
W = 3.25  # CVPR column width in inches
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
DIGIT_FIELDS = [("amount_usd", "amount", BLUE, "o"), ("hours_since_last_txn", "hours", ORANGE, "s"),
                ("distance_from_home_km", "distance", AQUA, "^")]  # short legend names; the caption gives the full ones
BANDS = [(18, 33), (34, 49), (50, 65), (66, 81)]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
ap.add_argument("--out", default="report_tmp/figs")
a = ap.parse_args()
cols = [c for c, *_ in DIGIT_FIELDS] + ["customer_age", "is_fraud"]
tr = pd.read_csv(Path(a.data) / "train.csv", usecols=cols)
og = pd.read_csv(a.orig, usecols=cols)
print(f"competition rows {len(tr)}, frauds {int(tr.is_fraud.sum())} ({tr.is_fraud.mean() * 100:.3f}%); "
      f"source rows {len(og)}, frauds {int(og.is_fraud.sum())} ({og.is_fraud.mean() * 100:.3f}%)")


def rates(k, y):
    g = pd.DataFrame({"k": np.asarray(k), "y": np.asarray(y)}).groupby("k")["y"].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    g["se"] = np.sqrt(g["rate"] * (1 - g["rate"]) / g["size"])
    return g


def chi_square(g, p):
    e1, e0 = g["size"] * p, g["size"] * (1 - p)
    stat = (((g["sum"] - e1) ** 2 / e1) + ((g["size"] - g["sum"] - e0) ** 2 / e0)).sum()
    return stat, len(g) - 1, chi2.sf(stat, len(g) - 1)


first_decimal = lambda x: (np.round(np.asarray(x, dtype=np.float64) * 100).astype(np.int64) // 10) % 10
fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 1.95), layout="constrained", gridspec_kw={"width_ratios": [1.35, 1]})

# ---- left: customer age
gc, gs = rates(tr.customer_age, tr.is_fraud), rates(og.customer_age, og.is_fraud)
print(f"customer age: {len(gc)} ages ({gc.index.min()}-{gc.index.max()}); competition rows per age {int(gc['size'].min())}-"
      f"{int(gc['size'].max())}, source rows per age {int(gs['size'].min())}-{int(gs['size'].max())}")
print("  competition rate (%) by age: " + " ".join(f"{k}:{v * 100:.2f}" for k, v in gc["rate"].items()))
print("  competition rows by age:     " + " ".join(f"{k}:{int(v)}" for k, v in gc["size"].items()))
print("  source rows by age:          " + " ".join(f"{k}:{int(v)}" for k, v in gs["size"].items()))
print(f"  competition, largest standard error of a single-age rate: {gc['se'].max() * 100:.2f} points (age {gc['se'].idxmax()})")
for lo3 in (35, 69):
    print("  " + ", ".join(f"age {k}: {gc.loc[k, 'rate'] * 100:.2f}% ({int(gc.loc[k, 'sum'])} of {int(gc.loc[k, 'size'])})" for k in range(lo3, lo3 + 3)))
stat, df, pv = chi_square(gc, tr.is_fraud.mean())
print(f"  competition, equal rates at all ages: chi-square {stat:.1f} on {df} df, p {pv:.1e}")
inside = [chi_square(rates(d.customer_age[m], d.is_fraud[m]), d.is_fraud[m].mean())
          for d in (tr, og) for lo in range(18, 82, 8) for m in [(d.customer_age >= lo) & (d.customer_age < lo + 8)]]
for name, part in (("competition", inside[:8]), ("source", inside[8:])):
    s, d = sum(p[0] for p in part), sum(p[1] for p in part)
    print(f"  {name}, equal rates inside each of eight 8-year bands: chi-square {s:.1f} on {d} df, p {chi2.sf(s, d):.1e}")
ax.plot(gc.index, gc["rate"] * 100, color=BLUE, lw=0.8, marker="o", ms=1.9, mew=0, zorder=3)
for lo, hi in BANDS:
    for name, d in (("competition", tr), ("source", og)):
        m = (d.customer_age >= lo) & (d.customer_age <= hi)
        n, s = int(m.sum()), int(d.is_fraud[m].sum())
        p = s / n
        half = 1.96 * np.sqrt(p * (1 - p) / n)
        print(f"  ages {lo}-{hi} {name:12s} rows {n:7d} frauds {s:5d} rate {p * 100:.2f}% (95% interval {100 * (p - half):.2f} to {100 * (p + half):.2f})")
        if name == "source":
            ax.plot([lo - 0.4, hi + 0.4], [p * 100] * 2, color=INK2, lw=1.3, solid_capstyle="butt", zorder=4)
            ax.plot([(lo + hi) / 2] * 2, [100 * (p - half), 100 * (p + half)], color=INK2, lw=0.7, zorder=4)
ax.set_xlim(16, 83)
ax.set_ylim(0, 7.4)
ax.set_xticks([20, 40, 60, 80])
ax.set_xlabel("customer age")
ax.set_ylabel("fraud rate (%)")
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.text(17.5, 7.25, "competition data,\neach age", ha="left", va="top", fontsize=6.5, color=BLUE)
ax.text(82.5, 0.12, "source table,\n16-year bands", ha="right", va="bottom", fontsize=6.5, color=INK2)

# ---- right: first decimal digit of three fields
prof = {}
for c, lab, col, mk in DIGIT_FIELDS:
    g = rates(first_decimal(tr[c]), tr.is_fraud)
    prof[c] = g["rate"].to_numpy()
    stat, df, pv = chi_square(g, tr.is_fraud.mean())
    go = rates(first_decimal(og[c]), og.is_fraud)
    so, dfo, pvo = chi_square(go, og.is_fraud.mean())
    print(f"first decimal digit of {c}: competition rate (%) " + " ".join(f"{v * 100:.2f}" for v in g["rate"])
          + f"; largest se {g['se'].max() * 100:.3f}; chi-square {stat:.1f} on {df} df, p {pv:.1e}")
    print(f"  source rate (%) " + " ".join(f"{v * 100:.2f}" for v in go["rate"]) + f"; rows per digit {int(go['size'].min())}-"
          f"{int(go['size'].max())}; largest se {go['se'].max() * 100:.3f}; chi-square {so:.1f} on {dfo} df, p {pvo:.3f}")
    bx.plot(g.index, g["rate"] * 100, color=col, lw=0.9, marker=mk, ms=2.8, mew=0, label=lab, zorder=3)
names = [c for c, *_ in DIGIT_FIELDS]
print("correlation between the digit profiles of two fields (competition): "
      + ", ".join(f"{names[i]} / {names[j]} {np.corrcoef(prof[names[i]], prof[names[j]])[0, 1]:.3f}" for i in range(3) for j in range(i + 1, 3)))
bx.axhline(tr.is_fraud.mean() * 100, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
bx.text(9.45, tr.is_fraud.mean() * 100 + 0.10, "all rows", ha="right", va="bottom", fontsize=6.5, color=INK2)
bx.set_xlim(-0.5, 9.5)
bx.set_ylim(1.1, 2.72)
bx.set_xticks(range(10))
bx.set_xlabel("first decimal digit")
bx.grid(axis="y", color=GRID, lw=0.5, zorder=0)
bx.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.0, labelspacing=0.2, handletextpad=0.5)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"pattern.{ext}")
print(f"wrote {out / 'pattern'}.pdf/.png")
