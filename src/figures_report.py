"""Column-width figures for the CVPR-format report, drawn at their printed size (3.25 in), plus numbers quoted in the text.

Usage (project root, on the machine that holds runs/ and data/):
  .venv/bin/python -I src/figures_report.py --out report_tmp/figs \
      --eval_boost report_tmp/eval_boost_s42.json --eval_glm report_tmp/eval_glm_3seeds.json
Writes <out>/{depth,lbnoise,transfer}.pdf (+ .png previews):
  depth     per-fold AUC change vs depth 2, LightGBM depths 1-6 on the raw features (learning rate 0.05)
  lbnoise   AUC of the main single-seed model on 300 random public-board-sized (60k-row) subsets of its OOF predictions
  transfer  transfer coefficient beta with 95% intervals: competition OOF vs the original (source) rows
Prints: per-fold depth differences, data facts (row and fraud counts, mean amount, flag rates in competition vs
source), exact-value coverage inside the competition folds, and the columns that the placebo control shuffles
(numeric columns with more than 32 distinct values, and their digit keys). The printed output of the run behind the
report is saved as results/figures_report.log.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette (validated light mode)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
W = 3.25  # CVPR column width in inches
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
DEPTH = {d: f"lgbm_learning_rate0.05_max_depth{d}_num_leaves{2 ** d}_base_orig0_s42" for d in (1, 2, 3, 4, 6)}
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out / name}.pdf/.png")


def fig_depth(log, out):
    ref = np.array(log[DEPTH[2]]["fold_auc"])
    ds = sorted(DEPTH)
    fig, ax = plt.subplots(figsize=(W, 1.7), layout="constrained")
    for d in ds:
        delta = (np.array(log[DEPTH[d]]["fold_auc"]) - ref) * 1e3
        ax.scatter(np.full(len(delta), d), delta, s=10, color=BLUE, alpha=0.4, linewidths=0, zorder=2)
        print(f"depth {d}: pooled {log[DEPTH[d]]['cv_auc']:.5f}  per-fold vs depth 2 (x1e-3) "
              + " ".join(f"{v:+.2f}" for v in delta) + f"  folds below depth 2: {int(np.sum(delta < 0))}/5")
    pooled = [(log[DEPTH[d]]["cv_auc"] - log[DEPTH[2]]["cv_auc"]) * 1e3 for d in ds]
    ax.plot(ds, pooled, color=BLUE, lw=1.4, marker="o", ms=3.5, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.set_xticks(ds)
    ax.set_xlabel(r"tree depth (leaves $= 2^{\mathrm{depth}}$)")
    ax.set_ylabel("ΔAUC vs depth 2 (×10⁻³)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    save(fig, out, "depth")


def fig_lbnoise(y, p, threshold, out):
    rng = np.random.default_rng(0)
    draws = np.array([roc_auc_score(y[i], p[i]) for i in (rng.choice(len(y), 60000, replace=False)
                                                          for _ in range(300))])
    full = roc_auc_score(y, p)
    share = np.mean(draws >= threshold) * 100
    lo, hi = np.percentile(draws, [5, 95])
    print(f"lbnoise: full OOF {full:.5f}; 300 draws sd {draws.std(ddof=1):.5f}, 5-95% [{lo:.4f}, {hi:.4f}], "
          f"share >= {threshold} {share:.1f}%")
    fig, ax = plt.subplots(figsize=(W, 1.65), layout="constrained")
    ax.hist(draws, bins=24, color=BLUE, edgecolor="white", linewidth=0.5, zorder=2)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.32)
    ax.axvline(full, color=INK, lw=0.9, zorder=3)
    ax.annotate(f"full OOF\n{full:.5f}", xy=(full, top * 1.3), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="top", fontsize=6.5, color=INK)
    ax.axvline(threshold, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=3)
    # The cut-off is other teams' score on the public rows and would move with the sample, so it is shown for scale only.
    ax.annotate(f"public top-10% score\n{threshold:.5f} (15 Nov)", xy=(threshold, top * 1.3),
                xytext=(3, 0), textcoords="offset points", ha="left", va="top", fontsize=6.5, color=INK2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("AUC on a random 60,000-row subset of the OOF predictions")
    ax.set_ylabel("draws (of 300)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    save(fig, out, "lbnoise")


def fig_transfer(ev_boost, ev_glm, out):
    rows = [("competition OOF", "Competition OOF\n(699,960 rows)"),
            ("original, all rows", "Source, all rows\n(20,000 rows)"),
            ("original, covered", "Source, amount and\nbalance seen (10,729)"),
            ("original, not covered", "Source, amount or\nbalance unseen (9,271)")]
    series = [(ev_boost, "GLM + 4-leaf LightGBM, seed 42", BLUE, "o", -0.13),
              (ev_glm, "GLM only, 3 seeds", ORANGE, "s", 0.13)]
    fig, ax = plt.subplots(figsize=(W, 2.05), layout="constrained")
    ys = np.arange(len(rows))[::-1].astype(float)
    for ev, lab, col, mk, dy in series:
        byset = {r["set"]: r for r in ev["rows"]}
        for (key, _), yv in zip(rows, ys):
            r = byset[key]
            lo, hi = r.get("beta_boot_ci", r["beta_wald_ci"])
            ax.plot([lo, hi], [yv + dy] * 2, color=col, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(r["beta"], yv + dy, marker=mk, ms=4, color=col, mec="white", mew=0.6, ls="none", zorder=3)
            print(f"transfer {lab:36s} {key:22s} beta {r['beta']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        ax.plot([], [], marker=mk, ms=4, color=col, mec="white", mew=0.6, lw=1.1, label=lab)
    for xv, txt in ((0, "ignored"), (1, "used in full")):
        ax.plot([xv, xv], [-0.6, len(rows) - 0.5], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)  # stops below its label
        ax.text(xv, len(rows) - 0.45, txt, ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=6.5)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlim(-0.3, 1.12)
    ax.set_xlabel(r"weight $\beta$ (95% interval)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.legend(loc="outside upper center", ncol=2, frameon=False, handlelength=1.6, columnspacing=1.2)
    save(fig, out, "transfer")


def coverage_and_placebo(data_dir, orig_path):
    tr = pd.read_csv(Path(data_dir) / "train.csv")
    te = pd.read_csv(Path(data_dir) / "test.csv")
    og = pd.read_csv(orig_path)
    y = tr["is_fraud"].to_numpy()
    print(f"rows: train {len(tr)}, test {len(te)}, source {len(og)}; frauds: train {int(y.sum())} "
          f"({y.mean() * 100:.2f}%), source {int(og['is_fraud'].sum())} ({og['is_fraud'].mean() * 100:.2f}%); "
          f"missing values: train {int(tr.isna().sum().sum())}, source {int(og.isna().sum().sum())}")
    print(f"mean amount_usd: competition train {tr['amount_usd'].mean():.1f}, source {og['amount_usd'].mean():.1f}")
    for c in BOOL:
        a, b = tr[c].astype(float).mean(), og[c].astype(float).mean()
        print(f"flag {c:30s} competition {a * 100:6.2f}%  source {b * 100:6.2f}%  ratio {b / a:5.2f}")
    feats = [c for c in tr.columns if c not in ("id", "is_fraud")]
    allx = pd.concat([tr[feats], te[feats]], ignore_index=True)
    num = [c for c in feats if c not in CAT + BOOL]
    wide = [c for c in num if allx[c].nunique() > 32]
    print(f"numeric columns: {len(num)}; with > 32 distinct values (placebo-shuffled, with their digit keys): {wide}")
    print("distinct values over train+test: " + ", ".join(f"{c} {allx[c].nunique()}" for c in num))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for c in FLOATS:
        cents = np.round(tr[c].to_numpy(dtype=np.float64) * 100).astype(np.int64)
        shares = []
        for tri, vai in skf.split(np.zeros(len(y)), y):
            shares.append(np.isin(cents[vai], np.unique(cents[tri])).mean())
        tcent = np.round(te[c].to_numpy(dtype=np.float64) * 100).astype(np.int64)
        print(f"exact-value coverage {c:24s} validation fold rows seen in training folds {np.mean(shares) * 100:6.2f}%"
              f"  test rows seen in train {np.isin(tcent, np.unique(cents)).mean() * 100:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="report_tmp/figs")
    ap.add_argument("--best", default="v2_glm_lgbm_base+digits+te_m20_s42")
    ap.add_argument("--threshold", type=float, default=0.95349)
    ap.add_argument("--eval_boost", required=True)
    ap.add_argument("--eval_glm", required=True)
    ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
    a = ap.parse_args()
    out = Path(a.out)
    log = {}
    for line in (Path(a.runs) / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        log[r["name"]] = r
    fig_depth(log, out)
    y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
    fig_lbnoise(y, np.load(Path(a.runs) / a.best / "oof.npy"), a.threshold, out)
    fig_transfer(json.loads(Path(a.eval_boost).read_text()), json.loads(Path(a.eval_glm).read_text()), out)
    coverage_and_placebo(a.data, a.orig)


if __name__ == "__main__":
    main()
