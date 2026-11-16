"""Report figures from runs/log.jsonl (PDF for LaTeX + PNG for preview) into figs/.

Usage (project root): .venv/bin/python -I src/figures.py --best RUN_NAME [--threshold 0.95349]
  fig_depth   per-fold AUC change vs depth 2 for LightGBM depths 1-6 (fixed learning rate)
  fig_ladder  pooled CV AUC of each step of the feature ladder (dot plot; AUC axis does not start at 0)
  fig_lbnoise AUC of the best run on 300 random public-board-sized subsets of its OOF predictions
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

DEPTH = {1: "lgbm_learning_rate0.05_max_depth1_num_leaves2_base_orig0_s42",
         2: "lgbm_learning_rate0.05_max_depth2_num_leaves4_base_orig0_s42",
         3: "lgbm_learning_rate0.05_max_depth3_num_leaves8_base_orig0_s42",
         4: "lgbm_learning_rate0.05_max_depth4_num_leaves16_base_orig0_s42",
         6: "lgbm_learning_rate0.05_max_depth6_num_leaves64_base_orig0_s42"}
LADDER = [("Logistic regression, raw features", "lr_base_orig0_s42"),
          ("LightGBM, 63 leaves", "lgbm_base_orig0_s42"),
          ("LightGBM, 4 leaves", "v2_lgbm_base_m20_s42"),
          ("4 leaves + digit keys", "v2_lgbm_base+digits_m20_s42"),
          ("4 leaves + value target encoding", "v2_lgbm_base+te_m20_s42"),
          ("4 leaves + both", "v2_lgbm_base+digits+te_m20_s42"),
          ("GLM on the encodings", "v2_glm_base+digits+te_m20_s42"),
          ("GLM-initialised boosting", "v2_glm_lgbm_base+digits+te_m20_s42")]


def save(fig, name):
    Path("figs").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figs/{name}.{ext}")
    plt.close(fig)
    print("wrote", f"figs/{name}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--best", required=True)
    ap.add_argument("--threshold", type=float, default=0.95349)
    a = ap.parse_args()
    log = {}
    for line in (Path(a.runs) / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        log[r["name"]] = r

    # depth: per-fold difference to depth 2
    if all(n in log for n in DEPTH.values()):
        ref = np.array(log[DEPTH[2]]["fold_auc"])
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ds = sorted(DEPTH)
        for d in ds:
            delta = np.array(log[DEPTH[d]]["fold_auc"]) - ref
            ax.scatter([d] * len(delta), delta * 1e3, s=14, color=BLUE, alpha=0.45, linewidths=0)
        pooled = [(log[DEPTH[d]]["cv_auc"] - log[DEPTH[2]]["cv_auc"]) * 1e3 for d in ds]
        ax.plot(ds, pooled, color=BLUE, lw=2, marker="o", ms=5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(ds)
        ax.set_xlabel("tree depth (leaves = 2^depth, learning rate 0.05)")
        ax.set_ylabel("AUC change vs depth 2 (×10⁻³)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        save(fig, "fig_depth")

    # ladder dot plot
    steps = [(lab, log[n]["cv_auc"]) for lab, n in LADDER if n in log]
    if steps:
        fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(steps) + 0.6))
        ys = np.arange(len(steps))[::-1]
        vals = [v for _, v in steps]
        ax.hlines(ys, min(vals) - 0.0005, vals, color=GRID, lw=1)
        ax.scatter(vals, ys, color=BLUE, s=28, zorder=3)
        for yv, v in zip(ys, vals):
            ax.text(v + 0.0002, yv, f"{v:.4f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for lab, _ in steps])
        ax.set_xlabel("5-fold out-of-fold ROC AUC")
        ax.set_xlim(min(vals) - 0.0005, max(vals) + 0.0016)
        save(fig, "fig_ladder")

    # public-board noise
    y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
    p = np.load(Path(a.runs) / a.best / "oof.npy")
    rng = np.random.default_rng(0)
    draws = np.array([roc_auc_score(y[i], p[i]) for i in (rng.choice(len(y), 60000, replace=False)
                                                          for _ in range(300))])
    full = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(draws, bins=24, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.25)
    ax.axvline(full, color=INK, lw=1.2)
    ax.text(full, top * 1.22, f"full OOF\n{full:.4f}", fontsize=7.5, color=INK, va="top", ha="right")
    ax.axvline(a.threshold, color=ORANGE, lw=1.2, ls="--")
    ax.text(a.threshold, top * 1.22, f" top-10% public\n {a.threshold:.5f}", fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("AUC on a random 60,000-row subset (public-board size)")
    ax.set_ylabel("draws")
    save(fig, "fig_lbnoise")
    print(f"public-sized draws: sd {draws.std(ddof=1):.5f}; share above threshold "
          f"{np.mean(draws >= a.threshold) * 100:.1f}%")


if __name__ == "__main__":
    main()
