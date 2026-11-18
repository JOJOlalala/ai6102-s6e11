"""Does down-weighting the value-level signal recover AUC on the source table? No retraining.

Usage (project root): .venv/bin/python -I src/xsource_shrink.py [--boot 2000]
For the boosted model (Codex's cross-source runs, seed 42) and the GLM alone (runs/xsource, three seeds), mix the
logits of the placebo and real arms as
    s_b = z_placebo + b * (z_real - z_placebo),
so that b = 0 is the placebo arm and b = 1 the real arm, and report the AUC of s_b on both tables.
z = logit of the prediction, averaged over seeds (competition: OOF; source: also over the five fold models), as in
src/xsource_beta.py. AUCs at b = 0 and b = 1 therefore differ slightly from src/xsource_eval.py, which averages
probabilities.
On the source rows three choices of b are compared with the placebo by a label-stratified paired bootstrap:
  b = 0.25              a fixed value, picked after seeing the source coefficient (post hoc);
  b from competition    beta / gamma of the logistic fit on the competition rows (the weight the competition supports);
  b cross-fitted        beta / gamma fitted on four fifths of the source rows and applied to the remaining fifth.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
ap.add_argument("--boost", default="codex/source_transfer/runs")
ap.add_argument("--glm", default="runs/xsource")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
y_tr = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
y_og = pd.read_csv(a.orig, usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)


def fit(yv, X):
    """Unpenalised logistic regression by Newton's method; returns the coefficients."""
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ w)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (yv - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def load(model):
    if model == "boost":
        R = Path(a.boost)
        zr_c, zp_c = lg(np.load(R / "real" / "oof.npy").astype(float)), lg(np.load(R / "placebo" / "oof.npy").astype(float))
        zr_o = np.mean([lg(np.load(R / "real" / f"source_fold{k}.npy")) for k in range(1, 6)], axis=0)
        zp_o = np.mean([lg(np.load(R / "placebo" / f"source_fold{k}.npy")) for k in range(1, 6)], axis=0)
    else:
        R, seeds = Path(a.glm), (42, 7, 2026)
        zr_c = np.mean([lg(np.load(R / f"glm_real_s{s}" / "oof.npy").astype(float)) for s in seeds], axis=0)
        zp_c = np.mean([lg(np.load(R / f"glm_placebo_s{s}" / "oof.npy").astype(float)) for s in seeds], axis=0)
        zr_o = np.mean([lg(np.load(R / f"glm_real_s{s}" / f"source_fold{k}.npy")) for s in seeds for k in range(1, 6)], axis=0)
        zp_o = np.mean([lg(np.load(R / f"glm_placebo_s{s}" / f"source_fold{k}.npy")) for s in seeds for k in range(1, 6)], axis=0)
    assert len(zr_c) == len(y_tr) and len(zr_o) == len(y_og) == 20000
    return zr_c, zp_c, zr_o, zp_o


for model, label in (("boost", "GLM-initialised 4-leaf LightGBM, seed 42"), ("glm", "GLM only, seeds 42, 7 and 2026")):
    zr_c, zp_c, zr_o, zp_o = load(model)
    wc = fit(y_tr, np.column_stack([np.ones(len(y_tr)), zp_c, zr_c - zp_c]))
    wo = fit(y_og, np.column_stack([np.ones(len(y_og)), zp_o, zr_o - zp_o]))
    print(f"== {label}")
    print(f"   competition: gamma {wc[1]:.3f}  beta {wc[2]:.3f}  beta/gamma {wc[2] / wc[1]:.3f}")
    print(f"   source:      gamma {wo[1]:.3f}  beta {wo[2]:.3f}  beta/gamma {wo[2] / wo[1]:.3f}")
    print("   competition AUC by b: " + "  ".join(f"{b:.2f}: {roc_auc_score(y_tr, zp_c + b * (zr_c - zp_c)):.5f}" for b in GRID))
    print("   source AUC by b:      " + "  ".join(f"{b:.2f}: {roc_auc_score(y_og, zp_o + b * (zr_o - zp_o)):.5f}" for b in GRID))
    b_comp = wc[2] / wc[1]
    s_cf, b_cf = np.zeros(len(y_og)), []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(zp_o, y_og):
        w = fit(y_og[tr], np.column_stack([np.ones(len(tr)), zp_o[tr], (zr_o - zp_o)[tr]]))
        b_cf.append(w[2] / w[1])
        s_cf[va] = zp_o[va] + b_cf[-1] * (zr_o - zp_o)[va]
    print("   b cross-fitted on the source rows, by fold: " + " ".join(f"{b:.3f}" for b in b_cf))
    scores = {"real arm (b = 1)": zr_o, "b = 0.25 (fixed, post hoc)": zp_o + 0.25 * (zr_o - zp_o),
              f"b from competition ({b_comp:.2f})": zp_o + b_comp * (zr_o - zp_o), "b cross-fitted on source": s_cf}
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(y_og == 1), np.flatnonzero(y_og == 0)
    idx = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(a.boot)]
    auc0 = roc_auc_score(y_og, zp_o)
    base = np.array([roc_auc_score(y_og[i], zp_o[i]) for i in idx])
    print(f"   source, placebo arm (b = 0): AUC {auc0:.5f}")
    for name, s in scores.items():
        full = roc_auc_score(y_og, s)
        d = np.array([roc_auc_score(y_og[i], s[i]) for i in idx]) - base
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"   source, {name:28s} AUC {full:.5f}  minus placebo {full - auc0:+.5f}  bootstrap 95% [{lo:+.5f}, {hi:+.5f}]")
