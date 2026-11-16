"""Transfer coefficient for Codex's cross-source runs (codex/source_transfer), no retraining.

Usage (project root): .venv/bin/python -I src/xsource_beta.py [--runs codex/source_transfer/runs]
Model, fitted separately on each source:
  logit P(y=1) = alpha + gamma * z_placebo + beta * (z_real - z_placebo)
z = logit of the prediction (competition: OOF; original data: mean over the five fold models of the logit).
beta = 1: the real-minus-placebo signal holds at full strength; 0: it adds nothing; < 0: it points the wrong way.
alpha and gamma absorb any shift in base rate and overall calibration between the sources.
Reports the Wald 95% interval, the likelihood-ratio test of beta = 0 and a label-stratified bootstrap interval.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="codex/source_transfer/runs")
ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
R = Path(a.runs)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def fit(yv, X):
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (yv - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    return w, se, ll


def report(label, yv, zr, zp, boot):
    X1 = np.column_stack([np.ones(len(yv)), zp, zr - zp])
    w, se, ll1 = fit(yv, X1)
    _, _, ll0 = fit(yv, X1[:, :2])
    lr = 2 * (ll1 - ll0)
    b, s = w[2], se[2]
    line = (f"{label:34s} n={len(yv)} pos={int(yv.sum())}  beta {b:+.3f}  Wald 95% [{b - 1.96 * s:+.3f}, {b + 1.96 * s:+.3f}]"
            f"  z {b / s:+.2f}  LR chi2 {lr:.2f} (p={chi2.sf(lr, 1):.4f})")
    if boot:
        rng = np.random.default_rng(42)
        pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
        bs = []
        for _ in range(boot):
            i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            bs.append(fit(yv[i], X1[i])[0][2])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        line += f"  bootstrap 95% [{lo:+.3f}, {hi:+.3f}]"
    print(line, flush=True)


y_tr = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
y_og = pd.read_csv(a.orig, usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
zr_c, zp_c = lg(np.load(R / "real" / "oof.npy").astype(np.float64)), lg(np.load(R / "placebo" / "oof.npy").astype(np.float64))
report("competition OOF (seed 42)", y_tr, zr_c, zp_c, boot=0)
zr_o = np.mean([lg(np.load(R / "real" / f"source_fold{k}.npy")) for k in range(1, 6)], axis=0)
zp_o = np.mean([lg(np.load(R / "placebo" / f"source_fold{k}.npy")) for k in range(1, 6)], axis=0)
assert len(zr_o) == len(y_og) == 20000
report("original 20k rows (5-model mean)", y_og, zr_o, zp_o, boot=a.boot)
for k in range(1, 6):
    report(f"original 20k rows, fold model {k}", y_og, lg(np.load(R / "real" / f"source_fold{k}.npy")),
           lg(np.load(R / "placebo" / f"source_fold{k}.npy")), boot=0)
print("one-sided p for beta < 0 on the original rows equals half the two-sided Wald p when beta is negative;"
      f" reference: z = -1.645 is p = {norm.cdf(-1.645):.3f}")
