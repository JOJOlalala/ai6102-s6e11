"""Two read-only checks for the report text (no training, nothing written).

1. How close are 2,048 quantile bins to exact values? For each wide field: distinct values over train+test, bins that
   pd.qcut(..., 2048, duplicates="drop") really produces, and the share of rows in a bin that holds a single value.
2. Sample-size check for the weight beta with the seed-42 boosted predictions of the transfer table: 300 subsets of the
   competition rows with the source table's class counts (339 / 19,661), stratified, without replacement, rng seed 1,
   the same fit as src/xsource_precheck.py.
Usage (project root): .venv/bin/python -I src/check_bins_beta.py > results/check_bins_beta.log
--boost is the folder with the real/ and placebo/ runs of the boosted model scored on the source table (the folder
that this repository calls codex/ was codex_review/ when the runs were made).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--boost", default="codex_review/source_transfer/runs")
a = ap.parse_args()
WIDE = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score",
        "merchant_risk_score", "card_age_months", "customer_age"]
tr = pd.read_csv(Path(a.data) / "train.csv", usecols=WIDE + ["is_fraud"])
te = pd.read_csv(Path(a.data) / "test.csv", usecols=WIDE)
allx = pd.concat([tr[WIDE], te[WIDE]], ignore_index=True)
print(f"rows train+test: {len(allx)}")
for k in (2048, 256, 32):
    print(f"--- {k} quantile bins")
    for c in WIDE:
        b = pd.qcut(allx[c], k, labels=False, duplicates="drop")
        per_bin = allx[c].groupby(b).nunique()
        single = per_bin.index[per_bin == 1]
        share = b.isin(single).mean() * 100
        print(f"{c:24s} distinct {allx[c].nunique():6d}  bins {b.nunique():5d}  single-value bins {len(single):5d}  "
              f"rows in single-value bins {share:6.2f}%")

y = tr["is_fraud"].to_numpy(dtype=int)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
R = Path(a.boost)
zr = lg(np.load(R / "real" / "oof.npy").astype(np.float64))
zp = lg(np.load(R / "placebo" / "oof.npy").astype(np.float64))
assert len(zr) == len(zp) == len(y)


def fit_beta(yv, z_p, z_r):
    X = np.column_stack([np.ones(len(yv)), z_p, z_r - z_p])
    w = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (yv - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(X @ w)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    return w[2], se[2]


b_all, se_all = fit_beta(y, zp, zr)
print(f"--- weight beta, boosted model, seed 42 (real vs shuffled): all competition rows {b_all:.5f} (se {se_all:.5f})")
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
n1, n0 = 339, 19661
rng = np.random.default_rng(1)
bs = []
for _ in range(300):
    i = np.r_[rng.choice(pos, n1, replace=False), rng.choice(neg, n0, replace=False)]
    bs.append(fit_beta(y[i], zp[i], zr[i])[0])
bs = np.array(bs)
print(f"300 subsets with {n1} frauds and {n0} normal rows: beta mean {bs.mean():.5f}, sd {bs.std(ddof=1):.5f}, "
      f"min {bs.min():.3f}, 2.5% {np.percentile(bs, 2.5):.3f}, 97.5% {np.percentile(bs, 97.5):.3f}, max {bs.max():.3f}")
for src in (0.21761, 0.26):
    print(f"source beta {src}: {(bs.mean() - src) / bs.std(ddof=1):.2f} sd below the mean; subsets at or below it: {(bs <= src).sum()}")
