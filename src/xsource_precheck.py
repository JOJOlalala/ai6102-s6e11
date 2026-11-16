"""Pre-checks for a cross-source test (score competition-trained models on the 20k-row original dataset).

Usage (project root): .venv/bin/python -I src/xsource_precheck.py [--orig data/credit_card_fraud_2026.csv]
1. Overlap: original rows whose 24 features exactly match a competition train/test row, rows matching on the six
   float columns, and per-column exact-value coverage (share of original rows whose value occurs in train.csv).
2. Power: if the real-vs-placebo gain transferred unchanged, how often would a set of the original's size
   (same number of positives and negatives) show it? Uses the strict three-seed OOF predictions on competition rows.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
ap.add_argument("--draws", type=int, default=1000)
a = ap.parse_args()

tr = pd.read_csv(Path(a.data) / "train.csv")
te = pd.read_csv(Path(a.data) / "test.csv")
og = pd.read_csv(a.orig)
feats = [c for c in tr.columns if c not in ("id", "is_fraud")]
assert set(feats) <= set(og.columns), set(feats) - set(og.columns)
y_tr = tr["is_fraud"].to_numpy(dtype=int)
y_og = og["is_fraud"].to_numpy(dtype=int)
print(f"original: {len(og)} rows, {y_og.sum()} positives ({y_og.mean() * 100:.2f}%)")


def norm_frame(df):
    out = pd.DataFrame(index=df.index)
    for c in feats:
        v = df[c]
        out[c] = v.round(2).map(lambda x: f"{x:.2f}") if v.dtype.kind == "f" else v.astype(str)
    return out


N_tr, N_te, N_og = norm_frame(tr), norm_frame(te), norm_frame(og)
key = lambda d, cols: d[cols].agg("|".join, axis=1)
for label, cols in (("all 24 features", feats), ("six float columns", FLOATS)):
    k_og = key(N_og, cols)
    in_tr, in_te = k_og.isin(set(key(N_tr, cols))), k_og.isin(set(key(N_te, cols)))
    print(f"original rows matching a competition row on {label}: train {in_tr.sum()}, test {in_te.sum()}, "
          f"either {(in_tr | in_te).sum()} (positives among them: {y_og[(in_tr | in_te).to_numpy()].sum()})")
print("exact-value coverage of original rows in train.csv (share of original rows whose value occurs in train):")
num = [c for c in feats if tr[c].dtype.kind in "if" and tr[c].nunique() > 2]
for c in num:
    seen = N_og[c].isin(set(N_tr[c]))
    print(f"  {c:24s} unique in original {og[c].nunique():6d}  covered {seen.mean() * 100:6.2f}%  "
          f"(positives covered {seen[y_og == 1].mean() * 100:6.2f}%)")

# power on competition OOF rows
rank01 = lambda p: (rankdata(p) - 1.0) / (len(p) - 1.0)
name = lambda kind, s: f"v2_glm_lgbm_base+digits+te{'_placebo' if kind == 'placebo' else ''}_strict_m20_esin_s{s}"
seeds = (42, 7, 2026)
real = np.mean([rank01(np.load(Path(a.runs) / name("real", s) / "oof.npy")) for s in seeds], axis=0)
plac = np.mean([rank01(np.load(Path(a.runs) / name("placebo", s) / "oof.npy")) for s in seeds], axis=0)
full = roc_auc_score(y_tr, real) - roc_auc_score(y_tr, plac)
pos, neg = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
n1, n0 = int(y_og.sum()), int(len(y_og) - y_og.sum())
rng = np.random.default_rng(0)
d = []
for _ in range(a.draws):
    i = np.r_[rng.choice(pos, n1, replace=False), rng.choice(neg, n0, replace=False)]
    d.append(roc_auc_score(y_tr[i], real[i]) - roc_auc_score(y_tr[i], plac[i]))
d = np.array(d)
sd = d.std(ddof=1)
print(f"competition 3-seed rank mean: real - placebo = {full:+.5f}")
print(f"original-sized subsets ({n1} positives, {n0} negatives, {a.draws} draws): mean {d.mean():+.5f}, sd {sd:.5f}, "
      f"share > 0 {np.mean(d > 0) * 100:.1f}%, share > 1.645 sd {np.mean(d > 1.645 * sd) * 100:.1f}%")
print(f"one-sided 5% test, power if the full effect transferred: {norm.cdf(full / sd - 1.645) * 100:.0f}%; "
      f"if half transferred: {norm.cdf(0.5 * full / sd - 1.645) * 100:.0f}%; "
      f"smallest effect detectable with 80% power: {(1.645 + 0.842) * sd:.5f}")

# 3. a one-parameter alternative: the transfer coefficient beta in
#    logit P(y=1) = alpha + gamma * z_placebo + beta * (z_real - z_placebo),   z = seed-mean logit of the OOF probability
#    beta = 0: the value-level signal adds nothing; beta = 1: it holds at full strength. Wald test of beta = 0.
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
zr = np.mean([lg(np.load(Path(a.runs) / name("real", s) / "oof.npy").astype(np.float64)) for s in seeds], axis=0)
zp = np.mean([lg(np.load(Path(a.runs) / name("placebo", s) / "oof.npy").astype(np.float64)) for s in seeds], axis=0)


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


b_all, se_all = fit_beta(y_tr, zp, zr)
print(f"transfer coefficient on all competition OOF rows: beta {b_all:.3f} (se {se_all:.3f})")
bs, zs = [], []
rng = np.random.default_rng(1)
for _ in range(300):
    i = np.r_[rng.choice(pos, n1, replace=False), rng.choice(neg, n0, replace=False)]
    b, se = fit_beta(y_tr[i], zp[i], zr[i])
    bs.append(b), zs.append(b / se)
bs, zs = np.array(bs), np.array(zs)
print(f"original-sized subsets (300 draws): beta mean {bs.mean():.3f}, sd {bs.std(ddof=1):.3f}; "
      f"power of one-sided 5% Wald test if the full effect transferred: {np.mean(zs > 1.645) * 100:.0f}%")
