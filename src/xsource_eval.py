"""Cross-source evaluation of real vs placebo encodings: competition OOF vs the 20k-row original (source) data.

Usage (project root):
  .venv/bin/python -I src/xsource_eval.py --label "GLM only" \
      --pairs runs/xsource/glm_real_s42:runs/xsource/glm_placebo_s42,runs/xsource/glm_real_s7:runs/xsource/glm_placebo_s7
Each run directory holds oof.npy (competition OOF) and source_fold1..5.npy (the five fold models scored on the
original rows; written by src/xsource_train.py = src/train2.py + codex/source_transfer/transfer.patch).
With several pairs (seeds) the predictions are averaged over seeds and fold models.

Reported for each source:
  AUC of real and placebo and their difference; on the original rows a label-stratified paired bootstrap interval
  (the models stay fixed, rows are resampled, the same resample is used for both arms).
  Transfer coefficient beta in  logit P(y=1) = alpha + gamma * z_placebo + beta * (z_real - z_placebo).
Coverage strata on the original rows: "covered" = both amount_usd and account_balance_usd take a value that occurs
in train.csv (the two columns whose exact-value coverage drops on the original data).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--orig", default="data/credit_card_fraud_2026.csv")
ap.add_argument("--pairs", required=True, help="comma-separated real_dir:placebo_dir")
ap.add_argument("--label", default="")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--json", default=None)
a = ap.parse_args()

lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
pairs = [tuple(Path(x) for x in p.split(":")) for p in a.pairs.split(",") if p]


def fit_beta(yv, zr, zp):
    X = np.column_stack([np.ones(len(yv)), zp, zr - zp])
    w = np.zeros(3)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ w)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (yv - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    ll1 = float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    X0 = X[:, :2]
    w0 = np.zeros(2)
    for _ in range(100):
        p0 = 1 / (1 + np.exp(-(X0 @ w0)))
        step = np.linalg.solve(X0.T @ (X0 * (p0 * (1 - p0))[:, None]), X0.T @ (yv - p0))
        w0 += step
        if np.abs(step).max() < 1e-10:
            break
    p0 = np.clip(1 / (1 + np.exp(-(X0 @ w0))), 1e-12, 1 - 1e-12)
    ll0 = float(np.sum(yv * np.log(p0) + (1 - yv) * np.log(1 - p0)))
    return float(w[2]), float(se[2]), float(chi2.sf(2 * (ll1 - ll0), 1))


def boot_idx(yv, rng):
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    return np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]


def evaluate(name, yv, pr, pp, zr, zp, boot, seed=42):
    ar, apl = roc_auc_score(yv, pr), roc_auc_score(yv, pp)
    b, se, p_lr = fit_beta(yv, zr, zp)
    row = {"set": name, "n": int(len(yv)), "positives": int(yv.sum()), "auc_real": ar, "auc_placebo": apl,
           "delta": ar - apl, "beta": b, "beta_se": se, "beta_wald_ci": [b - 1.96 * se, b + 1.96 * se],
           "beta_lr_p": p_lr}
    msg = (f"  {name:30s} n={len(yv):6d} pos={int(yv.sum()):5d}  AUC real {ar:.5f} placebo {apl:.5f} "
           f"delta {ar - apl:+.5f}")
    if boot:
        rng = np.random.default_rng(seed)
        d, bb = [], []
        for _ in range(boot):
            i = boot_idx(yv, rng)
            d.append(roc_auc_score(yv[i], pr[i]) - roc_auc_score(yv[i], pp[i]))
            bb.append(fit_beta(yv[i], zr[i], zp[i])[0])
        row["delta_boot_ci"] = list(np.percentile(d, [2.5, 97.5]))
        row["beta_boot_ci"] = list(np.percentile(bb, [2.5, 97.5]))
        msg += f" [{row['delta_boot_ci'][0]:+.5f}, {row['delta_boot_ci'][1]:+.5f}]"
    msg += f" | beta {b:+.3f} (Wald [{b - 1.96 * se:+.3f}, {b + 1.96 * se:+.3f}], LR p {p_lr:.4f}"
    msg += f"; boot [{row['beta_boot_ci'][0]:+.3f}, {row['beta_boot_ci'][1]:+.3f}])" if boot else ")"
    print(msg, flush=True)
    return row


tr = pd.read_csv(Path(a.data) / "train.csv", usecols=["amount_usd", "account_balance_usd", "is_fraud"])
og = pd.read_csv(a.orig, usecols=["amount_usd", "account_balance_usd", "is_fraud"])
y_tr, y_og = tr["is_fraud"].to_numpy(dtype=int), og["is_fraud"].to_numpy(dtype=int)
fmt = lambda s: s.round(2).map(lambda x: f"{x:.2f}")
covered = np.ones(len(og), dtype=bool)
for c in ("amount_usd", "account_balance_usd"):
    covered &= fmt(og[c]).isin(set(fmt(tr[c]))).to_numpy()

# competition OOF: probabilities and logits averaged over seeds
pr_c = np.mean([np.load(r / "oof.npy").astype(np.float64) for r, _ in pairs], axis=0)
pp_c = np.mean([np.load(p / "oof.npy").astype(np.float64) for _, p in pairs], axis=0)
zr_c = np.mean([lg(np.load(r / "oof.npy").astype(np.float64)) for r, _ in pairs], axis=0)
zp_c = np.mean([lg(np.load(p / "oof.npy").astype(np.float64)) for _, p in pairs], axis=0)
# original rows: probabilities and logits averaged over seeds and the five fold models
src = lambda d: [np.load(d / f"source_fold{k}.npy").astype(np.float64) for k in range(1, 6)]
pr_o = np.mean([q for r, _ in pairs for q in src(r)], axis=0)
pp_o = np.mean([q for _, p in pairs for q in src(p)], axis=0)
zr_o = np.mean([lg(q) for r, _ in pairs for q in src(r)], axis=0)
zp_o = np.mean([lg(q) for _, p in pairs for q in src(p)], axis=0)
assert len(pr_o) == len(y_og)

print(f"== {a.label}: {len(pairs)} seed pair(s); original rows covered on amount and balance: "
      f"{covered.sum()} ({covered.mean() * 100:.1f}%), positives {int(y_og[covered].sum())} of {int(y_og.sum())}")
rows = [evaluate("competition OOF", y_tr, pr_c, pp_c, zr_c, zp_c, boot=0),
        evaluate("original, all rows", y_og, pr_o, pp_o, zr_o, zp_o, boot=a.boot)]
for flag, nm in ((True, "original, covered"), (False, "original, not covered")):
    m = covered == flag
    rows.append(evaluate(nm, y_og[m], pr_o[m], pp_o[m], zr_o[m], zp_o[m], boot=a.boot))
b1, s1, b0, s0 = rows[2]["beta"], rows[2]["beta_se"], rows[3]["beta"], rows[3]["beta_se"]
z = (b1 - b0) / np.hypot(s1, s0)
print(f"  beta covered - not covered: {b1 - b0:+.3f} (se {np.hypot(s1, s0):.3f}, z {z:+.2f}, "
      f"two-sided p {2 * norm.sf(abs(z)):.3f})")
if a.json:
    Path(a.json).write_text(json.dumps({"label": a.label, "pairs": [f"{r}:{p}" for r, p in pairs], "rows": rows,
                                        "beta_strata_diff": {"diff": b1 - b0, "z": float(z)}}, indent=1))
