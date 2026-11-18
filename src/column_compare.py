"""Where does the value-level gain sit, and does it survive a second fold split? Reads the runs of src/grid_v7.txt.

Usage (project root): .venv/bin/python -I src/column_compare.py [--runs runs]
Main model, strict protocol, inner early stopping, seed 42.
  loo_<col>: all encodings except those of one wide column (its exact value and, for two-decimal columns, its digit keys);
             real - loo_<col>  = what is lost without that column's encodings.
  aoi_<col>: the ten narrow columns plus the encodings of one wide column;
             aoi_<col> - narrow = what that column's encodings add on their own.
Differences are pooled OOF AUCs with a paired DeLong 95% interval (rows resampled, predictions held fixed) and the
number of folds with a positive difference. The fs7_* runs repeat the three arms of the placebo comparison on a second
outer fold split (--fold_seed 7).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--json", default="results/column_compare.json")
a = ap.parse_args()
R = Path(a.runs)
WIDE = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score",
        "merchant_risk_score", "card_age_months", "customer_age"]
y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)


def folds_of(seed):
    f = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y), 1):
        f[va] = k
    return f


def oof(path):
    p = np.load(R / path / "oof.npy").astype(float)
    meta = json.loads((R / path / "meta.json").read_text())
    assert len(p) == len(y) and abs(meta["cv_auc"] - roc_auc_score(y, p)) < 1e-6, path
    return p


def placements(query, reference):
    s = np.sort(reference)
    return (np.searchsorted(s, query, side="left") + np.searchsorted(s, query, side="right")) / (2.0 * len(s))


def compare(pa, pb, folds):
    """AUC(pa) - AUC(pb): pooled difference, paired DeLong 95% interval, folds with a positive difference."""
    def comp(p):
        pos, neg = p[y == 1], p[y == 0]
        return placements(pos, neg), 1 - placements(neg, pos)
    (ua, va), (ub, vb) = comp(pa), comp(pb)
    d, e = ua - ub, va - vb
    delta = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    assert abs(d.mean() - delta) < 1e-10 and abs(e.mean() - delta) < 1e-10
    se = np.sqrt(d.var(ddof=1) / len(d) + e.var(ddof=1) / len(e))
    wins = sum(roc_auc_score(y[folds == k], pa[folds == k]) > roc_auc_score(y[folds == k], pb[folds == k]) for k in range(1, 6))
    return dict(delta=float(delta), low=float(delta - 1.96 * se), high=float(delta + 1.96 * se), folds_positive=int(wins))


fmt = lambda r: f"{r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}] ({r['folds_positive']}/5)"
f42 = folds_of(42)
real = oof("v2_glm_lgbm_base+digits+te_strict_m20_esin_s42")
narrow = oof("v2_glm_lgbm_narrowte_strict_m20_esin_s42")
placebo = oof("v2_glm_lgbm_base+digits+te_placebo_strict_m20_esin_s42")
out = {"reference": {"real": roc_auc_score(y, real), "narrow": roc_auc_score(y, narrow), "placebo": roc_auc_score(y, placebo),
                     "real_minus_narrow": compare(real, narrow, f42), "real_minus_placebo": compare(real, placebo, f42)},
       "columns": {}}
print(f"reference (seed 42): real {out['reference']['real']:.5f}  narrow-only {out['reference']['narrow']:.5f}  "
      f"placebo {out['reference']['placebo']:.5f}")
print(f"  real - narrow-only {fmt(out['reference']['real_minus_narrow'])}")
print(f"{'column':24s} {'AUC aoi':>8s}  {'aoi - narrow-only: added alone':40s} {'AUC loo':>8s}  real - loo: lost when removed")
for c in WIDE:
    pa, pl = oof(f"v7/aoi_{c}"), oof(f"v7/loo_{c}")
    r_aoi, r_loo = compare(pa, narrow, f42), compare(real, pl, f42)
    out["columns"][c] = {"aoi_auc": roc_auc_score(y, pa), "loo_auc": roc_auc_score(y, pl), "added_alone": r_aoi, "lost_when_removed": r_loo}
    print(f"{c:24s} {roc_auc_score(y, pa):8.5f}  {fmt(r_aoi):40s} {roc_auc_score(y, pl):8.5f}  {fmt(r_loo)}")
print(f"sum over the eight columns: added alone {sum(v['added_alone']['delta'] for v in out['columns'].values()):+.5f}, "
      f"lost when removed {sum(v['lost_when_removed']['delta'] for v in out['columns'].values()):+.5f}; "
      f"real - narrow-only {out['reference']['real_minus_narrow']['delta']:+.5f}")

f7 = folds_of(7)
r7, p7, n7 = oof("v7/fs7_real"), oof("v7/fs7_placebo"), oof("v7/fs7_narrow")
out["fold_seed_7"] = {"real": roc_auc_score(y, r7), "narrow": roc_auc_score(y, n7), "placebo": roc_auc_score(y, p7),
                      "real_minus_placebo": compare(r7, p7, f7), "real_minus_narrow": compare(r7, n7, f7),
                      "narrow_minus_placebo": compare(n7, p7, f7)}
s = out["fold_seed_7"]
print(f"second fold split (fold seed 7, model seed 42): real {s['real']:.5f}  narrow-only {s['narrow']:.5f}  placebo {s['placebo']:.5f}")
print(f"  real - placebo {fmt(s['real_minus_placebo'])};  real - narrow-only {fmt(s['real_minus_narrow'])};  "
      f"narrow-only - placebo {fmt(s['narrow_minus_placebo'])}")
Path(a.json).parent.mkdir(parents=True, exist_ok=True)
Path(a.json).write_text(json.dumps(out, indent=1))
