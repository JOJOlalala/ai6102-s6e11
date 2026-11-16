"""Split the real-vs-placebo gap with a clean "no value-level encoding" control (src/grid_v6.txt).

Usage (project root): .venv/bin/python -I src/narrow_compare.py [--seeds 42,7,2026] [--draws 300]
All arms are the main model (GLM-initialised 4-leaf LightGBM, digit keys as raw features, strict encoding protocol,
early stopping on the held-out 10% of the outer-training rows):
  real     v2_glm_lgbm_base+digits+te_strict_m20_esin_s<seed>          all 48 keys target-encoded
  narrow   v2_glm_lgbm_narrowte_strict_m20_esin_s<seed>                only the 10 narrow columns encoded (5 numeric
                                                                       columns with <= 32 distinct values, 5 categoricals)
  placebo  v2_glm_lgbm_base+digits+te_placebo_strict_m20_esin_s<seed>  the 38 wide-column keys encoded from labels
                                                                       shuffled within 32 quantile bins
real - narrow   = net gain of the value-level encodings over not having them
narrow - placebo = what adding within-bin-shuffled encodings costs (or gains)
For each seed and for the three-seed rank mean: pooled OOF AUC, per-fold differences, and the paired difference on
random 60,000-row subsets (public-board size; row sampling only, the seeds show the training randomness).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--seeds", default="42,7,2026")
ap.add_argument("--draws", type=int, default=300)
a = ap.parse_args()

y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
seeds = [int(s) for s in a.seeds.split(",")]
NAME = {"real": "v2_glm_lgbm_base+digits+te_strict_m20_esin_s{}",
        "narrow": "v2_glm_lgbm_narrowte_strict_m20_esin_s{}",
        "placebo": "v2_glm_lgbm_base+digits+te_placebo_strict_m20_esin_s{}"}
load = lambda arm, s: np.load(Path(a.runs) / NAME[arm].format(s) / "oof.npy").astype(np.float64)
rank01 = lambda p: (rankdata(p) - 1.0) / (len(p) - 1.0)
PAIRS = [("real", "narrow"), ("narrow", "placebo"), ("real", "placebo")]


def compare(p1, p2, label, rng):
    f1 = np.array([roc_auc_score(y[va], p1[va]) for _, va in folds])
    f2 = np.array([roc_auc_score(y[va], p2[va]) for _, va in folds])
    full = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    d = np.array([roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i])
                  for i in (rng.choice(len(y), 60000, replace=False) for _ in range(a.draws))])
    row = {"label": label, "auc_1": roc_auc_score(y, p1), "auc_2": roc_auc_score(y, p2), "diff": full,
           "fold_diff": (f1 - f2).tolist(), "folds_positive": int((f1 > f2).sum()),
           "paired_sd_60k": float(d.std(ddof=1)), "same_sign_pct": float(np.mean(np.sign(d) == np.sign(full)) * 100)}
    print(f"  {label:20s} {row['auc_1']:.5f} - {row['auc_2']:.5f} = {full:+.5f}  folds > 0: {row['folds_positive']}/5"
          f"  per fold " + " ".join(f"{v:+.5f}" for v in f1 - f2)
          + f"  | 60k paired sd {row['paired_sd_60k']:.5f}, same sign {row['same_sign_pct']:.0f}%")
    return row


rng = np.random.default_rng(0)
preds = {arm: {s: load(arm, s) for s in seeds} for arm in NAME}
out = {}
for s in seeds:
    print(f"seed {s}: real {roc_auc_score(y, preds['real'][s]):.5f}, narrow {roc_auc_score(y, preds['narrow'][s]):.5f}, "
          f"placebo {roc_auc_score(y, preds['placebo'][s]):.5f}")
    out[f"seed {s}"] = [compare(preds[p][s], preds[q][s], f"{p} - {q}", rng) for p, q in PAIRS]
rm = {arm: np.mean([rank01(preds[arm][s]) for s in seeds], axis=0) for arm in NAME}
print("3-seed rank mean: " + ", ".join(f"{arm} {roc_auc_score(y, rm[arm]):.5f}" for arm in NAME))
out["3-seed rank mean"] = [compare(rm[p], rm[q], f"{p} - {q}", rng) for p, q in PAIRS]
for k, (p, q) in enumerate(PAIRS):
    d = np.array([out[f"seed {s}"][k]["diff"] for s in seeds])
    print(f"{p} - {q} over seeds: mean {d.mean():+.5f}, sd {d.std(ddof=1):.5f}, min {d.min():+.5f}, max {d.max():+.5f}")
with open(Path(a.runs) / "narrow_compare.json", "w") as f:
    json.dump(out, f, indent=1)
