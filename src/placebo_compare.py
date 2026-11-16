"""Real vs placebo target encoding under the strict protocol, per seed and for the three-seed rank mean.

Usage (project root): .venv/bin/python -I src/placebo_compare.py [--seeds 42,7,2026] [--draws 300]
Reads runs/v2_glm_lgbm_base+digits+te{,_placebo}_strict_m20_esin_s<seed>/oof.npy and the legacy pair for reference.
Paired resampling: AUC difference on random 60,000-row subsets (public-board size) of the same OOF rows; this
covers row sampling only, while the three seeds show the training randomness.
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
name = lambda kind, s: f"v2_glm_lgbm_base+digits+te{'_placebo' if kind == 'placebo' else ''}_strict_m20_esin_s{s}"
load = lambda n: np.load(Path(a.runs) / n / "oof.npy").astype(np.float64)
rank01 = lambda p: (rankdata(p) - 1.0) / (len(p) - 1.0)


def describe(p_real, p_plac, label, rng):
    fr = np.array([roc_auc_score(y[va], p_real[va]) for _, va in folds])
    fp = np.array([roc_auc_score(y[va], p_plac[va]) for _, va in folds])
    full = roc_auc_score(y, p_real) - roc_auc_score(y, p_plac)
    d = []
    for _ in range(a.draws):
        i = rng.choice(len(y), 60000, replace=False)
        d.append(roc_auc_score(y[i], p_real[i]) - roc_auc_score(y[i], p_plac[i]))
    d = np.array(d)
    row = {"label": label, "real": roc_auc_score(y, p_real), "placebo": roc_auc_score(y, p_plac), "diff": full,
           "fold_diff": (fr - fp).tolist(), "folds_improved": int((fr > fp).sum()),
           "paired_sd_60k": float(d.std(ddof=1)), "same_sign_pct": float(np.mean(np.sign(d) == np.sign(full)) * 100)}
    print(f"{label:22s} real {row['real']:.5f}  placebo {row['placebo']:.5f}  diff {full:+.5f}  "
          f"folds improved {row['folds_improved']}/5  per fold " + " ".join(f"{v:+.5f}" for v in fr - fp)
          + f"  | 60k paired sd {row['paired_sd_60k']:.5f}, same sign {row['same_sign_pct']:.0f}%")
    return row


rng = np.random.default_rng(0)
rows = []
real, plac = {}, {}
for s in seeds:
    real[s], plac[s] = load(name("real", s)), load(name("placebo", s))
    rows.append(describe(real[s], plac[s], f"strict seed {s}", rng))
rm_real = np.mean([rank01(real[s]) for s in seeds], axis=0)
rm_plac = np.mean([rank01(plac[s]) for s in seeds], axis=0)
rows.append(describe(rm_real, rm_plac, "strict 3-seed rank mean", rng))
legacy = ("v2_glm_lgbm_base+digits+te_m20_esin_s42", "v2_glm_lgbm_base+digits+te_placebo_m20_esin_s42")
if all((Path(a.runs) / n / "oof.npy").exists() for n in legacy):
    rows.append(describe(load(legacy[0]), load(legacy[1]), "legacy seed 42", rng))
    print(f"strict vs legacy, seed 42: real {rows[0]['real'] - rows[-1]['real']:+.5f}, "
          f"placebo {rows[0]['placebo'] - rows[-1]['placebo']:+.5f}")
diffs = np.array([r["diff"] for r in rows[:len(seeds)]])
print(f"per-seed differences: mean {diffs.mean():+.5f}, sd {diffs.std(ddof=1):.5f}, min {diffs.min():+.5f}")
with open(Path(a.runs) / "placebo_compare.json", "w") as f:
    json.dump(rows, f, indent=1)
