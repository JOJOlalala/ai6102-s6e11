"""How noisy is a public-leaderboard-sized AUC? Resample OOF predictions at the public board's size.

Usage (project root): .venv/bin/python -I src/lb_noise.py --runs runs --a RUN_A [--b RUN_B] [--n 60000] [--draws 300]
Reports the AUC of RUN_A on random public-sized subsets (mean, sd, 5-95% range) and, with --b, the paired
difference A - B on the same subsets: how often a public-sized sample ranks the two runs the same way as CV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--a", required=True)
ap.add_argument("--b", default=None)
ap.add_argument("--n", type=int, default=60000, help="public board ~= 20% of 299,984 test rows")
ap.add_argument("--draws", type=int, default=300)
a = ap.parse_args()

y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
pa = np.load(Path(a.runs) / a.a / "oof.npy")
pb = np.load(Path(a.runs) / a.b / "oof.npy") if a.b else None
rng = np.random.default_rng(0)
sa, sd = [], []
for _ in range(a.draws):
    idx = rng.choice(len(y), size=a.n, replace=False)
    sa.append(roc_auc_score(y[idx], pa[idx]))
    if pb is not None:
        sd.append(sa[-1] - roc_auc_score(y[idx], pb[idx]))
sa = np.array(sa)
print(f"{a.a}: full OOF AUC {roc_auc_score(y, pa):.5f}; public-sized ({a.n} rows, {a.draws} draws): "
      f"mean {sa.mean():.5f} sd {sa.std(ddof=1):.5f} 5-95% [{np.percentile(sa, 5):.5f}, {np.percentile(sa, 95):.5f}]")
if pb is not None:
    sd = np.array(sd)
    full = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    print(f"A - B: full OOF {full:+.5f}; public-sized mean {sd.mean():+.5f} sd {sd.std(ddof=1):.5f}; "
          f"same sign as full OOF in {np.mean(np.sign(sd) == np.sign(full)) * 100:.0f}% of draws")
