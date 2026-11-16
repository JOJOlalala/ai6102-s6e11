"""Univariate out-of-fold test for residual signal in exact numeric values.

Start from 32 training quantile bins. Shrink each exact value's training
fraud rate toward its bin. A placebo shuffles training labels within bins
while retaining the number of observations per value and positives per bin.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

FIELDS = ("merchant_risk_score", "velocity_score", "amount_usd")


def run(archive, out):
    with ZipFile(archive) as z, z.open("train.csv") as f:
        df = pd.read_csv(f, usecols=[*FIELDS, "is_fraud"])
    y = df["is_fraud"].to_numpy(dtype=int)
    rng = np.random.default_rng(6102)
    summaries, folds = [], []
    for field in FIELDS:
        x = df[field].to_numpy()
        pred = {m: np.empty(len(y)) for m in ("bin", "exact", "placebo")}
        supported = np.zeros(len(y), dtype=bool)
        for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y), 1):
            cuts = np.unique(np.quantile(x[tr], np.linspace(0, 1, 33)))[1:-1]
            btr, bva = np.searchsorted(cuts, x[tr], side="right"), np.searchsorted(cuts, x[va], side="right")
            size = np.bincount(btr, minlength=len(cuts)+1)
            positives = np.bincount(btr, weights=y[tr], minlength=len(size))
            base = (positives + 20*y[tr].mean())/(size+20)
            pred["bin"][va] = base[bva]
            vals, inv, counts = np.unique(x[tr], return_inverse=True, return_counts=True)
            ix = np.searchsorted(vals, x[va]).clip(0, len(vals)-1)
            seen = vals[ix] == x[va]
            supported[va] = seen
            fake = y[tr].copy()
            for group in np.unique(btr):
                mask = btr == group
                fake[mask] = rng.permutation(fake[mask])
            for name, target in (("exact", y[tr]), ("placebo", fake)):
                pos = np.bincount(inv, weights=target)
                score = base[bva].copy()
                score[seen] = (pos[ix[seen]] + 20*score[seen]) / (counts[ix[seen]] + 20)
                pred[name][va] = score
            for name, p in pred.items():
                folds.append(dict(field=field, method=name, fold=k, auc=roc_auc_score(y[va], p[va])))
        for name, p in pred.items():
            summaries.append(dict(field=field, method=name, coverage=supported.mean(),
                                  auc=roc_auc_score(y,p), ap=average_precision_score(y,p)))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(folds).to_csv(out/"value_folds.csv",index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/"value_summary.csv",index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    run(args.archive,args.output)
