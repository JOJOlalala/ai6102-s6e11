"""How fine does a fraud-rate encoding have to be? Reads the coarse-encoding runs of src/grid_v9.txt.

Usage (project root): .venv/bin/python -I src/bins_compare.py [--runs runs]
Main model, strict protocol, inner early stopping, seed 42. binsK encodes each of the eight wide numeric columns by one
of K quantile bins instead of by its exact value and encodes no digit key. The reference arms are narrow-only (no
encodings for the wide columns), the within-bin shuffle and the real encodings (exact values and digit keys).
Differences are pooled OOF AUCs with a paired DeLong 95% interval and the number of folds with a positive difference.
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
ap.add_argument("--json", default="results/bins_compare.json")
a = ap.parse_args()
R = Path(a.runs)
y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
folds = np.zeros(len(y), dtype=int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y), 1):
    folds[va] = k


def oof(path):
    p = np.load(R / path / "oof.npy").astype(float)
    meta = json.loads((R / path / "meta.json").read_text())
    assert len(p) == len(y) and abs(meta["cv_auc"] - roc_auc_score(y, p)) < 1e-6, path
    return p


def placements(query, reference):
    s = np.sort(reference)
    return (np.searchsorted(s, query, side="left") + np.searchsorted(s, query, side="right")) / (2.0 * len(s))


def compare(pa, pb):
    def comp(p):
        pos, neg = p[y == 1], p[y == 0]
        return placements(pos, neg), 1 - placements(neg, pos)
    (ua, va), (ub, vb) = comp(pa), comp(pb)
    d, e = ua - ub, va - vb
    delta = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    se = np.sqrt(d.var(ddof=1) / len(d) + e.var(ddof=1) / len(e))
    wins = sum(roc_auc_score(y[folds == k], pa[folds == k]) > roc_auc_score(y[folds == k], pb[folds == k]) for k in range(1, 6))
    return dict(delta=float(delta), low=float(delta - 1.96 * se), high=float(delta + 1.96 * se), folds_positive=int(wins))


fmt = lambda r: f"{r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}] ({r['folds_positive']}/5)"
arms = {"narrow-only": oof("v2_glm_lgbm_narrowte_strict_m20_esin_s42"),
        "within-bin shuffle": oof("v2_glm_lgbm_base+digits+te_placebo_strict_m20_esin_s42"),
        "bins32": oof("v9/bins32"), "bins256": oof("v9/bins256"), "bins2048": oof("v9/bins2048"),
        "real": oof("v2_glm_lgbm_base+digits+te_strict_m20_esin_s42")}
out = {"auc": {k: float(roc_auc_score(y, p)) for k, p in arms.items()}, "differences": {}}
print("  ".join(f"{k} {v:.5f}" for k, v in out["auc"].items()))
for a_, b_ in (("bins32", "narrow-only"), ("bins256", "narrow-only"), ("bins2048", "narrow-only"), ("real", "narrow-only"),
               ("bins32", "within-bin shuffle"), ("bins256", "bins32"), ("bins2048", "bins256"), ("real", "bins2048"),
               ("real", "bins32")):
    out["differences"][f"{a_} - {b_}"] = r = compare(arms[a_], arms[b_])
    print(f"{a_:9s} - {b_:19s} {fmt(r)}")
Path(a.json).write_text(json.dumps(out, indent=1))
