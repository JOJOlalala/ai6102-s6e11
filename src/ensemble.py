"""Blend saved OOF/test predictions; compare blending methods honestly; optionally write a submission.

Usage (project root):
  .venv/bin/python -I src/ensemble.py --data data --runs runs --names runA,runB,runC [--write hill]

Methods compared on pooled out-of-fold AUC:
  best_single  the single run with the highest OOF AUC
  mean         average of probabilities                      (pre-specified, nothing fitted)
  rank_mean    average of per-model ranks                    (pre-specified, nothing fitted)
  fixed        rank blend with weights given by --weights    (pre-specified, nothing fitted)
  hill_xfit2   greedy forward selection with replacement on ranks (Caruana et al., 2004)
  lr_xfit2     logistic regression on logit(OOF) of every run
Main comparisons in the report use the pre-specified blends. hill_xfit2 / lr_xfit2 choose weights on
4/5 of the OOF rows and score the other 1/5: this is second-level cross-fitting only. The base OOF
predictions were produced by models that saw the labels of the meta-validation rows, so these two
numbers can still be slightly optimistic and are reported as such.
The submission (if --write) refits the chosen method on all OOF rows and applies it to test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rank01(a):
    return (rankdata(a) - 1.0) / (len(a) - 1.0)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def hill_fit(R, y, max_iter=60, tol=1e-6):
    m = R.shape[1]
    aucs = np.array([roc_auc_score(y, R[:, j]) for j in range(m)])
    counts = np.zeros(m)
    best = int(aucs.argmax())
    counts[best] = 1
    cur, cur_auc, n = R[:, best].copy(), aucs[best], 1
    for _ in range(max_iter):
        scores = [roc_auc_score(y, (cur * n + R[:, j]) / (n + 1)) for j in range(m)]
        j = int(np.argmax(scores))
        if scores[j] - cur_auc < tol:
            break
        cur, cur_auc, n = (cur * n + R[:, j]) / (n + 1), scores[j], n + 1
        counts[j] += 1
    return counts / counts.sum()


def lr_fit(L, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(L, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--names", required=True, help="comma-separated run directory names")
    ap.add_argument("--weights", default=None, help="pre-specified rank-blend weights, aligned with --names")
    ap.add_argument("--write", default=None, help="method to refit on all rows and write as a submission")
    ap.add_argument("--tag", default="blend")
    a = ap.parse_args()

    names = [n for n in a.names.split(",") if n]
    y = pd.read_csv(Path(a.data) / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(dtype=int)
    test_ids = pd.read_csv(Path(a.data) / "test.csv", usecols=["id"])["id"].to_numpy()
    P = np.column_stack([np.load(Path(a.runs) / n / "oof.npy") for n in names]).astype(np.float64)
    T = np.column_stack([np.load(Path(a.runs) / n / "test.npy") for n in names]).astype(np.float64)
    R = np.column_stack([rank01(P[:, j]) for j in range(P.shape[1])])
    RT = np.column_stack([rank01(T[:, j]) for j in range(T.shape[1])])
    L, LT = logit(P), logit(T)

    single = {n: roc_auc_score(y, P[:, j]) for j, n in enumerate(names)}
    res = {"best_single": max(single.values()),
           "mean": roc_auc_score(y, P.mean(axis=1)),
           "rank_mean": roc_auc_score(y, R.mean(axis=1))}
    w_fixed = None
    if a.weights:
        w_fixed = np.array([float(v) for v in a.weights.split(",")])
        if len(w_fixed) != len(names):
            raise ValueError("--weights must have one value per run")
        w_fixed = w_fixed / w_fixed.sum()
        res["fixed"] = roc_auc_score(y, R @ w_fixed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_hill, oof_lr = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in skf.split(R, y):
        w = hill_fit(R[tr], y[tr])
        oof_hill[va] = R[va] @ w
        oof_lr[va] = lr_fit(L[tr], y[tr]).decision_function(L[va])
    res["hill_xfit2"] = roc_auc_score(y, oof_hill)
    res["lr_xfit2"] = roc_auc_score(y, oof_lr)

    print("single-run OOF AUC:")
    for n, v in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"  {v:.6f}  {n}")
    print("blend OOF AUC (*_xfit2 = second-level cross-fitting, may be slightly optimistic):")
    for k, v in res.items():
        print(f"  {v:.6f}  {k}")

    w_all = hill_fit(R, y)
    print("hill weights on all rows:", {n: round(float(w), 3) for n, w in zip(names, w_all) if w > 0})

    rec = {"tag": a.tag, "names": names, "single": single, "blend": res,
           "hill_weights": dict(zip(names, map(float, w_all)))}
    with open(Path(a.runs) / "ensemble_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

    if a.write:
        if a.write in ("hill", "hill_xfit2"):
            pred = RT @ w_all
        elif a.write in ("lr_stack", "lr_xfit2"):
            pred = lr_fit(L, y).predict_proba(LT)[:, 1]
        elif a.write == "fixed":
            pred = RT @ w_fixed
        elif a.write == "rank_mean":
            pred = RT.mean(axis=1)
        elif a.write == "mean":
            pred = T.mean(axis=1)
        elif a.write == "best_single":
            pred = T[:, names.index(max(single, key=single.get))]
        else:
            raise ValueError(a.write)
        out = Path("subs")
        out.mkdir(exist_ok=True)
        path = out / f"{a.tag}_{a.write}.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": pred}).to_csv(path, index=False)
        print("wrote", path, "rows", len(pred))


if __name__ == "__main__":
    main()
