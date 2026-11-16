"""Compare additive logistic scores with three prespecified pair interactions.

Run with Python, NumPy, SciPy, pandas and scikit-learn. Accepts the official ZIP.
Outputs aligned fold predictions, per-fold metrics, and pooled metrics.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

PAIRS = [
    ("velocity_score", "txn_count_last_24h"),
    ("channel", "auth_method"),
    ("is_foreign_transaction", "ip_country_mismatch"),
]


def transform(frame, train):
    """Fit quantile boundaries on training rows only. Preserve discrete values."""
    levels = {}
    for col in frame:
        values = frame[col]
        fitting = values.iloc[train]
        if pd.api.types.is_numeric_dtype(values) and fitting.nunique() > 32:
            cuts = np.unique(fitting.quantile(np.linspace(0, 1, 33)).to_numpy())
            levels[col] = np.searchsorted(cuts[1:-1], values, side="right")
        else:
            categories = pd.Index(fitting.unique())
            levels[col] = categories.get_indexer(values)
    return pd.DataFrame(levels)


def one_hot(levels, train, valid):
    encoder = OneHotEncoder(handle_unknown="ignore")
    return encoder.fit_transform(levels.iloc[train]), encoder.transform(levels.iloc[valid])


def scores(y, p):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p))


def run(archive, output):
    with ZipFile(archive) as z, z.open("train.csv") as source:
        df = pd.read_csv(source)
    y = df.pop("is_fraud").astype("int8").to_numpy()
    ids = df.pop("id").to_numpy()
    if df.isna().any().any():
        raise ValueError("Missing features require an explicit separate level.")
    result = {}
    records = []
    folds = np.empty(len(y), dtype=np.int8)
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y), 1):
        levels = transform(df, tr)
        base_tr, base_va = one_hot(levels, tr, va)
        pairs = {}
        for a, b in PAIRS:
            # A tuple encoded as a string avoids collisions even for unseen levels (-1).
            pairs[f"{a}*{b}"] = levels[a].astype(str) + "/" + levels[b].astype(str)
        cross_tr, cross_va = one_hot(pd.DataFrame(pairs), tr, va)
        folds[va] = k
        for name in ("additive", "additive_plus_pairs"):
            xtr, xva = base_tr, base_va
            if name == "additive_plus_pairs":
                xtr = sparse.hstack((xtr, cross_tr), format="csr")
                xva = sparse.hstack((xva, cross_va), format="csr")
            fit = LogisticRegression(C=1, solver="newton-cholesky", max_iter=40, tol=1e-7)
            fit.fit(xtr, y[tr])
            pred = fit.predict_proba(xva)[:, 1]
            result.setdefault(name, np.empty(len(y)))[va] = pred
            records.append(dict(fold=k, model=name, n_valid=len(va), positives=int(y[va].sum()),
                                iterations=int(fit.n_iter_[0]), **scores(y[va], pred)))
            print(records[-1], flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "interaction_folds.csv", index=False)
    np.savez_compressed(output / "interaction_oof.npz", id=ids, fold=folds, **result)
    summary = {name: scores(y, p) for name, p in result.items()}
    (output / "interaction_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.archive, args.output)
