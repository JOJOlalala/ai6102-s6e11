"""Run from the project root after both strict five-fold fits. Score source labels only here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def placements(scores, reference):
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, scores, side="left")
    right = np.searchsorted(ordered, scores, side="right")
    return (left + right) / (2 * len(ordered))


def components(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    return placements(pos, neg), 1 - placements(neg, pos)


def paired_interval(labels, real, placebo):
    # The same source transactions are scored by all five models. Average
    # per-row DeLong placements before estimating the conditional variance.
    pos_diff, neg_diff = [], []
    for r, p in zip(real, placebo):
        ur, vr = components(labels, r)
        up, vp = components(labels, p)
        pos_diff.append(ur - up)
        neg_diff.append(vr - vp)
    u, v = np.mean(pos_diff, axis=0), np.mean(neg_diff, axis=0)
    delta = float(u.mean())
    np.testing.assert_allclose(delta, v.mean(), atol=1e-10)
    se = np.sqrt(u.var(ddof=1) / len(u) + v.var(ddof=1) / len(v))
    return [delta, float(delta - 1.96 * se), float(delta + 1.96 * se)]


def bootstrap_source(labels, real, placebo, n_boot=2000):
    # Sample the same positive and negative row IDs in every arm and fold.
    # Fixed score order allows exact bootstrap AUC, with a half-credit for ties.
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    scores = [*real, *placebo, real.mean(axis=0), placebo.mean(axis=0)]
    order = []
    for pred in scores:
        ns = np.argsort(pred[neg], kind="stable")
        ordered = pred[neg][ns]
        lower = np.searchsorted(ordered, pred[pos], side="left")
        upper = np.searchsorted(ordered, pred[pos], side="right")
        order.append((ns, lower, upper))
    rng = np.random.default_rng(42)
    differences = np.empty((n_boot, 2))
    for b in range(n_boot):
        wp = np.bincount(rng.integers(len(pos), size=len(pos)), minlength=len(pos))
        wn = np.bincount(rng.integers(len(neg), size=len(neg)), minlength=len(neg))
        aucs = []
        for ns, lower, upper in order:
            cumul = np.r_[0, np.cumsum(wn[ns])]
            aucs.append(wp @ (cumul[lower] + cumul[upper]) / (2.0 * len(pos) * len(neg)))
        k = len(real)
        differences[b] = (np.mean(aucs[:k]) - np.mean(aucs[k:2*k]), aucs[-2] - aucs[-1])
    return dict(replicates=n_boot, seed=42, mean_fold_ci=np.quantile(differences[:,0], [.025,.975]).tolist(),
                ensemble_ci=np.quantile(differences[:,1], [.025,.975]).tolist())


def row(scope, fold, labels, real, placebo):
    auc_r, auc_p = [float(roc_auc_score(labels, p)) for p in (real, placebo)]
    return dict(domain=scope, fold=fold, n=len(labels), positives=int(labels.sum()),
                real=auc_r, placebo=auc_p, delta=auc_r-auc_p)


def main():
    train = pd.read_csv(ROOT / "data/train.csv")
    source = pd.read_csv(ROOT / "data/credit_card_fraud_2026.csv")
    cols = train.columns.difference(["id", "is_fraud"]).tolist()
    assert len(cols) == 24
    assert not source.duplicated(cols).any()
    assert source[cols].merge(train[cols].drop_duplicates()).empty
    y, ys = train.is_fraud.to_numpy(), source.is_fraud.to_numpy()
    source_scores, oof = {}, {}
    for arm in ("real", "placebo"):
        path = HERE / "runs" / arm
        source_scores[arm] = np.stack([np.load(path / f"source_fold{k}.npy") for k in range(1, 6)])
        oof[arm] = np.load(path / "oof.npy")
        assert source_scores[arm].shape == (5, len(source))
        assert np.isfinite(source_scores[arm]).all()
        np.testing.assert_allclose(source_scores[arm].mean(axis=0), np.load(path / "test.npy")[-len(source):], atol=1e-7)
    rows, coverage = [], []
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        rows += [row("competition", k+1, y[va], oof["real"][va], oof["placebo"][va]),
                 row("source", k+1, ys, source_scores["real"][k], source_scores["placebo"][k])]
        fit, _ = train_test_split(tr, stratify=y[tr], test_size=0.1, random_state=42+200+k)
        for col in ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]:
            seen = set(train.iloc[fit][col].astype(str))
            coverage.append(dict(fold=k+1,column=col,competition=train.iloc[va][col].astype(str).isin(seen).mean(),
                                 source=source[col].astype(str).isin(seen).mean()))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("domain")[["real","placebo","delta"]].mean().to_dict(orient="index")
    real, placebo = source_scores["real"], source_scores["placebo"]
    summary["source_mean_delta_ci"] = paired_interval(ys, real, placebo)
    summary["source_ensemble"] = row("source", "ensemble", ys, real.mean(axis=0), placebo.mean(axis=0))
    summary["source_stratified_paired_bootstrap"] = bootstrap_source(ys, real, placebo)
    summary["source_ensemble_ci"] = paired_interval(ys, [real.mean(axis=0)], [placebo.mean(axis=0)])
    summary["competition_pooled_oof"] = row("competition","pooled", y, oof["real"], oof["placebo"])
    frame.to_csv(HERE / "folds.csv", index=False)
    pd.DataFrame(coverage).to_csv(HERE / "exact_value_coverage.csv", index=False)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
