"""Stack our model with the public S6E11 OOF Bank on the same five folds; write the stack and the bonus blend.

Usage (project root):
  .venv/bin/python -I src/stack_bank.py --bank public/oof_bank \
      --ours v2_glm_lgbm_base+digits+te_m20_esin_s42,v2_glm_lgbm_base+digits+te_m20_esin_s7,v2_glm_lgbm_base+digits+te_m20_esin_s2026 \
      --public public/lm_ladder/lm_ladder_submission_v7.csv --public_weight 0.8

Sources (credit both in the report):
  goodpjw2008/s6e11-oof-bank, Kaggle dataset, CC0, version 1: 70 models with out-of-fold predictions on
    StratifiedKFold(5, shuffle=True, random_state=42), the split every run in this project uses.
  goodpjw2008, "S6E11 LM Ladder | Forecast the Board | LB 0.95417", notebook version 7, Apache 2.0:
    its submission.csv. Its mixes were chosen with public-leaderboard probes, so it has no usable CV.

Fixed before any result was seen:
  members   the 20 'core' bank members (manifest.csv: one per recipe, no near-copies) plus one column for our
            model: the rank mean of three seeds of the main model trained with inner-split early stopping
  features  logit of each column's rank, as in the bank's README
  stacker   LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky"), as in the README
  scoring   second-level cross-fitting on the same five folds: the stacker scored on fold k is fitted on the
            other four folds' OOF rows. The base OOF predictions were made on these folds too.
  outputs   subs/sub2_bankstack.csv        the stack refitted on all OOF rows, applied to test
            subs/sub3_bonus_public80.csv   public_weight * rank(public submission) + (1 - public_weight) * rank(sub2)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

N_TRAIN, N_TEST = 699_960, 299_984


def rank01(a):
    return (rankdata(a) - 1.0) / (len(a) - 1.0)


def logit_rank(r):
    r = np.clip(r, 1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


def stacker():
    return LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky")


def check_sub(path, ids):
    s = pd.read_csv(path)
    ok = (list(s.columns) == ["id", "is_fraud"] and len(s) == len(ids) and s["id"].is_unique
          and np.array_equal(s["id"].to_numpy(), ids) and s["is_fraud"].between(0, 1).all()
          and np.isfinite(s["is_fraud"]).all())
    print(f"check {path}: rows {len(s)}, ids match test order {np.array_equal(s['id'].to_numpy(), ids)}, "
          f"range [{s['is_fraud'].min():.4f}, {s['is_fraud'].max():.4f}] -> {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(f"{path} failed the format check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bank", default="public/oof_bank")
    ap.add_argument("--ours", required=True, help="comma-separated run names of our model (seeds)")
    ap.add_argument("--public", default=None, help="public submission CSV for the bonus blend")
    ap.add_argument("--public_weight", type=float, default=0.8)
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    bank = Path(a.bank)

    tr = pd.read_csv(Path(a.data) / "train.csv", usecols=["id", "is_fraud"])
    y = tr["is_fraud"].to_numpy(dtype=int)
    test_ids = pd.read_csv(Path(a.data) / "test.csv", usecols=["id"])["id"].to_numpy()
    assert len(y) == N_TRAIN and len(test_ids) == N_TEST

    # 1. the bank must use exactly our folds
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fb = pd.read_csv(bank / "folds.csv")
    print("folds.csv columns:", list(fb.columns))
    fold_col = [c for c in fb.columns if c.lower() != "id"][0]
    if "id" in fb.columns and not np.array_equal(fb["id"].to_numpy(), tr["id"].to_numpy()):
        raise SystemExit("folds.csv is not in train.csv order")
    fv = fb[fold_col].to_numpy()
    same = all(len(np.unique(fv[va])) == 1 and (fv == fv[va][0]).sum() == len(va) for _, va in folds)
    print("bank folds identical to StratifiedKFold(5, shuffle=True, random_state=42):", same)
    if not same:
        raise SystemExit("fold mismatch: stop, the stack would not be cross-fitted")

    # 2. core members, with their reported OOF AUC re-computed from the files
    man = pd.read_csv(bank / "manifest.csv")
    core = man["core"].astype(str).str.lower().isin(["true", "1", "yes"])
    names = man.loc[core, "name"].tolist()
    auc_col = next((c for c in man.columns if c.lower() in ("oof_auc", "auc", "oof auc")), None)
    P = np.column_stack([np.load(bank / f"oof_{n}.npy", allow_pickle=False) for n in names]).astype(np.float64)
    T = np.column_stack([np.load(bank / f"test_{n}.npy", allow_pickle=False) for n in names]).astype(np.float64)
    assert P.shape == (N_TRAIN, len(names)) and T.shape == (N_TEST, len(names))
    assert np.isfinite(P).all() and np.isfinite(T).all()
    print(f"{len(names)} core members")
    worst = 0.0
    for j, n in enumerate(names):
        auc = roc_auc_score(y, P[:, j])
        rep = float(man.loc[man["name"] == n, auc_col].iloc[0]) if auc_col else float("nan")
        worst = max(worst, abs(auc - rep)) if auc_col else worst
        print(f"  {auc:.5f}  (manifest {rep:.5f})  {n}")
    print(f"largest |recomputed - manifest| OOF AUC: {worst:.6f}")

    # 3. our model: rank mean of the seeds, as one column
    ours = [n for n in a.ours.split(",") if n]
    po = np.mean([rank01(np.load(Path(a.runs) / n / "oof.npy")) for n in ours], axis=0)
    to = np.mean([rank01(np.load(Path(a.runs) / n / "test.npy")) for n in ours], axis=0)
    for n in ours:
        print(f"  ours {roc_auc_score(y, np.load(Path(a.runs) / n / 'oof.npy')):.5f}  {n}")
    print(f"  ours, rank mean of {len(ours)} seeds: {roc_auc_score(y, po):.5f}")

    R = np.column_stack([rank01(P[:, j]) for j in range(P.shape[1])] + [rank01(po)])
    RT = np.column_stack([rank01(T[:, j]) for j in range(T.shape[1])] + [rank01(to)])
    L, LT = logit_rank(R), logit_rank(RT)
    cols = {"bank20": list(range(len(names))), "bank20+ours": list(range(len(names) + 1))}

    # 4. second-level cross-fitting on the same folds
    nested, fold_auc = {}, {}
    for key, c in cols.items():
        out = np.zeros(len(y))
        for trn, va in folds:
            out[va] = stacker().fit(L[np.ix_(trn, c)], y[trn]).decision_function(L[np.ix_(va, c)])
        nested[key] = out
        fold_auc[key] = [roc_auc_score(y[va], out[va]) for _, va in folds]
        print(f"{key:12s} pooled OOF AUC {roc_auc_score(y, out):.6f}  per fold "
              + " ".join(f"{v:.5f}" for v in fold_auc[key]))
    d = np.array(fold_auc["bank20+ours"]) - np.array(fold_auc["bank20"])
    print("adding ours, per-fold change:", " ".join(f"{v:+.6f}" for v in d), f"mean {d.mean():+.6f}")
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(a.draws):
        i = rng.choice(len(y), 60000, replace=False)
        diffs.append(roc_auc_score(y[i], nested["bank20+ours"][i]) - roc_auc_score(y[i], nested["bank20"][i]))
    diffs = np.array(diffs)
    full = roc_auc_score(y, nested["bank20+ours"]) - roc_auc_score(y, nested["bank20"])
    print(f"public-sized paired difference: sd {diffs.std(ddof=1):.6f}, same sign as pooled "
          f"({full:+.6f}) in {np.mean(np.sign(diffs) == np.sign(full)) * 100:.0f}% of {a.draws} draws")

    # 5. refit on all rows, write sub2, save artefacts
    out_runs = Path(a.runs)
    rec = {"bank": str(bank), "core_members": names, "ours": ours, "folds_identical": same,
           "pooled": {k: roc_auc_score(y, v) for k, v in nested.items()}, "fold_auc": fold_auc}
    preds = {}
    for key, c in cols.items():
        m = stacker().fit(L[:, c], y)
        preds[key] = m.decision_function(LT[:, c])
        labels = names + ["ours"] if key == "bank20+ours" else names
        coef = dict(zip(labels, map(float, m.coef_[0])))
        rec[f"coef_{key}"] = coef
        run = out_runs / f"stack_{key.replace('+', '_')}"
        run.mkdir(parents=True, exist_ok=True)
        np.save(run / "oof.npy", nested[key])
        np.save(run / "test.npy", preds[key])
    print("stack weights (bank20+ours, all rows), largest first:")
    for n, w in sorted(rec["coef_bank20+ours"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {w:+.4f}  {n}")
    Path("subs").mkdir(exist_ok=True)
    s2 = rank01(preds["bank20+ours"])
    p2 = Path("subs") / "sub2_bankstack.csv"
    pd.DataFrame({"id": test_ids, "is_fraud": s2}).to_csv(p2, index=False)
    check_sub(p2, test_ids)

    # 6. bonus blend with the public submission (no CV possible: it has no OOF)
    if a.public:
        pub = pd.read_csv(a.public)
        pub = pub.set_index("id").loc[test_ids, "is_fraud"].to_numpy(dtype=np.float64)
        assert np.isfinite(pub).all()
        rp = rank01(pub)
        s3 = a.public_weight * rp + (1 - a.public_weight) * s2
        p3 = Path("subs") / f"sub3_bonus_public{round(a.public_weight * 100)}.csv"
        pd.DataFrame({"id": test_ids, "is_fraud": s3}).to_csv(p3, index=False)
        check_sub(p3, test_ids)
        sub1 = Path("subs") / "sub1_glmlgbm3seeds_fixed.csv"
        corr = {"public_vs_sub2": spearmanr(pub, s2)[0], "public_vs_ours": spearmanr(pub, to)[0],
                "sub2_vs_ours": spearmanr(s2, to)[0]}
        if sub1.exists():
            corr["public_vs_sub1"] = spearmanr(pub, pd.read_csv(sub1)["is_fraud"])[0]
        print("Spearman on test:", {k: round(float(v), 5) for k, v in corr.items()})
        rec.update({"public": a.public, "public_weight": a.public_weight, "spearman_test": corr})

    with open(out_runs / "stack_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
