"""Train one model with stratified K-fold CV on Kaggle S6E11; save OOF/test predictions; log CV AUC.

Usage (from the project root on the 5090):
  .venv/bin/python -I src/train.py --data data --out runs --model lgbm --fe base --orig 1 --seed 42

--model  lgbm | xgb | cat | lr
--fe     comma list of feature blocks: base (always), flags, ratio, freq, origte
--orig   1 = append the original dataset (credit_card_fraud_2026.csv) to every training fold;
         original rows never enter a validation fold, so CV AUC is measured on competition rows only.
Every run appends one JSON line to <out>/log.jsonl; that file is the ablation table for the report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "is_fraud"
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]


def build(data_dir, fe):
    d = Path(data_dir)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv")
    orig = pd.read_csv(d / "credit_card_fraud_2026.csv").drop(columns=["transaction_id"])
    feats = [c for c in tr.columns if c not in ("id", TARGET)]
    src = np.r_[np.zeros(len(tr)), np.ones(len(te)), np.full(len(orig), 2)].astype(np.int8)
    allx = pd.concat([tr[feats], te[feats], orig[feats]], ignore_index=True)
    y_tr = tr[TARGET].to_numpy(dtype=np.int8)
    y_orig = orig[TARGET].to_numpy(dtype=np.int8)

    X = allx.copy()
    for c in BOOL:
        X[c] = X[c].astype(np.int8)
    for c in CAT:
        X[c] = pd.Categorical(X[c].astype(str))
    num = [c for c in feats if c not in CAT + BOOL]

    if "flags" in fe:  # count of binary risk signals
        X["n_flags"] = (X[BOOL].sum(axis=1) + (X["cvv_retry_count"] > 0)
                        + (allx["auth_method"].astype(str) == "No Authentication")).astype(np.int8)
    if "ratio" in fe:
        X["amt_to_bal"] = (X["amount_usd"] / (X["account_balance_usd"] + 1.0)).astype(np.float32)
        X["amt_x_risk"] = (X["amount_usd"] * X["merchant_risk_score"]).astype(np.float32)
    if "freq" in fe:  # value frequency over competition train+test (unsupervised, no labels)
        comp = src < 2
        for c in num:
            vc = X.loc[comp, c].value_counts()
            X[f"{c}_freq"] = X[c].map(vc).fillna(0).astype(np.float32)
    if "origte" in fe:  # fraud rate of each value in the ORIGINAL data (labels of competition rows unused)
        prior, k = y_orig.mean(), 20.0
        o = allx.loc[src == 2].assign(_y=y_orig)
        for c in num + CAT:
            g = o.groupby(o[c].astype(str))["_y"].agg(["sum", "count"])
            te_map = (g["sum"] + prior * k) / (g["count"] + k)
            X[f"{c}_ote"] = allx[c].astype(str).map(te_map).fillna(prior).astype(np.float32)

    X_tr, X_te, X_orig = X[src == 0].reset_index(drop=True), X[src == 1].reset_index(drop=True), X[src == 2].reset_index(drop=True)
    return X_tr, y_tr, X_te, X_orig, y_orig, te["id"].to_numpy()


def fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed, threads, params):
    if model == "lgbm":
        import lightgbm as lgb
        kw = dict(n_estimators=20000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=seed, n_jobs=threads, verbose=-1)
        kw.update(params)
        m = lgb.LGBMClassifier(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.best_iteration_)
    if model == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=20000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                  device="cuda", enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
                  early_stopping_rounds=300, random_state=seed, n_jobs=threads)
        kw.update(params)
        m = xgb.XGBClassifier(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.best_iteration)
    if model == "cat":
        from catboost import CatBoostClassifier
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        conv = lambda df: df.assign(**{c: df[c].astype(str) for c in cats})
        kw = dict(iterations=20000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                  eval_metric="AUC", task_type="GPU", devices="0", random_seed=seed,
                  od_type="Iter", od_wait=300, verbose=0, thread_count=threads)
        kw.update(params)
        m = CatBoostClassifier(**kw)
        m.fit(conv(Xtr), ytr, cat_features=cats, eval_set=(conv(Xva), yva), use_best_model=True)
        return m.predict_proba(conv(Xva))[:, 1], m.predict_proba(conv(Xte))[:, 1], int(m.get_best_iteration())
    if model in ("lr", "gam"):
        # lr: linear in standardized features; gam: additive logistic model with cubic splines per numeric
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        nums = [c for c in Xtr.columns if c not in cats]
        if model == "lr":
            num_tf = [("sc", StandardScaler(), nums)]
        else:
            smooth = [c for c in nums if Xtr[c].nunique() > 12]
            small = [c for c in nums if c not in smooth]
            num_tf = [("sp", SplineTransformer(n_knots=params.get("n_knots", 10), degree=3,
                                               knots="quantile", extrapolation="constant"), smooth),
                      ("oh_small", OneHotEncoder(handle_unknown="ignore"), small)]
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cats)] + num_tf)
        m = make_pipeline(pre, LogisticRegression(C=params.get("C", 1.0), max_iter=3000))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], 0
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--fe", default="base")
    ap.add_argument("--orig", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42, help="keep fixed so OOF files are comparable")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--params", default="{}", help='JSON overrides, e.g. {"num_leaves": 7}')
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    fe = set(a.fe.split(","))
    params = json.loads(a.params)
    ptag = "".join(f"_{k}{v}" for k, v in params.items())
    name = a.name or f"{a.model}{ptag}_{a.fe.replace(',', '+')}_orig{a.orig}_s{a.seed}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X_tr, y_tr, X_te, X_orig, y_orig, test_ids = build(a.data, fe)
    print(f"[{name}] features={X_tr.shape[1]} train={len(X_tr)} test={len(X_te)} orig={len(X_orig)}", flush=True)

    oof = np.zeros(len(X_tr), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    fold_auc, iters = [], []
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.fold_seed)
    for k, (tri, vai) in enumerate(skf.split(X_tr, y_tr)):
        Xtr, ytr = X_tr.iloc[tri], y_tr[tri]
        if a.orig:
            Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
            ytr = np.r_[ytr, y_orig]
        p_va, p_te, it = fit_predict(a.model, Xtr, ytr, X_tr.iloc[vai], y_tr[vai], X_te, a.seed + k,
                                     a.threads, params)
        oof[vai] = p_va
        test_pred += p_te / a.folds
        fold_auc.append(roc_auc_score(y_tr[vai], p_va))
        iters.append(it)
        print(f"  fold {k + 1}: AUC {fold_auc[-1]:.5f}  best_iter {it}  ({time.time() - t0:.0f}s)", flush=True)

    cv = roc_auc_score(y_tr, oof)
    np.save(out / "oof.npy", oof.astype(np.float32))
    np.save(out / "test.npy", test_pred.astype(np.float32))
    rec = {"name": name, "model": a.model, "params": params, "fe": sorted(fe), "orig": a.orig, "seed": a.seed,
           "folds": a.folds, "fold_seed": a.fold_seed, "cv_auc": round(cv, 6),
           "fold_auc": [round(v, 6) for v in fold_auc], "fold_sd": round(float(np.std(fold_auc, ddof=1)), 6),
           "best_iter": iters, "n_features": X_tr.shape[1], "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(rec, indent=1))
    with open(Path(a.out) / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{name}] CV AUC {cv:.5f} (fold sd {rec['fold_sd']:.5f}) in {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
