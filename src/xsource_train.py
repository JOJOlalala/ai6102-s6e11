"""v2 trainer for Kaggle S6E11: in-fold per-value target encoding, digit keys, GLM, GLM-initialised boosting.

Usage (project root):
  .venv/bin/python -I src/train2.py --model glm_lgbm --fe base,digits,te,freq --seed 42

--model  lgbm | glm | glm_lgbm | xgb
           glm       logistic regression (newton-cholesky) on target-encoding logits + one-hot categoricals
           glm_lgbm  shallow LightGBM whose boosting starts at the GLM's logit (init_score)
--fe     base (raw columns), digits (digit keys of float columns), te (per-value target encoding),
         freq (value counts over train+test, no labels)

Target-encoding protocol (no label leakage into the scored fold):
  outer fold k: training rows T_k, validation rows V_k (same StratifiedKFold(5, shuffle, 42) as train.py)
  - rows of T_k get encodings cross-fitted with an inner StratifiedKFold on T_k (a row never sees its own label)
  - rows of V_k and the test set get encodings from all of T_k only
  - smoothing: (sum + m * prior) / (count + m), prior = fraud rate of T_k; unseen values fall back to the prior
--te_strict 1 (run names get "_strict"; pointed out by Codex's review): with --es inner the early-stopping rows E_k
  are split off T_k before any encoding, and every encoding uses only the rows it is fitted on:
  - fit rows F_k = T_k minus E_k get encodings cross-fitted with an inner StratifiedKFold on F_k
  - E_k, V_k and the test set get encodings from all of F_k
  - the prior and, for --te_placebo, the within-bin label shuffle are computed inside each fitting subset only.
  The legacy path shuffles placebo labels once over all of T_k, so shuffled labels can cross inner folds, and it
  encodes E_k by inner cross-fitting over T_k.
Writes runs/<name>/{oof,test}.npy and appends one line to runs/log.jsonl, like train.py.
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
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
          "velocity_score", "merchant_risk_score"]


def digit_keys(x):
    c = np.round(np.asarray(x, dtype=np.float64) * 100).astype(np.int64)
    ip = c // 100
    lead = np.where(ip > 0, ip // (10 ** np.floor(np.log10(np.maximum(ip, 1))).astype(np.int64)), 0)
    return {"int": ip, "int10": ip // 10, "d1": (c // 10) % 10, "d2": c % 10, "lead": lead}


def build_static(data_dir, fe):
    d = Path(data_dir)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv")
    feats = [c for c in tr.columns if c not in ("id", TARGET)]
    source = pd.read_csv(d / "credit_card_fraud_2026.csv", usecols=feats)
    official_count = len(tr) + len(te)
    official = pd.concat([tr[feats], te[feats]], ignore_index=True)
    for col in CAT:
        assert set(source[col].dropna()).issubset(set(official[col].dropna()))
    allx = pd.concat([official, source[feats]], ignore_index=True)
    y = tr[TARGET].to_numpy(dtype=np.int8)
    n_tr = len(tr)

    X = pd.DataFrame(index=allx.index)
    num = [c for c in feats if c not in CAT + BOOL]
    for c in num:
        X[c] = allx[c].astype(np.float32)
    for c in BOOL:
        X[c] = allx[c].astype(np.int8)
    for c in CAT:
        X[c] = pd.Categorical(allx[c].astype(str))

    keys = {}  # name -> integer codes over train+test, used for target encoding
    bins = {}  # name -> 32 quantile bins of the raw column (placebo control for exact-value keys)
    for c in num + CAT:
        keys[f"v_{c}"] = pd.factorize(allx[c].astype(str))[0]
        if c in num and official[c].nunique() > 32:
            bins[f"v_{c}"] = pd.qcut(official[c].rank(method="first"), 32, labels=False).to_numpy()
    if "digits" in fe:
        for c in FLOATS:
            for k, v in digit_keys(allx[c]).items():
                if len(np.unique(v[:official_count])) > 1:
                    X[f"{c}__{k}"] = v.astype(np.int32)
                    keys[f"d_{c}__{k}"] = pd.factorize(v)[0]
                    if f"v_{c}" in bins:  # placebo for a digit key: shuffle within the parent column's bins
                        bins[f"d_{c}__{k}"] = bins[f"v_{c}"]
    if "freq" in fe:
        for c in num:
            X[f"{c}__freq"] = allx[c].map(allx[c].value_counts()).astype(np.float32)
    return X, y, n_tr, keys, bins, te["id"].to_numpy()


def within_bin_permute(yv, b, rng):
    """Shuffle labels inside each bin: keeps every bin's fraud count and every value's frequency."""
    out = yv.copy()
    order = np.argsort(b, kind="stable")
    for grp in np.split(order, np.flatnonzero(np.diff(b[order])) + 1):
        out[grp] = yv[rng.permutation(grp)]
    return out


def te_block(keys, y, tr_idx, apply_idx, m, inner, seed, placebo_bins=None):
    """Target-encoding logits for rows tr_idx (inner cross-fitted) and apply_idx (from all tr_idx).
    placebo_bins: {key: bin codes}; labels of tr_idx are shuffled within those bins before encoding."""
    prior = y[tr_idx].mean()
    lo = lambda p: np.log(p / (1 - p))
    out_tr = np.empty((len(tr_idx), len(keys)), dtype=np.float32)
    out_ap = np.empty((len(apply_idx), len(keys)), dtype=np.float32)
    inner_split = list(StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed).split(tr_idx, y[tr_idx]))
    rng = np.random.default_rng(seed + 7)
    for j, (name, codes) in enumerate(keys.items()):
        K = codes.max() + 1
        ct, yt = codes[tr_idx], y[tr_idx].astype(np.float64)
        if placebo_bins is not None and name in placebo_bins:
            yt = within_bin_permute(yt, placebo_bins[name][tr_idx], rng)
        S = np.bincount(ct, weights=yt, minlength=K)
        C = np.bincount(ct, minlength=K).astype(np.float64)
        ca = codes[apply_idx]
        out_ap[:, j] = lo((S[ca] + m * prior) / (C[ca] + m))
        for _, fold_rows in inner_split:
            cf = ct[fold_rows]
            Sf = np.bincount(cf, weights=yt[fold_rows], minlength=K)
            Cf = np.bincount(cf, minlength=K).astype(np.float64)
            out_tr[fold_rows, j] = lo(((S - Sf)[cf] + m * prior) / ((C - Cf)[cf] + m))
    cols = [f"te_{n}" for n in keys]
    return pd.DataFrame(out_tr, columns=cols), pd.DataFrame(out_ap, columns=cols)


def te_block_strict(keys, y, fit_idx, apply_idx, m, inner, seed, placebo_bins=None):
    """Strict variant of te_block: each encoding is fitted on one subset and nothing else.
    fit_idx rows: inner fold f is encoded from fit_idx minus f. apply_idx rows: encoded from all of fit_idx.
    The prior and the placebo shuffle (within bins) are computed inside that same subset."""
    lo = lambda p: np.log(p / (1 - p))
    out_fit = np.empty((len(fit_idx), len(keys)), dtype=np.float32)
    out_ap = np.empty((len(apply_idx), len(keys)), dtype=np.float32)
    yf = y[fit_idx].astype(np.float64)
    inner_split = list(StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed).split(fit_idx, yf))
    subsets = [(np.arange(len(fit_idx)), None)] + inner_split  # (positions fitted on, positions encoded)
    rng = np.random.default_rng(seed + 7)
    for j, (name, codes) in enumerate(keys.items()):
        K = codes.max() + 1
        cf = codes[fit_idx]
        for fit_pos, enc_pos in subsets:
            cs, ys = cf[fit_pos], yf[fit_pos]
            if placebo_bins is not None and name in placebo_bins:
                ys = within_bin_permute(ys, placebo_bins[name][fit_idx][fit_pos], rng)
            prior = ys.mean()
            S = np.bincount(cs, weights=ys, minlength=K)
            C = np.bincount(cs, minlength=K).astype(np.float64)
            if enc_pos is None:
                ca = codes[apply_idx]
                out_ap[:, j] = lo((S[ca] + m * prior) / (C[ca] + m))
            else:
                ce = cf[enc_pos]
                out_fit[enc_pos, j] = lo((S[ce] + m * prior) / (C[ce] + m))
    cols = [f"te_{n}" for n in keys]
    return pd.DataFrame(out_fit, columns=cols), pd.DataFrame(out_ap, columns=cols)


def glm_design(Xs, te_df):
    """GLM inputs: TE logits + booleans + one-hot categoricals (dense float32)."""
    parts = [te_df.reset_index(drop=True)]
    parts.append(Xs[BOOL].reset_index(drop=True).astype(np.float32))
    parts.append(pd.get_dummies(Xs[CAT].reset_index(drop=True), dtype=np.float32))
    return pd.concat(parts, axis=1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--model", default="glm_lgbm")
    ap.add_argument("--fe", default="base,digits,te")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--te_m", type=float, default=20.0)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--glm_C", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--name", default=None)
    ap.add_argument("--te_cols", default="all", help="comma list of raw columns whose keys get target-encoded")
    ap.add_argument("--te_placebo", type=int, default=0, help="1 = shuffle labels within 32 quantile bins before encoding")
    ap.add_argument("--te_strict", type=int, default=0, help="1 = strict encoding protocol (see module docstring)")
    ap.add_argument("--es", default="outer", choices=["outer", "inner"], help="which rows drive early stopping")
    ap.add_argument("--es_frac", type=float, default=0.1, help="share of outer-training rows held out with --es inner")
    ap.add_argument("--max_folds", type=int, default=0, help="smoke test: stop after this many folds, save nothing")
    ap.add_argument("--shuffle_labels", type=int, default=0, help="leakage test: permute labels; CV should be ~0.5")
    a = ap.parse_args()
    fe = set(a.fe.split(","))
    params = json.loads(a.params)
    ptag = "".join(f"_{k}{v}" for k, v in params.items())
    estag = "_esin" if a.es == "inner" else ""
    tetag = (("" if a.te_cols == "all" else "_tc-" + a.te_cols.replace(",", "-")) + ("_placebo" if a.te_placebo else "")
             + ("_strict" if a.te_strict else ""))
    name = a.name or f"v2_{a.model}{ptag}_{a.fe.replace(',', '+')}{tetag}_m{a.te_m:g}{estag}_s{a.seed}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, y, n_tr, keys, bins, test_ids = build_static(a.data, fe)
    if a.shuffle_labels:
        y = np.random.default_rng(a.seed).permutation(y)
        name, out = name + "_SHUFFLED", Path(a.out) / (name + "_SHUFFLED")
        out.mkdir(parents=True, exist_ok=True)
    if "te" not in fe:
        keys = {}
    if a.te_cols != "all":  # keep exact-value (v_) and digit (d_) keys of the listed raw columns only
        wanted = set(a.te_cols.split(","))
        keys = {kname: v for kname, v in keys.items() if kname[2:].split("__")[0] in wanted}
    placebo_bins = bins if a.te_placebo else None
    if "base" not in fe:
        X = X.drop(columns=[c for c in X.columns if "__" not in c])
    idx_te = np.arange(n_tr, len(X))
    print(f"[{name}] static features={X.shape[1]} te keys={len(keys)} train={n_tr} test={len(idx_te)}", flush=True)

    oof = np.zeros(n_tr)
    test_pred = np.zeros(len(idx_te))
    fold_auc, iters = [], []
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.fold_seed)
    for k, (tri, vai) in enumerate(skf.split(np.zeros(n_tr), y)):
        apply_idx = np.r_[vai, idx_te]
        nva = len(vai)
        # Early-stopping rows: "outer" = the scored fold (legacy, slightly optimistic);
        # "inner" = a stratified slice of the outer-training rows, so the scored fold's labels stay unused.
        if a.es == "inner":
            from sklearn.model_selection import train_test_split
            fit_pos, es_pos = train_test_split(np.arange(len(tri)), test_size=a.es_frac, stratify=y[tri],
                                               random_state=a.seed + 200 + k)
        else:
            fit_pos, es_pos = np.arange(len(tri)), None
        if a.te_strict and es_pos is not None:
            # reorder the outer-training rows as [fit rows, early-stopping rows]; the same rows as the legacy split
            fit_idx, es_idx = tri[fit_pos], tri[es_pos]
            tri = np.r_[fit_idx, es_idx]
            fit_pos, es_pos = np.arange(len(fit_idx)), np.arange(len(fit_idx), len(tri))
        if keys and a.te_strict:
            fit_idx = tri[fit_pos]
            n_es = 0 if es_pos is None else len(es_pos)
            te_fit, te_rest = te_block_strict(keys, y, fit_idx, np.r_[tri[len(fit_idx):], apply_idx], a.te_m,
                                              a.inner, a.seed + 100 + k, placebo_bins)
            te_tr = pd.concat([te_fit, te_rest.iloc[:n_es]], ignore_index=True)
            te_ap = te_rest.iloc[n_es:].reset_index(drop=True)
        elif keys:
            te_tr, te_ap = te_block(keys, y, tri, apply_idx, a.te_m, a.inner, a.seed + 100 + k, placebo_bins)
        else:
            te_tr = te_ap = pd.DataFrame(index=range(0))
        Xtr = pd.concat([X.iloc[tri].reset_index(drop=True), te_tr], axis=1)
        Xap = pd.concat([X.iloc[apply_idx].reset_index(drop=True), te_ap], axis=1)
        ytr, yva = y[tri], y[vai]

        init_tr = init_ap = None
        if a.model in ("glm", "glm_lgbm"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            Gtr, Gap = glm_design(X.iloc[tri], te_tr), glm_design(X.iloc[apply_idx], te_ap)
            Gap = Gap.reindex(columns=Gtr.columns, fill_value=0.0)
            sc = StandardScaler().fit(Gtr.iloc[fit_pos])
            glm = LogisticRegression(C=a.glm_C, solver="newton-cholesky", max_iter=100)
            glm.fit(sc.transform(Gtr.iloc[fit_pos]), ytr[fit_pos])
            init_tr = glm.decision_function(sc.transform(Gtr))
            init_ap = glm.decision_function(sc.transform(Gap))
            it = int(np.max(glm.n_iter_))
            p_ap = sigmoid(init_ap)
        if es_pos is None:
            es_X, es_y = Xap.iloc[:nva], yva
            es_init = None if init_ap is None else init_ap[:nva]
        else:
            es_X, es_y = Xtr.iloc[es_pos], ytr[es_pos]
            es_init = None if init_tr is None else init_tr[es_pos]
        if a.model in ("lgbm", "glm_lgbm"):
            import lightgbm as lgb
            kw = dict(n_estimators=20000, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                      random_state=a.seed + k, n_jobs=a.threads, verbose=-1)
            kw.update(params)
            mdl = lgb.LGBMClassifier(**kw)
            fit_kw = dict(eval_set=[(es_X, es_y)], eval_metric="auc",
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            if init_tr is not None:
                fit_kw.update(init_score=init_tr[fit_pos], eval_init_score=[es_init])
            mdl.fit(Xtr.iloc[fit_pos], ytr[fit_pos], **fit_kw)
            raw = mdl.predict(Xap, raw_score=True)
            p_ap = sigmoid(raw + (init_ap if init_ap is not None else 0.0))
            it = int(mdl.best_iteration_)
        if a.model == "xgb":
            import xgboost as xgb
            kw = dict(n_estimators=20000, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
                      colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist", device="cuda",
                      enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
                      early_stopping_rounds=300, random_state=a.seed + k, n_jobs=a.threads)
            kw.update(params)
            mdl = xgb.XGBClassifier(**kw)
            mdl.fit(Xtr.iloc[fit_pos], ytr[fit_pos], eval_set=[(es_X, es_y)], verbose=False)
            p_ap = mdl.predict_proba(Xap)[:, 1]
            it = int(mdl.best_iteration)

        # The source rows follow the official test set, in their original file order.
        np.save(out / f"source_fold{k+1}.npy", p_ap[nva + len(test_ids):])
        oof[vai] = p_ap[:nva]
        test_pred += p_ap[nva:] / a.folds
        fold_auc.append(roc_auc_score(yva, p_ap[:nva]))
        iters.append(it)
        print(f"  fold {k + 1}: AUC {fold_auc[-1]:.5f}  iter {it}  ({time.time() - t0:.0f}s)", flush=True)
        if a.max_folds and k + 1 >= a.max_folds:
            print(f"[{name}] smoke test stopped after {k + 1} fold(s); nothing saved", flush=True)
            return

    cv = roc_auc_score(y, oof)
    np.save(out / "oof.npy", oof.astype(np.float32))
    np.save(out / "test.npy", test_pred.astype(np.float32))
    rec = {"name": name, "model": a.model, "params": params, "fe": sorted(fe), "orig": 0, "seed": a.seed,
           "te_m": a.te_m, "inner": a.inner, "glm_C": a.glm_C, "es": a.es, "folds": a.folds,
           "te_placebo": a.te_placebo, "te_strict": a.te_strict,
           "fold_seed": a.fold_seed,
           "cv_auc": round(cv, 6), "fold_auc": [round(v, 6) for v in fold_auc],
           "fold_sd": round(float(np.std(fold_auc, ddof=1)), 6), "best_iter": iters,
           "n_features": int(Xtr.shape[1]), "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(rec, indent=1))
    with open(Path(a.out) / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{name}] CV AUC {cv:.5f} (fold sd {rec['fold_sd']:.5f}) in {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
