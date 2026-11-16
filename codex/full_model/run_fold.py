"""From the project root, run with .venv/bin/python codex/full_model/run_fold.py --fold 1."""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from train2 import build_static, glm_design, sigmoid, within_bin_permute

MERCHANT = "v_merchant_risk_score"
FIELDS = {MERCHANT, "v_velocity_score", "v_amount_usd"}
ARMS = {"A": set(), "B": {MERCHANT}, "C": {MERCHANT}, "D": FIELDS}


def encode_keys(keys, y, fit, apply, k, merchant_values):
    """Construct cross-fitted training columns; never use stop/validation labels to encode."""
    splits = list(StratifiedKFold(5, shuffle=True, random_state=142+k).split(fit, y[fit]))
    folds = [(fit[inside], fit[held], held) for inside, held in splits]
    folds.append((fit, apply, None))
    names = list(keys) + ["placebo"]
    training = np.empty((len(fit),len(names)), dtype=np.float32)
    scored = np.empty((len(apply),len(names)), dtype=np.float32)
    for part, (ref, dest, rows) in enumerate(folds):
        target = y[ref]
        prior = target.mean()
        cut = np.unique(np.quantile(merchant_values[ref], np.linspace(0,1,33)))[1:-1]
        bins = np.searchsorted(cut, merchant_values[ref], side="right")
        shuffled = within_bin_permute(target, bins, np.random.default_rng(6102+10*k+part))
        for j, name in enumerate(names):
            codes = keys[MERCHANT if name == "placebo" else name]
            ref_codes, dest_codes = codes[ref], codes[dest]
            size = int(codes.max()) + 1
            counts = np.bincount(ref_codes, minlength=size)
            labels = shuffled if name == "placebo" else target
            positive = np.bincount(ref_codes, weights=labels, minlength=size)
            p = (positive[dest_codes]+20*prior)/(counts[dest_codes]+20)
            values = np.log(p/(1-p))
            if rows is None:
                scored[:, j] = values
            else:
                training[rows, j] = values
    return pd.DataFrame(training, columns=names), pd.DataFrame(scored, columns=names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1,6), required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"runs")
    args=parser.parse_args()
    k=args.fold-1
    X,y,n,keys,_,_=build_static(ROOT/"data", {"base","digits","te"})
    X=X.iloc[:n].copy()
    keys={name:arr[:n] for name,arr in keys.items()}
    outer, valid = list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))[k]
    fit,stop = train_test_split(outer,test_size=0.10,stratify=y[outer],random_state=242+k)
    apply=np.r_[stop,valid]
    enc_fit, enc_apply=encode_keys(keys,y,fit,apply,k,X["merchant_risk_score"].to_numpy())
    raw_fit=X.iloc[fit].reset_index(drop=True)
    raw_apply=X.iloc[apply].reset_index(drop=True)
    output={"index":valid,"label":y[valid]}
    audit={}
    for arm, extras in ARMS.items():
        names=[n for n in keys if n not in FIELDS or n in extras]
        sources=["placebo" if arm=="C" and n==MERCHANT else n for n in names]
        ef=enc_fit[sources].set_axis(["te_"+n for n in names],axis=1)
        ea=enc_apply[sources].set_axis(ef.columns,axis=1)
        gf,ga=glm_design(raw_fit,ef),glm_design(raw_apply,ea)
        scale=StandardScaler().fit(gf)
        glm=LogisticRegression(C=1,solver="newton-cholesky",max_iter=100)
        glm.fit(scale.transform(gf),y[fit])
        init_fit=glm.decision_function(scale.transform(gf))
        init_apply=glm.decision_function(scale.transform(ga))
        xf,xa=pd.concat([raw_fit,ef],axis=1),pd.concat([raw_apply,ea],axis=1)
        model=lgb.LGBMClassifier(n_estimators=20000,learning_rate=.03,num_leaves=4,max_depth=2,
            min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.5,reg_lambda=1,
            random_state=42+k,n_jobs=args.threads,verbose=-1)
        model.fit(xf,y[fit],init_score=init_fit,eval_set=[(xa.iloc[:len(stop)],y[stop])],
                  eval_init_score=[init_apply[:len(stop)]],eval_metric="auc",
                  callbacks=[lgb.early_stopping(300,verbose=False)])
        p=sigmoid(init_apply[len(stop):]+model.predict(xa.iloc[len(stop):],raw_score=True))
        output[arm]=p
        audit[arm]={"auc":roc_auc_score(y[valid],p),"iters":model.best_iteration_,
                    "feature_count":xf.shape[1],"te_count":len(names)}
        print(f"fold {args.fold} {arm} {audit[arm]}",flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/f"fold{args.fold}.npz",**output)
    (args.out/f"fold{args.fold}.json").write_text(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
