"""Recompute pooled and per-fold AUC from predictions, using the official ordered labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LEGACY={
    "A":"v2_lgbm_base_m20_esin_s42",
    "B":"v2_lgbm_base+te_tc-merchant_risk_score_m20_esin_s42",
    "C":"v2_lgbm_base+te_tc-merchant_risk_score_placebo_m20_esin_s42",
    "D":"v2_lgbm_base+te_tc-merchant_risk_score-velocity_score-amount_usd_m20_esin_s42",
}


def placements(query, reference):
    s=np.sort(reference)
    left=np.searchsorted(s,query,side="left")
    right=np.searchsorted(s,query,side="right")
    return (left+right)/(2.0*len(s))


def score(legacy):
    y=pd.read_csv(ROOT/"data/train.csv",usecols=["is_fraud"]).is_fraud.to_numpy(dtype=int)
    pred={arm:np.full(len(y),np.nan) for arm in LEGACY}
    folds=np.zeros(len(y),dtype=int)
    for k,(_,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y),1):
        folds[va]=k
        if not legacy:
            with np.load(HERE/f"runs/fold{k}.npz") as f:
                np.testing.assert_array_equal(va,f["index"])
                np.testing.assert_array_equal(y[va],f["label"])
                for arm in pred:pred[arm][va]=f[arm]
    if legacy:
        for arm, directory in LEGACY.items():
            path=ROOT/"runs"/directory
            pred[arm]=np.load(path/"oof.npy").astype(float)
            meta=json.loads((path/"meta.json").read_text())
            np.testing.assert_allclose(meta["cv_auc"],roc_auc_score(y,pred[arm]),atol=1e-6,rtol=0)
            np.testing.assert_allclose(meta["fold_auc"],[roc_auc_score(y[folds==k],pred[arm][folds==k]) for k in range(1,6)],atol=1e-6,rtol=0)
    assert np.all(folds>0)
    for p in pred.values():assert len(p)==len(y) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    rows=[]
    for k in range(6):
        mask=folds==k if k else np.ones(len(y),dtype=bool)
        auc={arm:roc_auc_score(y[mask],p[mask]) for arm,p in pred.items()}
        rows.append(dict(fold=k or "pooled",n=int(mask.sum()),positives=int(y[mask].sum()),**auc,
                         B_minus_A=auc["B"]-auc["A"],C_minus_A=auc["C"]-auc["A"],
                         D_minus_A=auc["D"]-auc["A"],B_minus_C=auc["B"]-auc["C"]))
    frame=pd.DataFrame(rows)
    tag="claude_lgbm" if legacy else "full_glm_lgbm"
    frame.to_csv(HERE/f"{tag}_auc.csv",index=False)
    estimates=[]
    for arm,control in [("B","A"),("C","A"),("D","A"),("B","C")]:
        def components(p):
            pos,neg=p[y==1],p[y==0]
            return placements(pos,neg),1-placements(neg,pos)
        u,v=components(pred[arm])
        r,s=components(pred[control])
        d=u-r;e=v-s
        delta=rows[0][arm]-rows[0][control]
        np.testing.assert_allclose([d.mean(),e.mean()],delta,atol=1e-12)
        se=np.sqrt(d.var(ddof=1)/len(d)+e.var(ddof=1)/len(e))
        estimates.append(dict(arm=arm,comparison=control,delta=delta,low=delta-1.96*se,high=delta+1.96*se))
    pd.DataFrame(estimates).to_csv(HERE/f"{tag}_conditional_intervals.csv",index=False)
    print(frame.to_string(index=False,float_format=lambda n:f"{n:.6f}"),flush=True)
    print(pd.DataFrame(estimates).to_string(index=False,float_format=lambda n:f"{n:.6f}"),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--legacy",action="store_true")
    score(parser.parse_args().legacy)
