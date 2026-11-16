"""Quick EDA for Kaggle Playground S6E11 (credit card fraud).

Usage: python -I src/eda.py <data_dir>
<data_dir> holds train.csv, test.csv, sample_submission.csv, credit_card_fraud_2026.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
train = pd.read_csv(d / "train.csv")
test = pd.read_csv(d / "test.csv")
orig = pd.read_csv(d / "credit_card_fraud_2026.csv")
sub = pd.read_csv(d / "sample_submission.csv")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

print("shapes  train", train.shape, " test", test.shape, " orig", orig.shape, " sub", sub.shape)
print("train id range", train["id"].min(), train["id"].max(), "| test id range", test["id"].min(), test["id"].max())
print("target rate  train %.4f  orig %.4f" % (train["is_fraud"].mean(), orig["is_fraud"].mean()))
print("\ncolumns only in train:", sorted(set(train.columns) - set(orig.columns)))
print("columns only in orig :", sorted(set(orig.columns) - set(train.columns)))

feats = [c for c in train.columns if c not in ("id", "is_fraud")]
rows = []
for c in feats:
    rows.append({
        "col": c, "dtype": str(train[c].dtype),
        "nuniq_tr": train[c].nunique(), "nuniq_te": test[c].nunique(),
        "nuniq_orig": orig[c].nunique() if c in orig else np.nan,
        "na_tr": int(train[c].isna().sum()), "na_te": int(test[c].isna().sum()),
        "mean_tr": train[c].mean() if train[c].dtype.kind in "biuf" else np.nan,
        "mean_orig": orig[c].mean() if c in orig and orig[c].dtype.kind in "biuf" else np.nan,
        "fraud_corr_tr": train[c].corr(train["is_fraud"]) if train[c].dtype.kind in "biuf" else np.nan,
    })
print("\n", pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

print("\n== categorical columns: top values and fraud rate (train)")
for c in feats:
    if train[c].dtype.kind not in "biuf" or train[c].nunique() <= 12:
        g = train.groupby(c)["is_fraud"].agg(["size", "mean"]).sort_values("size", ascending=False).head(12)
        print(f"\n[{c}]\n" + g.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n== rows of train that exactly match an original row (feature columns shared):")
shared = [c for c in feats if c in orig.columns]
m = train[shared].merge(orig[shared].drop_duplicates(), on=shared, how="inner")
print(len(m))
print("\n== head(3) train\n", train.head(3).T)
