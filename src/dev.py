"""Validation-only development harness for the overnight round.

dev_eval(X, fit_predict) trains on each primary fold's train cohorts and scores its
VAL cohort only. Test rows are dropped before anything is fitted or predicted, so
this code path cannot produce a test metric.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import make_folds, split, tune_threshold


def dev_eval(users, X, fit_predict, cfg, label="label_engaged"):
    """X: DataFrame indexed by uid. fit_predict(X_tr, y_tr, X_va, y_va, fold) -> p_va.
    Returns per-fold val metrics, plus the mean and the val predictions."""
    rows, preds = [], {}
    for fold in [f for f in make_folds(cfg) if f["kind"] == "primary"]:
        tr, va, _ = split(users, fold)                       # test part discarded
        p = fit_predict(X.loc[tr.uid], tr[label].to_numpy(), X.loc[va.uid], va[label].to_numpy(), fold)
        y = va[label].to_numpy()
        thr = tune_threshold(y, p, cfg)                       # in-sample F1 (optimistic)
        from sklearn.metrics import f1_score
        rows.append({"fold": fold["name"], "val_pr_auc": average_precision_score(y, p),
                     "val_roc_auc": roc_auc_score(y, p), "val_f1_insample": f1_score(y, p >= thr)})
        preds[fold["name"]] = pd.Series(p, index=va.uid.to_numpy())
    df = pd.DataFrame(rows)
    return df, df.mean(numeric_only=True), preds
