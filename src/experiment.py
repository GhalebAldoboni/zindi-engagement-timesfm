"""Shared plumbing for running models across folds, recording results, and
comparing models with the paired bootstrap.

Test metrics are recorded for two populations:
  all                  every test user
  excl_never_tracked   without users who have no row in any table in any month
                       (sensitivity only; that flag uses outcome-period data)
The model, threshold and training data are identical for both; only the
evaluated rows differ.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import make_folds, metrics, paired_bootstrap

POPULATIONS = ("all", "excl_never_tracked")


def config_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def pred_path(cfg, model, fold_name, label):
    suffix = "" if label == "label_engaged" else f"_{label}"
    d = Path(cfg["results_dir"]) / "cache" / "preds"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}__{fold_name}{suffix}.parquet"


def record(cfg, model, fold, label, va, te, p_val, p_test, thr, extra=None):
    """Save val/test predictions and return metric rows for val and both test populations."""
    y_va, y_te = va[label].to_numpy(), te[label].to_numpy()
    pd.DataFrame({"uid": np.r_[va.uid, te.uid],
                  "split": ["val"] * len(va) + ["test"] * len(te),
                  "never_tracked": np.r_[va.never_tracked, te.never_tracked],
                  "no_events": np.r_[va.no_events, te.no_events],
                  "y": np.r_[y_va, y_te], "p": np.r_[p_val, p_test],
                  "threshold": thr}).to_parquet(pred_path(cfg, model, fold["name"], label),
                                                index=False)
    base = {"model": model, "fold": fold["name"], "kind": fold["kind"], "label": label,
            "threshold": thr, **(extra or {}), "config_hash": config_hash(cfg)}
    rows = [{**base, "split": "val", "population": "all", "n": len(y_va),
             **metrics(y_va, p_val, thr)}]
    for pop in POPULATIONS:
        keep = np.ones(len(te), bool) if pop == "all" else ~te.never_tracked.to_numpy()
        rows.append({**base, "split": "test", "population": pop, "n": int(keep.sum()),
                     **metrics(y_te[keep], p_test[keep], thr)})
    return rows


def segment_rows(model, fold, label, te, p_test, thr):
    rows = []
    for name, mask in [("no_events", te.no_events.to_numpy() == 1),
                       ("has_events", te.no_events.to_numpy() == 0),
                       ("never_tracked", te.never_tracked.to_numpy())]:
        y, yhat = te[label].to_numpy()[mask], p_test[mask] >= thr
        rows.append({"model": model, "fold": fold["name"], "segment": name, "n": int(mask.sum()),
                     "positives": int(y.sum()), "pred_pos": int(yhat.sum()),
                     "true_pos": int((yhat & (y == 1)).sum())})
    return rows


def summarize(rows):
    """Mean and std over folds of the test metrics, per kind, population and model."""
    test = pd.DataFrame(rows).query("split == 'test'")
    return (test.groupby(["kind", "population", "model"])[["f1", "roc_auc", "pr_auc",
                                                             "precision", "recall"]]
            .agg(["mean", "std"]).round(4))


def compare(cfg, model_a, model_b, label="label_engaged", population="all", kind="primary"):
    """Paired bootstrap of A - B over the test folds of `kind`, from cached predictions."""
    folds = []
    for f in [f for f in make_folds(cfg) if f["kind"] == kind]:
        a = pd.read_parquet(pred_path(cfg, model_a, f["name"], label)).query("split == 'test'")
        b = pd.read_parquet(pred_path(cfg, model_b, f["name"], label)).query("split == 'test'")
        assert (a.uid.to_numpy() == b.uid.to_numpy()).all(), "predictions are not paired"
        if population == "excl_never_tracked":
            keep = ~a.never_tracked.to_numpy()
            a, b = a[keep], b[keep]
        folds.append({"y": a.y.to_numpy(), "p_a": a.p.to_numpy(), "thr_a": a.threshold.iloc[0],
                      "p_b": b.p.to_numpy(), "thr_b": b.threshold.iloc[0], "name": f["name"]})
    rows = paired_bootstrap(folds, cfg["bootstrap"]["n"], cfg["seed"])
    names = {f"fold{j}": f["name"] for j, f in enumerate(folds)}
    for r in rows:
        r.update({"a": model_a, "b": model_b, "label": label, "population": population,
                  "scope": names.get(r["scope"], r["scope"])})
    return rows
