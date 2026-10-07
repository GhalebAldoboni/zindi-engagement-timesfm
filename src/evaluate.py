"""Folds, and later threshold tuning, metrics and the paired bootstrap.

Rolling-origin folds by cohort: for test cohort k,
  train = cohorts 0 .. k-2,  val = cohort k-1,  test = cohort k.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def make_folds(cfg):
    folds = []
    for kind in ["primary", "supplementary"]:
        for k in cfg["folds"][f"{kind}_test_cohorts"]:
            if k < 2:
                raise ValueError(f"test cohort {k} leaves no train cohort")
            fold = {"name": f"test_c{k}", "kind": kind,
                    "train": list(range(k - 1)), "val": [k - 1], "test": [k]}
            used = set(fold["train"] + fold["val"] + fold["test"])
            if used & set(cfg["excluded_cohorts"]):
                raise ValueError(f"fold {fold['name']} uses an excluded cohort")
            folds.append(fold)
    return folds


def split(users, fold):
    """Return (train, val, test) rows of `users` for a fold."""
    return tuple(users[users.cohort.isin(fold[part])] for part in ["train", "val", "test"])


# ---- threshold tuning and metrics -------------------------------------------


def threshold_grid(cfg):
    lo, hi, n = cfg["threshold_grid"]
    return np.linspace(lo, hi, int(n))


def tune_threshold(y_val, p_val, cfg):
    """Threshold maximising F1 on the val fold (first maximum on the grid)."""
    grid = threshold_grid(cfg)
    f1s = [f1_score(y_val, p_val >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics(y, p, threshold):
    yhat = p >= threshold
    return {"f1": f1_score(y, yhat, zero_division=0),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "pred_pos_rate": float(yhat.mean()), "pos_rate": float(np.mean(y))}


# ---- paired bootstrap ---------------------------------------------------------
METRICS = ("f1", "roc_auc", "pr_auc")


def _metric(name, y, p, thr):
    if name == "f1":
        return f1_score(y, p >= thr, zero_division=0)
    if name == "roc_auc":
        return roc_auc_score(y, p)
    return average_precision_score(y, p)


def paired_bootstrap(folds, n, seed, metrics_=METRICS):
    """Paired bootstrap of metric(A) - metric(B).

    folds: list of dicts with y, p_a, thr_a, p_b, thr_b (one per test fold). Each
    resample draws users with replacement within every fold, the same rows for A
    and B. Per-fold differences are reported, and the pooled statistic is the mean
    of the per-fold differences (equal fold weight). Returns one row per
    (metric, scope) with the observed difference, 95% percentile CI, half-width and
    one-sided p = share of resamples with difference <= 0.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics_:
        obs = [_metric(m, f["y"], f["p_a"], f["thr_a"]) - _metric(m, f["y"], f["p_b"], f["thr_b"])
               for f in folds]
        boot = np.empty((n, len(folds)))
        for j, f in enumerate(folds):
            y, k = f["y"], len(f["y"])
            for i in range(n):
                idx = rng.integers(0, k, k)
                if y[idx].min() == y[idx].max():      # AUC undefined; redraw once
                    idx = rng.integers(0, k, k)
                boot[i, j] = (_metric(m, y[idx], f["p_a"][idx], f["thr_a"])
                              - _metric(m, y[idx], f["p_b"][idx], f["thr_b"]))
        scopes = [(f"fold{j}", obs[j], boot[:, j]) for j in range(len(folds))]
        scopes.append(("pooled", float(np.mean(obs)), boot.mean(axis=1)))
        for scope, o, b in scopes:
            lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append({"metric": m, "scope": scope, "diff": float(o), "ci_lo": float(lo),
                         "ci_hi": float(hi), "half_width": float(hi - lo) / 2, "p_one_sided": float(np.mean(b <= 0))})
    return rows


# ---- pre-registered H3 verdict (REPORT.md, approved 2026-10-04) -------------------
def h3_verdict(pooled):
    """pooled: {metric: (diff, ci_lo, ci_hi)} for best P - B3, metrics f1/roc_auc/pr_auc.
    Order (O2): SUPPORTED -> PARTIAL -> NOT SUPPORTED -> INCONCLUSIVE.
    F1-only CI > 0 is INCONCLUSIVE (O3). Returns (verdict, harm notes (O4))."""
    above = {m: lo > 0 for m, (_, lo, _) in pooled.items()}
    below = [m for m, (_, _, hi) in pooled.items() if hi < 0]
    if above["pr_auc"] and above["f1"]:
        verdict = "SUPPORTED"
    elif above["pr_auc"] or above["roc_auc"]:
        verdict = "PARTIAL"
    elif all(d <= 0 for d, _, _ in pooled.values()):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, [f"significantly worse on {m}" for m in below]
