"""Phase 2: classical baselines B0-B3 on every fold.

B0  heuristic: any feature-window event in the last 7 days of the month -> 1
B1  logistic regression (log1p, impute, scale, balanced) on numeric handcrafted
B2  random forest (500 trees, balanced) on numeric handcrafted
B3  LightGBM on all handcrafted incl. categoricals, with the hyperparameters frozen in
    cfg["lightgbm_frozen"], early stopping on val, and the mean of 5 seeds (D3)
B4  MiniRocket + LR on the raw daily series (N, 4, 31); needs 02b_build_series.py

Thresholds are tuned on val (F1) and applied unchanged to test. B1 is the
co-reference baseline: B3 vs B1 is bootstrapped, and its pooled CI half-width
is the smallest difference the pooled test can resolve.

  --retune   rerun the 30-trial val search (writes b3_tuning.csv and
             b3_params_retuned.csv; never touches the frozen config)

Writes (results/, suffixed by label if not label_engaged)
  baselines.csv            model x fold x split x population
  baselines_summary.csv    mean, std over folds
  b3_importance.csv        B3 gain importance per fold (seed-averaged, sums to 1)
  segments_baselines.csv   no_events / has_events / never_tracked test segments
  bootstrap_b3_vs_b1.csv   paired bootstrap B3 - B1, both populations
  cache/preds/*.parquet    val/test predictions
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_folds, split, tune_threshold  # noqa: E402
from src.experiment import (POPULATIONS, compare, record, segment_rows,  # noqa: E402
                            summarize)
from src.features import (build_features, last7_heuristic, numeric_cols,  # noqa: E402
                          static_cols)
from src.io import load_config  # noqa: E402
from src.models import (lightgbm_fit, lightgbm_search_space, lightgbm_seed_avg,  # noqa: E402
                        logreg, minirocket_logreg, random_forest, to_categorical)


def retune_b3(X_tr, y_tr, X_va, y_va, cfg, fold_name):
    trials = []
    for i, params in enumerate(lightgbm_search_space(cfg, cfg["seed"])):
        m = lightgbm_fit(X_tr, y_tr, X_va, y_va, params, cfg, cfg["seed"])
        trials.append({"fold": fold_name, "trial": i, **params, "best_iter": m.best_iteration_,
                       "val_pr_auc": average_precision_score(y_va, m.predict_proba(X_va)[:, 1])})
    return pd.DataFrame(trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--label", default="label_engaged", choices=["label_engaged", "label_any"])
    ap.add_argument("--retune", action="store_true")
    args = ap.parse_args()
    cfg, label = load_config(args.config), args.label
    res = Path(cfg["results_dir"])
    sfx = "" if label == "label_engaged" else f"_{label}"

    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    window = pd.read_parquet(Path(cfg["processed_dir"]) / "window_events.parquet")
    X_all = build_features(users, window, cfg).loc[users.uid]
    users["no_events"] = X_all.no_events.to_numpy()
    num, cat = numeric_cols(X_all, cfg), static_cols(cfg)
    b0_all = pd.Series(last7_heuristic(window, users, cfg), index=users.uid.to_numpy())
    series_dir = Path(cfg["results_dir"]) / "cache" / "series"
    raw = np.load(series_dir / "raw.npy") if (series_dir / "raw.npy").exists() else None
    if raw is not None:
        assert (np.load(series_dir / "uids.npy") == users.uid.to_numpy()).all()
    row = {u: i for i, u in enumerate(users.uid)}

    rows, segs, imps, tuning = [], [], [], []
    for fold in make_folds(cfg):
        t0 = time.time()
        tr, va, te = split(users, fold)
        X_tr, X_va, X_te = (X_all.loc[d.uid] for d in (tr, va, te))
        y_tr, y_va = tr[label].to_numpy(), va[label].to_numpy()
        Xc_tr, Xc_va, Xc_te = to_categorical(X_tr, [X_va, X_te], cat)

        preds = {"B0": (b0_all.loc[va.uid].to_numpy(float), b0_all.loc[te.uid].to_numpy(float))}
        m = logreg(cfg["seed"]).fit(X_tr[num], y_tr)
        preds["B1"] = (m.predict_proba(X_va[num])[:, 1], m.predict_proba(X_te[num])[:, 1])
        m = random_forest(cfg["seed"]).fit(X_tr[num], y_tr)
        preds["B2"] = (m.predict_proba(X_va[num])[:, 1], m.predict_proba(X_te[num])[:, 1])
        params = cfg["lightgbm_frozen"][fold["name"]]
        pv, pt, models = lightgbm_seed_avg(Xc_tr, y_tr, Xc_va, y_va, Xc_te, params, cfg)
        preds["B3"] = (pv, pt)
        imp = sum(pd.Series(m.booster_.feature_importance("gain"), index=Xc_tr.columns)
                  for m in models)
        imps.append((imp / imp.sum()).rename(fold["name"]))
        extras = {"B3": {"best_iters": ",".join(str(m.best_iteration_) for m in models)}}
        if raw is not None:
            R = [raw[[row[u] for u in d.uid]] for d in (tr, va, te)]
            pv, pt, C = minirocket_logreg(R[0], y_tr, R[1], y_va, R[2], cfg, cfg["seed"])
            preds["B4"] = (pv, pt)
            extras["B4"] = {"C": C}
        if args.retune:
            tuning.append(retune_b3(Xc_tr, y_tr, Xc_va, y_va, cfg, fold["name"]))

        for model, (p_val, p_test) in preds.items():
            thr = 0.5 if model == "B0" else tune_threshold(y_va, p_val, cfg)
            rows += record(cfg, model, fold, label, va, te, p_val, p_test, thr,
                           extras.get(model))
            segs += segment_rows(model, fold, label, te, p_test, thr)
        print(f"{fold['name']}: {', '.join(preds)} in {time.time() - t0:.0f}s")

    out = pd.DataFrame(rows)
    out.to_csv(res / f"baselines{sfx}.csv", index=False)
    summ = summarize(rows)
    summ.to_csv(res / f"baselines_summary{sfx}.csv")
    pd.concat(imps, axis=1).to_csv(res / f"b3_importance{sfx}.csv")
    pd.DataFrame(segs).to_csv(res / f"segments_baselines{sfx}.csv", index=False)
    if tuning:
        t = pd.concat(tuning)
        t.to_csv(res / f"b3_tuning{sfx}.csv", index=False)
        t.loc[t.groupby("fold").val_pr_auc.idxmax()].to_csv(res / f"b3_params_retuned{sfx}.csv",
                                                             index=False)

    boot = pd.DataFrame([r for pop in POPULATIONS for r in compare(cfg, "B3", "B1", label, pop)])
    boot.to_csv(res / f"bootstrap_b3_vs_b1{sfx}.csv", index=False)

    pd.set_option("display.width", 220)
    test = out[(out.split == "test")]
    print(test[["model", "fold", "population", "n", "threshold", "f1", "roc_auc", "pr_auc"]]
          .round(4).to_string(index=False))
    print(summ.to_string())
    print(boot[boot.scope == "pooled"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
