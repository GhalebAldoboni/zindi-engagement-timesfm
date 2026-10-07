"""Overnight round N: one-time test evaluation of the pre-committed final list
(results/overnight_protocol.md). N1 CatBoost-v2; N2 N1 refit on train+val;
N3 rank-avg(CatBoost-v2, LightGBM-v2, P1); N4 N3 with every member refit on train+val.
Writes results/overnight_final.csv and bootstrap_overnight.csv (vs B3 and E2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_folds, split, tune_threshold  # noqa: E402
from src.experiment import compare, record, summarize  # noqa: E402
from src.features import build_features, static_cols  # noqa: E402
from src.io import load_config  # noqa: E402
from src.models import lightgbm_fit, logreg_dense_tuned, to_categorical  # noqa: E402

CB = dict(learning_rate=0.03, depth=4, l2_leaf_reg=5, verbose=0)


def pct(p):
    return rankdata(p) / len(p)


def main():
    cfg = load_config()
    res, label, seeds = Path(cfg["results_dir"]), "label_engaged", cfg["lightgbm_seeds"]
    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    window = pd.read_parquet(Path(cfg["processed_dir"]) / "window_events.parquet")
    X = build_features(users, window, cfg).loc[users.uid]
    users["no_events"] = X.no_events.to_numpy()
    V = pd.read_parquet(Path(cfg["processed_dir"]) / "features_v2.parquet").drop(
        columns=["platform_live_comps"])
    XV = X.join(V)
    cats = static_cols(cfg)
    E = pd.DataFrame(np.load(res / "cache" / "emb" / "raw__univariate__late__mean.npy").astype(np.float32),
                     index=users.uid.to_numpy())

    rows = []
    for fold in make_folds(cfg):
        tr, va, te = split(users, fold)
        trva = pd.concat([tr, va])
        y_tr, y_va, y_trva = (d[label].to_numpy() for d in (tr, va, trva))
        A, B, T = to_categorical(XV.loc[tr.uid], [XV.loc[va.uid], XV.loc[te.uid]], cats)
        AB, T2 = to_categorical(XV.loc[trva.uid], [XV.loc[te.uid]], cats)
        # CatBoost takes the raw string categories (as in development)
        cA, cB, cT, cAB = (XV.loc[d.uid] for d in (tr, va, te, trva))

        cb = [CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=100, random_seed=s,
                                 cat_features=cats, **CB).fit(cA, y_tr, eval_set=(cB, y_va),
                                                              use_best_model=True) for s in seeds]
        cb_va = np.mean([m.predict_proba(cB)[:, 1] for m in cb], 0)
        cb_te = np.mean([m.predict_proba(cT)[:, 1] for m in cb], 0)
        it = max(int(np.mean([m.get_best_iteration() + 1 for m in cb])), 1)
        cb_te_refit = np.mean([CatBoostClassifier(iterations=it, random_seed=s, cat_features=cats, **CB)
                               .fit(cAB, y_trva).predict_proba(cT)[:, 1] for s in seeds], 0)

        params = cfg["lightgbm_frozen"][fold["name"]]
        lg = [lightgbm_fit(A, y_tr, B, y_va, params, cfg, s) for s in seeds]
        lg_va = np.mean([m.predict_proba(B)[:, 1] for m in lg], 0)
        lg_te = np.mean([m.predict_proba(T)[:, 1] for m in lg], 0)
        lg_it = max(int(np.mean([m.best_iteration_ for m in lg])), 1)
        lg_te_refit = []
        for s in seeds:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=lg_it, subsample_freq=1, random_state=s, verbose=-1,
                                   n_jobs=-1, **params).fit(AB, y_trva)
            lg_te_refit.append(m.predict_proba(T2)[:, 1])
        lg_te_refit = np.mean(lg_te_refit, 0)

        Z = [E.loc[d.uid].to_numpy() for d in (tr, va, te, trva)]
        p1_va, p1_te, C = logreg_dense_tuned(Z[0], y_tr, Z[1], y_va, Z[2],
                                             cfg["embedding_logreg"]["C_grid"], cfg["seed"])
        sc = StandardScaler().fit(Z[3])
        p1_te_refit = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                         random_state=cfg["seed"]).fit(sc.transform(Z[3]), y_trva) \
            .predict_proba(sc.transform(Z[2]))[:, 1]

        thr_n1 = tune_threshold(y_va, cb_va, cfg)
        rows += record(cfg, "N1", fold, label, va, te, cb_va, cb_te, thr_n1, {"cb_iters": it})
        rows += record(cfg, "N2", fold, label, va, te, cb_va, cb_te_refit, thr_n1,
                       {"cb_iters": it, "note": "val row = N1 (refit has no held-out val)"})
        n3_va = np.mean([pct(cb_va), pct(lg_va), pct(p1_va)], 0)
        n3_te = np.mean([pct(cb_te), pct(lg_te), pct(p1_te)], 0)
        n4_te = np.mean([pct(cb_te_refit), pct(lg_te_refit), pct(p1_te_refit)], 0)
        thr_n3 = tune_threshold(y_va, n3_va, cfg)
        rows += record(cfg, "N3", fold, label, va, te, n3_va, n3_te, thr_n3)
        rows += record(cfg, "N4", fold, label, va, te, n3_va, n4_te, thr_n3,
                       {"note": "val row = N3 (refit has no held-out val)"})
        print(fold["name"], "done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(res / "overnight_final.csv", index=False)
    boot = [r for m in ("N1", "N2", "N3", "N4") for ref in ("B3", "E2")
            for r in compare(cfg, m, ref, label, "all")]
    pd.DataFrame(boot).to_csv(res / "bootstrap_overnight.csv", index=False)
    pd.set_option("display.width", 200)
    print(summarize(rows).to_string())
    b = pd.DataFrame(boot)
    print(b[b.scope == "pooled"][["a", "b", "metric", "diff", "ci_lo", "ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
