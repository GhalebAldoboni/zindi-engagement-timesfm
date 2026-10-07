"""Validation-only development experiments (exploratory; never touch test cohorts).

Every experiment trains on each primary fold's train cohorts and scores its val
cohort (src/dev.py). Numbers are reported in results/overnight_log.md and
results/report_notes.md.

  python scripts/dev/dev_experiments.py --exp catboost      # CatBoost on v2 features (current best)
  python scripts/dev/dev_experiments.py --exp features     # v2 vs + v3 (KKBox/CFIN-style) vs + v4 (timing/entry/momentum)
  python scripts/dev/dev_experiments.py --exp balancing    # CatBoost class weights
  python scripts/dev/dev_experiments.py --exp xgboost      # XGBoost grid (needs xgboost<2.1, see setup_env.sh)
  python scripts/dev/dev_experiments.py --exp kernels      # Nystroem RBF/Laplacian, splines, poly-2
  python scripts/dev/dev_experiments.py --exp stacking     # OOF stacking of CatBoost, LightGBM, LR, P1
  python scripts/dev/dev_experiments.py --exp catboost_search
  python scripts/dev/dev_experiments.py --exp all
Feature tables are built first by: python scripts/dev/dev_experiments.py --build-features
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dev import dev_eval  # noqa: E402
from src.features import build_features, numeric_cols, static_cols  # noqa: E402
from src.io import load_config, load_events  # noqa: E402

CFG = load_config(str(ROOT / "configs/default.yaml"))
PROC = ROOT / CFG["processed_dir"]


def load_xv(extra=()):
    u = pd.read_parquet(PROC / "users.parquet")
    w = pd.read_parquet(PROC / "window_events.parquet")
    X = build_features(u, w, CFG).loc[u.uid]
    XV = X.join(pd.read_parquet(PROC / "features_v2.parquet").drop(columns=["platform_live_comps"]))
    for name in extra:
        XV = XV.join(pd.read_parquet(PROC / f"features_{name}.parquet").loc[u.uid])
    return u, X, XV


def build_all_features():
    from src.features_v2 import build_features_v2, build_features_v3, build_features_v4, competition_sets
    u = pd.read_parquet(PROC / "users.parquet")
    w = pd.read_parquet(PROC / "window_events.parquet")
    ev = load_events(CFG)
    X = build_features(u, w, CFG).loc[u.uid]
    build_features_v2(u, w, ev, CFG).loc[u.uid].to_parquet(PROC / "features_v2.parquet")
    build_features_v3(u, w, CFG, X).to_parquet(PROC / "features_v3.parquet")
    build_features_v4(u, w, ev, CFG).loc[u.uid].to_parquet(PROC / "features_v4.parquet")
    A, M = competition_sets(u, w, ev, CFG)
    np.save(PROC / "comp_sets.npy", A)
    np.save(PROC / "comp_mask.npy", M)
    print("feature tables written to", PROC)


def catboost_fp(cats, **kw):
    from catboost import CatBoostClassifier
    params = dict(iterations=3000, learning_rate=0.03, depth=4, l2_leaf_reg=5, verbose=0,
                  od_type="Iter", od_wait=100)
    params.update(kw)

    def fp(a, ya, b, yb, f):
        return np.mean([CatBoostClassifier(random_seed=s, cat_features=cats, **params)
                        .fit(a, ya, eval_set=(b, yb), use_best_model=True).predict_proba(b)[:, 1]
                        for s in CFG["lightgbm_seeds"]], 0)
    return fp


def report(name, m):
    print(f"{name:40s} val PR {m.val_pr_auc:.4f} ROC {m.val_roc_auc:.4f}", flush=True)


def exp_catboost():
    u, _, XV = load_xv()
    report("CatBoost-v2", dev_eval(u, XV, catboost_fp(static_cols(CFG)), CFG)[1])


def exp_features():
    u, _, XV = load_xv()
    V3 = pd.read_parquet(PROC / "features_v3.parquet").loc[u.uid]
    V4 = pd.read_parquet(PROC / "features_v4.parquet").loc[u.uid]
    cats = static_cols(CFG)
    for name, M in [("v2", XV), ("v2 + v3", XV.join(V3)), ("v2 + v4", XV.join(V4))]:
        report(f"CatBoost {name}", dev_eval(u, M, catboost_fp(cats), CFG)[1])


def exp_balancing():
    u, _, XV = load_xv()
    for acw in ["SqrtBalanced", "Balanced"]:
        report(f"CatBoost class weights {acw}",
               dev_eval(u, XV, catboost_fp(static_cols(CFG), auto_class_weights=acw), CFG)[1])


def exp_xgboost():
    import xgboost as xgb
    from src.models import to_categorical
    u, _, XV = load_xv()
    cats = static_cols(CFG)

    def make(md, lr):
        def fp(a, ya, b, yb, f):
            A, B = to_categorical(a, [b], cats)
            for c in cats:                                   # train levels; unseen -> -1
                A[c], B[c] = A[c].cat.codes.astype(float), B[c].cat.codes.astype(float)
            ps = []
            for s in CFG["lightgbm_seeds"]:
                m = xgb.XGBClassifier(n_estimators=3000, learning_rate=lr, max_depth=md, min_child_weight=5,
                                      subsample=0.8, colsample_bytree=0.6, reg_lambda=5, tree_method="hist",
                                      early_stopping_rounds=100, eval_metric="logloss", random_state=s, n_jobs=-1)
                m.fit(A.to_numpy(np.float32), ya, eval_set=[(B.to_numpy(np.float32), yb)], verbose=False)
                ps.append(m.predict_proba(B.to_numpy(np.float32))[:, 1])
            return np.mean(ps, 0)
        return fp
    for md in (3, 4, 6):
        for lr in (0.02, 0.05):
            report(f"XGBoost depth {md} lr {lr}", dev_eval(u, XV, make(md, lr), CFG)[1])


def exp_kernels():
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures, SplineTransformer,
                                       StandardScaler)
    u, _, XV = load_xv(["v4"])
    X = XV[[c for c in XV.columns if c not in static_cols(CFG)]].astype(float)
    slog = lambda Z: np.sign(Z) * np.log1p(np.abs(Z))  # noqa: E731

    def tuned(build, grid):
        def fp(a, ya, b, yb, f):
            best = (-1, None)
            for g in grid:
                p = build(g).fit(a, ya).predict_proba(b)[:, 1]
                best = max(best, (average_precision_score(yb, p), p), key=lambda t: t[0])
            return best[1]
        return fp
    pre = lambda: [FunctionTransformer(slog), StandardScaler()]  # noqa: E731
    nys = lambda k, g: Nystroem(kernel=k, gamma=g, n_components=1500, random_state=0)  # noqa: E731
    models = {
        "linear LR": tuned(lambda C: make_pipeline(*pre(), LogisticRegression(C=C, max_iter=5000)), [0.01, 0.1, 1]),
        "RBF kernel (Nystroem) + LR": tuned(lambda g: make_pipeline(*pre(), nys("rbf", g[0]), LogisticRegression(C=g[1], max_iter=5000)),
                                            [(gm, C) for gm in (0.002, 0.005, 0.01, 0.03) for C in (0.1, 1)]),
        "Laplacian kernel (Nystroem) + LR": tuned(lambda g: make_pipeline(*pre(), nys("laplacian", g[0]), LogisticRegression(C=g[1], max_iter=5000)),
                                                  [(gm, C) for gm in (0.002, 0.005, 0.01) for C in (0.1, 1)]),
        "splines (GAM-style) + LR": tuned(lambda C: make_pipeline(*pre(), SplineTransformer(n_knots=6, degree=3), LogisticRegression(C=C, max_iter=5000)),
                                          [0.003, 0.01, 0.03, 0.1]),
        "poly-2 interactions + LR": tuned(lambda C: make_pipeline(*pre(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=C, max_iter=5000)),
                                          [0.0003, 0.001, 0.003]),
    }
    for name, fp in models.items():
        report(name, dev_eval(u, X, fp, CFG)[1])


def exp_stacking():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from src.evaluate import make_folds, split
    from src.models import logreg, to_categorical
    u, X, XV = load_xv()
    cats, num = static_cols(CFG), numeric_cols(X, CFG)
    E = pd.DataFrame(np.load(ROOT / CFG["results_dir"] / "cache/emb/raw__univariate__late__mean.npy").astype(np.float32),
                     index=u.uid.to_numpy())
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731

    def base(a, ya, b, fold):
        A, B = to_categorical(XV.loc[a], [XV.loc[b]], cats)
        sc = StandardScaler().fit(E.loc[a])
        return {"cb": np.mean([CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=5, random_seed=s,
                                                  verbose=0, cat_features=cats).fit(XV.loc[a], ya).predict_proba(XV.loc[b])[:, 1]
                               for s in range(3)], 0),
                "lgb": np.mean([lgb.LGBMClassifier(n_estimators=80, subsample_freq=1, random_state=s, verbose=-1,
                                                   **CFG["lightgbm_frozen"][fold]).fit(A, ya).predict_proba(B)[:, 1] for s in range(3)], 0),
                "lr": logreg(0).fit(X.loc[a, num], ya).predict_proba(X.loc[b, num])[:, 1],
                "p1": LogisticRegression(C=1e-4, class_weight="balanced", max_iter=5000)
                      .fit(sc.transform(E.loc[a]), ya).predict_proba(sc.transform(E.loc[b]))[:, 1]}
    pr, roc = [], []
    for f in [f for f in make_folds(CFG) if f["kind"] == "primary"]:
        tr, va, _ = split(u, f)
        ids, y = tr.uid.to_numpy(), tr.label_engaged.to_numpy()
        oof = {k: np.zeros(len(ids)) for k in ("cb", "lgb", "lr", "p1")}
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y):
            for k, v in base(ids[a], y[a], ids[b], f["name"]).items():
                oof[k][b] = v
        stack = LogisticRegression().fit(np.column_stack([logit(oof[k]) for k in oof]), y)
        full = base(ids, y, va.uid.to_numpy(), f["name"])
        p = stack.predict_proba(np.column_stack([logit(full[k]) for k in oof]))[:, 1]
        yv = va.label_engaged.to_numpy()
        pr.append(average_precision_score(yv, p))
        roc.append(roc_auc_score(yv, p))
        print(f["name"], "stacker weights", dict(zip(oof, stack.coef_[0].round(2))))
    print(f"{'OOF stacking':40s} val PR {np.mean(pr):.4f} ROC {np.mean(roc):.4f}")


def exp_catboost_search():
    from sklearn.model_selection import ParameterSampler
    u, _, XV = load_xv()
    space = {"depth": [3, 4, 5, 6, 8], "l2_leaf_reg": [1, 3, 5, 10, 20], "learning_rate": [0.02, 0.03, 0.05],
             "random_strength": [0.5, 1, 2], "bagging_temperature": [0, 0.5, 1]}
    for ps in ParameterSampler(space, n_iter=14, random_state=1):
        report(f"CatBoost {ps}", dev_eval(u, XV, catboost_fp(static_cols(CFG), **ps), CFG)[1])


EXPS = {"catboost": exp_catboost, "features": exp_features, "balancing": exp_balancing,
        "xgboost": exp_xgboost, "kernels": exp_kernels, "stacking": exp_stacking,
        "catboost_search": exp_catboost_search}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=[*EXPS, "all"])
    ap.add_argument("--build-features", action="store_true")
    a = ap.parse_args()
    if a.build_features:
        build_all_features()
    for name in (EXPS if a.exp == "all" else [a.exp] if a.exp else []):
        print(f"== {name}")
        EXPS[name]()
