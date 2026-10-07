"""Model wrappers. Every fit sees train rows only; LightGBM additionally uses the
val rows for early stopping. Categorical levels come from train only."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def logreg(seed):
    """B1: log1p (fixed; all handcrafted numerics are >= 0) -> impute -> scale -> LR."""
    return make_pipeline(FunctionTransformer(np.log1p), SimpleImputer(strategy="median"),
                         StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000,
                                            random_state=seed))


def random_forest(seed):
    """B2."""
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1,
                                  random_state=seed)


def minirocket_logreg(X_tr, y_tr, X_va, y_va, X_te, cfg, seed):
    """B4: MiniRocket (fit on train series only) -> scale -> balanced LR, with C chosen
    on val PR-AUC from cfg["minirocket"]["C_grid"]. Inputs are (n, C, T) arrays."""
    from sklearn.metrics import average_precision_score
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    mr = MiniRocketMultivariate(random_state=seed).fit(X_tr.astype(np.float64))
    Z_tr, Z_va, Z_te = (mr.transform(x.astype(np.float64)).to_numpy() for x in (X_tr, X_va, X_te))
    scaler = StandardScaler().fit(Z_tr)
    Z_tr, Z_va, Z_te = (scaler.transform(z) for z in (Z_tr, Z_va, Z_te))
    best = None
    for C in cfg["minirocket"]["C_grid"]:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        m.fit(Z_tr, y_tr)
        score = average_precision_score(y_va, m.predict_proba(Z_va)[:, 1])
        if best is None or score > best[0]:
            best = (score, C, m)
    _, C, m = best
    return m.predict_proba(Z_va)[:, 1], m.predict_proba(Z_te)[:, 1], C


def to_categorical(train, others, cols):
    """Categorical dtype with levels seen in train; unseen levels become NaN."""
    out = []
    for df in [train, *others]:
        df = df.copy()
        for c in cols:
            levels = sorted(train[c].unique())
            df[c] = pd.Categorical(df[c].where(df[c].isin(levels)), categories=levels)
        out.append(df)
    return out


def lightgbm_fit(X_tr, y_tr, X_va, y_va, params, cfg, seed):
    """B3 / P3 / P4: LightGBM with early stopping on val. Returns the fitted model."""
    lc = cfg["lightgbm"]
    model = lgb.LGBMClassifier(
        n_estimators=lc["n_estimators"], learning_rate=lc["learning_rate"],
        num_leaves=lc["num_leaves"], subsample=lc["subsample"],
        colsample_bytree=lc["colsample_bytree"], subsample_freq=1,
        random_state=seed, verbose=-1, n_jobs=-1)
    model.set_params(**params)
    model.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,), eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(lc["early_stopping"], verbose=False)])
    return model


def lightgbm_seed_avg(X_tr, y_tr, X_va, y_va, X_te, params, cfg):
    """Mean predicted probability over cfg["lightgbm_seeds"] (decision D3).
    Returns (p_val, p_test, fitted models)."""
    models = [lightgbm_fit(X_tr, y_tr, X_va, y_va, params, cfg, s) for s in cfg["lightgbm_seeds"]]
    p_val = np.mean([m.predict_proba(X_va)[:, 1] for m in models], axis=0)
    p_test = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)
    return p_val, p_test, models


def lightgbm_search_space(cfg, seed):
    """Random search configs for the val-fold tuning (sampled, not hand-picked)."""
    from sklearn.model_selection import ParameterSampler
    space = {"num_leaves": [3, 4, 7, 15, 31, 63], "min_child_samples": [10, 20, 50, 100],
             "learning_rate": [0.01, 0.03, 0.05], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}
    return list(ParameterSampler(space, n_iter=cfg["lightgbm_tuning"]["n_trials"],
                                 random_state=seed))


def logreg_dense_tuned(Z_tr, y_tr, Z_va, y_va, Z_te, C_grid, seed):
    """P1/P2: scale -> balanced LR on dense embeddings, C chosen on val PR-AUC.
    Returns (p_val, p_test, C)."""
    from sklearn.metrics import average_precision_score
    scaler = StandardScaler().fit(Z_tr)
    Z_tr, Z_va, Z_te = (scaler.transform(z) for z in (Z_tr, Z_va, Z_te))
    best = None
    for C in C_grid:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        m.fit(Z_tr, y_tr)
        score = average_precision_score(y_va, m.predict_proba(Z_va)[:, 1])
        if best is None or score > best[0]:
            best = (score, C, m)
    _, C, m = best
    return m.predict_proba(Z_va)[:, 1], m.predict_proba(Z_te)[:, 1], C


def pca_block(Z_tr, others, k, seed, prefix):
    """Standardise + PCA(k), both fitted on the train rows only. Returns DataFrames."""
    from sklearn.decomposition import PCA
    scaler = StandardScaler().fit(Z_tr)
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(Z_tr))
    cols = [f"{prefix}_pc{i}" for i in range(k)]
    out = [pd.DataFrame(pca.transform(scaler.transform(z)), columns=cols)
           for z in (Z_tr, *others)]
    return out, float(pca.explained_variance_ratio_.sum())
