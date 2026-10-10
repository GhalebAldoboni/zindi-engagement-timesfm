"""Phase 5: proposed models on TimesFM-3 embeddings.

P1        LR (scaled, balanced, C on val PR-AUC) on embeddings of the raw series
P2        same on embeddings of the smoothed series
P3_pca{k} LightGBM (B3's frozen params, mean of 5 seeds) on handcrafted features +
          PCA(k) of standardised raw-series embeddings
P4_pca{k} same with smoothed-series embeddings
k in cfg["embedding"]["pca_dims"] (default 8, 32). Scaler and PCA are fitted on
the train fold only; thresholds are tuned on val.

  --raw-emb / --smooth-emb   embedding file stems in results/cache/emb/ (ablations)
  --tag                      suffix for model names and output files (ablations)
  --label                    label_engaged (default) or label_any (sensitivity)
  --pca-dims                 override pca_dims (-1 = full embedding, no PCA)
  --models                   subset, e.g. P4 (default: all)

Writes results/proposed{tag}{label sfx}.csv, segments_proposed*.csv, and
bootstrap_proposed*.csv (every P model vs B3 and vs B1, both populations; this is
descriptive, and the verdict is applied separately under the pre-registered rule).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_folds, split, tune_threshold  # noqa: E402
from src.experiment import (POPULATIONS, compare, record, segment_rows,  # noqa: E402
                            summarize)
from src.features import build_features, static_cols  # noqa: E402
from src.io import load_config  # noqa: E402
from src.models import (lightgbm_seed_avg, logreg_dense_tuned, pca_block,  # noqa: E402
                        to_categorical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--label", default="label_engaged", choices=["label_engaged", "label_any"])
    ap.add_argument("--raw-emb", default="raw__univariate__late__mean")
    ap.add_argument("--smooth-emb", default="smooth_bw12__univariate__late__mean")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pca-dims", type=int, nargs="*")
    ap.add_argument("--models", nargs="*", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="handcrafted column prefixes or names to drop (feature-group ablation)")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    cfg, label, tag = load_config(args.config), args.label, args.tag
    res = Path(cfg["results_dir"])
    sfx = tag + ("" if label == "label_engaged" else f"_{label}")
    dims = args.pca_dims or cfg["embedding"]["pca_dims"]

    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    window = pd.read_parquet(Path(cfg["processed_dir"]) / "window_events.parquet")
    X_all = build_features(users, window, cfg).loc[users.uid]
    users["no_events"] = X_all.no_events.to_numpy()
    if args.drop_features:
        drop = [c for c in X_all.columns
                if any(c == d or c.startswith(d + "_") for d in args.drop_features)]
        assert drop, f"nothing matched {args.drop_features}"
        X_all = X_all.drop(columns=drop)
    cats = [c for c in static_cols(cfg) if c in X_all.columns]
    uids = np.load(res / "cache" / "series" / "uids.npy")
    assert (uids == users.uid.to_numpy()).all(), "embedding rows must follow users.parquet order"
    emb = {"raw": np.load(res / "cache" / "emb" / f"{args.raw_emb}.npy").astype(np.float32),
           "smooth": np.load(res / "cache" / "emb" / f"{args.smooth_emb}.npy").astype(np.float32)}
    row = {u: i for i, u in enumerate(users.uid)}
    C_grid = cfg["embedding_logreg"]["C_grid"]

    rows, segs = [], []
    for fold in make_folds(cfg):
        t0 = time.time()
        tr, va, te = split(users, fold)
        idx = [np.array([row[u] for u in d.uid]) for d in (tr, va, te)]
        y_tr, y_va = tr[label].to_numpy(), va[label].to_numpy()
        X_tr, X_va, X_te = (X_all.loc[d.uid].reset_index(drop=True) for d in (tr, va, te))
        Xc = to_categorical(X_tr, [X_va, X_te], cats)
        params = cfg["lightgbm_frozen"][fold["name"]]
        preds, extras = {}, {}
        for p_name, rep, pref in [("P1", "raw", "P3"), ("P2", "smooth", "P4")]:
            Z = [emb[rep][i] for i in idx]
            if p_name in args.models:
                pv, pt, C = logreg_dense_tuned(Z[0], y_tr, Z[1], y_va, Z[2], C_grid, cfg["seed"])
                preds[p_name], extras[p_name] = (pv, pt), {"C": C}
            if pref in args.models:
                for k in dims:
                    if k == -1:                      # full embedding, no PCA (standardised on train)
                        from sklearn.preprocessing import StandardScaler
                        sc = StandardScaler().fit(Z[0])
                        cols = [f"{rep}_emb{i}" for i in range(Z[0].shape[1])]
                        P_tr, P_va, P_te = (pd.DataFrame(sc.transform(z), columns=cols) for z in Z)
                        evr = 1.0
                    else:
                        (P_tr, P_va, P_te), evr = pca_block(Z[0], Z[1:], k, cfg["seed"], f"{rep}_emb")
                    H = [pd.concat([x, p], axis=1) for x, p in zip(Xc, (P_tr, P_va, P_te))]
                    pv, pt, models = lightgbm_seed_avg(H[0], y_tr, H[1], y_va, H[2], params, cfg)
                    name = f"{pref}_full" if k == -1 else f"{pref}_pca{k}"
                    preds[name] = (pv, pt)
                    extras[name] = {"pca_explained_var": round(evr, 4),
                                    "best_iters": ",".join(str(m.best_iteration_) for m in models)}
        for model, (p_val, p_test) in preds.items():
            thr = tune_threshold(y_va, p_val, cfg)
            rows += record(cfg, model + tag, fold, label, va, te, p_val, p_test, thr,
                           extras.get(model))
            segs += segment_rows(model + tag, fold, label, te, p_test, thr)
        print(f"{fold['name']}: {', '.join(preds)} in {time.time() - t0:.0f}s")

    out = pd.DataFrame(rows)
    out.to_csv(res / f"proposed{sfx}.csv", index=False)
    pd.DataFrame(segs).to_csv(res / f"segments_proposed{sfx}.csv", index=False)
    if not args.no_bootstrap:
        boot = [r for m in sorted(out.model.unique()) for ref in ("B3", "B1") for pop in POPULATIONS
                for r in compare(cfg, m, ref, label, pop)]
        pd.DataFrame(boot).to_csv(res / f"bootstrap_proposed{sfx}.csv", index=False)

    pd.set_option("display.width", 220)
    print(summarize(rows).to_string())
    val = out[(out.split == "val") & (out.kind == "primary")].groupby("model")[["pr_auc", "roc_auc", "f1"]].mean()
    print("\nmean VAL metrics over primary folds (candidate selection, O1):\n", val.round(4).to_string())


if __name__ == "__main__":
    main()
