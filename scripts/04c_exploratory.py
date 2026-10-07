"""Exploratory round E (pre-committed in results/checkpoints.md before any test score).

E1 rank-avg(B3, P1)   E2 rank-avg(B3, P1, B4)   E4 rank-avg(P4_pca8, P1)
E3 LR on [log1p handcrafted numerics | raw-series embeddings], C on val PR-AUC

Rank-average: mean of within-split percentile ranks (no labels used). The winner is
chosen by mean val PR-AUC over the primary folds. Writes results/exploratory.csv,
exploratory_selection.csv and bootstrap_exploratory.csv (winner vs B3 and B1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_folds, split, tune_threshold  # noqa: E402
from src.experiment import POPULATIONS, compare, pred_path, record, summarize  # noqa: E402
from src.features import build_features, numeric_cols  # noqa: E402
from src.io import load_config  # noqa: E402
from src.models import logreg_dense_tuned  # noqa: E402

ENSEMBLES = {"E1": ["B3", "P1"], "E2": ["B3", "P1", "B4"], "E4": ["P4_pca8", "P1"]}


def pct_rank(p):
    return rankdata(p) / len(p)


def main():
    cfg = load_config()
    res, label = Path(cfg["results_dir"]), "label_engaged"
    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    window = pd.read_parquet(Path(cfg["processed_dir"]) / "window_events.parquet")
    X_all = build_features(users, window, cfg).loc[users.uid]
    users["no_events"] = X_all.no_events.to_numpy()
    num = numeric_cols(X_all, cfg)
    emb = np.load(res / "cache" / "emb" / "raw__univariate__late__mean.npy").astype(np.float32)
    row = {u: i for i, u in enumerate(users.uid)}

    rows = []
    for fold in make_folds(cfg):
        tr, va, te = split(users, fold)
        y_va = va[label].to_numpy()
        for name, members in ENSEMBLES.items():
            ps = [pd.read_parquet(pred_path(cfg, m, fold["name"], label)) for m in members]
            for p in ps[1:]:
                assert (p.uid.to_numpy() == ps[0].uid.to_numpy()).all()
            out = {}
            for s, d in (("val", va), ("test", te)):
                sub = [p[p.split == s] for p in ps]
                assert (sub[0].uid.to_numpy() == d.uid.to_numpy()).all()
                out[s] = np.mean([pct_rank(x.p.to_numpy()) for x in sub], axis=0)
            thr = tune_threshold(y_va, out["val"], cfg)
            rows += record(cfg, name, fold, label, va, te, out["val"], out["test"], thr)
        # E3
        Z = [np.hstack([np.log1p(X_all.loc[d.uid, num].to_numpy(np.float32)),
                        emb[[row[u] for u in d.uid]]]) for d in (tr, va, te)]
        pv, pt, C = logreg_dense_tuned(Z[0], tr[label].to_numpy(), Z[1], y_va, Z[2],
                                       cfg["embedding_logreg"]["C_grid"], cfg["seed"])
        rows += record(cfg, "E3", fold, label, va, te, pv, pt, tune_threshold(y_va, pv, cfg),
                       {"C": C})
        print(fold["name"], "done")

    out = pd.DataFrame(rows)
    out.to_csv(res / "exploratory.csv", index=False)
    sel = (out[(out.split == "val") & (out.kind == "primary")]
           .groupby("model")[["pr_auc", "roc_auc", "f1"]].mean()
           .sort_values("pr_auc", ascending=False))
    sel.round(4).to_csv(res / "exploratory_selection.csv")
    winner = sel.index[0]
    print("VAL selection:\n", sel.round(4).to_string(), "\nwinner:", winner)
    boot = [r for ref in ("B3", "B1") for pop in POPULATIONS
            for r in compare(cfg, winner, ref, label, pop)]
    pd.DataFrame(boot).to_csv(res / "bootstrap_exploratory.csv", index=False)
    pd.set_option("display.width", 200)
    print(summarize(rows).to_string())
    b = pd.DataFrame(boot)
    print(b[b.scope == "pooled"][["a", "b", "population", "metric", "diff", "ci_lo", "ci_hi",
                                  "p_one_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
