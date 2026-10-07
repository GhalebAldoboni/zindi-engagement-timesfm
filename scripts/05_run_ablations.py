"""Phase 6: ablations, each a call to 04_run_proposed.py with --no-bootstrap and a tag.

On the pre-registered candidate P4_pca8 (smoothed-series hybrid):
  bandwidth 6/24/48 h; layer early/middle; last-token pooling; PCA 64;
  5 channels; multivariate embedding; drop static attributes; drop each handcrafted
  feature group (the embedding is kept).
Post hoc, on P1 (LR on raw-series embeddings, the strongest single P model on test):
  layer early/middle; 5 channels; multivariate.

Each variant is bootstrapped against its default (P4_pca8 or P1) on the primary
folds. Writes results/ablations.csv and results/bootstrap_ablations.csv.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import compare  # noqa: E402
from src.io import load_config  # noqa: E402

S = "smooth_bw12__univariate__late__mean"
GROUPS = ["browse", "learn", "compete", "contribute", "account", "onboarding", "badges", "comments"]
RUNS = (  # (study, tag, ablated default, extra args)
    [("bandwidth", f"_bw{b}", "P4_pca8", ["--smooth-emb", f"smooth_bw{b}__univariate__late__mean"])
     for b in (6, 24, 48)]
    + [("layer", f"_L{l}", "P4_pca8", ["--smooth-emb", f"smooth_bw12__univariate__{l}__mean"])
       for l in ("early", "middle")]
    + [("pooling", "_poolLast", "P4_pca8", ["--smooth-emb", "smooth_bw12__univariate__late__last"]),
       ("pca_dim", "_abl", "P4_pca8", ["--pca-dims", "64"]),
       ("channels", "_5ch", "P4_pca8", ["--smooth-emb", "5ch_smooth_bw12__univariate__late__mean"]),
       ("multivariate", "_mv", "P4_pca8", ["--smooth-emb", "smooth_bw12__multivariate__late__mean"]),
       ("features", "_noStatic", "P4_pca8", ["--drop-features", "FeatureX", "FeatureY", "Countries_ID"])]
    + [("features", f"_no_{g}", "P4_pca8", ["--drop-features", g]) for g in GROUPS]
    + [("layer", f"_L{l}", "P1", ["--raw-emb", f"raw__univariate__{l}__mean"]) for l in ("early", "middle")]
    + [("channels", "_5ch", "P1", ["--raw-emb", "5ch_raw__univariate__late__mean"]),
       ("multivariate", "_mv", "P1", ["--raw-emb", "raw__multivariate__late__mean"])]
)


def main():
    cfg = load_config()
    res = Path(cfg["results_dir"])
    rows, boots = [], []
    for study, tag, base, extra in RUNS:
        models = ["P4"] if base.startswith("P4") else ["P1"]
        dims = ["--pca-dims", "8"] if base == "P4_pca8" and "--pca-dims" not in extra else []
        tag_full = f"_{base[:2]}{tag}"
        cmd = [sys.executable, "scripts/04_run_proposed.py", "--no-bootstrap", "--tag", tag_full,
               "--models", *models, *dims, *extra]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = pd.read_csv(res / f"proposed{tag_full}.csv")
        variant = out.model.iloc[0]
        t = out[(out.kind == "primary") & (out.population == "all")]
        m = t.groupby("split")[["f1", "roc_auc", "pr_auc"]].mean()
        rows.append({"study": study, "default": base, "variant": variant, "args": " ".join(extra),
                     **{f"test_{k}": v for k, v in m.loc["test"].items()},
                     **{f"val_{k}": v for k, v in m.loc["val"].items()}})
        for r in compare(cfg, variant, base):
            if r["scope"] == "pooled":
                boots.append({"study": study, **r})
        print(f"{study:12s} {variant:22s} test F1 {m.loc['test','f1']:.4f} "
              f"ROC {m.loc['test','roc_auc']:.4f} PR {m.loc['test','pr_auc']:.4f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(res / "ablations.csv", index=False)
    pd.DataFrame(boots).round(4).to_csv(res / "bootstrap_ablations.csv", index=False)


if __name__ == "__main__":
    main()
