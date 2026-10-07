"""Apply the pre-registered H3 rule (REPORT.md, approved 2026-10-04).

1. Candidate (O1): the P model with the highest mean VAL PR-AUC over the primary
   folds, from results/proposed.csv (only rows with split == "val" are read).
2. Verdict: src.evaluate.h3_verdict on the pooled bootstrap of candidate - B3
   (results/bootstrap_proposed.csv). Primary population: all; sensitivity:
   excl_never_tracked.

Writes results/h3_verdict.csv and results/candidate_selection.csv.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import h3_verdict  # noqa: E402
from src.io import load_config  # noqa: E402

CANDIDATES = ["P1", "P2", "P3_pca8", "P3_pca32", "P4_pca8", "P4_pca32"]


def main():
    cfg = load_config()
    res = Path(cfg["results_dir"])
    prop = pd.read_csv(res / "proposed.csv")
    val = prop[(prop.split == "val") & (prop.kind == "primary") & prop.model.isin(CANDIDATES)]
    sel = val.groupby("model")[["pr_auc", "roc_auc", "f1"]].mean().sort_values("pr_auc",
                                                                                ascending=False)
    sel.round(4).to_csv(res / "candidate_selection.csv")
    cand = sel.index[0]

    boot = pd.read_csv(res / "bootstrap_proposed.csv")
    rows = []
    for pop in ["all", "excl_never_tracked"]:
        b = boot[(boot.a == cand) & (boot.b == "B3") & (boot.population == pop)
                 & (boot.scope == "pooled")].set_index("metric")
        pooled = {m: (b.loc[m, "diff"], b.loc[m, "ci_lo"], b.loc[m, "ci_hi"])
                  for m in ["f1", "roc_auc", "pr_auc"]}
        verdict, harm = h3_verdict(pooled)
        rows.append({"population": pop, "candidate": cand, "verdict": verdict,
                     "harm": "; ".join(harm) or "none",
                     **{f"{m}_{k}": round(v, 4) for m, t in pooled.items()
                        for k, v in zip(["diff", "lo", "hi"], t)}})
    out = pd.DataFrame(rows)
    out.to_csv(res / "h3_verdict.csv", index=False)
    print("candidate selection (mean val over primary folds):\n", sel.round(4).to_string())
    print("\n", out.T.to_string())


if __name__ == "__main__":
    main()
