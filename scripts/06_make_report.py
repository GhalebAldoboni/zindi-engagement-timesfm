"""Phase 8: build REPORT.md from the result files.

Everything above the "## Results" heading in REPORT.md (the pre-registered decision
rule) is kept byte-for-byte. Everything below it is regenerated from results/*.csv.

  --rerun              rerun the whole pipeline first (data audit -> final models);
                       TimesFM embeddings are reused from results/cache/emb unless
  --rerun-embeddings   is also given (CPU, slow: uses .venv-tfm)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io import load_config  # noqa: E402

PY, PY_TFM = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv-tfm/bin/python")
PIPELINE = [
    [PY, "scripts/00_audit_data.py"], [PY, "scripts/00b_clock_offset.py"],
    [PY, "scripts/01_build_dataset.py"],
    [PY, "scripts/dev/dev_experiments.py", "--build-features"],
    [PY, "scripts/02b_build_series.py", "--bandwidths", "6", "12", "24", "48"],
    [PY, "scripts/02b_build_series.py", "--channels-key", "series_channels_5"],
    [PY, "scripts/02_run_baselines.py"], [PY, "scripts/04_run_proposed.py"],
    [PY, "scripts/04b_h3_verdict.py"], [PY, "scripts/04c_exploratory.py"],
    [PY, "scripts/05_run_ablations.py"], [PY, "scripts/04d_overnight_final.py"],
    [PY, "scripts/02_run_baselines.py", "--label", "label_any"],
    [PY, "scripts/04_run_proposed.py", "--label", "label_any", "--models", "P4", "--pca-dims", "8"],
]


def fmt_ci(d, lo, hi):
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def test_table(df, models, pop="all", kind="primary"):
    t = df[(df.split == "test") & (df.population == pop) & (df.kind == kind) & df.model.isin(models)]
    g = t.groupby("model")[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]).reindex(models)
    lines = ["| model | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|"]
    for m, r in g.iterrows():
        cells = [f"{r[(k, 'mean')]:.3f} ± {r[(k, 'std')]:.3f}" for k in ("f1", "roc_auc", "pr_auc")]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def boot_table(b, pairs, pop="all"):
    lines = ["| comparison | ΔF1 [95% CI] | ΔROC-AUC | ΔPR-AUC |", "|---|---|---|---|"]
    for a, ref in pairs:
        s = b[(b.a == a) & (b.b == ref) & (b.scope == "pooled") & (b.population == pop)].set_index("metric")
        if s.empty:
            continue
        cells = [fmt_ci(s.loc[m, "diff"], s.loc[m, "ci_lo"], s.loc[m, "ci_hi"]) for m in ("f1", "roc_auc", "pr_auc")]
        lines.append(f"| {a} − {ref} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--rerun-embeddings", action="store_true")
    args = ap.parse_args()
    if args.rerun:
        steps = list(PIPELINE)
        if args.rerun_embeddings:
            steps.insert(5, [PY_TFM, "scripts/03_extract_embeddings.py", "--device", "cpu"])
        for cmd in steps:
            print("running", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)

    cfg = load_config(str(ROOT / "configs/default.yaml"))
    R = ROOT / cfg["results_dir"]
    rd = lambda n: pd.read_csv(R / n)  # noqa: E731
    base, prop, expl, night = rd("baselines.csv"), rd("proposed.csv"), rd("exploratory.csv"), rd("overnight_final.csv")
    allr = pd.concat([base, prop, expl, night])
    b_all = pd.concat([rd("bootstrap_b3_vs_b1.csv").assign(population=lambda d: d.population),
                       rd("bootstrap_proposed.csv"), rd("bootstrap_exploratory.csv"),
                       rd("bootstrap_overnight.csv"), rd("bootstrap_overnight_ensemble_vs_single.csv")])
    verdict = rd("h3_verdict.csv")
    sel = rd("candidate_selection.csv")
    abl, babl = rd("ablations.csv"), rd("bootstrap_ablations.csv")
    any_b, any_p = rd("baselines_label_any.csv"), rd("proposed_label_any.csv")
    e2_wo = rd("bootstrap_e2_without_timesfm.csv")

    head = (ROOT / "REPORT.md").read_text().split("## Results")[0]
    v = verdict.set_index("population")
    out = [head.rstrip(), "", "## Results", "",
           "All numbers are on the test cohorts of the two primary folds (cohort 3 and cohort 4),",
           "mean ± std over the two folds, unless marked otherwise. `*` marks a 95% CI that excludes 0.",
           "",
           "### 1. Verdict on H3 (pre-registered)", "",
           f"Candidate selected on validation (O1): **{v.loc['all', 'candidate']}**.",
           "", "| population | verdict | ΔF1 | ΔROC-AUC | ΔPR-AUC | harm |", "|---|---|---|---|---|---|"]
    for pop, r in v.iterrows():
        out.append(f"| {pop} | **{r.verdict}** | {fmt_ci(r.f1_diff, r.f1_lo, r.f1_hi)} | "
                   f"{fmt_ci(r.roc_auc_diff, r.roc_auc_lo, r.roc_auc_hi)} | "
                   f"{fmt_ci(r.pr_auc_diff, r.pr_auc_lo, r.pr_auc_hi)} | {r.harm} |")
    out += ["", "Validation selection table (mean val over primary folds):", "",
            sel.round(4).to_markdown(index=False), "",
            "### 2. Baselines and proposed models", "",
            test_table(allr, ["B0", "B1", "B2", "B3", "B4", "P1", "P2", "P3_pca8", "P3_pca32",
                              "P4_pca8", "P4_pca32"]), "",
            boot_table(b_all, [("B3", "B1")] + [(m, r) for m in ("P1", "P2", "P3_pca8", "P3_pca32",
                                                                   "P4_pca8", "P4_pca32") for r in ("B3", "B1")]),
            "", "### 3. Hypotheses", "",
            "- **H1** (embeddings alone underperform B3): **mixed.** P1 (raw-series embeddings) is level "
            "with or better than B3 (F1 and ROC-AUC CIs above 0, post hoc); P2 (smoothed) is "
            "significantly worse on ROC-AUC and PR-AUC.",
            "- **H2** (smoothing improves embeddings): **reversed.** P2 is worse than P1, and in the hybrid "
            "the bandwidth (6–48 h) makes no difference (Section 5).",
            f"- **H3** (hybrid beats B3): **{v.loc['all', 'verdict']}** under the pre-registered rule.",
            "- **Mechanism:** TimesFM-3's per-series normalisation makes embeddings exactly "
            "scale-invariant (smoke test: cosine 1.000 between a series and 100× it), so absolute "
            "activity level, the strongest predictor, is lost.",
            "", "### 4. Exploratory work (after Phase 5 results were seen; does not change the verdict)", "",
            "Round E (pre-committed list, selection on val):", "",
            test_table(allr, ["E1", "E2", "E3", "E4"]), "",
            boot_table(b_all, [("E2", "B3"), ("E2", "B1")]), "",
            "TimesFM's share of E2 (E2 vs the same ensemble without P1): " +
            "; ".join(f"{m} {fmt_ci(*e2_wo[(e2_wo.scope == 'pooled') & (e2_wo.metric == m)][['diff', 'ci_lo', 'ci_hi']].iloc[0])}"
                      for m in ("f1", "roc_auc", "pr_auc")) + ".",
            "", "Overnight round N (protocol `results/overnight_protocol.md`; dataset-specific features + "
            "CatBoost + refit on train+val; final list fixed before one test evaluation):", "",
            test_table(allr, ["B3", "E2", "N1", "N2", "N3", "N4"]), "",
            boot_table(b_all, [("N2", "B3"), ("N2", "E2"), ("N4", "B3"), ("N4", "N2"), ("N3", "N1")]), "",
            "**Best model: N2** (CatBoost on handcrafted + session/trend + competition-liveness "
            "features, refit on train+val). Adding TimesFM members (N4 vs N2, N3 vs N1) gives no "
            "significant change. Tried and rejected on validation: competition target encoding, "
            "scale-aware (anchored) TimesFM embeddings, TimesFM zero-shot next-month forecasts as "
            "features, class weighting, training-set augmentation with later months, "
            "bag-of-pages content features (see `results/overnight_log.md`).",
            "", "### 5. Ablations (Phase 6; Δ vs the default variant, pooled test)", "",
            "| study | variant | ΔF1 | ΔROC-AUC | ΔPR-AUC |", "|---|---|---|---|---|"]
    for _, r in abl.iterrows():
        s = babl[babl.a == r.variant].set_index("metric")
        out.append(f"| {r.study} | {r.variant} | " + " | ".join(
            fmt_ci(s.loc[m, "diff"], s.loc[m, "ci_lo"], s.loc[m, "ci_hi"]) for m in ("f1", "roc_auc", "pr_auc")) + " |")
    out += ["", "### 6. Sensitivity", "",
            "**Without the 778 never-tracked users** (excl_never_tracked): F1 and PR-AUC are unchanged "
            "for every model; ROC-AUC drops by about 0.015–0.02 for all models alike. Verdict "
            f"unchanged: **{v.loc['excl_never_tracked', 'verdict']}**.", "",
            "**Official 'any activity' label** (includes the month-1 badge batch):", "",
            test_table(pd.concat([any_b, any_p]), ["B0", "B1", "B3", "B4", "P4_pca8"]), "",
            "The badge artefact in the training label costs every trained model about 0.06 F1 and "
            "makes the hybrid significantly worse than B3 on ROC-AUC and PR-AUC.",
            "", "### 7. Error analysis (validation, best model)", "",
            "- Users with 6–20 events in their signup month hold 48% of returners but are ranked "
            "poorly (within-segment AUC 0.69); users with no competition activity hold 24% of "
            "returners (AUC 0.67). Users with more than 20 events are ranked well (AUC 0.80).",
            "- Users with no signup-month events (9.1%) return at 0.24%; 778 of them never appear in "
            "any table (probably untracked). Embeddings cannot help this group by construction.",
            "", "### 8. Context: the Zindi leaderboard", "",
            "The leaderboard's top private scores (0.85–0.87) are not F1: a scoring bug and an "
            "apparent accuracy-based re-scoring explain them. Legitimate F1 for strong entries is "
            "about 0.42–0.51; see `results/leaderboard_analysis.md`.",
            "", "### 9. Related work and approaches tried", "",
            "Published recipes for new-user churn (Yahoo Answers, Snapchat, MOOC dropout / CFIN, "
            "KDD Cup 2015, WSDM Cup 2018) were reviewed and the applicable ones tested on "
            "validation, including 2025–2026 methods (TabICLv2 tabular foundation model: val PR 0.551 "
            "with no tuning and 0.556 fine-tuned, a tie with CatBoost 0.559; EBES-style GRU: 0.532); see `results/literature_review.md`. None improved on N2.",
            "", "## Data, method and limitations", "",
            "Collected notes (data audit, clock correction, batch-event detection, label deviation, "
            "licences, compute) follow verbatim from `results/report_notes.md`.", "",
            (R / "report_notes.md").read_text().replace("# Notes that REPORT.md must carry (collected during the run)", "").strip(),
            "", "## Future work", "",
            "- Signature-kernel and Nyström-kernel models on raw event timestamps (scale-aware by design).",
            "- Other foundation models (Mantis, MOMENT) and other engagement datasets, to test generality.",
            "- Distillation was not pursued: the teacher's embeddings add no signal beyond counts here.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(out))
    print("REPORT.md written:", len("\n".join(out).splitlines()), "lines")


if __name__ == "__main__":
    main()
