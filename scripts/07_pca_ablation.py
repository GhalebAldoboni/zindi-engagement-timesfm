"""Ablation: number of PCA components of the TimesFM-3 embedding vs the full embedding.

Inputs: results/proposed_sweep.csv and cached predictions from
    python scripts/04_run_proposed.py --models P3 P4 --pca-dims 2 4 8 16 32 64 128 -1 --tag _sweep --no-bootstrap
k = 0 is the tuned LightGBM baseline (no embedding); "full" = all 5,120 standardised dimensions.
Writes results/bootstrap_pca_sweep.csv, report/figures/fig_pca_ablation.png, report/ablation_pca.tex.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import compare  # noqa: E402
from src.io import load_config  # noqa: E402

KS = [2, 4, 8, 16, 32, 64, 128, "full"]
REPS = {"P3": "raw daily series", "P4": "smoothed curves"}
PARAGRAPH = r"""No compression level turns the embedding into a reliable gain in ranking quality: every PR-AUC confidence interval includes zero and all differences are within $\pm0.004$, except that adding more components of the smoothed-curve embedding steadily \emph{hurts} (down to $-0.014$ with the full embedding), consistent with extra dimensions adding noise. For the raw-series embedding, F1 drifts upward with $k$ (from $+0.001$ at $k=2$ to $+0.024$ at $k=128$), and two settings ($k=64$, $k=128$) have F1 intervals just above zero. We read this cautiously: the trend does not appear in the threshold-free PR-AUC, validation PR-AUC \emph{decreases} with $k$ (so these settings would not have been selected), and with 16 settings tested about one such interval is expected by chance. Overall, the ablation is consistent with the explanation offered in Section~\ref{sec:why}: if normalisation has removed the activity level, neither compressing nor expanding the embedding can restore it."""


def name(rep, k):
    return f"{rep}_{'full' if k == 'full' else f'pca{k}'}_sweep"


def main():
    cfg = load_config(str(ROOT / "configs/default.yaml"))
    res = ROOT / cfg["results_dir"]
    boot_f = res / "bootstrap_pca_sweep.csv"
    if boot_f.exists():
        boot = pd.read_csv(boot_f)
    else:
        rows = []
        for rep in REPS:
            for k in KS:
                rows += [r for r in compare(cfg, name(rep, k), "B3") if r["scope"] == "pooled"]
                print("bootstrapped", name(rep, k), flush=True)
        boot = pd.DataFrame(rows)
        boot.to_csv(boot_f, index=False)

    sw = pd.read_csv(res / "proposed_sweep.csv")
    base = pd.read_csv(res / "baselines.csv")
    prim = lambda d: d[(d.kind == "primary") & (d.population == "all")]  # noqa: E731
    b3_test = prim(base)[(base.model == "B3") & (base.split == "test")][["f1", "roc_auc", "pr_auc"]].mean()
    b3_val = prim(base)[(base.model == "B3") & (base.split == "val")]["pr_auc"].mean()

    table = []
    for rep, lab in REPS.items():
        for k in KS:
            m = name(rep, k)
            s = prim(sw)[sw.model == m]
            t = s[s.split == "test"][["f1", "roc_auc", "pr_auc"]].mean()
            evr = s[s.split == "test"]["pca_explained_var"].mean()
            bb = boot[(boot.a == m)].set_index("metric")
            table.append({"rep": rep, "label": lab, "k": k, "evr": evr,
                          "val_pr": s[s.split == "val"]["pr_auc"].mean(),
                          **{f"test_{c}": t[c] for c in t.index},
                          **{f"d_{c}": bb.loc[c, "diff"] for c in ("f1", "pr_auc")},
                          **{f"lo_{c}": bb.loc[c, "ci_lo"] for c in ("f1", "pr_auc")},
                          **{f"hi_{c}": bb.loc[c, "ci_hi"] for c in ("f1", "pr_auc")}})
    T = pd.DataFrame(table)
    T.to_csv(res / "pca_ablation_summary.csv", index=False)

    # ---- figure ----
    SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"P3": "#2a78d6", "P4": "#eb6834"}
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    xs = list(range(len(KS)))
    for ax, (m, title) in zip(axes, [("pr_auc", "Δ PR-AUC vs LightGBM (no embedding)"),
                                     ("f1", "Δ F1 vs LightGBM (no embedding)")]):
        ax.axhline(0, color=T1, lw=1)
        for j, rep in enumerate(REPS):
            d = T[T.rep == rep]
            off = (j - 0.5) * 0.18
            ax.errorbar([x + off for x in xs], d[f"d_{m}"], yerr=[d[f"d_{m}"] - d[f"lo_{m}"], d[f"hi_{m}"] - d[f"d_{m}"]],
                        fmt="o-", color=COL[rep], lw=1.6, ms=5, capsize=0, elinewidth=2, label=f"TimesFM on {REPS[rep]}")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("PCA components of the 5,120-d embedding (full = no PCA)")
        ax.set_title(title, loc="left", fontsize=11.5, color=T1, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(0.01, 0.01, "Test cohorts 3–4, LightGBM with frozen B3 settings, mean of 5 seeds; bars: 95% paired-bootstrap CI vs the baseline (k = 0).",
             fontsize=8.5, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "report/figures/fig_pca_ablation.png", dpi=200)

    # ---- LaTeX section ----
    def fmt_ci(d, lo, hi):
        star = "$^{*}$" if (lo > 0 or hi < 0) else ""
        return f"${d:+.3f}$\\,[${lo:+.3f}$, ${hi:+.3f}$]{star}"
    lines = [r"\subsection{Ablation: PCA dimension versus the full embedding}\label{sec:pca}",
             r"To test whether the inconclusive result depends on how strongly the embedding is compressed, we varied the",
             r"number of PCA components appended to the handcrafted features ($k\in\{2,4,8,16,32,64,128\}$) and also used the",
             r"full standardised 5{,}120-dimensional embedding without PCA, for both the raw-series and smoothed-series",
             r"embeddings. All other settings (frozen LightGBM hyperparameters, five seeds, folds) were unchanged, and",
             r"$k=0$ is the LightGBM baseline (Figure~\ref{fig:pca}, Table~\ref{tab:pca}).",
             r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{fig_pca_ablation.png}",
             r"\caption{Effect of the number of PCA components of the TimesFM-3 embedding on test PR-AUC and F1,",
             r"relative to the LightGBM baseline without embeddings. Bars are 95\% paired-bootstrap confidence intervals.}",
             r"\label{fig:pca}\end{figure}",
             r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccccc}\toprule",
             r"Embedding & $k$ & Expl.\ var. & Val.\ PR-AUC & Test PR-AUC & $\Delta$PR-AUC [95\% CI] & $\Delta$F1 [95\% CI]\\\midrule",
             f"none (LightGBM) & 0 & -- & {b3_val:.3f} & {b3_test['pr_auc']:.3f} & -- & --\\\\\\midrule"]
    for rep, lab in REPS.items():
        for _, r in T[T.rep == rep].iterrows():
            evr = "1.00" if r.k == "full" else f"{r.evr:.2f}"
            lines.append(f"{lab} & {r.k} & {evr} & {r.val_pr:.3f} & {r.test_pr_auc:.3f} & "
                         f"{fmt_ci(r.d_pr_auc, r.lo_pr_auc, r.hi_pr_auc)} & {fmt_ci(r.d_f1, r.lo_f1, r.hi_f1)}\\\\")
        lines.append(r"\midrule" if rep == "P3" else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{PCA-dimension ablation (test cohorts 3--4). $^{*}$: confidence interval excludes zero.",
              r"Explained variance is the share of embedding variance kept by the PCA components (mean over folds).}",
              r"\label{tab:pca}\end{table}"]
    lines.append(PARAGRAPH)          # interpretation written after inspecting the numbers
    (ROOT / "report/ablation_pca.tex").write_text("\n".join(lines) + "\n")
    print(T[["rep", "k", "evr", "val_pr", "test_pr_auc", "d_pr_auc", "lo_pr_auc", "hi_pr_auc", "d_f1", "lo_f1", "hi_f1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
