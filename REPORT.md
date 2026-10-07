# Foundation-Model Embeddings for New-User Engagement Prediction (PoC)

> Draft. Sections are filled by `scripts/06_make_report.py`. The pre-registered
> decision rule below was written on 2026-10-01, **before any P model was run**,
> and must not be edited after P results exist. Additions go in a dated addendum.

## Pre-registered decision rule for H3 (fixed 2026-10-01)

**Comparison.** The pre-selected best hybrid model (see "Candidate" below) against B3
(LightGBM, handcrafted features, hyperparameters frozen at STOP 2, mean of 5 seeds).
Both use the same folds and seeds. B1 (logistic regression) appears alongside as a
co-reference in every comparison table.

**Test.** A paired bootstrap of (model − B3) on the two primary test cohorts (3 and 4),
with 2,000 resamples. Users are resampled within each fold, the same users for both
models. The pooled statistic is the mean of the two per-fold differences. 95%
percentile CIs are computed for **F1, ROC-AUC and PR-AUC**. Per-fold CIs and one-sided
p-values are also reported. Primary population: all test users. Sensitivity
population: without the 778 never-tracked users.

**Verdict (as specified by the project lead):**
- **SUPPORTED:** the pooled 95% CI of the difference is above 0 for PR-AUC **and** for F1.
- **PARTIAL:** the CI is above 0 for PR-AUC or ROC-AUC only.
- **INCONCLUSIVE:** all CIs include 0.
- **NOT SUPPORTED:** point estimates ≤ 0 on all three metrics.

**Resolution.** The smallest difference the pooled test can resolve is the CI
half-width of B3 vs B1 (results/bootstrap_b3_vs_b1.csv): F1 ±0.021, ROC-AUC ±0.013,
PR-AUC ±0.022. An INCONCLUSIVE verdict means any true effect is smaller than about
these values. This is a proxy: a hybrid model is more correlated with B3 than B1 is,
so its own CI against B3 will probably be somewhat narrower.

**Candidate (O1, approved 2026-10-04).** The P model with the highest mean **val**
PR-AUC over the two primary folds, chosen among P1, P2, P3_pca{8,32} and P4_pca{8,32}.
Test results are never used for selection. All other P models are reported
descriptively, without a verdict.

**Open points O1–O4: APPROVED by the project lead on 2026-10-04, before any P model
was run.** Applied exactly as written below.
- **O1, candidate selection:** which P model counts as "best". Proposal: the P model
  with the highest mean **val** PR-AUC over the two primary folds, chosen among P1–P4 ×
  PCA {8, 32}. Test results are never used for selection. All other P models are
  reported descriptively, without a verdict.
- **O2, overlap:** "all CIs include 0" and "all point estimates ≤ 0" can both hold.
  Proposal: check in the order SUPPORTED → PARTIAL → NOT SUPPORTED → INCONCLUSIVE.
- **O3, gap:** a CI above 0 for F1 only, with neither AUC, fits no category.
  Proposal: INCONCLUSIVE, with the F1 result stated.
- **O4, harm:** a CI entirely below 0 on any metric is reported as "significantly
  worse on <metric>" next to the verdict.

## Results

All numbers are on the test cohorts of the two primary folds (cohort 3 and cohort 4),
mean ± std over the two folds, unless marked otherwise. `*` marks a 95% CI that excludes 0.

### 1. Verdict on H3 (pre-registered)

Candidate selected on validation (O1): **P4_pca8**.

| population | verdict | ΔF1 | ΔROC-AUC | ΔPR-AUC | harm |
|---|---|---|---|---|---|
| all | **INCONCLUSIVE** | +0.013 [-0.001, +0.027] | +0.001 [-0.002, +0.004] | +0.002 [-0.003, +0.008] | none |
| excl_never_tracked | **INCONCLUSIVE** | +0.013 [-0.001, +0.027] | +0.001 [-0.002, +0.004] | +0.002 [-0.003, +0.008] | none |

Validation selection table (mean val over primary folds):

| model    |   pr_auc |   roc_auc |     f1 |
|:---------|---------:|----------:|-------:|
| P4_pca8  |   0.5394 |    0.7776 | 0.5036 |
| P4_pca32 |   0.5382 |    0.7767 | 0.502  |
| P3_pca8  |   0.5368 |    0.7763 | 0.5083 |
| P3_pca32 |   0.5354 |    0.7723 | 0.5093 |
| P1       |   0.5147 |    0.778  | 0.5006 |
| P2       |   0.4675 |    0.7419 | 0.4598 |

### 2. Baselines and proposed models

| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B0 | 0.394 ± 0.021 | 0.691 ± 0.021 | 0.233 ± 0.028 |
| B1 | 0.453 ± 0.010 | 0.778 ± 0.011 | 0.456 ± 0.030 |
| B2 | 0.443 ± 0.012 | 0.774 ± 0.027 | 0.430 ± 0.030 |
| B3 | 0.446 ± 0.002 | 0.794 ± 0.033 | 0.472 ± 0.045 |
| B4 | 0.457 ± 0.005 | 0.788 ± 0.030 | 0.467 ± 0.018 |
| P1 | 0.476 ± 0.022 | 0.808 ± 0.030 | 0.469 ± 0.037 |
| P2 | 0.435 ± 0.002 | 0.779 ± 0.049 | 0.417 ± 0.051 |
| P3_pca8 | 0.454 ± 0.000 | 0.795 ± 0.031 | 0.475 ± 0.043 |
| P3_pca32 | 0.456 ± 0.006 | 0.793 ± 0.031 | 0.474 ± 0.044 |
| P4_pca8 | 0.459 ± 0.000 | 0.795 ± 0.030 | 0.474 ± 0.054 |
| P4_pca32 | 0.445 ± 0.010 | 0.793 ± 0.032 | 0.470 ± 0.050 |

| comparison | ΔF1 [95% CI] | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| B3 − B1 | -0.008 [-0.029, +0.013] | +0.016 [+0.003, +0.029] * | +0.016 [-0.006, +0.037] |
| P1 − B3 | +0.030 [+0.005, +0.053] * | +0.014 [+0.004, +0.024] * | -0.003 [-0.026, +0.020] |
| P1 − B1 | +0.022 [-0.005, +0.049] | +0.030 [+0.016, +0.044] * | +0.013 [-0.018, +0.042] |
| P2 − B3 | -0.011 [-0.037, +0.015] | -0.014 [-0.027, -0.001] * | -0.055 [-0.082, -0.023] * |
| P2 − B1 | -0.018 [-0.044, +0.009] | +0.001 [-0.015, +0.018] | -0.039 [-0.069, -0.004] * |
| P3_pca8 − B3 | +0.008 [-0.003, +0.020] | +0.001 [-0.002, +0.004] | +0.004 [-0.001, +0.008] |
| P3_pca8 − B1 | +0.001 [-0.019, +0.021] | +0.017 [+0.003, +0.030] * | +0.019 [-0.003, +0.040] |
| P3_pca32 − B3 | +0.010 [-0.006, +0.027] | -0.000 [-0.005, +0.004] | +0.002 [-0.006, +0.009] |
| P3_pca32 − B1 | +0.003 [-0.017, +0.023] | +0.015 [+0.002, +0.029] * | +0.018 [-0.005, +0.039] |
| P4_pca8 − B3 | +0.013 [-0.001, +0.027] | +0.001 [-0.001, +0.004] | +0.002 [-0.003, +0.008] |
| P4_pca8 − B1 | +0.005 [-0.015, +0.028] | +0.017 [+0.004, +0.030] * | +0.018 [-0.005, +0.039] |
| P4_pca32 − B3 | -0.001 [-0.017, +0.015] | -0.000 [-0.005, +0.005] | -0.002 [-0.010, +0.006] |
| P4_pca32 − B1 | -0.009 [-0.032, +0.017] | +0.015 [+0.002, +0.029] * | +0.014 [-0.010, +0.036] |

### 3. Hypotheses

- **H1** (embeddings alone underperform B3): **mixed.** P1 (raw-series embeddings) is level with or better than B3 (F1 and ROC-AUC CIs above 0, post hoc); P2 (smoothed) is significantly worse on ROC-AUC and PR-AUC.
- **H2** (smoothing improves embeddings): **reversed.** P2 is worse than P1, and in the hybrid the bandwidth (6–48 h) makes no difference (Section 5).
- **H3** (hybrid beats B3): **INCONCLUSIVE** under the pre-registered rule.
- **Mechanism:** TimesFM-3's per-series normalisation makes embeddings exactly scale-invariant (smoke test: cosine 1.000 between a series and 100× it), so absolute activity level, the strongest predictor, is lost.

### 4. Exploratory work (after Phase 5 results were seen; does not change the verdict)

Round E (pre-committed list, selection on val):

| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| E1 | 0.468 ± 0.026 | 0.807 ± 0.030 | 0.482 ± 0.043 |
| E2 | 0.475 ± 0.018 | 0.804 ± 0.027 | 0.487 ± 0.043 |
| E3 | 0.476 ± 0.036 | 0.809 ± 0.028 | 0.475 ± 0.039 |
| E4 | 0.466 ± 0.034 | 0.807 ± 0.028 | 0.483 ± 0.050 |

| comparison | ΔF1 [95% CI] | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| E2 − B3 | +0.029 [+0.014, +0.044] * | +0.010 [+0.004, +0.017] * | +0.016 [+0.002, +0.028] * |
| E2 − B1 | +0.022 [-0.001, +0.044] | +0.026 [+0.014, +0.038] * | +0.032 [+0.009, +0.052] * |

TimesFM's share of E2 (E2 vs the same ensemble without P1): f1 +0.018 [+0.001, +0.034] *; roc_auc +0.008 [+0.005, +0.011] *; pr_auc +0.008 [-0.001, +0.017].

Overnight round N (protocol `results/overnight_protocol.md`; dataset-specific features + CatBoost + refit on train+val; final list fixed before one test evaluation):

| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B3 | 0.446 ± 0.002 | 0.794 ± 0.033 | 0.472 ± 0.045 |
| E2 | 0.475 ± 0.018 | 0.804 ± 0.027 | 0.487 ± 0.043 |
| N1 | 0.481 ± 0.038 | 0.812 ± 0.024 | 0.499 ± 0.058 |
| N2 | 0.482 ± 0.026 | 0.812 ± 0.021 | 0.505 ± 0.062 |
| N3 | 0.488 ± 0.031 | 0.812 ± 0.025 | 0.498 ± 0.051 |
| N4 | 0.496 ± 0.038 | 0.813 ± 0.025 | 0.501 ± 0.055 |

| comparison | ΔF1 [95% CI] | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| N2 − B3 | +0.036 [+0.018, +0.055] * | +0.018 [+0.011, +0.025] * | +0.033 [+0.020, +0.045] * |
| N2 − E2 | +0.006 [-0.010, +0.024] | +0.008 [+0.002, +0.013] * | +0.018 [+0.003, +0.033] * |
| N4 − B3 | +0.050 [+0.031, +0.070] * | +0.019 [+0.012, +0.026] * | +0.029 [+0.016, +0.042] * |
| N4 − N2 | +0.014 [-0.004, +0.032] | +0.001 [-0.002, +0.004] | -0.004 [-0.013, +0.005] |
| N3 − N1 | +0.007 [-0.008, +0.022] | +0.001 [-0.002, +0.004] | -0.001 [-0.011, +0.008] |

**Best model: N2** (CatBoost on handcrafted + session/trend + competition-liveness features, refit on train+val). Adding TimesFM members (N4 vs N2, N3 vs N1) gives no significant change. Tried and rejected on validation: competition target encoding, scale-aware (anchored) TimesFM embeddings, TimesFM zero-shot next-month forecasts as features, class weighting, training-set augmentation with later months, bag-of-pages content features (see `results/overnight_log.md`).

### 5. Ablations (Phase 6; Δ vs the default variant, pooled test)

| study | variant | ΔF1 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| bandwidth | P4_pca8_P4_bw6 | -0.009 [-0.025, +0.006] | -0.002 [-0.004, +0.000] | -0.001 [-0.005, +0.003] |
| bandwidth | P4_pca8_P4_bw24 | +0.004 [-0.007, +0.015] | +0.003 [+0.001, +0.006] * | +0.007 [+0.002, +0.012] * |
| bandwidth | P4_pca8_P4_bw48 | -0.003 [-0.015, +0.009] | -0.000 [-0.003, +0.003] | -0.000 [-0.005, +0.005] |
| layer | P4_pca8_P4_Learly | -0.007 [-0.019, +0.006] | +0.000 [-0.002, +0.003] | +0.001 [-0.004, +0.005] |
| layer | P4_pca8_P4_Lmiddle | +0.005 [-0.005, +0.016] | +0.001 [-0.002, +0.003] | +0.003 [-0.002, +0.008] |
| pooling | P4_pca8_P4_poolLast | -0.008 [-0.023, +0.006] | +0.000 [-0.003, +0.003] | -0.001 [-0.007, +0.005] |
| pca_dim | P4_pca64_P4_abl | -0.003 [-0.016, +0.010] | -0.001 [-0.005, +0.003] | -0.005 [-0.012, +0.001] |
| channels | P4_pca8_P4_5ch | +0.004 [-0.007, +0.015] | -0.000 [-0.003, +0.002] | +0.002 [-0.002, +0.005] |
| multivariate | P4_pca8_P4_mv | +0.000 [-0.010, +0.011] | -0.001 [-0.003, +0.002] | -0.003 [-0.007, +0.001] |
| features | P4_pca8_P4_noStatic | +0.004 [-0.011, +0.018] | +0.002 [-0.002, +0.006] | +0.003 [-0.004, +0.009] |
| features | P4_pca8_P4_no_browse | -0.009 [-0.023, +0.006] | -0.003 [-0.006, +0.002] | -0.005 [-0.014, +0.004] |
| features | P4_pca8_P4_no_learn | +0.018 [+0.004, +0.032] * | -0.001 [-0.003, +0.002] | -0.002 [-0.008, +0.003] |
| features | P4_pca8_P4_no_compete | -0.046 [-0.069, -0.022] * | -0.008 [-0.013, -0.002] * | -0.032 [-0.047, -0.017] * |
| features | P4_pca8_P4_no_contribute | -0.004 [-0.015, +0.005] | -0.000 [-0.002, +0.001] | +0.003 [+0.000, +0.007] * |
| features | P4_pca8_P4_no_account | -0.004 [-0.018, +0.009] | -0.000 [-0.002, +0.002] | +0.001 [-0.002, +0.005] |
| features | P4_pca8_P4_no_onboarding | +0.000 [-0.015, +0.014] | +0.001 [-0.001, +0.003] | +0.000 [-0.003, +0.004] |
| features | P4_pca8_P4_no_badges | -0.011 [-0.024, +0.003] | -0.000 [-0.002, +0.001] | -0.002 [-0.006, +0.001] |
| features | P4_pca8_P4_no_comments | -0.000 [-0.007, +0.007] | -0.001 [-0.003, +0.000] | -0.001 [-0.004, +0.003] |
| layer | P1_P1_Learly | -0.021 [-0.042, -0.001] * | -0.009 [-0.015, -0.004] * | -0.002 [-0.018, +0.016] |
| layer | P1_P1_Lmiddle | -0.020 [-0.041, +0.000] | -0.002 [-0.008, +0.003] | -0.007 [-0.023, +0.009] |
| channels | P1_P1_5ch | -0.006 [-0.019, +0.006] | -0.003 [-0.008, +0.001] | -0.006 [-0.014, +0.002] |
| multivariate | P1_P1_mv | -0.033 [-0.054, -0.013] * | -0.009 [-0.016, -0.002] * | +0.001 [-0.018, +0.019] |

### 6. Sensitivity

**Without the 778 never-tracked users** (excl_never_tracked): F1 and PR-AUC are unchanged for every model; ROC-AUC drops by about 0.015–0.02 for all models alike. Verdict unchanged: **INCONCLUSIVE**.

**Official 'any activity' label** (includes the month-1 badge batch):

| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B0 | 0.394 ± 0.021 | 0.691 ± 0.021 | 0.233 ± 0.028 |
| B1 | 0.358 ± 0.056 | 0.721 ± 0.024 | 0.346 ± 0.063 |
| B3 | 0.386 ± 0.069 | 0.740 ± 0.065 | 0.340 ± 0.073 |
| B4 | 0.409 ± 0.013 | 0.755 ± 0.098 | 0.399 ± 0.041 |
| P4_pca8 | 0.384 ± 0.071 | 0.721 ± 0.089 | 0.314 ± 0.116 |

The badge artefact in the training label costs every trained model about 0.06 F1 and makes the hybrid significantly worse than B3 on ROC-AUC and PR-AUC.

### 7. Error analysis (validation, best model)

- Users with 6–20 events in their signup month hold 48% of returners but are ranked poorly (within-segment AUC 0.69); users with no competition activity hold 24% of returners (AUC 0.67). Users with more than 20 events are ranked well (AUC 0.80).
- Users with no signup-month events (9.1%) return at 0.24%; 778 of them never appear in any table (probably untracked). Embeddings cannot help this group by construction.

### 8. Context: the Zindi leaderboard

The leaderboard's top private scores (0.85–0.87) are not F1: a scoring bug and an apparent accuracy-based re-scoring explain them. Legitimate F1 for strong entries is about 0.42–0.51; see `results/leaderboard_analysis.md`.

### 9. Related work and approaches tried

Published recipes for new-user churn (Yahoo Answers, Snapchat, MOOC dropout / CFIN, KDD Cup 2015, WSDM Cup 2018) were reviewed and the applicable ones tested on validation, including 2025–2026 methods (TabICLv2 tabular foundation model: val PR 0.551 with no tuning and 0.556 fine-tuned, a tie with CatBoost 0.559; EBES-style GRU: 0.532); see `results/literature_review.md`. None improved on N2.

## Data, method and limitations

Collected notes (data audit, clock correction, batch-event detection, label deviation, licences, compute) follow verbatim from `results/report_notes.md`.

## Deviations from the official task
- **Label.** Our primary label `label_engaged` = any UserActivity row in the month after
  signup, excluding `badge_*` events and confirmed system-wide batch events. The
  official Zindi label counts any row. The reason: a one-day `badge_OCZE` award
  (month 1, day 15; 2,892 users; 99% of rows between 09:00 and 11:00; 94% of recipients
  did nothing else that day) makes 77% of cohort 1 "active" (27% without it).
  Sensitivity: B3 and the best P model are re-run with `label_any`.
- **Evaluation data.** Zindi's test set (cohort 5, month-4 signups, next-month activity
  removed) is unlabelled and unused. Cohort 6 has no next month. Neither is used in any fit.
- **Folds.** Primary: test on cohort 3 (train 0–1, val 2) and on
  cohort 4 (train 0–2, val 3). Supplementary: test on cohort 2 (train 0, val 1).
- **H3 criterion (revised at STOP 1).** Paired bootstrap of the F1 difference (best P
  minus B3), resampling within each fold and pooling over both primary test cohorts;
  H3 holds if the 95% CI is above 0. Per-fold results are reported as well.
  ROC-AUC and PR-AUC get equal weight with F1, because the active rate falls across
  cohorts (0.198 → 0.169 → 0.115) and a threshold tuned on val may not transfer.

## Data findings and corrections
- **Clock offset.** UserActivity timestamps run ahead of Users, CompetitionPartipation
  and Discussion: +2 h until month 12, day 4 around midday, then +1 h. Evidence:
  "Signed Up" vs Users.created, and "Joined Competition" vs participation rows.
  Corrected on load as a step at 14:00 on the activity clock; afterwards 99.94% of
  "Signed Up" events are within 1 min of the signup time.
- **Batch detection.** A robust z-score on distinct users per (event type, day) flagged
  7 spikes. Only the `badge_OCZE` award passed confirmation (time-concentrated and
  passive). The Dec-27 cluster (Signed Up, $identify, Confirmed Email, ...; 343 signups
  against a median of 48/day) is a genuine signup surge: every user in it also did
  other things that day. See results/batch_events.csv.
- 650 signup-month events fall before the signup time (probably anonymous activity
  merged via `$create_alias`). They are kept: same month, no leakage.
- 823 of 9,026 labelled users (9%) have no event in their signup month.

## Limitations
- Masked data; the month order is inferred, not confirmed by Zindi (strong evidence:
  0.071% pre-signup events under the chosen order, against 0.187% for the runner-up).
- `VariableDefinitions.csv` does not match the files: it documents `Day_of_week`
  columns, while the files have `Day_of_month` plus a clock time, and some column
  names differ.
- The 31-day month grid is an approximation; real month lengths are masked.
- Comments: 467 rows, 86% in month 1. Used in handcrafted features only, not as a series.

## Added at STOP 2
- **Clock:** the offset is a step (+2 h → +1 h) on month 12, day 4 around midday,
  not a monthly or DST change. Per-day evidence is in results/clock_offset_per_day.csv.
  Labels changed by the correction: label_engaged 5, label_any 2 (against no correction).
- **Month lengths** come from the data (30/31/30/31/31/30). The previous 31-day grid
  would have produced non-existent dates after the clock shift.
- "Joined Competition" was removed from features as a duplicate of participation rows (98.75%).
- **no_events segment:** 823 users (9.1%), positive rate 0.24%. 778 of them never
  appear in any table, so they are probably untracked. Embeddings cannot help this
  group by construction (empty series). It is reported as a separate error-analysis
  segment.
- **Phase 6** adds a 4- vs 5-channel ablation (contribute separate vs merged into compete).
- **B3 variance:** tuning-selection and seed noise (test F1 std about 0.01 across
  seeds) is comparable to the expected effect sizes; see the STOP 2 decision D3.

## Added after Phase 3
- **Smoothing edge correction:** each event's kernel is renormalised by its own mass
  inside the user's observation window [min(signup, first event), month end].
  Smoothed mass / raw count = 1.00 (5–95%: 0.987–1.013, resampling error). The
  rejected alternative, dividing by kernel mass at each evaluation point, lost about
  30% of the mass, because most events cluster right after signup.
- **Late-signup users** get short, very high-intensity bursts (e.g. about 130
  events/day squeezed into the last hour). This is correct under the definition;
  `days_observed` is in the features.
- **Series zero share** (4 approved channels): browse 13%, learn 22%, compete 47%,
  account 62%. Many channels are all-zero, which matters for TimesFM's per-series
  normalisation.
- **B4 (MiniRocket + LR, raw daily series):** about as strong as B1/B3 (F1 0.457,
  ROC-AUC 0.788, PR-AUC 0.467 on the primary folds). The LR C grid was extended
  downward after the first run picked the grid minimum. As with B3, test F1 moved
  about 0.017 with the grid.
- **TimesFM 3.0 licence:** weights under "TimesFM Non-Commercial License v1.0"
  (non-commercial / non-production); code under Apache-2.0. Use in this course
  project is non-commercial research. Source: huggingface.co/google/timesfm-3.0-pytorch,
  github.com/google-research/timesfm.
- **Environment:** numba 0.62.1 (the last release with x86-64 macOS wheels) pulled
  numpy back from 2.5.3 to 2.3.5. B0–B3 results reproduce exactly (max abs diff 0.0).

## Added at STOP 3
- **Embedding protocol:** forward hooks on TimesFM-3 blocks 4/9/19 during
  `decode(horizon=1)`, pooled over context patches, univariate per channel and then
  concatenated. Masked left-padding replaces the plan's zero padding.
- **Scale invariance (smoke test):** the embedding is identical for any rescaling of a
  series (cosine 1.000), and all constant series, including all-zero, share one
  embedding. On real data, raw-series embeddings separate only 39% of users
  (3,520 distinct), against 5,652 distinct count patterns.
- **Compute:** 828 s on one RTX A4000 for all 16 embedding sets.

## Context only: official Zindi leaderboard (read 2026-10-04, logged-in view; no names recorded)
- Final leaderboard, competition closed 13 Feb 2023, metric F1 on Zindi's hidden test set
  (cohort 5 = month-4 signups, "any activity" label). 220 teams on the leaderboard.
- Private F1, top 50: ranks 1–3 are shown as 0.87 / 0.86 / 0.85 (2 decimals; their public
  scores are 0.50 / 0.71 / 0.50, so these look like post-review final scores). Rank 4: 0.843.
  Rank 5: 0.665. Ranks 6–10: 0.509–0.487. Rank 20: 0.463. Rank 30: 0.440. Rank 50: 0.424.
- **Not comparable** to our numbers: different test cohort (ours: 3 and 4, labels
  observed), our primary label excludes the badge batch, and no tuning on test. Our test
  F1 range (B3 0.446, P4_pca8 0.459, P1 0.476) sits in the same band as the typical
  top-50 private scores (about 0.42–0.51). This is context only, not a ranking claim.

## Context: why the leaderboard's top scores are not F1 (from the competition's public discussion, read 2026-10-04)
- A confirmed scoring bug: duplicate rows in a submission inflated F1 (one participant
  showed 0.99 by padding a file with duplicates). Participants report that the final
  re-scoring of the top entries computed **accuracy rather than F1**. With 14.7%
  positives in the test set (197/1,340, per the discussion), predicting all-negative
  already gives 0.853 accuracy, which matches the top-3 private scores of 0.85–0.87.
- Legitimate F1 reported by participants: a 12th-place 4-model ensemble at 0.483
  private, and a top-20 notebook author who could not exceed about 0.50 without duplicates.
- So the realistic F1 ceiling on this task is about 0.48–0.51. Our E2 (0.475 on test
  cohorts 3–4) is in that range; 0.70 F1 is not supported by any evidence.

## Threshold policy under base-rate drift (exploratory, applied once)
The return rate drifts across cohorts (0.198 → 0.169 → 0.115). A drift-adjusted
decision rule was chosen on development transfer only: predict as positive the top
(val positive share × label-free drift estimate) of users. Applied once on test:
N2 F1 0.482 → **0.497** (+0.015 [−0.002, +0.033], CI includes 0); N4 0.496 → 0.496.
Per fold: cohort 3 ≈ 0.52–0.53, cohort 4 ≈ 0.47. ROC-AUC and PR-AUC are unaffected
(they are threshold-free).

## EDA findings on user behaviour (train cohorts 0–2; descriptive)
- **Delayed competition joiners return most.** Next-month return rate by time from
  signup to first competition join: within 1 h 32%; 1–24 h 46%; 1–10 days 60%; more than
  10 days 76%; never joined 14%. A deliberate later join signals commitment; a join
  within the signup session is often a one-off (e.g. arriving via a hackathon link).
- **Entry point:** users whose first non-onboarding action was a competition page
  return at only 4% (n = 186), against 26–28% for those who first browse or open
  learning pages.
- **Live competitions:** users with 0/1/2/3+ competitions still active at month end
  return at 13/35/43/65%.
- These are strong descriptive signals, but adding them as features gave no
  validation gain (val PR 0.558 vs 0.559): the model already captures them through
  recency, last-active position and competition liveness.

## Kernel methods, non-linear features and a new architecture (validation only)
| Model | val PR-AUC | val ROC-AUC |
|---|---|---|
| CatBoost-v2 (best) | **0.559** | 0.782 |
| Linear LR (signed-log, scaled) | 0.536 | 0.780 |
| RBF-kernel LR (Nyström, 1,500 components) | 0.534 | **0.784** |
| Laplacian-kernel LR (Nyström) | 0.542 | 0.782 |
| Spline (GAM-style) LR | 0.541 | 0.782 |
| Degree-2 polynomial interactions LR | 0.516 | 0.762 |
| CatBoost + 16 Laplacian-kernel components | 0.558 | 0.782 |
| **MVRN** (new multi-view network: MLP on features + GRU on the daily sequence + TimesFM projection, fused by learned per-user gating) | 0.538 | 0.776 |
| MVRN + CatBoost (rank-avg) | 0.554 | 0.782 |
| **MVRN-2** (+ competition-set view: per-competition rows with first-touch/join timing, user activity and competition liveness/momentum/age, attention-pooled; tab view with all v2–v4 features) | 0.541 | 0.780 |
| MVRN-2 + CatBoost (rank-avg) | 0.554 | 0.784 |

Kernels add non-linearity to linear models (Laplacian +0.006 PR over linear) but do not
reach the tree ensembles, which are already non-linear. MVRN's learned gate puts
**74–89% of its weight on the engineered-feature view, 7–21% on the daily sequence and
only 4–5% on the TimesFM embedding**. This is a model-internal confirmation that the
foundation-model view carries little extra signal here. In MVRN-2 the new competition-set
view takes 15–26% of the gate weight (second only to the engineered features; sequence
17–22%), while TimesFM stays at 5%.

## Future work

- Signature-kernel and Nyström-kernel models on raw event timestamps (scale-aware by design).
- Other foundation models (Mantis, MOMENT) and other engagement datasets, to test generality.
- Distillation was not pursued: the teacher's embeddings add no signal beyond counts here.
