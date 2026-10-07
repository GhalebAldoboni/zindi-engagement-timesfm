# Notes that REPORT.md must carry (collected during the run)

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
