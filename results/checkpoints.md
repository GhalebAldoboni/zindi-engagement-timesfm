# Checkpoints

## STOP 1 — Phase 0 go/no-go (2026-10-01)

Full evidence: `results/audit.md` (regenerate with `python scripts/00_audit_data.py`).

**Verdict: GO, but two decisions are needed (D1, D2).**

### Granularity
- Every event table has Year (always 1), Month, Day_of_month and a clock time
  (HH:MM:SS; Users has microseconds). `VariableDefinitions.csv` is out of date: it
  describes Day_of_week columns that do not exist in the files.
- Month and day are real values with a time of day, so `t_day` = day − 1 + hour/24 is
  computable and Phases 3–6 are feasible.

### Month order
- Codes: {11, 12, 1, 2, 3, 4, 5}. We scored all 5040 orders by the share of event rows
  dated in a month before the user's signup month.
- The community order [11, 12, 1, 2, 3, 4, 5] ranks first: 0.071% (232 rows).
  The runner-up swaps 4 and 5 (0.187%); the next order scores 1.14%.
- The order is also consistent at day level: of 229,729 signup-month activity rows,
  only 27 are dated before the signup day.
- Treated as verified for this study; still listed as a limitation.

### Tables
- Users: 12,413 (one signup month each).
- UserActivity: 317,292 rows, 397 event `Title`s, 10,400 users.
- CompetitionPartipation: 8,385 rows. Discussion: 1,439 rows.
- Comments: 467 rows, 4 months only, 86% of them in month 1.
- Every user_id in every event table exists in Users.
- No exact duplicate rows.

### Cohorts and labels (label = any UserActivity row in signup month + 1)
| cohort (month) | n | rate (any row) | rate (excl. badge_*) |
|---|---|---|---|
| 0 (11) | 1262 | 0.185 | 0.185 |
| 1 (12) | 1982 | **0.773** | 0.274 |
| 2 (1)  | 1504 | 0.199 | 0.198 |
| 3 (2)  | 2278 | 0.169 | 0.169 |
| 4 (3)  | 2000 | 0.115 | 0.115 |
| 5 (4)  | 1382 | 0.006 — hidden test labels | — |
| 6 (5)  | 2005 | no next month | — |

### Findings that change the plan
1. **The test set is cohort 5.** All 1,340 SampleSubmission ids (`_Month_5`) are month-4
   signups, and their month-5 activity has been removed. Labelled cohorts are therefore
   0–4 (5 cohorts), not 6. Cohort 5 also looks partly stripped: only 53% of its users
   have signup-month activity, against about 88–92% in other cohorts.
2. **A batch system event inflates cohort 1's label.** 3,005 `badge_OCZE` rows fall on
   month 1, day 15 (a one-day badge award). Under "any row", 75.5% of cohort-1 users
   look active next month only because of this award. Without `badge_*` events the rate
   is 0.274, in line with the other cohorts.
3. Other activity titles that are probably system-generated: `$identify`,
   `$create_alias`, `Signed Up`, `Confirmed Email`. They don't change any label rate,
   but they belong in the feature design.

### Decisions needed
- **D1 — label definition.** (a) Exclude `badge_*` events from the label. This is the
  recommended option: it measures real user engagement. (b) Keep "any row" and accept
  the cohort-1 artefact.
- **D2 — folds.** With 5 labelled cohorts, the plan's rule (last 3 test cohorts, at
  least 2 train cohorts) only allows 2 folds: test 3 (train 0–1, val 2) and test 4
  (train 0–2, val 3). Options: (a) 2 folds, recommended; (b) 3 folds, where the extra
  fold tests on cohort 2 with only cohort 0 for training. H3's "≥ 2 of 3 folds"
  criterion would then become "both folds".
- Cohort 5 (the Zindi test set) is unused because it has no labels.

### STOP 1 answers (user, 2026-10-01)
D1: the label excludes badge_*; confirmed batch events are excluded from labels AND
features; detection must be systematic; a label_any sensitivity run is required.
D2: two primary folds (test cohorts 3 and 4) plus a supplementary fold (test cohort 2).
H3 = pooled within-fold paired bootstrap CI > 0. ROC-AUC and PR-AUC weighted equally
with F1. Comments go in features but not in series. Cohorts 5 and 6 are excluded.
Channel grouping must be approved before Phase 3.

## Phase 1 done (2026-10-01): awaiting channel-grouping approval

- Built by `scripts/01_build_dataset.py`. 14/14 tests pass (`pytest`).
- **New finding: clock offset.** UserActivity is +60 min ahead of the other tables
  (+120 min in month_idx 0). This is corrected on load; afterwards 99.1% of
  "Signed Up" events fall within 1 min of the signup time.
- **Batch detection:** 7 spikes flagged, 1 confirmed (badge_OCZE, month 1, day 15).
  It is removed from labels and features. The Dec-27 signup surge is genuine and kept.
  Evidence: results/batch_events.csv.
- Labelled users: 9,026 (cohorts 0–4). label_engaged rates: 0.185 / 0.273 / 0.198 /
  0.169 / 0.115. 823 users (9%) have no signup-month event.
- Event types and the proposed channels: results/event_types.csv and
  results/channel_coverage.csv. Proposal (configs/default.yaml, channels_approved: false):
  browse 87% of users, learn 78%, compete 52%, account 38%, contribute 7%.

### Phase 1 checkpoint answers (user, 2026-10-01)
Clock: estimate the offset per day; report how many labels changed. Add a `no_events`
feature, report it, and make it an error-analysis segment. Channels approved with
contribute merged into compete (4 channels); contribute keeps its own handcrafted
features. Add a 4- vs 5-channel ablation to Phase 6. Run B0–B3; hold B4.

## STOP 2: classical baselines (2026-10-01)

### Data changes since Phase 1
- **Clock offset is a step, not a monthly value.** Per-day estimates
  (results/clock_offset_per_day.csv): +2 h through month 11 and month 12 days 1–3,
  mixed on month 12 day 4, +1 h from day 5 on. The switch falls between 11:31 and
  13:22 on the Users clock on month 12 day 4, and participation rows confirm it
  independently. A midday switch is not a DST change at 02:00, so it is more likely a
  server change. Implemented as a step at 14:00 on the activity clock. "Signed Up"
  within 1 min of signup: 99.94% (flat by-month offset: 99.11%; no correction: 0%).
- **Labels changed by the correction** (results/clock_correction.csv), against no
  correction: label_engaged 5 (all 1→0; cohorts 0–1), label_any 2.
  Against the flat by-month offset: 1 each.
- **Real month lengths** replace the 31-day grid: 30/31/30/31/31/30 days (the last
  month runs to day 27, where the data ends). Recency, last-7-days and clock shifts
  across month boundaries now use them.
- "Joined Competition" is dropped from features and series: 98.75% of these events
  duplicate a participation row. Labels are unchanged.
- Missing Countries_ID (3,507 users) is now its own category, "missing".

### no_events segment (823 users, 9.1%)
- Positive rate: 0.24% (2/823), against 20.5% for users with events.
- 778 of the 823 never appear in any table in any month, so they are most likely
  untracked users rather than inactive ones.
- In the test folds the segment has 0 positives (c3: 111 users, c4: 174), and every
  model predicts 0 for it. Embeddings cannot help this group by construction (empty
  series).

### Baselines: test, label_engaged (results/baselines*.csv)
| model | F1 c3 | F1 c4 | F1 mean±std | ROC-AUC mean±std | PR-AUC mean±std |
|---|---|---|---|---|---|
| B0 last-7-days | 0.409 | 0.380 | 0.395±0.021 | 0.691±0.021 | 0.233±0.028 |
| B1 LogReg | 0.461 | 0.446 | 0.453±0.010 | 0.778±0.011 | 0.456±0.030 |
| B2 RF | 0.435 | 0.451 | 0.443±0.012 | 0.774±0.027 | 0.430±0.030 |
| B3 LightGBM | 0.452 | 0.445 | 0.448±0.005 | **0.797±0.034** | **0.474±0.044** |
Supplementary fold c2 (train on cohort 0 only): B3 F1 0.483, ROC 0.766, PR 0.536.
The std is over 2 folds only.

### Findings
- On ROC-AUC and PR-AUC, B3 is the best baseline on both primary folds. On F1 it is
  level with B1 (0.448 vs 0.453).
- **B3 tuning noise is about the size of the effects we want to measure.** Tuning
  picks among near-identical configs (best val PR-AUC differs by ≤0.004 between
  two search grids), but test F1 on c3 moved 0.482 → 0.452. With params fixed,
  retraining across 10 seeds gives test F1 std 0.006–0.010 and PR-AUC std 0.003–0.009.
- Threshold transfer (results/threshold_transfer_baselines.csv): the val-tuned
  threshold loses 0.00–0.03 F1 against the best possible test threshold (B3: about 0.01).
- B3 importance by group (gain, mean over primary folds): compete 0.39, browse 0.30,
  global/static 0.13, onboarding 0.08, learn 0.07, account 0.02, badges, comments
  and contribute ≤0.005 each. Top features: compete_last7, compete_rate,
  browse_span, compete_recency, browse_rate.

### Protocol notes
- B3: 30-trial random search per fold on val PR-AUC; early stopping on val logloss.
  num_leaves grid widened to include 3 and 4 after the first run picked the grid
  minimum (7) on every fold. That decision was made on that ground alone, not on test.
- B1 uses a fixed log1p before impute/scale, because the counts are heavily skewed.
- LightGBM `subsample` takes effect only with `subsample_freq=1` (now set).
- Phase 6 gains a 4- vs 5-channel ablation (`series_channels_5` in config).

### Decisions needed
- **D3 (recommended): average over 5 training seeds** for B3 and every P model (same
  seeds), so the B3-vs-P bootstrap measures the representation, not seed luck.
  The cost is 5× LightGBM fits (about 1 min per run here).
- **D4:** B4 (MiniRocket) is held until the series are built (Phase 3), as agreed.

### STOP 2 answers (user, 2026-10-01)
D3 approved (5-seed averaging, identical seeds, B3 and all P models). B3 params frozen
in config and reused unchanged for P3/P4. B1 is a co-reference in every table. PCA 8
joins the default comparison. Decision rule pre-registered before any P run (AUC and
PR-AUC bootstrapped too). Report the resolution as the B3-vs-B1 CI half-width. Report
results with and without the 778 never-tracked users. Then Phase 3 with 4 channels.

## Pre-registered H3 decision rule (written 2026-10-01, before any P model was run)
The full text is in REPORT.md and is copied here unchanged:
- SUPPORTED: pooled paired-bootstrap 95% CI for (best P − B3) above 0 for PR-AUC AND F1.
- PARTIAL: CI above 0 for PR-AUC or ROC-AUC only.
- INCONCLUSIVE: all CIs include 0.
- NOT SUPPORTED: point estimates ≤ 0 on all three metrics.
Pooled = mean of the per-fold differences on test cohorts 3 and 4. Users are
resampled within each fold, 2,000 resamples.
Open points O1–O4 (candidate selection by val PR-AUC; precedence; F1-only case; harm
reporting) are listed in REPORT.md and await confirmation before Phase 5.

## Baselines after D3 (frozen params, mean of 5 seeds), test, primary folds
| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B1 LogReg (co-ref) | 0.453±0.010 | 0.778±0.011 | 0.456±0.030 |
| B3 LightGBM | 0.446±0.002 | 0.794±0.034 | 0.472±0.045 |
B3 − B1, pooled (results/bootstrap_b3_vs_b1.csv): F1 −0.008 [−0.029, 0.013];
ROC-AUC +0.016 [0.003, 0.029] (p=0.009); PR-AUC +0.016 [−0.006, 0.037] (p=0.07).
**Resolution (half-width): F1 0.021, ROC-AUC 0.013, PR-AUC 0.022.**
Without the 778 never-tracked users: F1 and PR-AUC are unchanged (all models already
predicted them negative); ROC-AUC is about 0.017 lower for every model.

## Phase 3 done (2026-10-01)
- Series (scripts/02b_build_series.py): raw (9026, 4, 31) daily counts; smoothed
  (9026, 4, 512), bandwidth 12 h, edge-corrected, mass-conserving (ratio 1.00 ± 0.013).
  Figures: results/figures/series_examples_bw12.png. 21/21 tests pass.
- Zero-series share: browse 13%, learn 22%, compete 47%, account 62%.
- **B4 (MiniRocket + LR on raw series), primary folds:** F1 0.457±0.006, ROC-AUC
  0.788±0.030, PR-AUC 0.467±0.018. That is level with B1/B3, so a time-series
  representation of the raw counts already matches the handcrafted features.
- TimesFM 3.0: 0.3B params, 20 layers × 1280, patch 32, `timesfm[torch]`,
  `TimesFM3Evaluator` (has a device argument), multivariate input
  (num_variates, context). The docs do not describe an embedding API, so a forward
  hook is needed, as planned. Weights: non-commercial licence (recorded).
- Phase 4 compute estimate: about 72k series for the defaults (9,026 users × 4 channels × raw + smoothed), about 180k with the
  bandwidth ablations, each up to 512 points (16 patches), on a 0.3B model. This
  takes hours on this CPU and minutes on a GPU.

## STOP 3: TimesFM-3 embeddings (2026-10-04)

### Model and API (from the timesfm 3.0.2 source, not guessed)
- `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda").model`
  is a TimesFM3Torch with **330.7M params and 20 blocks** (`transformer_stack.layers.0–19`,
  width 1280, causal sequence attention). Full `named_modules()` listing:
  results/embeddings_extract.log.
- Embeddings come from forward hooks on blocks 4 (early), 9 (middle) and 19 (late, the
  default) during `decode(target, horizon=1)`, pooled over context patches (mean;
  `last` also stored for smoothed).
- **Change from the plan:** short series are padded with the model's own masked
  positions, not zeros, because zeros would be read as real data. The raw series
  (31 points) is one patch, so 1 token per channel and mean == last.
- Licence: weights under the TimesFM Non-Commercial License v1.0; code Apache-2.0.

### Smoke test (results/embeddings_smoke.json), late layer, 96 synthetic series in 8 types
- Leave-one-out 1-NN type accuracy: **0.875** (raw values 0.81; chance 0.116).
  Embeddings separate shapes meaningfully.
- **Full scale invariance:** constants at levels 0, 0.5, 1, 5, 50 and 500 give
  identical embeddings (cosine 1.000), and x vs 100x gives cosine 1.000. Absolute
  activity level is discarded entirely (RevIN), which is direct evidence for H1.

### Real-data embeddings (results/embeddings_checks.csv, embeddings_runtime.csv)
- 16 files, (9026, C·1280) float16, all finite. Checksums match the GPU box.
- Univariate: every empty channel maps to one fixed vector (max spread ≤ 0.001).
- **Collapse:** raw embeddings have only 3,520 distinct users (39%), against 5,652
  distinct raw count patterns, because scale invariance merges patterns that differ
  only in level. Smoothed: 7,252 distinct (80%).
- Runtime: 828 GPU-seconds in total on an RTX A4000 (torch 2.11 / CUDA 12.8). Raw
  (36k series) 22 s; smoothed (36k × 512) about 128 s per bandwidth.
- Extracted: raw and bw12 at 3 layers (bw12 also last-token pooling); bw 6/24/48
  late-mean; 5-channel raw and bw12; multivariate raw and bw12 (optional extension).
  Everything the planned ablations need is in place.

### Data handling
Only the series arrays (no IDs, labels or CSVs) went to the GPU box. The box can now
be destroyed; that also deletes the copies of the series and embeddings on it.

### Before Phase 5
P1–P4 code is written and tested (scripts/04_run_proposed.py, 22/22 tests) but not
run. Open points O1–O4 of the pre-registered rule (REPORT.md) still need confirmation.

### STOP 3 answer (user, 2026-10-04)
O1–O4 approved as written in REPORT.md, before any P model was run: the candidate is
chosen by mean val PR-AUC; precedence SUPPORTED → PARTIAL → NOT SUPPORTED →
INCONCLUSIVE; an F1-only win is INCONCLUSIVE; any CI entirely below 0 is reported as
significantly worse.

## Phase 5 done (2026-10-04): H3 verdict under the pre-registered rule

**Candidate (O1, by val PR-AUC):** P4_pca8 (val PR-AUC 0.539; P4_pca32 0.538; P3_pca8 0.537).
**Verdict: INCONCLUSIVE** (both populations; no harm). Pooled P4_pca8 − B3:
F1 +0.013 [−0.001, +0.027]; ROC-AUC +0.001 [−0.002, +0.004]; PR-AUC +0.002 [−0.003, +0.008].
Any true benefit of the hybrid is below about 0.027 F1 and 0.008 PR-AUC.

Test, primary folds (mean; results/proposed.csv, bootstrap_proposed.csv):
| model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B1 | 0.453 | 0.778 | 0.456 |
| B3 | 0.446 | 0.794 | 0.472 |
| B4 MiniRocket | 0.457 | 0.788 | 0.467 |
| P1 LR on raw-series emb. | 0.476 | 0.808 | 0.469 |
| P2 LR on smoothed emb. | 0.435 | 0.779 | 0.417 |
| P3_pca8 | 0.454 | 0.795 | 0.475 |
| P3_pca32 | 0.456 | 0.793 | 0.474 |
| P4_pca8 (candidate) | 0.459 | 0.795 | 0.474 |
| P4_pca32 | 0.445 | 0.793 | 0.470 |

**Descriptive, post hoc (no verdict; 6 models × 3 metrics, so read with multiplicity in mind):**
- **H1 is not borne out for raw-series embeddings.** P1 − B3: F1 +0.030 [+0.005, +0.053],
  ROC-AUC +0.014 [+0.004, +0.024], PR-AUC −0.003 [−0.027, +0.020]. C is interior (1e-4).
  P1 was not the val pick (val PR-AUC 0.515 vs 0.539).
- **H2 is reversed:** smoothing hurts. P2 − B3: PR-AUC −0.055 [−0.082, −0.023] and
  ROC-AUC −0.014 [−0.027, −0.001], both significantly worse. P2 is also worse than P1.
- Every hybrid beats the co-reference B1 on ROC-AUC (CIs above 0, about +0.015–0.017), but
  not on F1 or PR-AUC.
- Never-tracked users: no P model predicts any of them positive.
- Hybrid PCA(8) explains about 62% of the embedding variance; LightGBM stops at 52–94 trees.

## Exploratory round E (pre-committed 2026-10-04, BEFORE any of these test scores were computed)
The user asked for the strongest results before ablations. These models are designed
AFTER the Phase 5 test results were seen (P1 strong), so they are **exploratory**: they
do not change the pre-registered H3 verdict (INCONCLUSIVE).
Fixed candidate list (no additions after test results are seen):
- E1  rank-average of B3 + P1
- E2  rank-average of B3 + P1 + B4
- E3  LR (log1p handcrafted numerics + raw-series embeddings, scaled, balanced, C on val PR-AUC)
- E4  rank-average of P4_pca8 + P1
Rank-average = mean of within-split percentile ranks (uses no labels). Threshold tuned
on val F1, as for every model.
Selection: highest mean val PR-AUC over the primary folds. The winner gets ONE test
evaluation plus a paired bootstrap against B3 and B1. All four are reported.

### Round E results (exploratory)
VAL selection (mean PR-AUC, primary folds): E2 0.540, E4 0.537, E1 0.536, E3 0.521, so **E2** wins.
Test, primary folds (mean): E2 = rank-avg(B3, P1, B4): **F1 0.475, ROC-AUC 0.804, PR-AUC 0.487**.
E2 − B3 pooled: F1 +0.030 [+0.014, +0.044]; ROC-AUC +0.010 [+0.004, +0.017];
PR-AUC +0.016 [+0.002, +0.028]. Same without never-tracked users.
E2 − B1: F1 +0.022 [−0.001, +0.044]; ROC-AUC +0.026 [+0.014, +0.039]; PR-AUC +0.032 [+0.009, +0.052].
(E1 0.468/0.807/0.482; E3 0.476/0.809/0.475; E4 0.466/0.807/0.483.)
This is exploratory and does not change the pre-registered H3 verdict (INCONCLUSIVE).

### Leaderboard-gap check
Zindi test users (1,340, cohort 5) have **no** month-5 rows in any table (activity,
participation, discussion, comments), so no label leak exists in the released files.
For reference, on cohorts 3 and 4, next-month non-activity rows alone give F1 0.49 / 0.34
for label_any (high precision, recall ≤ 0.33). Leaderboard ranks 6–50 sit at
0.42–0.51 private F1; ranks 1–4 (0.84–0.87) cannot be explained from the released data.

## Phase 6 ablations done (2026-10-04): results/ablations.csv, bootstrap_ablations.csv
Variant minus its default, pooled over the primary folds, test, 95% CI.
**On P4_pca8 (pre-registered candidate):** no TimesFM setting matters. Bandwidth 6/24/48,
layer early/middle, last-token pooling, PCA 64, 5 channels, multivariate: every F1 CI
includes 0, and all |ΔROC| ≤ 0.003. (bw24 has PR-AUC +0.007 [+0.002, +0.012], small; one of
about 66 comparisons, so read with multiplicity in mind.)
Feature groups: dropping **compete** hurts clearly (F1 −0.046 [−0.069, −0.022], PR-AUC −0.032
[−0.047, −0.017]); dropping learn gives F1 +0.018 [+0.004, +0.032] with no AUC change
(threshold effect). All other groups and static attributes: no effect.
**On P1 (post hoc):** the late layer is best (early: F1 −0.021, ROC −0.009, both CIs below 0);
multivariate is worse (F1 −0.033 [−0.054, −0.013], ROC −0.009); 5 channels: no effect.
**Reading:** the hybrid's performance comes from the handcrafted competition-activity
features; no choice of TimesFM representation moves it.
**TimesFM's contribution to E2** (E2 vs rank-avg(B3, B4), results/bootstrap_e2_without_timesfm.csv):
F1 +0.018 [+0.001, +0.034], ROC-AUC +0.008 [+0.005, +0.011], PR-AUC +0.008 [−0.001, +0.017].
Statistically above 0 on two metrics, **practically small** (lower bounds near 0).

## Overnight round N (exploratory; protocol: results/overnight_protocol.md, log: overnight_log.md)
Development on train/val only. Final list fixed before one test evaluation.
**Headline N2 = CatBoost on v2 features, refit on train+val** (test, primary folds, mean):
**F1 0.482, ROC-AUC 0.812, PR-AUC 0.505.**
N2 − B3 pooled: F1 **+0.036** [+0.018, +0.055]; ROC-AUC **+0.018** [+0.011, +0.026]; PR-AUC **+0.033** [+0.020, +0.045].
N2 − E2: F1 +0.006 [−0.010, +0.024]; ROC +0.008 [+0.002, +0.013]; PR +0.018 [+0.003, +0.033].
Others: N1 (no refit) 0.481/0.812/0.499; N3 ensemble 0.488/0.812/0.498; N4 ensemble refit 0.496/0.813/0.501.
**TimesFM's share: none.** N4 − N2 (adding LightGBM + P1): F1 +0.014 [−0.004, +0.032], ROC +0.001, PR −0.004; N3 − N1 likewise null.
Where the gain came from (val evidence): v2 features (last_active_pos, sessions, competition
liveness): val PR 0.537 → 0.549; CatBoost: → 0.559; refit on one more recent cohort:
about +0.02 PR in the data-size check. Rejected on val: competition target encoding, anchored
(scale-aware) TimesFM embeddings (scale-aware, cos 1→0.67 across levels, but no val gain).
Infrastructure: TimesFM now runs on this Mac's CPU (torch 2.2.2 + RMSNorm shim;
matches GPU embeddings, cosine 1.0000).

## label_any sensitivity (D1 / STOP 1 decision 1c), test, primary folds
Training on the official "any activity" label (cohort 1 is 77% positive because of the badge batch):
B3 F1 0.386 / ROC 0.740 / PR 0.340 (vs 0.446 / 0.794 / 0.472 with label_engaged);
B4 0.409 / 0.755 / 0.399; P4_pca8 0.384 / 0.721 / 0.314.
P4 − B3 (label_any): F1 −0.002 [−0.015, +0.010]; ROC −0.019 [−0.027, −0.010]; PR −0.026 [−0.040, −0.014].
**Reading:** (1) the badge artefact in the official label damages every trained model
(F1 −0.06), which justifies D1; (2) under the official label the hybrid is
significantly WORSE than B3 on ROC-AUC and PR-AUC, so the H3 conclusion does not improve.
B0 (no training) is unchanged at F1 0.395, and it beats every trained model under label_any.

## Continued exploration (validation only; nothing new taken to test)
All rejected for no val gain over CatBoost-v2 (val PR 0.559):
- TimesFM zero-shot next-month forecasts as features: 0.555 (single-feature AUC 0.720 < total_events 0.743).
- Class weights (Balanced 0.553, SqrtBalanced 0.554).
- Training augmentation with later-month rows of earlier users (labels ≤ k−1): 0.558–0.559.
- Bag-of-pages (specific titles) TF-IDF LR: 0.32 alone; it hurts ensembles.
- CatBoost random search (14 configs): best 0.560, a plateau.
Error analysis (val): moderate-activity browsers (6–20 events; 48% of returners) and
users with no competition activity (24%) are where the remaining errors sit
(within-segment AUC 0.69 / 0.67). Their signup month carries little information about returning.
**Conclusion:** N2 (F1 0.482 / ROC 0.812 / PR 0.505 on test) is the best this data
supports under an honest protocol; REPORT.md is generated.
