# Why the Zindi leaderboard is misleading for this task

*Zindi New User Engagement Prediction Challenge — final leaderboard read on 2026-10-04
(logged-in view) together with the competition's public discussion threads. Participant
names are deliberately omitted.*

## TL;DR

The top of the final leaderboard (private scores 0.84–0.87) does **not** reflect F1 on
the prediction task. A confirmed scoring bug and a re-scoring that appears to have used
**accuracy** instead of **F1** produced those numbers. The honest F1 range for strong
solutions is roughly **0.42–0.51**, which is where our models sit.

## 1. What the leaderboard shows

| Rank | Public score | Private score | Notes |
|---|---|---|---|
| 1 | 0.503 | **0.87** | private shown with 2 decimals only |
| 2 | 0.708 | **0.86** | 2 decimals |
| 3 | 0.497 | **0.85** | 2 decimals |
| 4 | 0.843 | 0.843 | public = private |
| 5 | 0.665 | 0.665 | |
| 6–10 | 0.509 → 0.487 | same | |
| 20 | 0.463 | same | |
| 30 | 0.440 | same | |
| 50 | 0.424 | same | |

The metric configured for the competition (Zindi API field `error_metric`) is **F1**.
220 teams appear on the leaderboard; submissions closed on 13 Feb 2023.

Two things stand out at once:
- Ranks 1–3 jump from public scores around 0.50 to private scores of 0.85–0.87, reported
  with a different precision from every other row.
- From rank 4 down, public and private scores are identical. The discussion suggests
  there was effectively a single test set rather than a separate private split.

## 2. Problem 1: duplicate rows inflated F1 (a confirmed scoring bug)

A participant showed in the discussion forum that the scorer did not de-duplicate
submission IDs. Appending many copies of one user predicted as positive pushed a
submission to **0.99**. The problem was raised before the competition closed, and the
suggested fix was to keep one row per ID before computing F1. Later posts disagree on
whether Zindi ever fully fixed it.

**Consequence:** any submission whose pipeline produced duplicate IDs, even by accident,
could have an inflated score.

## 3. Problem 2: the top scores look like accuracy, not F1

After the post-competition review, a participant pointed out that the corrected top
scores appear to have been computed as **accuracy** rather than F1. The arithmetic
supports this:

- The test set had **197 positives out of 1,340** users (14.7%), as reported in the
  discussion.
- Predicting **"nobody returns"** gives accuracy 1 − 0.147 = **0.853** with F1 = 0.
- That is exactly the range of the top-3 private scores (0.85–0.87).

We checked this with our own models on our test cohorts:

| Test cohort | Positive rate | Accuracy, predict all-negative | Accuracy, our best ensemble (threshold tuned for accuracy) | Our F1 |
|---|---|---|---|---|
| 3 | 16.9% | 0.831 | 0.859 | ≈ 0.49 |
| 4 | 11.5% | 0.885 | 0.895 | ≈ 0.46 |

Measured as accuracy, a mediocre or even trivial model reaches the 0.85–0.90 band.
On a dataset this imbalanced, accuracy rewards predicting the majority class and says
almost nothing about finding returning users. That is exactly why F1 was the chosen metric.

## 4. What a legitimate F1 looks like

- A 12th-place team published a four-model ensemble at **0.483** private F1.
- A top-20 participant wrote that they could not see how anyone exceeded about 0.50
  without duplicates.
- Ranks 6–50 cluster between **0.42 and 0.51**.

So the realistic ceiling for honest solutions is about **0.48–0.51 F1**.

## 5. Is there a data leak that could explain high scores?

We checked whether the released files leak the label for Zindi's test users (cohort 5,
signups in month code 4, label = activity in month code 5):

- Test users have **no month-5 rows in any table** (UserActivity, CompetitionPartipation,
  Discussion, Comments), so there is no direct label leak.
- Even the strongest "future" signal we could construct on labelled cohorts (next-month
  rows in the participation, discussion and comment tables) reaches only F1 0.34–0.49.

Nothing in the released data supports an F1 anywhere near 0.84–0.87.

## 6. How our results compare (context only, not a ranking)

| Model (our protocol) | Test F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| B3 tuned LightGBM | 0.446 | 0.794 | 0.472 |
| E2 ensemble incl. TimesFM | 0.475 | 0.804 | 0.487 |
| N2 CatBoost + dataset-specific features (refit) | **0.482** | **0.812** | **0.505** |

These numbers are **not directly comparable** to the leaderboard:
- different test cohorts (ours: 3 and 4 with observed labels; Zindi's: the hidden cohort 5);
- our primary label excludes a one-day badge batch that the official label counts;
- no tuning on test anywhere in our pipeline.

Within those caveats, our F1 sits in the same band as the best legitimate competition
entries.

## 7. Lessons

1. **Read the leaderboard against the metric.** With 14.7% positives, any score near
   0.85 should be checked against the all-negative accuracy baseline before being
   treated as F1.
2. **Inspect the scorer, not just the model.** Duplicate-ID handling broke the ranking.
3. **Trust cross-validation over the leaderboard.** Participants who followed time-based
   CV were not misled. Our evaluation uses rolling cohort folds, a frozen model list and a
   single test evaluation for the same reason.
4. **Report effect sizes with confidence intervals.** A leaderboard position without
   uncertainty hides how small the differences between rows 6 and 50 really are
   (0.42–0.51).

## Sources

- Zindi competition API (`api.zindi.world/v1/competitions/zindi-new-user-engagement-prediction-challenge`):
  metric F1, submissions closed 2023-02-12.
- Final leaderboard page (logged-in view), read 2026-10-04.
- Competition discussion threads, read 2026-10-04: *Confirmed: Error in score calculations*,
  *Winning solutions*, *Competition ended 10 days ago – any update?*, *Approach to top 20*.
- Our own accuracy/F1 check: `results/cache/preds/*`, `results/exploratory.csv`,
  `results/overnight_final.csv`.
