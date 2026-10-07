# Overnight development round N (started 2026-10-04 evening)

Goal (user): explore the data, tailor models to it (including TimesFM), and get the best
results this data can honestly support. "Improvement" means a practically meaningful gain,
not a CI that only just excludes 0.

Rules (fixed before any work):
1. **EDA uses train cohorts 0–2 only.** Labels of cohorts 3 and 4 are never inspected
   during development.
2. **Every modelling choice is made on validation:** the val cohorts of the primary
   folds (2 for test_c3, 3 for test_c4), or inner CV within train. No test metric is
   computed during development.
3. At the end, a **fixed final list** of at most 5 models is written to this file BEFORE
   the single test evaluation. The final pick is by mean val PR-AUC. Test is reported
   once, with a paired bootstrap against B3 and E2.
4. Every feature must be knowable at the end of the signup month: no events from month+1.
   Competition schedules (start/end dates) are public in advance, so they are allowed.
5. These results are exploratory. The pre-registered H3 verdict (INCONCLUSIVE) stands.

Progress log: results/overnight_log.md

## FINAL LIST (fixed at the end of development, BEFORE any test metric of these models)
Features "v2" = B3 handcrafted + sessions/trend + competition liveness (src/features_v2.py,
without platform_live_comps). Each LightGBM/CatBoost model = mean of seeds 0–4.
- **N1** CatBoost(depth 4, lr 0.03, l2 5, early stopping on val) on v2. Val PR 0.559.
- **N2** N1 refit on train+val cohorts, with iterations fixed at N1's mean best iteration.
- **N3** rank-avg(CatBoost-v2, LightGBM-v2 [B3 frozen params], P1). Val PR 0.553, ROC 0.785.
- **N4** N3 with every member refit on train+val (fixed iterations; P1 with C from val).
- reference: E2 (B3 + P1 + B4 rank-avg) and B3.
Thresholds: tuned on val for N1/N3. For refit models (no val left), the val-tuned threshold
of the matching non-refit model is reused (probability for N2; rank percentile for N4).
Selection: among non-refit N1/N3 by val PR, so **N1**. **Headline = N2** (its refit
version), as decided by the data-size check above, not by test.
Test: evaluated once for N1–N4, with paired bootstrap vs B3 and vs E2, all users.

## Threshold policy T (fixed before its test evaluation, 2026-10-05)
Chosen on development transfer only (tune on a past cohort, apply to the next past cohort;
test cohorts never used): "fraction × estimated drift" had mean dev F1 0.492, against 0.486
for the current probability threshold (oracle 0.503).
Rule, per fold, label-free on test:
1. frac = share of VAL users predicted positive at the val-F1-tuned threshold (N1 for N2;
   N3 for N4).
2. drift = mean N1 probability on TEST / mean N1 probability on VAL (N1 = non-refit
   CatBoost; no labels used).
3. On test, predict the top min(frac × drift, 1) share of users (ranked by the model's
   own test score) as positive.
Applied ONCE to N2 and N4 on test; reported whatever the outcome, with a bootstrap vs the
same models under the current threshold.
