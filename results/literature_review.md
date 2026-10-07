# Related work: new-user churn / engagement prediction, and what we tested here

| Source | Setting | Core approach | Applied here? | Result on our validation (CatBoost-v2 = PR-AUC 0.559) |
|---|---|---|---|---|
| Dror et al., *Churn prediction in new users of Yahoo! Answers* (WWW 2012 workshop) | new users who quit after one or two answers | activity counts, timing gaps and community feedback in the first days | partly: activity, gaps, sessions (v2/v3). Feedback received: too few comments (467) to use | gaps/sessions in v2 (part of the +0.012 from v2); v3 gaps: no further gain |
| Yang et al., *I Know You'll Be Back* (KDD 2018, Snapchat) | churn in week 2 after registration | daily multi-type activity sequences + ego-network; user-type clustering; LSTM + attention | sequence view via MiniRocket (B4) and TimesFM (P1/P2); no friend graph in the data | B4 ≈ B3; TimesFM adds nothing over v2 |
| Feng et al., *Understanding Dropouts in MOOCs* / CFIN (AAAI 2019) | MOOC dropout (KDD Cup 2015 data) | context smoothing of activity by course statistics; attention over user and course context; reports about +1 AUC point over GBDT (90.1 vs 89.1) | yes: competition context (liveness, v2) and within-cohort/obs-window percentiles (v3) | liveness helps (13% → 65% return by number of live competitions); percentiles: 0.556 (no gain) |
| KDD Cup 2015 winners (three-stage ensemble + feature engineering) | MOOC dropout | many engineered features + stacked ensembles | yes: OOF stacking of CatBoost, LightGBM, LR, P1 | 0.548 vs 0.547 CatBoost alone (stacker puts ~0.9 weight on CatBoost) |
| Gregory, *Predicting Customer Churn: XGBoost with Temporal Data* (WSDM Cup 2018, 1st) | KKBox subscription churn | ~250 temporal features: days since last key action; aggregates over several windows | yes (v3): last-1/3-day windows, days since 7 key actions, first-24 h vs later | 0.557 (no gain); the strongest single v3 feature is days since last competition-page view (AUC 0.746) |

**Takeaway.** The published recipe (engineered temporal and context features + gradient
boosting, optionally stacked) is what our best model N2 already is. Our gains came from
the dataset-specific parts (competition liveness, last-active position, sessions,
CatBoost, a more recent training cohort). The published extras (deep sequence models,
context smoothing, stacking) are reported elsewhere as about +1 AUC point and were neutral
here. The remaining errors sit in moderate-activity browsers, the same "one-or-two-actions
then quit" group that Dror et al. identify as the hard case.

Sources:
- Dror, Pelleg, Rokhlenko, Szpektor (2012), https://www.researchgate.net/publication/235479659_Churn_prediction_in_new_users_of_Yahoo_Answers
- Yang, Shi, Luo, Han (KDD 2018), https://arxiv.org/abs/1910.01447
- Feng, Tang, Liu (AAAI 2019), https://keg.cs.tsinghua.edu.cn/persons/jietang/publications/AAAI19-Feng-dropout-moocs.pdf
- KDD Cup 2015, https://www.biendata.xyz/competition/kddcup2015/
- Gregory (2018), https://arxiv.org/pdf/1802.03396

## 2025–2026 methods (tested 2026-10-05)

| Source | Approach | Our result (val PR-AUC; CatBoost-v2 = 0.559) |
|---|---|---|
| Hollmann et al., TabPFN v2 (*Nature*, 2025); TabPFN-2.5 (Nov 2025); TabPFN-3 (May 2026) | tabular foundation models with in-context learning; report beating tuned GBDTs on small tables | represented by TabICLv2 (fully open; TabPFN weights carry their own licence terms) |
| Qu et al., **TabICLv2** (arXiv 2602.11139, Feb 2026), checkpoint `tabicl-classifier-v2-20260212` | open tabular foundation model, column-then-row attention | zero-shot **0.551** (level with tuned LightGBM). **Fine-tuned** (prediction head only, CPU, early stop on val): **0.556**, ROC 0.785 vs CatBoost 0.559 / 0.782, a tie. Averaged with CatBoost: 0.560 / 0.785, negligible gain |
| Osin et al., **EBES** benchmark (KDD 2025) | event-sequence classification: GRU best on short sequences; aggregates + MLP within about 5% | GRU on daily 8-group sequences: **0.532** (optimistic early stopping) |
| Time-series foundation models (TimesFM-3, Aug 2026) | embeddings / zero-shot forecasts | four ways tested (embeddings, smoothing, scale-aware anchoring, forecasts): no gain over features |

**2026 takeaway.** The newest tabular foundation model matches our tuned gradient boosting
with zero tuning, which is impressive, but it does not exceed engineered features +
CatBoost on this data. That agrees with recent evaluations reporting TabPFN-style gains
mainly on small, clean tables without strong domain features. The sequence-model
evidence (EBES) agrees that aggregates suffice for short sequences.

Sources: https://www.nature.com/articles/s41586-024-08328-6 · https://arxiv.org/abs/2605.13986 ·
https://arxiv.org/pdf/2602.11139 · https://github.com/soda-inria/tabicl · https://arxiv.org/abs/2410.03399
