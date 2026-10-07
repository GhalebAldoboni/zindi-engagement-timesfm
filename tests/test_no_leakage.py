"""Leakage guards: features only from the signup month, labels only from the
next month, batch/badge events handled as specified. Synthetic tests always run;
the data tests run once scripts/01_build_dataset.py has been run."""
from pathlib import Path

import pandas as pd
import pytest

from src.batch_events import COLUMNS, detect_batch_events
from src.io import _align_clock, load_config
from src.labels import build_labels, feature_window, labelled_cohorts

CFG = load_config()
PROC = Path(CFG["processed_dir"])


def _ev(rows):
    ev = pd.DataFrame(rows, columns=["uid", "source", "title", "month_idx", "day"])
    ev["etype"] = ev.title
    ev["t_day"] = ev.day - 1 + 0.5
    return ev


USERS = pd.DataFrame({"uid": ["a", "b", "c"], "cohort": [2, 2, 3]})
EV = _ev([
    ("a", "activity", "Viewed All Competitions", 2, 3),   # a: window event
    ("a", "activity", "Viewed All Competitions", 3, 4),   # a: next month -> label 1
    ("b", "activity", "Viewed All Competitions", 1, 9),   # b: before signup month
    ("b", "activity", "badge_OCZE", 3, 2),                   # b: only a badge next month
    ("c", "activity", "Signed In", 3, 5),                 # c: window event
    ("c", "activity", "Signed In", 5, 1),                 # c: two months later
])
NO_BATCHES = pd.DataFrame(columns=COLUMNS).astype({"confirmed": bool})


def test_feature_window_is_signup_month_only():
    w = feature_window(EV, USERS, NO_BATCHES, CFG).merge(USERS, on="uid")
    assert (w.month_idx == w.cohort).all()
    assert len(w) == 2


def test_labels_use_next_month_and_engaged_excludes_badges():
    lab = build_labels(USERS, EV, NO_BATCHES, CFG).set_index("uid")
    assert lab.loc["a", "label_engaged"] == 1
    assert lab.loc["b", "label_any"] == 1 and lab.loc["b", "label_engaged"] == 0
    assert lab.loc["c", "label_engaged"] == 0 and lab.loc["c", "label_any"] == 0


def test_confirmed_batch_is_removed_from_window_and_label():
    batches = pd.DataFrame([["Viewed All Competitions", 3, 4, 0, 0, 0, 0, 1, 1, True]],
                           columns=COLUMNS)
    lab = build_labels(USERS, EV, batches, CFG).set_index("uid")
    assert lab.loc["a", "label_engaged"] == 0 and lab.loc["a", "label_any"] == 1
    w = feature_window(EV, USERS.assign(cohort=[3, 3, 3]), batches, CFG)
    assert not ((w.etype == "Viewed All Competitions") & (w.day == 4)).any()


def test_detector_confirms_concentrated_passive_spike_only():
    rows = [(f"u{i}", "activity", "badge_OCZE", 2, 10) for i in range(200)]       # batch
    rows += [(f"v{i}", "activity", "Signed Up", 1, 5) for i in range(200)]     # surge...
    rows += [(f"v{i}", "activity", "Updated Profile", 1, 5) for i in range(200)]  # ...with other activity
    rows += [("w", "activity", t, m, d) for t in ["badge_OCZE", "Signed Up"] for m in range(7) for d in (1, 15)]
    ev = _ev(rows)
    ev.loc[ev.uid.str.startswith("v"), "t_day"] = ev.day - 1 + (ev.index % 24) / 24
    b = detect_batch_events(ev, CFG).set_index("etype")
    assert b.loc["badge_OCZE", "confirmed"]
    assert not b.loc["Signed Up", "confirmed"]


def test_clock_shift_steps_and_crosses_month_boundary():
    # before the switch (+2 h): month_idx 1 day 1 00:30 -> month_idx 0 (30 days) day 30 22:30
    # after the switch (+1 h):  month_idx 2 day 1 00:30 -> month_idx 1 (31 days) day 31 23:30
    e = pd.DataFrame({"month_idx": [1, 2, 0], "day": [1, 1, 10], "t_day": [0.5 / 24, 0.5 / 24, 9.5]})
    out = _align_clock(e, CFG, "activity")
    assert out.month_idx.tolist() == [0, 1, 0] and out.day.tolist() == [30, 31, 10]
    assert abs(out.t_day.iloc[0] - (30 - 1.5 / 24)) < 1e-9
    assert abs(out.t_day.iloc[1] - (31 - 0.5 / 24)) < 1e-9
    assert abs(out.t_day.iloc[2] - (9.5 - 2 / 24)) < 1e-9


needs_data = pytest.mark.skipif(not (PROC / "users.parquet").exists(),
                                reason="run scripts/01_build_dataset.py first")


@needs_data
def test_processed_window_never_after_signup_month():
    users = pd.read_parquet(PROC / "users.parquet")
    w = pd.read_parquet(PROC / "window_events.parquet").merge(users[["uid", "cohort"]], on="uid")
    assert len(w) == len(pd.read_parquet(PROC / "window_events.parquet"))
    assert (w.month_idx == w.cohort).all()


@needs_data
def test_processed_users_only_labelled_cohorts():
    users = pd.read_parquet(PROC / "users.parquet")
    assert set(users.cohort) == set(labelled_cohorts(CFG))
    assert not set(users.cohort) & set(CFG["excluded_cohorts"])


@needs_data
def test_processed_window_has_no_confirmed_batch_events():
    b = pd.read_csv(Path(CFG["results_dir"]) / "batch_events.csv")
    w = pd.read_parquet(PROC / "window_events.parquet")
    for r in b[b.confirmed].itertuples():
        assert not ((w.etype == r.etype) & (w.month_idx == r.month_idx) & (w.day == r.day)).any()


def test_categorical_levels_come_from_train_only():
    from src.models import to_categorical
    tr = pd.DataFrame({"c": ["a", "b"]})
    va = pd.DataFrame({"c": ["a", "z"]})
    tr2, va2 = to_categorical(tr, [va], ["c"])
    assert list(va2.c.cat.categories) == ["a", "b"] and va2.c.isna().tolist() == [False, True]


@needs_data
def test_features_complete_and_no_events_consistent():
    from src.features import build_features
    users = pd.read_parquet(PROC / "users.parquet")
    w = pd.read_parquet(PROC / "window_events.parquet")
    X = build_features(users, w, CFG)
    assert not X.isna().any().any()
    assert set(X.index[X.no_events == 1]) == set(users.uid) - set(w.uid)
    assert (X.loc[X.no_events == 1, "total_events"] == 0).all()


def test_pca_block_is_fitted_on_train_only():
    import numpy as np
    from src.models import pca_block
    rng = np.random.default_rng(0)
    Z_tr = rng.normal(size=(200, 20))
    va_a, va_b = rng.normal(size=(50, 20)), 1000 * rng.normal(size=(50, 20))
    (tr_a, _), _ = pca_block(Z_tr, [va_a], 4, 0, "e")
    (tr_b, _), _ = pca_block(Z_tr, [va_b], 4, 0, "e")
    assert np.allclose(tr_a.to_numpy(), tr_b.to_numpy())


def test_competition_liveness_uses_only_same_month_last_week():
    from src.features_v2 import competition_liveness
    ev = pd.DataFrame({"uid": ["a", "b", "c", "d"], "comp": ["X", "X", "X", "X"],
                       "month_idx": [2, 2, 3, 2], "t_day": [27.0, 10.0, 28.0, 29.5]})
    live = competition_liveness(ev, CFG)          # month_idx 2 has 30 days -> last 7 = t_day >= 23
    assert live.loc[("X", 2)] == 2                # a and d; b too early; c is next month
    assert live.loc[("X", 3)] == 1


def test_dev_harness_never_passes_test_rows():
    from src.dev import dev_eval
    users = pd.DataFrame({"uid": [f"u{i}" for i in range(50)], "cohort": [i % 5 for i in range(50)],
                          "label_engaged": [i % 2 for i in range(50)]})
    X = pd.DataFrame({"f": range(50)}, index=users.uid)
    seen = []

    def fp(a, ya, b, yb, fold):
        seen.append(set(users.set_index("uid").loc[list(a.index) + list(b.index)].cohort))
        return b.f.to_numpy() / 50.0
    dev_eval(users, X, fp, CFG)
    assert seen[0] <= {0, 1, 2} and seen[1] <= {0, 1, 2, 3}     # test cohorts 3 / 4 never seen
