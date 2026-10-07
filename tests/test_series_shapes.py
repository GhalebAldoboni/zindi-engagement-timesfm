from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.io import load_config
from src.series import N_DAYS, raw_daily, smoothed

CFG = load_config()
L = 512
DX = N_DAYS / L

# u0: mid-month user (cohort 2, 30-day month); u1: no events; u2: late signup, near month end
USERS = pd.DataFrame({"uid": ["u0", "u1", "u2"], "cohort": [2, 3, 1],
                      "signup_t_day": [10.2, 5.0, 29.8]})
WINDOW = pd.DataFrame({
    "uid":   ["u0", "u0", "u0", "u0", "u2", "u2"],
    "group": ["browse", "browse", "contribute", "badges", "compete", "account"],
    "t_day": [10.25, 14.5, 20.0, 12.0, 29.81, 30.9],
})
WINDOW["day"] = WINDOW.t_day.astype(int) + 1


def test_shapes_and_channel_order():
    raw = raw_daily(USERS, WINDOW, CFG)
    sm = smoothed(USERS, WINDOW, CFG, 12, L)
    assert raw.shape == (3, 4, N_DAYS) and sm.shape == (3, 4, L)
    assert list(CFG["series_channels"]) == ["browse", "learn", "compete", "account"]


def test_raw_counts_merge_contribute_into_compete_and_skip_feature_only_groups():
    raw = raw_daily(USERS, WINDOW, CFG)
    assert raw[0, 0].sum() == 2          # browse
    assert raw[0, 2].sum() == 1          # contribute -> compete
    assert raw[0].sum() == 3             # badges is not a series channel
    assert raw[1].sum() == 0
    raw5 = raw_daily(USERS, WINDOW, CFG, "series_channels_5")
    assert raw5.shape[1] == 5 and raw5[0, 3].sum() == 1   # contribute separate


def test_smoothed_mass_equals_event_count_even_near_edges():
    sm = smoothed(USERS, WINDOW, CFG, 12, L)
    mass = sm.sum(-1) * DX
    raw = raw_daily(USERS, WINDOW, CFG)
    assert np.allclose(mass, raw.sum(-1), atol=0.03)


def test_smoothed_zero_outside_observation_window_and_for_no_event_users():
    sm = smoothed(USERS, WINDOW, CFG, 12, L)
    t = (np.arange(L) + 0.5) * DX
    assert (sm[0][:, t < 10.0] == 0).all()          # before signup hour
    assert (sm[0][:, t > 30.0] == 0).all()          # cohort 2 month has 30 days
    assert (sm[1] == 0).all()
    assert (sm >= 0).all()


@pytest.mark.skipif(not Path(CFG["results_dir"], "cache", "series", "raw.npy").exists(),
                    reason="run scripts/02b_build_series.py first")
def test_cached_series_match_users_and_window():
    d = Path(CFG["results_dir"], "cache", "series")
    users = pd.read_parquet(Path(CFG["processed_dir"]) / "users.parquet")
    w = pd.read_parquet(Path(CFG["processed_dir"]) / "window_events.parquet")
    assert (np.load(d / "uids.npy", allow_pickle=True) == users.uid.to_numpy()).all()
    raw = np.load(d / "raw.npy")
    assert raw.shape == (len(users), 4, N_DAYS)
    assert raw.sum() == (w.channel != "").sum()
