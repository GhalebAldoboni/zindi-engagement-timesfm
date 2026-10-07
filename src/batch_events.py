"""Systematic detection of system-wide batch events (e.g. a badge awarded to
thousands of users at once), which are not user engagement.

Step 1 (flag): for each event type, count distinct users per calendar day over
the whole period (days with no events count as 0). A (type, day) is flagged when
its robust z-score, (users - median) / (1.4826 * MAD) with MAD floored at 1,
exceeds `robust_z` and it has at least `min_users` users.

Step 2 (confirm): a flagged spike is a batch only if it is (a) concentrated in
time, with at least `min_window_share` of its rows inside one `max_window_hours`
window, and (b) passive, with at least `min_solo_share` of its users doing
nothing else that day. A genuine traffic surge (e.g. many signups on one day)
fails (b), because those users also do other things.
"""
import numpy as np
import pandas as pd

from src.io import from_abs, month_starts

COLUMNS = ["etype", "month_idx", "day", "users", "rows", "type_median_users",
           "robust_z", "window_share", "solo_share", "confirmed"]


def detect_batch_events(ev, cfg):
    p = cfg["batch_detection"]
    n_days = int(sum(cfg["month_lengths"]))
    gday = month_starts(cfg)[ev.month_idx.to_numpy()].astype(int) + ev.day - 1
    daily = ev.groupby([ev.etype, gday]).uid.nunique()
    rows = []
    for et, s in daily.groupby(level=0):
        x = s.droplevel(0).reindex(range(n_days), fill_value=0)
        med = x.median()
        mad = max(1.4826 * (x - med).abs().median(), 1.0)
        z = (x - med) / mad
        for d in x.index[(z > p["robust_z"]) & (x >= p["min_users"])]:
            on_day = gday == d
            spike = ev[on_day & (ev.etype == et)]
            hours = np.sort((spike.t_day - spike.day + 1).to_numpy() * 24)
            ends = np.searchsorted(hours, hours + p["max_window_hours"], side="right")
            window_share = (ends - np.arange(len(hours))).max() / len(hours)
            others = set(ev.loc[on_day & (ev.etype != et), "uid"])
            spike_users = spike.uid.drop_duplicates()
            solo_share = (~spike_users.isin(others)).mean()
            confirmed = (window_share >= p["min_window_share"]) and (solo_share >= p["min_solo_share"])
            m, t_day = from_abs(d, cfg)
            rows.append([et, int(m), int(t_day) + 1, int(x[d]), len(spike), med, round(z[d], 1),
                         round(window_share, 3), round(solo_share, 3), confirmed])
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("robust_z", ascending=False,
                                                           ignore_index=True)


def drop_batch_events(ev, batches):
    """Remove every event that belongs to a confirmed batch (same etype, month, day)."""
    conf = batches.loc[batches.confirmed, ["etype", "month_idx", "day"]]
    if conf.empty:
        return ev
    m = ev.merge(conf.assign(_batch=True), on=["etype", "month_idx", "day"], how="left")
    return ev[~m._batch.eq(True).to_numpy()]
