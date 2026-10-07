"""Load the Zindi CSVs through the config column mapping and put every event on
one time axis.

Time axis:
  month_idx = position of the masked month code in cfg["month_order"]
  t_day     = day_of_month - 1 + time_of_day / 24h   (fractional days, [0, month length))
  t_abs     = month_start[month_idx] + t_day           (one axis across months, in days)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def read_table(cfg, name):
    return pd.read_csv(Path(cfg["data_dir"]) / f"{name}.csv", low_memory=False)


def seconds_of_day(s):
    """'HH:MM:SS[.ffffff]' -> seconds since midnight."""
    return pd.to_timedelta(s).dt.total_seconds()


def month_starts(cfg):
    """Day offset of each month on the t_abs axis (cumulative month lengths)."""
    return np.concatenate([[0], np.cumsum(cfg["month_lengths"])[:-1]]).astype(float)


def to_abs(month_idx, t_day, cfg):
    return month_starts(cfg)[np.asarray(month_idx)] + np.asarray(t_day)


def from_abs(t_abs, cfg):
    """t_abs -> (month_idx, t_day)."""
    m = np.searchsorted(month_starts(cfg), np.asarray(t_abs), side="right") - 1
    return m, np.asarray(t_abs) - month_starts(cfg)[m]


def month_index(codes, cfg):
    pos = {m: i for i, m in enumerate(cfg["month_order"])}
    idx = codes.map(pos)
    if idx.isna().any():
        raise ValueError(f"month codes not in month_order: {sorted(codes[idx.isna()].unique())}")
    return idx.astype(int)


def event_type(title):
    """Collapse per-object page views (comp_ID_x, blog_ID_x, job_ID_x) into one type each."""
    return title.str.replace(r"^(comp|blog|job)_ID_.*$", r"\1_page", regex=True)


def load_users(cfg):
    m = cfg["users"]
    d = read_table(cfg, m["file"])
    users = pd.DataFrame({"uid": d[m["user_id"]], "signup_month": d[m["signup_month"]]})
    users["cohort"] = month_index(users.signup_month, cfg)
    users["signup_t_day"] = d[m["signup_day"]] - 1 + seconds_of_day(d[m["signup_time"]]) / 86400
    for c in cfg["static_cols"]:
        users[c] = d[c].astype(str).where(d[c].notna(), "missing")
    return users


def load_events(cfg):
    """All event sources in long format:
    uid, source, title, etype, month_idx, day, t_day, t_abs."""
    parts = []
    for src, m in cfg["events"].items():
        d = read_table(cfg, m["file"])
        title = d[m["type"]] if m.get("type") else pd.Series(src, index=d.index)
        e = pd.DataFrame({"uid": d[m["user_id"]], "source": src, "title": title,
                          "month_idx": month_index(d[m["month"]], cfg),
                          "day": d[m["day"]]})
        e["t_day"] = e.day - 1 + seconds_of_day(d[m["time"]]) / 86400
        for name, col in (m.get("extra") or {}).items():
            e[name] = d[col].to_numpy()
        parts.append(_align_clock(e, cfg, src))
    ev = pd.concat(parts, ignore_index=True)
    ev["etype"] = event_type(ev.title)
    if "comp" in ev:                       # competition page views: comp_ID_x -> ID_x
        page = ev.title.str.startswith("comp_ID_")
        ev.loc[page, "comp"] = ev.title[page].str.replace("comp_", "", regex=False)
    ev["t_abs"] = to_abs(ev.month_idx, ev.t_day, cfg)
    return ev


def _align_clock(e, cfg, src):
    """Shift a source onto the Users clock; an event can move into the previous day
    or month, so day, month_idx and t_day are recomputed."""
    schedule = cfg.get("clock_offset_hours", {}).get(src)
    if not schedule:
        return e
    t_raw = pd.Series(to_abs(e.month_idx, e.t_day, cfg), index=e.index)
    hours = pd.Series(0.0, index=e.index)
    for step in schedule:              # later steps override earlier ones
        f = step["from"]
        start = (-np.inf if f is None else
                 to_abs(cfg["month_order"].index(f["month"]), f["day"] - 1 + f["hour"] / 24, cfg))
        hours[t_raw >= start] = step["hours"]
    t_abs = t_raw - hours / 24
    e = e.copy()
    e["month_idx"], e["t_day"] = from_abs(t_abs, cfg)
    e["day"] = e.t_day.astype(int) + 1
    return e
