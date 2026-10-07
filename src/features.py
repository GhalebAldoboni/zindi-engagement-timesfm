"""Handcrafted per-user features from the feature window (signup month only).

Per feature group g (cfg["feature_groups"]):
  g_count        events
  g_active_days  distinct days with an event
  g_recency      days from the last event to the end of the signup month
                 (no event: days from signup to month end, i.e. censored at signup)
  g_span         days between first and last event (0 if fewer than 2)
  g_last7        events in the last 7 days of the month
  g_rate         events per observed day
Global: signup_day, days_observed, total_events, total_active_days,
n_event_types, n_groups, no_events, plus static user attributes (categorical).

Nothing here is fitted; categorical levels are set per fold in models.py.
"""
import numpy as np
import pandas as pd

PER_GROUP = ["count", "active_days", "recency", "span", "last7", "rate"]


def month_end(users, cfg):
    return np.asarray(cfg["month_lengths"], dtype=float)[users.cohort.to_numpy()]


def build_features(users, window, cfg):
    u = users.set_index("uid")
    end = pd.Series(month_end(users, cfg), index=users.uid)
    days_obs = (end - u.signup_t_day).clip(lower=0)
    w = window.assign(end=window.uid.map(end))

    X = pd.DataFrame(index=u.index)
    X["signup_day"] = u.signup_t_day
    X["days_observed"] = days_obs
    for g in cfg["feature_groups"]:
        e = w[w.group == g]
        by = e.groupby("uid")
        first, last = by.t_day.min(), by.t_day.max()
        X[f"{g}_count"] = by.size()
        X[f"{g}_active_days"] = by.day.nunique()
        X[f"{g}_recency"] = (end - last).reindex(X.index)
        X[f"{g}_span"] = (last - first).reindex(X.index)
        X[f"{g}_last7"] = e[e.t_day >= e.end - 7].groupby("uid").size()
        X[f"{g}_count"] = X[f"{g}_count"].fillna(0)
        X[f"{g}_rate"] = X[f"{g}_count"] / days_obs.clip(lower=1)
        X[f"{g}_recency"] = X[f"{g}_recency"].fillna(days_obs)
        X[[f"{g}_active_days", f"{g}_span", f"{g}_last7"]] = \
            X[[f"{g}_active_days", f"{g}_span", f"{g}_last7"]].fillna(0)

    by = w.groupby("uid")
    X["total_events"] = by.size().reindex(X.index).fillna(0)
    X["total_active_days"] = by.day.nunique().reindex(X.index).fillna(0)
    X["n_event_types"] = by.etype.nunique().reindex(X.index).fillna(0)
    X["n_groups"] = by.group.nunique().reindex(X.index).fillna(0)
    X["no_events"] = (X.total_events == 0).astype(int)
    for c in cfg["static_cols"]:
        X[c] = u[c]
    return X


def static_cols(cfg):
    return list(cfg["static_cols"])


def numeric_cols(X, cfg):
    return [c for c in X.columns if c not in cfg["static_cols"]]


def last7_heuristic(window, users, cfg):
    """B0: 1 if the user has any feature-window event in the last 7 days of the month."""
    end = pd.Series(month_end(users, cfg), index=users.uid)
    w = window[window.t_day >= window.uid.map(end) - 7]
    return users.uid.isin(set(w.uid)).astype(int).to_numpy()
