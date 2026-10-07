"""Cohorts, next-month labels, and the feature window.

cohort  = signup month index.
labels  = activity in month cohort + 1, under two definitions:
  label_engaged (primary): label-source events, excluding `label_exclude_prefixes`
                           (badge_*) and confirmed batch events.
  label_any (sensitivity): any label-source row, as in the official Zindi label.
Only cohorts with an observed next month that are not in `excluded_cohorts`
are labelled.
feature window = events in the signup month only, minus confirmed batch events and
`drop_etypes`, with each event tagged by its feature group and series channel.
"""
import pandas as pd

from src.batch_events import drop_batch_events


def labelled_cohorts(cfg):
    n = len(cfg["month_order"])
    return [c for c in range(n - 1) if c not in cfg["excluded_cohorts"]]


def build_labels(users, ev, batches, cfg):
    users = users[users.cohort.isin(labelled_cohorts(cfg))].copy()
    src = ev[ev.source.isin(cfg["label_sources"])].merge(users[["uid", "cohort"]], on="uid")
    nxt = src[src.month_idx == src.cohort + 1]
    engaged = drop_batch_events(nxt, batches)
    engaged = engaged[~engaged.title.str.startswith(tuple(cfg["label_exclude_prefixes"]))]
    users["label_engaged"] = users.uid.isin(set(engaged.uid)).astype(int)
    users["label_any"] = users.uid.isin(set(nxt.uid)).astype(int)
    return users.reset_index(drop=True)


def group_maps(cfg, channels_key="series_channels"):
    """etype -> feature group, and feature group -> series channel."""
    etype_group = {e: g for g, types in cfg["feature_groups"].items() for e in types}
    group_channel = {g: ch for ch, groups in cfg[channels_key].items() for g in groups}
    return etype_group, group_channel


def feature_window(ev, users, batches, cfg):
    """Events of each user in their signup month, without confirmed batch events or
    `drop_etypes`; adds `group` and `channel` ('' for feature-only groups)."""
    w = ev.merge(users[["uid", "cohort"]], on="uid")
    w = w[(w.month_idx == w.cohort) & ~w.etype.isin(cfg.get("drop_etypes", []))]
    w = drop_batch_events(w.drop(columns="cohort"), batches).reset_index(drop=True)
    etype_group, group_channel = group_maps(cfg)
    w["group"] = w.etype.map(etype_group)
    unmapped = sorted(w.etype[w.group.isna()].unique())
    if unmapped:
        raise ValueError(f"event types without a feature group: {unmapped}")
    w["channel"] = w.group.map(group_channel).fillna("")
    return w
