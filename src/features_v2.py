"""Dataset-specific features added in the overnight round (all knowable at the end of
the signup month; they use only events with month_idx <= the user's cohort month).

Sessions        n_sessions, events_per_session, sessions_last7, max_session_events
                (a session breaks after a 30-minute gap)
Trend           events in the 2nd half of the observed window minus the 1st half;
                last-active-day position relative to the observed window
Competition     for the competitions a user touched in the signup month (comp page
liveness        views, participation rows): how many distinct users of ANY cohort were
                active on that competition in the last 7 days of the same month
                (max / sum over the user's competitions), and how many of the
                user's competitions are "live" (>= 20 such users)
Platform        number of live competitions in the user's signup month (cohort-level
context         context that the per-user features cannot express)
"""
import numpy as np
import pandas as pd

SESSION_GAP_DAYS = 30 / 1440
LIVE_MIN_USERS = 20


def competition_liveness(events, cfg):
    """(comp, month_idx) -> distinct users active on it in the last 7 days of that month."""
    ends = np.asarray(cfg["month_lengths"], float)
    e = events.dropna(subset=["comp"])
    e = e[e.t_day >= ends[e.month_idx.to_numpy()] - 7]
    return e.groupby(["comp", "month_idx"]).uid.nunique()


def build_features_v2(users, window, all_events, cfg):
    X = pd.DataFrame(index=users.uid)
    w = window.sort_values(["uid", "t_abs"])
    gap = w.groupby("uid").t_abs.diff()
    w = w.assign(new_session=(gap.isna() | (gap > SESSION_GAP_DAYS)).astype(int))
    w["session"] = w.groupby("uid").new_session.cumsum()
    ends = np.asarray(cfg["month_lengths"], float)
    end = pd.Series(ends[users.cohort.to_numpy()], index=users.uid)
    start = users.set_index("uid").signup_t_day
    w["end"] = w.uid.map(end)
    sess = w.groupby(["uid", "session"]).agg(n=("t_abs", "size"), t=("t_day", "min"),
                                             end=("end", "first"))
    X["n_sessions"] = sess.groupby(level=0).size()
    X["events_per_session"] = sess.n.groupby(level=0).mean()
    X["max_session_events"] = sess.n.groupby(level=0).max()
    X["sessions_last7"] = sess[sess.t >= sess.end - 7].groupby(level=0).size()
    mid = (start + end) / 2
    w["second_half"] = (w.t_day >= w.uid.map(mid)).astype(int)
    h = w.groupby("uid").second_half.agg(["sum", "size"])
    X["trend_2nd_minus_1st"] = h["sum"] - (h["size"] - h["sum"])
    last = w.groupby("uid").t_day.max()
    X["last_active_pos"] = ((last - start.reindex(last.index)) /
                            (end - start).clip(lower=1e-3).reindex(last.index)).clip(0, 1)

    live = competition_liveness(all_events, cfg)
    wc = w.dropna(subset=["comp"])
    lv = pd.Series(live.reindex(pd.MultiIndex.from_arrays([wc.comp, wc.month_idx])).to_numpy(),
                   index=wc.index).fillna(0)
    wc = wc.assign(live=lv)
    per = wc.groupby(["uid", "comp"]).live.max()
    X["comp_live_max"] = per.groupby(level=0).max()
    X["comp_live_sum"] = per.groupby(level=0).sum()
    X["n_live_comps"] = (per >= LIVE_MIN_USERS).groupby(level=0).sum()
    X["n_comps_touched"] = per.groupby(level=0).size()
    month_live = (live >= LIVE_MIN_USERS).groupby(level="month_idx").sum()
    X["platform_live_comps"] = users.set_index("uid").cohort.map(month_live).fillna(0)
    return X.fillna(0)


def comp_target_encoding(window, users_tr, label, users_apply, prior_weight=20):
    """Smoothed next-month return rate of each competition's TRAIN participants
    (fitted on users_tr only), aggregated per user over the competitions they touched:
    max, mean and count of encoded competitions. Returns a DataFrame for users_apply."""
    wc = window.dropna(subset=["comp"])[["uid", "comp"]].drop_duplicates()
    lab = users_tr.set_index("uid")[label]
    t = wc[wc.uid.isin(lab.index)].assign(y=lambda d: d.uid.map(lab))
    prior = lab.mean()
    stats = t.groupby("comp").y.agg(["sum", "count"])
    enc = (stats["sum"] + prior * prior_weight) / (stats["count"] + prior_weight)
    a = wc[wc.uid.isin(set(users_apply.uid))].assign(e=lambda d: d.comp.map(enc))
    g = a.dropna(subset=["e"]).groupby("uid").e
    out = pd.DataFrame({"comp_te_max": g.max(), "comp_te_mean": g.mean()}).reindex(users_apply.uid)
    return out.fillna(prior)


# ---- v3: literature-inspired features (KKBox-style windows, CFIN-style context) ----
KEY_TYPES = ["Created Submission", "Downloaded Competition Datafile", "comp_page",
             "Viewed All Learning Pages", "Signed In", "participation", "Updated Profile"]


def build_features_v3(users, window, cfg, base):
    """Multi-window counts, first-day vs later activity, per-key-action recency,
    inter-day gaps (KKBox / KDD Cup 2015 style), and within-context percentile ranks of
    core features (CFIN-style context smoothing; contexts = signup cohort, and
    days-observed bucket within cohort). `base` = existing feature frame (indexed by uid)."""
    X = pd.DataFrame(index=users.uid)
    u = users.set_index("uid")
    end = pd.Series(np.asarray(cfg["month_lengths"], float)[u.cohort.to_numpy()], index=u.index)
    w = window.assign(end=window.uid.map(end), sday=window.uid.map(u.signup_t_day))
    for k in (1, 3):
        X[f"events_last{k}d"] = w[w.t_day >= w.end - k].groupby("uid").size()
    first = w[w.t_day < w.sday + 1]
    X["events_first24h"] = first.groupby("uid").size()
    X["events_after_first24h"] = w[w.t_day >= w.sday + 1].groupby("uid").size()
    X["active_days_after_signup_day"] = w[w.day - 1 > np.floor(w.sday)].groupby("uid").day.nunique()
    for t in KEY_TYPES:
        last = w[w.etype == t].groupby("uid").t_day.max()
        X[f"days_since_{t.replace(' ', '_')}"] = (end - last.reindex(X.index)).fillna(99)
    days = w.groupby("uid").day.apply(lambda s: np.diff(np.sort(s.unique())))
    X["max_gap_days"] = days.apply(lambda d: d.max() if len(d) else 0)
    X["mean_gap_days"] = days.apply(lambda d: d.mean() if len(d) else 0)
    X = X.fillna(0)
    # context percentiles
    core = ["total_events", "compete_count", "browse_count", "total_active_days", "compete_last7"]
    B = base.loc[X.index, core]
    coh = u.cohort.reindex(X.index)
    obs_bucket = pd.cut((end - u.signup_t_day).reindex(X.index), [-1, 3, 7, 14, 21, 32], labels=False)
    for c in core:
        X[f"{c}_pct_cohort"] = B[c].groupby(coh).rank(pct=True)
        X[f"{c}_pct_cohort_obs"] = B[c].groupby([coh, obs_bucket]).rank(pct=True)
    return X


# ---- v4: competition momentum/age, entry point, signup crowd, time of day ----
ONBOARDING = {"Signed Up", "$identify", "$create_alias", "Confirmed Email"}


def build_features_v4(users, window, all_events, cfg):
    """All inputs knowable at the end of the user's signup month (events with
    t_abs <= the end of that month only)."""
    X = pd.DataFrame(index=users.uid)
    u = users.set_index("uid")
    starts = np.concatenate([[0], np.cumsum(cfg["month_lengths"])[:-1]]).astype(float)
    m_end_abs = starts + np.asarray(cfg["month_lengths"], float)        # per month_idx
    # competition momentum and age, per (comp, month): users active in the last 7 days vs
    # the 7 days before; first appearance of the comp in the logs up to that month end
    e = all_events.dropna(subset=["comp"])
    rows = []
    for m in range(len(cfg["month_lengths"])):
        end = m_end_abs[m]
        em = e[e.t_abs < end]
        last = em[em.t_abs >= end - 7].groupby("comp").uid.nunique()
        prev = em[(em.t_abs >= end - 14) & (em.t_abs < end - 7)].groupby("comp").uid.nunique()
        first = em.groupby("comp").t_abs.min()
        d = pd.DataFrame({"last7": last, "prev7": prev, "age": end - first}).fillna(0)
        d["month_idx"] = m
        rows.append(d)
    cm = pd.concat(rows).rename_axis("comp").reset_index()
    wc = window.dropna(subset=["comp"])[["uid", "comp", "month_idx"]].drop_duplicates()
    j = wc.merge(cm, on=["comp", "month_idx"], how="left").fillna(0)
    j["momentum"] = (j.last7 - j.prev7) / (j.prev7 + 5)
    g = j.groupby("uid")
    X["comp_momentum_max"] = g.momentum.max()
    X["comp_age_min"] = g.age.min()
    X["n_young_live_comps"] = g.apply(lambda d: ((d.age < 30) & (d.last7 >= 20)).sum())
    # entry point: first non-onboarding action after signup, and time to first competition join
    w = window.sort_values("t_abs")
    act = w[~w.etype.isin(ONBOARDING)]
    first = act.groupby("uid").first()
    for grp in ["compete", "browse", "learn", "account", "badges"]:
        X[f"entry_{grp}"] = (first.group == grp).astype(float)
    sday = u.signup_t_day
    tj = w[w.source == "participation"].groupby("uid").t_day.min()
    X["hours_to_first_join"] = ((tj - sday.reindex(tj.index)) * 24).clip(lower=0)
    X["joined_within_1h"] = (X.hours_to_first_join <= 1).astype(float)
    X["hours_to_first_join"] = X.hours_to_first_join.fillna(999)
    # signup crowd (same calendar day, all users) and time of day
    sd = (u.cohort.astype(str) + "_" + np.floor(u.signup_t_day).astype(int).astype(str))
    X["signup_day_crowd"] = sd.map(sd.value_counts())
    X["signup_hour"] = (u.signup_t_day % 1) * 24
    X["active_hour_mean"] = (w.t_day % 1 * 24).groupby(w.uid).mean()
    return X.reindex(users.uid).fillna(0)


def competition_sets(users, window, all_events, cfg, k=8):
    """Per-user set of competitions touched in the signup month (up to k, most-active first).
    Row features: first touch (hours after signup), joined flag, hours signup->join,
    user's events on it (log1p), and the competition's last-7-day users, previous-7-day
    users, momentum and age at the end of that month. All computed from events up to
    the month end. Returns (array (N, k, 8), mask (N, k))."""
    u = users.set_index("uid")
    starts = np.concatenate([[0], np.cumsum(cfg["month_lengths"])[:-1]]).astype(float)
    m_end = starts + np.asarray(cfg["month_lengths"], float)
    e = all_events.dropna(subset=["comp"])
    stats = []
    for m in range(len(cfg["month_lengths"])):
        em = e[e.t_abs < m_end[m]]
        last = em[em.t_abs >= m_end[m] - 7].groupby("comp").uid.nunique()
        prev = em[(em.t_abs >= m_end[m] - 14) & (em.t_abs < m_end[m] - 7)].groupby("comp").uid.nunique()
        d = pd.DataFrame({"last7": last, "prev7": prev, "age": m_end[m] - em.groupby("comp").t_abs.min()}).fillna(0)
        stats.append(d.assign(month_idx=m))
    cs = pd.concat(stats).rename_axis("comp").reset_index()
    wc = window.dropna(subset=["comp"]).copy()
    wc["sday"] = wc.uid.map(u.signup_t_day)
    g = wc.groupby(["uid", "comp", "month_idx"])
    per = pd.DataFrame({"first_h": (g.t_day.min() - g.sday.first()) * 24,
                        "n_ev": g.size()}).reset_index()
    joins = wc[wc.source == "participation"].groupby(["uid", "comp"]).agg(join_t=("t_day", "min"), sday=("sday", "first"))
    per = per.merge(joins, on=["uid", "comp"], how="left")
    per["joined"] = per.join_t.notna().astype(float)
    per["join_h"] = ((per.join_t - per.sday) * 24).fillna(-1)
    per = per.merge(cs, on=["comp", "month_idx"], how="left").fillna(0)
    per["momentum"] = (per.last7 - per.prev7) / (per.prev7 + 5)
    cols = ["first_h", "joined", "join_h", "n_ev", "last7", "prev7", "momentum", "age"]
    per["n_ev"] = np.log1p(per.n_ev)
    for c in ["first_h", "join_h", "last7", "prev7", "age"]:
        per[c] = np.sign(per[c]) * np.log1p(np.abs(per[c]))
    per = per.sort_values(["uid", "n_ev"], ascending=[True, False])
    per["rank"] = per.groupby("uid").cumcount()
    per = per[per["rank"] < k]
    pos = {x: i for i, x in enumerate(users.uid)}
    A = np.zeros((len(users), k, len(cols)), np.float32)
    M = np.zeros((len(users), k), bool)
    ii, rr = per.uid.map(pos).to_numpy(), per["rank"].to_numpy()
    A[ii, rr] = per[cols].to_numpy(np.float32)
    M[ii, rr] = True
    return A, M
