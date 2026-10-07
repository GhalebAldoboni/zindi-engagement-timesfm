"""Training-set augmentation with later months of earlier users.

For a user u who signed up in cohort c, and a later month m > c (m <= max_feature_month),
one extra row is built: features from u's events in month m (the whole month observed,
so signup_t_day = 0), label = activity in month m + 1 under the same label definitions,
plus months_since_signup = m - c (0 for ordinary signup-month rows).
For a fold with test cohort k, only rows whose label month m + 1 <= k - 1 may be used for
training; the caller passes max_feature_month = k - 2.
"""
import pandas as pd

from src.batch_events import drop_batch_events
from src.features import build_features
from src.labels import feature_window


def augmented_rows(all_users, ev, batches, cfg, max_feature_month):
    rows = []
    for m in range(1, max_feature_month + 1):
        older = all_users[all_users.cohort < m]
        pseudo = older.assign(uid=older.uid + f"@m{m}", cohort=m, signup_t_day=0.0,
                              months_since_signup=m - older.cohort)
        rows.append(pseudo)
    if not rows:
        return None, None, None
    pseudo = pd.concat(rows, ignore_index=True)
    orig = pseudo.uid.str.split("@").str[0]

    # events re-keyed to pseudo users: each original event is copied to every pseudo
    # row of its user; feature_window then keeps only those in the pseudo row's month
    key = pd.DataFrame({"uid_orig": orig, "uid": pseudo.uid, "cohort": pseudo.cohort})
    e = ev.merge(key[["uid_orig", "uid", "cohort"]].rename(columns={"uid": "puid"}),
                 left_on="uid", right_on="uid_orig")
    e = e[(e.month_idx == e.cohort) | (e.month_idx == e.cohort + 1)]
    feat_ev = e.assign(uid=e.puid).drop(columns=["puid", "uid_orig", "cohort"])

    window = feature_window(feat_ev, pseudo, batches, cfg)
    nxt = feat_ev.merge(pseudo[["uid", "cohort"]], on="uid")
    nxt = nxt[(nxt.month_idx == nxt.cohort + 1) & nxt.source.isin(cfg["label_sources"])]
    eng = drop_batch_events(nxt, batches)
    eng = eng[~eng.title.str.startswith(tuple(cfg["label_exclude_prefixes"]))]
    pseudo["label_engaged"] = pseudo.uid.isin(set(eng.uid)).astype(int)
    pseudo["label_any"] = pseudo.uid.isin(set(nxt.uid)).astype(int)
    X = build_features(pseudo, window, cfg)
    return pseudo, window, X
