"""Evidence for the UserActivity clock correction.

Writes
  results/clock_offset_per_day.csv  per signup day: matched "Signed Up" events and the
                                    share at +1 h / +2 h (raw activity minus Users time)
  results/clock_correction.csv      per correction variant: share of "Signed Up" events
                                    within 1 min of signup, and labels changed vs the
                                    configured (final) correction
"""
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.batch_events import detect_batch_events  # noqa: E402
from src.io import load_config, load_events, load_users, to_abs  # noqa: E402
from src.labels import build_labels  # noqa: E402

FLAT = [{"from": None, "hours": 2}, {"from": {"month": 12, "day": 1, "hour": 0.0}, "hours": 1}]


def signup_offset_minutes(ev, users, cfg):
    t_users = pd.Series(to_abs(users.cohort, users.signup_t_day, cfg), index=users.uid)
    t_signed = ev[ev.title == "Signed Up"].groupby("uid").t_abs.min()
    return (t_signed - t_users.reindex(t_signed.index)) * 1440


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    res = Path(cfg["results_dir"])
    users = load_users(cfg)

    variants = {"none": {}, "flat_by_month": {"activity": FLAT},
                "configured": cfg["clock_offset_hours"]}
    rows, labels = [], {}
    for name, off in variants.items():
        c = copy.deepcopy(cfg)
        c["clock_offset_hours"] = off
        ev = load_events(c)
        d = signup_offset_minutes(ev, users, cfg)
        if name == "none":
            u = users.set_index("uid").loc[d.index]
            per_day = pd.DataFrame({"month": u.signup_month, "day": u.signup_t_day.astype(int) + 1,
                                    "offset_h": (d / 60).round()})
            per_day = per_day.groupby(["month", "day"]).offset_h.agg(
                n="size", share_1h=lambda s: (s == 1).mean(), share_2h=lambda s: (s == 2).mean())
            per_day.round(3).to_csv(res / "clock_offset_per_day.csv")
        labels[name] = build_labels(users, ev, detect_batch_events(ev, c), c).set_index("uid")
        rows.append({"variant": name, "signed_up_within_1min": round((d.abs() < 1).mean(), 4)})

    final = labels["configured"]
    for r in rows:
        for col in ["label_engaged", "label_any"]:
            o = labels[r["variant"]][col]
            r[f"{col}_changed"] = int((o != final[col]).sum())
            r[f"{col}_1to0"] = int(((o == 1) & (final[col] == 0)).sum())
    out = pd.DataFrame(rows)
    out.to_csv(res / "clock_correction.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
