"""Phase 0 data audit.

Part A (schema-agnostic, always runs): shape, columns, dtypes, null rates,
sample rows, duplicates, and the distinct values of any column whose name
looks time-related, for every CSV in data_dir.

Part B (needs the column mapping in configs/default.yaml): user-ID overlap,
month-order verification over all permutations of the observed month codes,
cohort sizes and next-month label rate.

Output: results/audit.md (plus stdout).
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

TIME_PAT = re.compile(r"month|day|date|hour|time|year|week|minute", re.I)


def load_all(data_dir):
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        sys.exit(f"No CSVs in {data_dir}. Download the Zindi files there first.")
    return {f.stem: pd.read_csv(f, low_memory=False) for f in files}


def part_a(tables, out):
    for name, df in tables.items():
        out.append(f"## {name}.csv\n")
        out.append(f"shape: {df.shape}, exact duplicate rows: {df.duplicated().sum()}\n")
        info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                             "null_rate": df.isna().mean().round(4),
                             "n_unique": df.nunique()})
        out.append(info.to_markdown() + "\n")
        out.append("sample rows:\n\n" + df.head(3).to_markdown(index=False) + "\n")
        for col in df.columns:
            if TIME_PAT.search(str(col)):
                vc = df[col].value_counts(dropna=False).sort_index()
                shown = vc if len(vc) <= 40 else vc.head(40)
                out.append(f"time-like column `{col}`: {len(vc)} distinct values"
                           + ("" if len(vc) <= 40 else " (first 40 shown)") + "\n\n"
                           + shown.to_frame("count").to_markdown() + "\n")


def part_b(tables, cfg, out):
    u, ev = cfg["users"], cfg["events"] or {}
    if not (u.get("file") and u.get("user_id") and u.get("signup_month") and ev):
        out.append("\n# Part B skipped\nFill `users` and `events` in configs/default.yaml "
                   "from Part A, then rerun.\n")
        return
    users = tables[u["file"]][[u["user_id"], u["signup_month"]]].drop_duplicates()
    users.columns = ["uid", "signup"]
    out.append(f"\n# Part B\nusers: {len(users)}, users with >1 signup month: "
               f"{users.uid.duplicated().sum()}\n")

    events = []
    for src, m in ev.items():
        e = tables[m["file"]][[m["user_id"], m["month"]]].copy()
        e.columns = ["uid", "month"]
        missing = ~e.uid.isin(users.uid)
        out.append(f"- {src}: {len(e)} rows, {e.uid.nunique()} users, "
                   f"{missing.mean():.2%} of rows have a user_id not in the users table; "
                   f"day column: {m.get('day')}, time column: {m.get('time')}\n")
        e["src"] = src
        events.append(e[~missing])
    events = pd.concat(events).merge(users, on="uid")

    codes = sorted(set(events.month.dropna()) | set(users.signup.dropna()))
    out.append(f"\nobserved month codes: {codes}\n")

    def before_signup_share(order):
        pos = {c: i for i, c in enumerate(order)}
        return (events.month.map(pos) < events.signup.map(pos)).mean()

    if len(codes) <= 8:
        scores = sorted((before_signup_share(p), p) for p in itertools.permutations(codes))
        out.append("\nshare of events dated before the user's signup month, "
                   f"best 10 of {len(scores)} orders:\n\n")
        out.append(pd.DataFrame([(list(p), round(s, 5)) for s, p in scores[:10]],
                                columns=["order", "share_before_signup"]).to_markdown(index=False) + "\n")
    cand = cfg["month_order"]
    if set(cand) == set(codes):
        out.append(f"\nconfigured order {cand}: share = {before_signup_share(cand):.5f}\n")
    else:
        out.append(f"\nconfigured order {cand} does not match observed codes {codes}\n")
        return

    pos = {c: i for i, c in enumerate(cand)}
    users["cohort"] = users.signup.map(pos)
    lab_events = events[events.src.isin(cfg["label_sources"])]
    active = set(zip(lab_events.uid, lab_events.month.map(pos)))
    users["label"] = [(uid, c + 1) in active for uid, c in zip(users.uid, users.cohort)]
    labelled = users[users.cohort < len(cand) - 1]
    tab = labelled.groupby("cohort").agg(n=("uid", "size"), label_rate=("label", "mean"))
    out.append("\ncohort sizes and next-month label rate (label source: "
               f"{cfg['label_sources']}):\n\n" + tab.round(4).to_markdown() + "\n")
    out.append(f"\noverall label rate: {labelled.label.mean():.4f}\n")
    part_c(tables, cfg, users, pos, out)


def part_c(tables, cfg, users, pos, out):
    """Checks added after the first Part B run: test-set identity, batch system
    events that distort the label, and pre-signup timing at day resolution."""
    out.append("\n# Part C\n")
    sub = next((t for n, t in tables.items() if n.lower().startswith("samplesub")), None)
    if sub is not None:
        ids = sub.iloc[:, 0].str.replace(r"_Month_\d+$", "", regex=True)
        test = users[users.uid.isin(ids)]
        suffixes = sub.iloc[:, 0].str.extract(r"(Month_\d+)")[0].value_counts().to_dict()
        out.append(f"SampleSubmission: {len(sub)} ids, suffixes {suffixes}; "
                   f"their signup months {test.signup.value_counts().to_dict()}\n")
    m = cfg["events"]["activity"]
    a = tables[m["file"]][[m["user_id"], m["month"], m["day"], m["type"]]].copy()
    a.columns = ["uid", "month", "day", "type"]
    a["m"] = a.month.map(pos)
    a = a.merge(users[["uid", "cohort"]], on="uid")
    burst = (a.groupby(["type", "month", "day"]).size().sort_values(ascending=False).head(5))
    out.append("\nlargest single-day bursts of one activity type:\n\n"
               + burst.to_frame("rows").to_markdown() + "\n")
    def rates(keep):
        act = set(a[keep & (a.m == a.cohort + 1)].uid)
        lab = users[users.cohort < len(pos) - 1]
        return lab.groupby("cohort").uid.apply(lambda s: round(s.isin(act).mean(), 4))
    tab = pd.DataFrame({"any_row": rates(a.type.notna()),
                        "excl_badge_*": rates(~a.type.str.startswith("badge_"))})
    out.append("\nlabel rate per cohort under two label definitions:\n\n" + tab.to_markdown() + "\n")
    sd = cfg["users"].get("signup_day")
    if sd:
        u = tables[cfg["users"]["file"]][[cfg["users"]["user_id"], sd]]
        u.columns = ["uid", "sday"]
        same = a[a.m == a.cohort].merge(u, on="uid")
        out.append(f"\nsignup-month activity rows dated before the signup day: "
                   f"{(same.day < same.sday).sum()} / {len(same)}\n")
    has = users.uid.isin(a[a.m == a.cohort].uid)
    out.append("\nshare of users with any signup-month activity, per cohort: "
               f"{has.groupby(users.cohort).mean().round(3).to_dict()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    tables = load_all(cfg["data_dir"])
    out = ["# Data audit\n"]
    part_a(tables, out)
    part_b(tables, cfg, out)
    text = "\n".join(out)
    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    Path(cfg["results_dir"], "audit.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
