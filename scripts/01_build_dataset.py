"""Phase 1: build users/labels, the feature-window events, and fold assignments.

Writes
  data/processed/users.parquet          labelled cohorts only, both label definitions, and
                                        never_tracked (no row in any table in any month;
                                        uses outcome-period data, so sensitivity use only)
  data/processed/window_events.parquet  signup-month events, batch events removed
  results/batch_events.csv              every flagged spike with its evidence
  results/dataset_summary.csv           per cohort: size, label rates, fold roles
  results/event_types.csv               event types in the feature window + group/channel
  results/group_coverage.csv            per feature group and per series channel:
                                        users reached, volume, events per active user
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.batch_events import detect_batch_events  # noqa: E402
from src.evaluate import make_folds  # noqa: E402
from src.io import load_config, load_events, load_users  # noqa: E402
from src.labels import build_labels, feature_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    proc, res = Path(cfg["processed_dir"]), Path(cfg["results_dir"])
    proc.mkdir(parents=True, exist_ok=True)

    all_users, ev = load_users(cfg), load_events(cfg)
    batches = detect_batch_events(ev, cfg)
    users = build_labels(all_users, ev, batches, cfg)
    users["never_tracked"] = ~users.uid.isin(set(ev.uid))
    window = feature_window(ev, users, batches, cfg)
    users.to_parquet(proc / "users.parquet", index=False)
    window.to_parquet(proc / "window_events.parquet", index=False)
    batches.to_csv(res / "batch_events.csv", index=False)

    summary = all_users.groupby("cohort").agg(
        signup_month=("signup_month", "first"), n=("uid", "size"))
    summary = summary.join(users.groupby("cohort")[["label_engaged", "label_any"]].mean().round(4))
    summary["with_window_events"] = (users.uid.isin(window.uid)
                                     .groupby(users.cohort).mean().round(3))
    summary["role"] = ["excluded" if c in cfg["excluded_cohorts"] else "" for c in summary.index]
    for f in make_folds(cfg):
        for part in ["train", "val", "test"]:
            for c in f[part]:
                summary.loc[c, "role"] += f"{f['name']}({f['kind'][:4]}):{part} "
    summary.to_csv(res / "dataset_summary.csv")

    types = window.groupby(["group", "channel", "source", "etype"]).agg(
        rows=("uid", "size"), users=("uid", "nunique"))
    types["user_share"] = (types.users / len(users)).round(4)
    types.reset_index().sort_values(["group", "rows"], ascending=[True, False]).to_csv(
        res / "event_types.csv", index=False)

    cov = []
    for kind in ["group", "channel"]:
        w = window[window[kind] != ""]
        per_user = w.groupby([kind, "uid"]).size()
        c = per_user.groupby(level=0).agg(users="size", median_events_if_any="median",
                                          p90_events_if_any=lambda s: s.quantile(0.9))
        c["user_share"] = (c.users / len(users)).round(3)
        c["rows"] = w[kind].value_counts()
        cov.append(c.rename_axis("name").reset_index().assign(kind=kind))
    cov = pd.concat(cov)[["kind", "name", "users", "user_share", "rows",
                          "median_events_if_any", "p90_events_if_any"]]
    cov.to_csv(res / "group_coverage.csv", index=False)

    print(batches.to_string(), "\n")
    print(summary.to_string(), "\n")
    print(cov.to_string())
    print(f"\nlabelled users: {len(users)}, window events: {len(window)}, "
          f"users with no window event: {(~users.uid.isin(window.uid)).sum()}")


if __name__ == "__main__":
    main()
