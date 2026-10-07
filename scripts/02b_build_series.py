"""Phase 3: build raw and smoothed series for every labelled user, plus sanity plots.

Writes
  results/cache/series/uids.npy                     row order (= users.parquet order)
  results/cache/series/raw.npy                      (N, C, 31) daily counts
  results/cache/series/smooth_bw{H}.npy             (N, C, resample_len), events/day
  results/cache/series/meta.json                    channels, shapes, bandwidths
  results/figures/series_examples_bw{H}.png         10 random users with events
  results/series_stats.csv                          per channel: zero share, mass checks

  --bandwidths 6 12 24 48   build several bandwidths (default: cfg smoothing.bandwidth_hours)
  --channels-key            series_channels (default) or series_channels_5 (ablation)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io import load_config  # noqa: E402
from src.series import N_DAYS, channel_names, raw_daily, smoothed  # noqa: E402


def plot_examples(users, raw, smooth, chans, bw, path, seed):
    rng = np.random.default_rng(seed)
    has = np.where(raw.sum(axis=(1, 2)) > 0)[0]
    pick = rng.choice(has, 10, replace=False)
    fig, axes = plt.subplots(5, 2, figsize=(14, 14), sharex=True)
    days = np.arange(N_DAYS) + 0.5
    grid = (np.arange(smooth.shape[-1]) + 0.5) * N_DAYS / smooth.shape[-1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, i in zip(axes.flat, pick):
        for c, name in enumerate(chans):
            ax.bar(days + (c - 1.5) * 0.2, raw[i, c], width=0.2, color=colors[c], alpha=0.45)
            ax.plot(grid, smooth[i, c], color=colors[c], lw=1.4, label=name)
        u = users.iloc[i]
        ax.axvline(u.signup_t_day, color="k", ls=":", lw=1)
        ax.set_title(f"user {i} | cohort {u.cohort} | signup day {u.signup_t_day:.1f} | "
                     f"label {u.label_engaged}", fontsize=9)
        ax.set_ylabel("events / day")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("day of signup month")
    fig.suptitle(f"Raw daily counts (bars) vs smoothed intensity (lines), bandwidth {bw} h; "
                 "dotted line = signup")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--bandwidths", type=float, nargs="*")
    ap.add_argument("--channels-key", default="series_channels")
    args = ap.parse_args()
    cfg = load_config(args.config)
    key = args.channels_key
    bws = args.bandwidths or [cfg["smoothing"]["bandwidth_hours"]]
    L = cfg["smoothing"]["resample_len"]
    res = Path(cfg["results_dir"])
    out = res / "cache" / ("series" if key == "series_channels" else f"series_{key}")
    out.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(exist_ok=True)

    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    window = pd.read_parquet(Path(cfg["processed_dir"]) / "window_events.parquet")
    chans = channel_names(cfg, key)
    np.save(out / "uids.npy", users.uid.to_numpy().astype(str))
    raw = raw_daily(users, window, cfg, key)
    np.save(out / "raw.npy", raw)

    stats = []
    for bw in bws:
        sm = smoothed(users, window, cfg, bw, L, key)
        tag = f"{bw:g}"
        np.save(out / f"smooth_bw{tag}.npy", sm)
        plot_examples(users, raw, sm, chans, tag, res / "figures" /
                      f"series_examples{'' if key == 'series_channels' else '_' + key}_bw{tag}.png",
                      cfg["seed"])
        mass = sm.sum(-1) * N_DAYS / L               # integral of events/day over days
        for c, name in enumerate(chans):
            n = raw[:, c].sum(-1)
            act = n > 0
            stats.append({"bandwidth_h": bw, "channel": name,
                          "users_with_events": int(act.sum()), "zero_series_share": 1 - act.mean(),
                          "raw_events": int(n.sum()),
                          "smoothed_mass_over_raw_median": float(np.median(mass[act, c] / n[act])),
                          "smoothed_mass_over_raw_p05": float(np.percentile(mass[act, c] / n[act], 5)),
                          "smoothed_mass_over_raw_p95": float(np.percentile(mass[act, c] / n[act], 95))})
        print(f"bw {tag}: smoothed {sm.shape}")
    pd.DataFrame(stats).round(4).to_csv(res / f"series_stats{'' if key == 'series_channels' else '_' + key}.csv",
                                        index=False)
    json.dump({"channels": chans, "raw_shape": raw.shape, "resample_len": L,
               "bandwidths": bws, "n_users": len(users)}, open(out / "meta.json", "w"), indent=1)
    print(f"raw {raw.shape}")
    print(pd.DataFrame(stats).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
