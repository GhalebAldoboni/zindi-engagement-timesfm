"""Per-user series from the feature window, one row per channel in
cfg["series_channels"] (approved: browse, learn, compete, account).

raw       daily event counts on the calendar grid, shape (N, C, 31); day 31 of a
          30-day month is always 0.
smoothed  hourly histogram (31 * 24 bins) -> Gaussian kernel (sigma = bandwidth hours)
          with edge correction -> resampled to `resample_len` points over the same
          31 days. Units: events per day.

Edge correction: each user is observed from obs_start = min(signup time, first
event) to the end of their signup month. Each event's kernel is renormalised by its
own mass inside that window (weight 1 / (mask * K)(t_i)), and intensity outside the
window is set to 0. The integral of the curve therefore equals the event count, and
events near either edge are not shrunk. (Dividing by the kernel mass at each
evaluation point instead loses about 30% of the mass here, because most events
cluster right after signup.)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

N_DAYS = 31
N_HOURS = N_DAYS * 24


def channel_names(cfg, key="series_channels"):
    return list(cfg[key])


def _channel_index(window, cfg, key):
    """Map each window event to its channel index (-1 = not in any channel)."""
    group_to_idx = {g: i for i, ch in enumerate(channel_names(cfg, key)) for g in cfg[key][ch]}
    return window.group.map(group_to_idx).fillna(-1).astype(int).to_numpy()


def _user_index(window, users):
    pos = {u: i for i, u in enumerate(users.uid)}
    return window.uid.map(pos).to_numpy()


def raw_daily(users, window, cfg, key="series_channels"):
    C = len(cfg[key])
    out = np.zeros((len(users), C, N_DAYS), np.float32)
    ch = _channel_index(window, cfg, key)
    keep = ch >= 0
    np.add.at(out, (_user_index(window, users)[keep], ch[keep],
                    window.day.to_numpy()[keep] - 1), 1)
    return out


def observation_mask(users, window, cfg):
    """(N, N_HOURS) 1 inside [obs_start, month end), else 0."""
    first = window.groupby("uid").t_day.min().reindex(users.uid).to_numpy()
    start = np.fmin(users.signup_t_day.to_numpy(), first)        # fmin ignores NaN
    end = np.asarray(cfg["month_lengths"], float)[users.cohort.to_numpy()]
    hours = np.arange(N_HOURS)
    return ((hours >= np.floor(start * 24)[:, None]) &
            (hours < (end * 24)[:, None])).astype(np.float32)


def smoothed(users, window, cfg, bandwidth_hours, resample_len, key="series_channels"):
    C = len(cfg[key])
    hist = np.zeros((len(users), C, N_HOURS), np.float32)
    ch = _channel_index(window, cfg, key)
    keep = ch >= 0
    ui = _user_index(window, users)[keep]
    hour = np.minimum((window.t_day.to_numpy() * 24).astype(int), N_HOURS - 1)[keep]

    mask = observation_mask(users, window, cfg)
    in_window = gaussian_filter1d(mask, bandwidth_hours, axis=-1, mode="constant")
    np.add.at(hist, (ui, ch[keep], hour), 1.0 / in_window[ui, hour])
    dens = gaussian_filter1d(hist, bandwidth_hours, axis=-1, mode="constant")
    dens = dens * mask[:, None, :] * 24                                      # events/day

    # resample hour-bin centres -> resample_len evenly spaced points over 31 days
    src = np.arange(N_HOURS) + 0.5
    dst = (np.arange(resample_len) + 0.5) * N_HOURS / resample_len
    j = np.clip(np.searchsorted(src, dst) - 1, 0, N_HOURS - 2)
    w = ((dst - src[j]) / (src[j + 1] - src[j])).clip(0, 1)
    return (dens[..., j] * (1 - w) + dens[..., j + 1] * w).astype(np.float32)
