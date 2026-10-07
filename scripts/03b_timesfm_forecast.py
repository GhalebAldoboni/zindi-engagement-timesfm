"""Zero-shot TimesFM-3 FORECASTS of next-month activity as features (runs on CPU).

For each user and channel, the context is the daily count series of the signup month,
with days before the signup day and days after the month end MASKED (the model
ignores them instead of reading them as zeros). TimesFM forecasts the next 31 days
with quantiles; forecasts are de-normalised by the model, so they keep the absolute
activity scale. Features per channel: sum of the median forecast over the horizon,
sum of the 0.9 quantile, and the median forecast for the first 7 days.

Writes results/cache/timesfm_forecast_features.parquet (index = uid).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embed_timesfm import load_model  # noqa: E402
from src.io import load_config  # noqa: E402
from src.series import channel_names  # noqa: E402

H = 31


def main():
    torch.set_num_threads(8)
    cfg = load_config()
    res = Path(cfg["results_dir"])
    users = pd.read_parquet(Path(cfg["processed_dir"]) / "users.parquet")
    raw = np.load(res / "cache" / "series" / "raw.npy")            # (N, C, 31)
    N, C, T = raw.shape
    lengths = np.asarray(cfg["month_lengths"])[users.cohort.to_numpy()]
    first_day = np.floor(users.signup_t_day.to_numpy()).astype(int)  # 0-based signup day
    day = np.arange(T)
    mask = (day[None, :] < first_day[:, None]) | (day[None, :] >= lengths[:, None])   # True = masked
    model = load_model(cfg["embedding"]["checkpoint"], "cpu")
    q = list(model.quantiles)
    i50, i90 = q.index(0.5), q.index(0.9)

    feats = np.zeros((N, C, 3), np.float32)
    t0 = time.time()
    bs = 512
    with torch.inference_mode():
        for s in range(0, N, bs):
            x = torch.from_numpy(raw[s:s + bs].astype(np.float32))                    # (b, C, T)
            m = torch.from_numpy(mask[s:s + bs])                                       # (b, T)
            b = x.shape[0]
            xu = x.reshape(b * C, 1, T)                                                # univariate
            mu = m.repeat_interleave(C, dim=0)
            out = model.decode(target=xu, horizon=H, mask=mu)                          # (b*C, 1, H, Q)
            out = out[:, 0].clamp(min=0).reshape(b, C, H, len(q)).numpy()
            feats[s:s + b, :, 0] = out[..., i50].sum(-1)
            feats[s:s + b, :, 1] = out[..., i90].sum(-1)
            feats[s:s + b, :, 2] = out[:, :, :7, i50].sum(-1)
            if s == 0:
                print(f"first batch {time.time() - t0:.1f}s", flush=True)
    cols = [f"tfm_{ch}_{k}" for ch in channel_names(cfg) for k in ("next31_med", "next31_q90", "next7_med")]
    df = pd.DataFrame(feats.reshape(N, -1), columns=cols, index=users.uid.to_numpy())
    df["tfm_all_next31_med"] = feats[:, :, 0].sum(1)
    assert np.isfinite(df.to_numpy()).all()
    df.to_parquet(res / "cache" / "timesfm_forecast_features.parquet")
    print(f"done in {(time.time() - t0) / 60:.1f} min; shape {df.shape}")


if __name__ == "__main__":
    main()
