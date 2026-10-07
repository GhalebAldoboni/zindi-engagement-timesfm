"""Phase 4: TimesFM-3 embeddings of the series (runs on the GPU box).

Needs only numpy, pyyaml, torch and timesfm[torch], plus the series arrays in
results/cache/series*/ (no raw data, no user IDs).

  --smoke   synthetic smoke test only: results/embeddings_smoke.json
  (default) smoke test, then every job in JOBS -> results/cache/emb/*.npy
            (float16, shape (N, C * 1280), rows in users.parquet order)
            and results/embeddings_runtime.csv

Univariate mode embeds each channel as its own series and concatenates the
channels; multivariate mode (optional extension) passes all channels of a user
jointly, so TimesFM-3's variate attention can mix them.
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embed_timesfm import (chance_rate, embed, knn_type_accuracy,  # noqa: E402
                               load_model, synthetic_series)

# (series dir, file, mode, layers, poolings)
ALL_LAYERS = ("early", "middle", "late")
JOBS = [
    ("series", "raw", "univariate", ALL_LAYERS, ("mean",)),          # 1 token: mean == last
    ("series", "smooth_bw12", "univariate", ALL_LAYERS, ("mean", "last")),
    ("series", "smooth_bw6", "univariate", ("late",), ("mean",)),
    ("series", "smooth_bw24", "univariate", ("late",), ("mean",)),
    ("series", "smooth_bw48", "univariate", ("late",), ("mean",)),
    ("series_series_channels_5", "raw", "univariate", ("late",), ("mean",)),
    ("series_series_channels_5", "smooth_bw12", "univariate", ("late",), ("mean",)),
    ("series", "raw", "multivariate", ("late",), ("mean",)),
    ("series", "smooth_bw12", "multivariate", ("late",), ("mean",)),
]


def smoke_test(model, cfg, layer_idx):
    X, y = synthetic_series()
    late = layer_idx["late"]
    out, secs = embed(model, X, [late], horizon=cfg["embedding"]["horizon"], poolings=("mean",))
    E = out[(late, "mean")][:, 0]
    res = {"n_series": len(y), "types": sorted(set(y)), "layer": late, "seconds": secs,
           "knn1_acc_embedding": knn_type_accuracy(E, y),
           "knn1_acc_raw_values": knn_type_accuracy(X[:, 0], y),
           "chance": chance_rate(y), "all_finite": bool(np.isfinite(E).all())}
    # scale invariance: constants at different levels, and the all-zero series
    T = X.shape[-1]
    lv = np.array([0.0, 0.5, 1, 5, 50, 500], np.float32)
    Xc = np.repeat(lv[:, None, None], T, axis=2)
    Ec = embed(model, Xc, [late], poolings=("mean",))[0][(late, "mean")][:, 0]
    En = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)
    res["constant_levels"] = lv.tolist()
    res["constant_cosine_vs_level_1"] = (En @ En[2]).round(4).tolist()
    # same shape at two scales (x and 100x): RevIN should make them near-identical
    E1 = embed(model, X[:12], [late], poolings=("mean",))[0][(late, "mean")][:, 0]
    E2 = embed(model, 100 * X[:12], [late], poolings=("mean",))[0][(late, "mean")][:, 0]
    cos = (E1 * E2).sum(1) / np.linalg.norm(E1, axis=1) / np.linalg.norm(E2, axis=1)
    res["cosine_x_vs_100x_mean"] = float(cos.mean())
    # different types: mean cosine between type centroids, to show the scale of differences
    cents = np.stack([E[y == t].mean(0) for t in res["types"]])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    res["min_centroid_cosine"] = float((cents @ cents.T)[np.triu_indices(len(cents), 1)].min())
    res["max_centroid_cosine"] = float((cents @ cents.T)[np.triu_indices(len(cents), 1)].max())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ec = cfg["embedding"]
    res = Path(cfg["results_dir"])
    emb_dir = res / "cache" / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)

    import torch
    model = load_model(ec["checkpoint"], args.device)
    gpu = torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor()
    print(f"loaded {ec['checkpoint']} on {gpu}; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"{len(model.transformer_stack.layers)} layers")
    print("\n".join(n for n, _ in model.named_modules() if n.count(".") <= 2))

    smoke = smoke_test(model, cfg, ec["layers"])
    smoke.update({"gpu": gpu, "torch": torch.__version__})
    json.dump(smoke, open(res / "embeddings_smoke.json", "w"), indent=1)
    print(json.dumps(smoke, indent=1))
    if args.smoke:
        return

    rows = []
    for sdir, name, mode, layer_names, pools in JOBS:
        X = np.load(res / "cache" / sdir / f"{name}.npy")
        N, C, T = X.shape
        idx = [ec["layers"][n] for n in layer_names]
        Xin = X.reshape(N * C, 1, T) if mode == "univariate" else X
        out, secs = embed(model, Xin, idx, horizon=ec["horizon"],
                          batch_size=ec["batch_size"] if mode == "univariate" else ec["batch_size"] // C,
                          poolings=pools)
        tag = "" if sdir == "series" else "5ch_"
        for ln, i in zip(layer_names, idx):
            for p in pools:
                E = out[(i, p)].reshape(N, C * 1280)
                assert np.isfinite(E).all(), (name, mode, ln, p)
                f = emb_dir / f"{tag}{name}__{mode}__{ln}__{p}.npy"
                np.save(f, E.astype(ec["storage_dtype"]))
                rows.append({"file": f.name, "series": f"{sdir}/{name}", "mode": mode,
                             "layer": ln, "layer_idx": i, "pooling": p, "N": N, "C": C, "T": T,
                             "dim": C * 1280, "seconds_for_job": round(secs, 1), "gpu": gpu})
        print(f"{tag}{name} {mode}: {N}x{C}x{T} in {secs:.0f}s")
    with open(res / "embeddings_runtime.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
