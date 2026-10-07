"""TimesFM-3 hidden-state embeddings (inference only).

API taken from the timesfm 3.0.2 source (not guessed):
  TimesFM3Forecaster.from_pretrained(checkpoint, device=...).model -> TimesFM3Torch
  TimesFM3Torch.decode(target=(b, v, T), horizon, mask=(b, T) or None)
    pads T to a multiple of 32 with masked positions, runs forward(), and the
    transformer blocks live in model.transformer_stack.layers[i]; each block
    returns a tuple whose [0] is (b, v, n_patches, 1280).
The first ceil(T / 32) patches are context; the rest are horizon patches, which
causal sequence attention keeps from influencing the context tokens.

Pooling per series: mean over context patches that are not fully masked, or
the last context patch.
"""
import math
import time

import numpy as np

PATCH = 32


def load_model(checkpoint, device):
    import torch
    import src.torch_compat  # noqa: F401  (nn.RMSNorm for torch < 2.4)
    from timesfm3 import TimesFM3Forecaster
    torch.backends.cuda.matmul.allow_tf32 = False      # keep fp32 numerics
    fc = TimesFM3Forecaster.from_pretrained(checkpoint, device=device)
    return fc.model


def _context_patches(T):
    pad = (PATCH - T % PATCH) % PATCH
    n_ctx = (T + pad) // PATCH
    first_valid = pad // PATCH          # leading patches that are fully padding
    return n_ctx, first_valid


def embed(model, X, layers, horizon=1, batch_size=512, poolings=("mean", "last")):
    """X: (N, V, T) float array; all V series of a row go through one forward pass
    (V=1: univariate). Returns {(layer, pooling): (N, V, D) float32} and the seconds taken."""
    import torch
    N, V, T = X.shape
    n_ctx, first = _context_patches(T)
    captured = {}
    hooks = [model.transformer_stack.layers[i].register_forward_hook(
        lambda m, inp, out, i=i: captured.__setitem__(i, out[0])) for i in layers]
    out = {(i, p): np.empty((N, V, 1280), np.float32) for i in layers for p in poolings}
    device = next(model.parameters()).device
    t0 = time.time()
    try:
        with torch.inference_mode():
            for s in range(0, N, batch_size):
                x = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size], np.float32)).to(device)
                model.decode(target=x, horizon=horizon)
                for i in layers:
                    h = captured[i][:, :, first:n_ctx, :]            # (b, V, ctx, D)
                    if "mean" in poolings:
                        out[(i, "mean")][s:s + len(x)] = h.mean(dim=2).float().cpu().numpy()
                    if "last" in poolings:
                        out[(i, "last")][s:s + len(x)] = h[:, :, -1, :].float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
    finally:
        for h in hooks:
            h.remove()
    return out, time.time() - t0


def synthetic_series(T=512, per_type=12, seed=0):
    """Smoke-test families: sines at 4 frequencies, constants, single spikes,
    sparse bursts, linear ramps. Returns (X (n, 1, T), type labels)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X, y = [], []
    for k, period in enumerate([16, 64, 128, 256]):
        for _ in range(per_type):
            amp, phase = rng.uniform(0.5, 5), rng.uniform(0, 2 * np.pi)
            X.append(amp * np.sin(2 * np.pi * t / period + phase) + rng.normal(0, 0.05 * amp, T))
            y.append(f"sine_p{period}")
    for _ in range(per_type):
        X.append(np.full(T, rng.uniform(0.1, 20)))
        y.append("constant")
    for _ in range(per_type):
        x = np.zeros(T)
        x[rng.integers(T // 4, T)] = rng.uniform(1, 20)
        X.append(x)
        y.append("spike")
    for _ in range(per_type):
        x = np.zeros(T)
        x[rng.choice(T, 8, replace=False)] = rng.uniform(1, 5, 8)
        X.append(x)
        y.append("sparse_bursts")
    for _ in range(per_type):
        X.append(rng.uniform(-3, 3) * t / T + rng.uniform(-5, 5))
        y.append("ramp")
    return np.asarray(X, np.float32)[:, None, :], np.asarray(y)


def knn_type_accuracy(E, y):
    """Leave-one-out 1-NN accuracy (cosine) of type labels."""
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    return float((y[S.argmax(1)] == y).mean())


def chance_rate(y):
    _, c = np.unique(y, return_counts=True)
    return float(((c - 1) / (len(y) - 1) * (c / len(y))).sum())


def n_batches(N, batch_size):
    return math.ceil(N / batch_size)
