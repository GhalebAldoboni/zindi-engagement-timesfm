"""Validation-only neural / foundation-model experiments. Run with the torch env:
    .venv-tfm/bin/python scripts/dev/dev_neural.py --exp {tabicl,tabicl_ft,mvrn,mvrn2,anchored}

tabicl     TabICLv2 zero-shot (in-context, no training)
tabicl_ft  TabICLv2 fine-tuned (column/row modules frozen; early stopping on val)
mvrn       Multi-View Retention Network (features MLP + daily-sequence GRU + TimesFM projection, gated)
mvrn2      MVRN + competition-set view (attention-pooled per-competition rows)
anchored   scale-aware TimesFM embeddings: a fixed reference patch is prepended to every
           raw series so per-series normalisation keeps the activity level
Requires the feature tables from scripts/dev/dev_experiments.py --build-features.
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.torch_compat  # noqa: E402,F401  (nn.RMSNorm / GradScaler for torch 2.2)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dev import dev_eval  # noqa: E402
from src.evaluate import make_folds, split  # noqa: E402
from src.features import build_features, static_cols  # noqa: E402
from src.io import load_config  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
CFG = load_config(str(ROOT / "configs/default.yaml"))
PROC, RES = ROOT / CFG["processed_dir"], ROOT / CFG["results_dir"]


def frame(extra=()):
    u = pd.read_parquet(PROC / "users.parquet")
    w = pd.read_parquet(PROC / "window_events.parquet")
    XV = build_features(u, w, CFG).loc[u.uid].join(
        pd.read_parquet(PROC / "features_v2.parquet").drop(columns=["platform_live_comps"]))
    for name in extra:
        XV = XV.join(pd.read_parquet(PROC / f"features_{name}.parquet").loc[u.uid])
    return u, w, XV


def exp_tabicl(finetune):
    from tabicl import FinetunedTabICLClassifier, TabICLClassifier
    u, _, XV = frame()
    for c in static_cols(CFG):
        XV[c] = XV[c].astype("category").cat.codes.astype(float)

    def fp(a, ya, b, yb, f):
        A, B = a.to_numpy(np.float32), b.to_numpy(np.float32)
        if not finetune:
            return TabICLClassifier(device="cpu", n_estimators=8, random_state=0).fit(A, ya).predict_proba(B)[:, 1]
        m = FinetunedTabICLClassifier(epochs=30, learning_rate=1e-5, device="cpu", freeze_col=True, freeze_row=True,
                                      time_limit=25 * 60, patience=4, verbose=True, random_state=0)
        m.fit(A, ya, X_val=B, y_val=yb, output_dir=RES / "cache" / f"tabicl_ft_{f['name']}")
        return m.predict_proba(B)[:, 1]
    m = dev_eval(u, XV, fp, CFG)[1]
    print(f"TabICLv2 {'fine-tuned' if finetune else 'zero-shot'}: val PR {m.val_pr_auc:.4f} ROC {m.val_roc_auc:.4f}")


def exp_mvrn(version):
    from src.mvrn import train_mvrn, train_mvrn2
    u, w, XV = frame(["v3", "v4"] if version == 2 else [])
    cats = static_cols(CFG)
    num = [c for c in XV.columns if c not in cats]
    row = {x: i for i, x in enumerate(u.uid)}
    groups = list(CFG["feature_groups"])
    gi = {g: i for i, g in enumerate(groups)}
    S = np.zeros((len(u), 31, len(groups) + 1), np.float32)
    np.add.at(S, (w.uid.map(row).to_numpy(), w.day.to_numpy() - 1, w.group.map(gi).to_numpy()), 1)
    S[:, :, :-1] = np.log1p(S[:, :, :-1])
    S[:, :, -1] = np.arange(31)[None, :] >= np.floor(u.signup_t_day.to_numpy())[:, None]
    E = np.load(RES / "cache/emb/raw__univariate__late__mean.npy").astype(np.float32)
    CA, CM = np.load(PROC / "comp_sets.npy"), np.load(PROC / "comp_mask.npy")
    slog = lambda Z: np.sign(Z) * np.log1p(np.abs(Z))  # noqa: E731
    res = []
    for f in [f for f in make_folds(CFG) if f["kind"] == "primary"]:
        tr, va, _ = split(u, f)
        itr = np.array([row[x] for x in tr.uid])
        iva = np.array([row[x] for x in va.uid])
        sc = StandardScaler().fit(slog(XV.loc[tr.uid, num].to_numpy(float)))
        es = StandardScaler().fit(E[itr])
        real = CA[itr][CM[itr]]
        cmu, csd = real.mean(0), real.std(0) + 1e-6

        def pack(idx, ids):
            cat = np.stack([pd.Categorical(XV.loc[ids, c], categories=sorted(XV.loc[tr.uid, c].unique())).codes + 1
                            for c in cats], 1)
            d = {"tab": torch.tensor(np.clip(sc.transform(slog(XV.loc[ids, num].to_numpy(float))), -6, 6), dtype=torch.float32),
                 "cats": torch.tensor(cat, dtype=torch.long), "seq": torch.tensor(S[idx]),
                 "emb": torch.tensor(np.clip(es.transform(E[idx]), -6, 6), dtype=torch.float32)}
            if version == 2:
                comps = np.where(CM[idx][..., None], (CA[idx] - cmu) / csd, 0).astype(np.float32)
                d["comps"], d["cmask"] = torch.tensor(np.clip(comps, -6, 6)), torch.tensor(CM[idx])
            return d
        dtr, dva = pack(itr, tr.uid), pack(iva, va.uid)
        ytr, yva = tr.label_engaged.to_numpy(), va.label_engaged.to_numpy()
        train = train_mvrn2 if version == 2 else train_mvrn
        outs = [train(dtr, dva, ytr, yva, s) for s in range(5)]
        p = np.mean([o[0] for o in outs], 0)
        gate = np.mean([o[1] for o in outs], 0)
        res.append((average_precision_score(yva, p), roc_auc_score(yva, p)))
        views = "(features, sequence, TimesFM" + (", competitions)" if version == 2 else ")")
        print(f"{f['name']}: MVRN-{version} val PR {res[-1][0]:.4f} ROC {res[-1][1]:.4f} gate {views} = {np.round(gate, 2)}")
    print(f"MVRN-{version} mean: val PR {np.mean([r[0] for r in res]):.4f} ROC {np.mean([r[1] for r in res]):.4f}")


def exp_anchored():
    from src.embed_timesfm import embed, load_model
    m = load_model(CFG["embedding"]["checkpoint"], "cpu")
    raw = np.load(RES / "cache/series/raw.npy")
    N, C, T = raw.shape
    ref = np.tile([0.0, 1.0], 16).astype(np.float32)          # fixed reference patch (mean .5, std .5)
    X = raw.reshape(N * C, 1, T)
    X = np.concatenate([np.repeat(ref[None, None, :], len(X), 0), X], axis=2)
    out, secs = embed(m, X, [9, 19], batch_size=256, poolings=("last",))
    for L, name in [(9, "middle"), (19, "late")]:
        np.save(RES / f"cache/emb/raw_anchored__univariate__{name}__last.npy",
                out[(L, "last")].reshape(N, C * 1280).astype(np.float16))
    print(f"anchored embeddings written ({secs / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["tabicl", "tabicl_ft", "mvrn", "mvrn2", "anchored"])
    e = ap.parse_args().exp
    {"tabicl": lambda: exp_tabicl(False), "tabicl_ft": lambda: exp_tabicl(True),
     "mvrn": lambda: exp_mvrn(1), "mvrn2": lambda: exp_mvrn(2), "anchored": exp_anchored}[e]()
