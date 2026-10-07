"""Multi-View Retention Network (MVRN): a new architecture for new-user engagement.

Three views of the signup month, fused by learned per-user gating:
  tab  : engineered features (signed-log + standardised; scaler fitted on train) -> MLP
  seq  : daily counts of the 8 feature groups + observed-day mask, 31 days -> GRU
  emb  : TimesFM-3 raw-series embedding (4 x 1280) -> dropout + linear projection
  cat  : learned embeddings for the static categorical attributes (added to tab view)
Fusion: a gate network outputs softmax weights over the three view vectors (per user);
the head sees the weighted sum concatenated with the raw views.
Training: BCE with pos_weight, AdamW, early stopping on the validation fold
(the same protocol as CatBoost's early stopping), mean of several seeds.
"""
import numpy as np
import torch
import torch.nn as nn


class MVRN(nn.Module):
    def __init__(self, d_tab, d_seq, d_emb, cat_cards, h=64, p_emb_drop=0.5):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(c + 1, min(8, c + 1)) for c in cat_cards])
        d_cat = sum(min(8, c + 1) for c in cat_cards)
        self.tab = nn.Sequential(nn.Linear(d_tab + d_cat, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, h), nn.ReLU())
        self.seq = nn.GRU(d_seq, h, batch_first=True)
        self.emb = nn.Sequential(nn.Dropout(p_emb_drop), nn.Linear(d_emb, h), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(3 * h, 32), nn.ReLU(), nn.Linear(32, 3))
        self.head = nn.Sequential(nn.Linear(4 * h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, tab, cats, seq, emb, return_gate=False):
        c = torch.cat([e(cats[:, i]) for i, e in enumerate(self.cat_emb)], dim=1)
        v_tab = self.tab(torch.cat([tab, c], dim=1))
        _, hs = self.seq(seq)
        v_seq = hs[-1]
        v_emb = self.emb(emb)
        views = torch.stack([v_tab, v_seq, v_emb], dim=1)                 # (b, 3, h)
        g = torch.softmax(self.gate(views.flatten(1)), dim=1)              # (b, 3)
        fused = (g.unsqueeze(-1) * views).sum(1)
        out = self.head(torch.cat([fused, views.flatten(1)], dim=1)).squeeze(-1)
        return (out, g) if return_gate else out


def train_mvrn(data_tr, data_va, y_tr, y_va, seed, epochs=60, patience=8, lr=1e-3, bs=128):
    """data_*: dict of tensors tab, cats, seq, emb. Returns (val probabilities, mean gate)."""
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    cards = [int(max(data_tr["cats"][:, i].max(), data_va["cats"][:, i].max())) + 1
             for i in range(data_tr["cats"].shape[1])]
    m = MVRN(data_tr["tab"].shape[1], data_tr["seq"].shape[2], data_tr["emb"].shape[1], cards)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y_tr.mean()) / y_tr.mean()))
    n = len(y_tr)
    best = (-1.0, None, None, 0)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = m(data_tr["tab"][b], data_tr["cats"][b], data_tr["seq"][b], data_tr["emb"][b])
            lossf(out, yt[b]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            logit, g = m(data_va["tab"], data_va["cats"], data_va["seq"], data_va["emb"], return_gate=True)
        p = torch.sigmoid(logit).numpy()
        ap = average_precision_score(y_va, p)
        if ap > best[0]:
            best = (ap, p, g.mean(0).numpy(), ep)
        elif ep - best[3] >= patience:
            break
    return best[1], best[2]


# ---- MVRN-2: adds a competition-set view with attention pooling ----------------------
class CompSetEncoder(nn.Module):
    """Encodes a user's set of competitions (up to K rows of per-competition features,
    padded rows masked) with a shared MLP and attention pooling (DeepSets + attention)."""
    def __init__(self, d_in, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, h), nn.ReLU())
        self.att = nn.Linear(h, 1)
        self.empty = nn.Parameter(torch.zeros(h))           # learned vector for "no competition"

    def forward(self, comps, mask):                         # comps (b, K, d), mask (b, K) True = real
        z = self.phi(comps)
        a = self.att(z).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1) * mask
        pooled = (w.unsqueeze(-1) * z).sum(1)
        has = mask.any(1, keepdim=True).float()
        return has * pooled + (1 - has) * self.empty, w


class MVRN2(nn.Module):
    def __init__(self, d_tab, d_seq, d_emb, d_comp, cat_cards, h=64, p_emb_drop=0.5):
        super().__init__()
        self.base = MVRN(d_tab, d_seq, d_emb, cat_cards, h, p_emb_drop)
        self.comp = CompSetEncoder(d_comp, h)
        self.gate = nn.Sequential(nn.Linear(4 * h, 32), nn.ReLU(), nn.Linear(32, 4))
        self.head = nn.Sequential(nn.Linear(5 * h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, tab, cats, seq, emb, comps, cmask, return_gate=False):
        b = self.base
        c = torch.cat([e(cats[:, i]) for i, e in enumerate(b.cat_emb)], dim=1)
        v_tab = b.tab(torch.cat([tab, c], dim=1))
        _, hs = b.seq(seq)
        v_seq, v_emb = hs[-1], b.emb(emb)
        v_comp, _ = self.comp(comps, cmask)
        views = torch.stack([v_tab, v_seq, v_emb, v_comp], dim=1)
        g = torch.softmax(self.gate(views.flatten(1)), dim=1)
        fused = (g.unsqueeze(-1) * views).sum(1)
        out = self.head(torch.cat([fused, views.flatten(1)], dim=1)).squeeze(-1)
        return (out, g) if return_gate else out


def train_mvrn2(dtr, dva, y_tr, y_va, seed, epochs=60, patience=8, lr=1e-3, bs=128):
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    cards = [int(max(dtr["cats"][:, i].max(), dva["cats"][:, i].max())) + 1 for i in range(dtr["cats"].shape[1])]
    m = MVRN2(dtr["tab"].shape[1], dtr["seq"].shape[2], dtr["emb"].shape[1], dtr["comps"].shape[2], cards)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y_tr.mean()) / y_tr.mean()))
    keys = ["tab", "cats", "seq", "emb", "comps", "cmask"]
    best = (-1.0, None, None, 0)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(m(*[dtr[k][b] for k in keys]), yt[b]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            logit, g = m(*[dva[k] for k in keys], return_gate=True)
        p = torch.sigmoid(logit).numpy()
        ap = average_precision_score(y_va, p)
        if ap > best[0]:
            best = (ap, p, g.mean(0).numpy(), ep)
        elif ep - best[3] >= patience:
            break
    return best[1], best[2]
