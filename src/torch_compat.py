"""Shim for running timesfm 3.0.2 on torch 2.2.x (the last Intel-macOS build), which
lacks nn.RMSNorm (added in torch 2.4). Same parameter name ('weight') and formula
as torch's RMSNorm, so the checkpoint loads unchanged. Import before timesfm3."""
import torch
from torch import nn

if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype)) \
                if elementwise_affine else None

        def forward(self, x):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            dims = tuple(range(-len(self.normalized_shape), 0))
            y = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + eps)
            return y * self.weight if self.weight is not None else y

    nn.RMSNorm = RMSNorm

if not hasattr(torch, "GradScaler"):               # torch >= 2.3 API; used by tabicl fine-tuning
    def GradScaler(device="cuda", enabled=True, **kw):
        return torch.cuda.amp.GradScaler(enabled=enabled and device == "cuda" and torch.cuda.is_available(), **kw)
    torch.GradScaler = GradScaler
