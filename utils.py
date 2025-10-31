# File: utils.py

from __future__ import annotations
import math
import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def noise_schedule(t: torch.Tensor, kind: str = "poly2",
                   min_m: float = 0.02, max_m: float = 0.95) -> torch.Tensor:
    """
    Map t in [0,1] -> masking prob m(t). We clip to avoid empty/all-masked batches.
    kind="poly2": m(t) = t^2
    """
    if kind == "poly2":
        m = t ** 2
    elif kind == "linear":
        m = t
    else:
        raise ValueError(f"unknown schedule kind: {kind}")
    return m.clamp(min_m, max_m)

def fourier_time_embed(t: torch.Tensor, dim: int = 32, max_freq: float = 10.0) -> torch.Tensor:
    """Sinusoidal time embedding: (B,) -> (B, 2*dim)."""
    device = t.device
    freqs = torch.exp(torch.linspace(0, math.log(max_freq), dim, device=device))  # (dim,)
    phases = t[:, None] * freqs[None, :]  # (B,dim)
    emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (B,2*dim)
    return emb


def mask_with_schedule(labels: torch.Tensor, m: float) -> torch.Tensor:
    """Mask only {1,2} tokens to 3 with probability m. Keep 0s visible.
    labels: (B,V,V) int64 in {0,1,2} -> returns obs_tokens in {0,1,2,3}
    """
    B, V, _ = labels.shape
    obs = labels.clone()
    maskable = (labels == 1) | (labels == 2)
    rand = torch.rand_like(labels.float())
    to_mask = (rand < m) & maskable
    obs[to_mask] = 3
    # Diagonal stays zero (no self-edges)
    diag = torch.arange(V, device=labels.device)
    obs[:, diag, diag] = 0
    return obs


def assert_shapes_and_types(base_adj: torch.Tensor, labels: torch.Tensor, node_feats: torch.Tensor):
    B, V, V2 = base_adj.shape
    assert V == V2, "base_adj must be square"
    assert labels.shape == (B, V, V), "labels shape mismatch"
    assert node_feats.shape[0] == B and node_feats.shape[1] == V, "node_feats batch or V mismatch"
    assert base_adj.dtype in (torch.float32, torch.float64), "base_adj must be float"
    assert labels.dtype == torch.long, "labels must be int64"