# File: model.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fourier_time_embed

class GCNBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin_self = nn.Linear(hidden, hidden)
        self.lin_neigh = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden)

    def forward(self, A_norm: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        neigh = torch.matmul(A_norm, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = self.act(out)
        out = self.norm(out)
        return out

class SimpleGraphDenoiser(nn.Module):
    def __init__(self, node_in: int = 1, hidden: int = 64, time_dim: int = 64, layers: int = 3, edge_hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * (time_dim // 2), hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.node_proj = nn.Linear(node_in, hidden)
        self.layers = nn.ModuleList([GCNBlock(hidden) for _ in range(layers)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1 + 4 + hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 4)
        )

    def forward(self, base_adj: torch.Tensor, node_feats: torch.Tensor, obs_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, V, _ = node_feats.shape
        A = base_adj
        h = self.node_proj(node_feats)
        t_emb = self.time_mlp(fourier_time_embed(t, dim=32))  # (B,H)
        h = h + t_emb[:, None, :]
        # Normalize A
        deg = A.sum(dim=-1).clamp(min=1.0)
        D_inv_sqrt = (1.0 / torch.sqrt(deg))
        A_norm = D_inv_sqrt[:, :, None] * A * D_inv_sqrt[:, None, :]
        for layer in self.layers:
            h = layer(A_norm, h)
        hi = h[:, :, None, :].expand(B, V, V, self.hidden)
        hj = h[:, None, :, :].expand(B, V, V, self.hidden)
        edge_exists = A[:, :, :, None]
        obs_onehot = F.one_hot(obs_tokens, num_classes=4).float()
        t_edge = t_emb[:, None, None, :].expand(B, V, V, self.hidden)
        x = torch.cat([hi, hj, edge_exists, obs_onehot, t_edge], dim=-1)
        logits = self.edge_mlp(x)
        return logits  # (B,V,V,4)