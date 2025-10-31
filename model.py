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
        self.time_pairs = 16  # i.e., total fourier features = 2 * 16 = 32
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * self.time_pairs, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
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
        t_feats = fourier_time_embed(t, dim=self.time_pairs)   # (B, 2*time_pairs)
        t_emb = self.time_mlp(t_feats)     
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

# --- New model: EdgeTokenTransformer -----------------------------------------
class EdgeTokenTransformer(nn.Module):
    """
    Transformer over edge tokens (i<j where base_adj=1). For item b:
      * Build E tokens (E = # undirected edges) with embeddings from:
        - obs token id {0,1,2,3}  (on edges: {1,2,3})
        - i and j index embeddings
        - degree pair (deg[i], deg[j]) via small MLP
        - time embedding from Fourier(t_b) via MLP
      * TransformerEncoder -> per-token classifier -> 4 logits
      * Write logits into (V,V,4) symmetrically.

    If you truly want all pairs (upper triangle), set edge_only=False (heavier).
    """
    def __init__(self, V: int, hidden: int = 64, nhead: int = 4, layers: int = 3,
                 edge_hidden: int = 128, time_dim: int = 32, edge_only: bool = True):
        super().__init__()
        assert hidden % nhead == 0, "hidden must be divisible by nhead"
        self.V = V
        self.hidden = hidden
        self.edge_only = edge_only

        # Embeddings / projections
        self.tok_emb   = nn.Embedding(4, hidden)              # obs_tokens {0,1,2,3}
        self.idx_emb_i = nn.Embedding(V, hidden)
        self.idx_emb_j = nn.Embedding(V, hidden)
        self.deg_mlp   = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.time_pairs = 16  # total fourier features = 32
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * self.time_pairs, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )

        # Transformer encoder
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden*4,
            batch_first=True, activation='gelu', dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

        # Token classifier
        self.classifier = nn.Sequential(nn.Linear(hidden, edge_hidden), 
                                        nn.GELU(), 
                                        nn.Linear(edge_hidden, 4))

    def _edge_index(self, base_adj_b: torch.Tensor):
        """Upper-triangular indices (i<j). If edge_only, keep only existing edges."""
        V = base_adj_b.shape[0]
        iu, ju = torch.triu_indices(V, V, offset=1, device=base_adj_b.device)
        if self.edge_only:
            mask = base_adj_b[iu, ju] > 0.5
            iu, ju = iu[mask], ju[mask]
        return iu, ju

    def forward(self, base_adj: torch.Tensor, node_feats: torch.Tensor,
                obs_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, V, _ = base_adj.shape
        device = base_adj.device
        logits_full = torch.zeros((B, V, V, 4), device=device)

        for b in range(B):
            A   = base_adj[b]       # (V,V)
            obs = obs_tokens[b]     # (V,V)
            # degrees (normalize by max to ~[0,1])
            deg = A.sum(dim=-1)
            deg_norm = deg / deg.clamp(min=1.0).max()

            iu, ju = self._edge_index(A)  # (E,), (E,)
            if iu.numel() == 0:
                continue

            tok = self.tok_emb(obs[iu, ju].clamp(min=0, max=3))              # (E,H)
            ei  = self.idx_emb_i(iu)                                         # (E,H)
            ej  = self.idx_emb_j(ju)                                         # (E,H)
            deg_pair = torch.stack([deg_norm[iu], deg_norm[ju]], dim=-1)     # (E,2)
            deg_vec  = self.deg_mlp(deg_pair)                                # (E,H)
            t_feats = fourier_time_embed(t[b:b+1], dim=self.time_pairs)  # (1, 2*time_pairs)
            t_vec = self.time_mlp(t_feats).squeeze(0)                    # (hidden,)
            t_vec = t_vec.unsqueeze(0).expand(tok.size(0), -1)           # (E, hidden)

            x = tok + ei + ej + deg_vec + t_vec                               # (E,H)
            x = self.encoder(x.unsqueeze(0)).squeeze(0)                       # (E,H)
            out = self.classifier(x)                                          # (E,4)

            logits_full[b, iu, ju, :] = out
            logits_full[b, ju, iu, :] = out  # symmetric copy

        return logits_full  # (B,V,V,4)