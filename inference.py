# File: inference.py

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

def choose_edge(mask: torch.Tensor, scores: torch.Tensor, mode: str = 'sample', temperature: float = 1.0) -> Tuple[int,int]:
    """Select which masked edge to reveal next.
    mask: (V,V) bool; scores: (V,V) float (e.g., max prob over {1,2}).
    mode='greedy' picks argmax; mode='sample' samples proportionally to scores^1/temperature.
    Returns (i,j) index in [0,V).
    """
    V = mask.shape[0]
    flat_scores = scores.clone().masked_fill(~mask, -1e9).flatten()
    if mode == 'greedy':
        idx = torch.argmax(flat_scores).item()
        return idx // V, idx % V
    else:
        # Sampling over masked edges using softmax on (score/temperature)
        logits = flat_scores / max(1e-6, temperature)
        # Avoid all -inf: replace -1e9 with very small number before softmax
        logits = torch.where(flat_scores > -1e8, logits, torch.full_like(logits, -1e9))
        probs = F.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return idx // V, idx % V

def sample_token_from_probs(prob_vec: torch.Tensor, classes=(1,2), temperature: float = 1.0) -> int:
    """Given per-class probs (size 4 over {0,1,2,3}), sample from the subset `classes` after
    restricting to those classes and re-normalizing with temperature.
    Returns the chosen class id.
    """
    # Extract and temperature-scale
    sel = torch.tensor(classes, device=prob_vec.device)
    p = prob_vec[sel].clamp(min=1e-12)
    if temperature != 1.0:
        p = p.pow(1.0 / max(1e-6, temperature))
    p = p / p.sum()
    k = torch.multinomial(p, 1).item()
    return int(sel[k].item())

def sequential_unmask_unconstrained_sampling(model,
                                             base_adj: torch.Tensor,
                                             node_feats: torch.Tensor,
                                             device: torch.device,
                                             edge_select: str = 'greedy',
                                             edge_select_temp: float = 1.0,
                                             token_temp: float = 1.0) -> np.ndarray:
    """Start with all existing edges masked (3), non-edges fixed at 0. No constraints.
    Each step: choose a masked edge (greedy or sampled by confidence), then sample its
    token from {1,2} using temperature. Return (V,V) tokens numpy.
    """
    model.eval()
    V = base_adj.shape[0]
    obs = torch.where(base_adj > 0.5, torch.full((V, V), 3, dtype=torch.long, device=device), torch.zeros((V, V), dtype=torch.long, device=device))
    idx = torch.arange(obs.size(0), device=obs.device)
    obs[idx, idx] = 0

    steps = int((base_adj > 0.5).sum().item())
    for step in range(steps):
        tval = torch.tensor([min(1.0, step / max(1, steps - 1))], device=device)
        logits = model(base_adj[None, :, :], node_feats[None, :, :], obs[None, :, :], tval)  # (1,V,V,4)
        probs = F.softmax(logits[0], dim=-1)  # (V,V,4)
        mask = (obs == 3)
        if mask.sum().item() == 0:
            break
        # Confidence per edge = max over {1,2}
        conf_12 = probs[..., 1:3].max(dim=-1).values  # (V,V)
        i, j = choose_edge(mask, conf_12, mode=edge_select, temperature=edge_select_temp)
        # Sample token from {1,2}
        chosen_class = sample_token_from_probs(probs[i, j], classes=(1,2), temperature=token_temp)
        obs[i, j] = chosen_class
        obs[j, i] = chosen_class

    return obs.detach().cpu().numpy()