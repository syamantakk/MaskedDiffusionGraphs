# File: train.py

from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from utils import mask_with_schedule

def cls2_metrics(logits: torch.Tensor, labels: torch.Tensor, obs_tokens: torch.Tensor):
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B,V,V)
        mask = (obs_tokens == 3) & ((labels == 1) | (labels == 2))
        if mask.sum().item() == 0:
            return {"cls2_prec": 0.0, "cls2_rec": 0.0, "masked_frac_2": 0.0}

        # masked ground-truth 2s
        gt2 = (labels == 2) & mask
        pred2 = (preds == 2) & mask

        tp = (pred2 & gt2).sum().item()
        fp = (pred2 & ~gt2).sum().item()
        fn = (~pred2 & gt2).sum().item()
        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))
        frac2 = gt2.sum().item() / max(1, mask.sum().item())
        return {"cls2_prec": prec, "cls2_rec": rec, "masked_frac_2": frac2}

def build_balanced_obs(labels_b: torch.Tensor,
                       p_mask_2: float = 1.0,
                       rng: torch.Generator | None = None) -> torch.Tensor:
    """
    For ONE item (V,V):
      - leave class-0 visible (0)
      - mask class-2 with prob p_mask_2 (default: all 2's)
      - mask the same number of class-1 positions (randomly sampled)
      - keep diagonal = 0
    Returns obs: int {0,1,2,3}
    """
    if rng is None:
        rng = torch.Generator(device=labels_b.device)
    V = labels_b.shape[0]
    obs_b = labels_b.clone()

    idx2 = (labels_b == 2)
    idx1 = (labels_b == 1)

    # mask all / some 2s
    if p_mask_2 >= 1.0:
        mask2 = idx2
    else:
        mask2 = idx2 & (torch.rand_like(labels_b, dtype=torch.float32) < p_mask_2)

    n2 = int(mask2.sum().item())
    obs_b[mask2] = 3

    # mask exactly n2 of the available 1s (if possible)
    if n2 > 0:
        flat1 = idx1.view(-1).nonzero(as_tuple=True)[0]
        if flat1.numel() > 0:
            k = min(n2, flat1.numel())
            perm = torch.randperm(flat1.numel(), generator=rng, device=labels_b.device)
            pick = flat1[perm[:k]]
            mask1 = torch.zeros_like(labels_b, dtype=torch.bool)
            mask1.view(-1)[pick] = True
            obs_b[mask1] = 3

    # diagonal stays 0
    d = torch.arange(V, device=labels_b.device)
    obs_b[d, d] = 0
    return obs_b

# def compute_loss(logits: torch.Tensor, labels: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
#     B, V, _, C = logits.shape
#     logits = logits.view(B * V * V, C)
#     labels = labels.view(B * V * V)
#     obs = obs_tokens.view(B * V * V)

#     # Only compute loss on masked positions with true label in {1,2}
#     mask = (obs == 3) & ((labels == 1) | (labels == 2))
#     if mask.sum() == 0:
#         return torch.tensor(0.0, device=logits.device, requires_grad=True)

#     logits_m = logits[mask]
#     labels_m = labels[mask]

#     # (labels_m is always {1,2} here, so entries 0 and 3 are unused but must be present)
#     class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0], device=logits.device)
#     return F.cross_entropy(logits_m, labels_m, weight=class_weights)

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
    B, V, _, C = logits.shape
    logits = logits.view(B * V * V, C)
    labels = labels.view(B * V * V)
    obs = obs_tokens.view(B * V * V)

    mask = (obs == 3) & ((labels == 1) | (labels == 2))
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_m = logits[mask]
    labels_m = labels[mask]  # {1,2}

    n1 = (labels_m == 1).sum().item()
    n2 = (labels_m == 2).sum().item()
    total = max(1, n1 + n2)
    f1, f2 = n1 / total, n2 / total

    # inverse-frequency weights, clipped for stability
    w1 = float(min(max(0.5 / max(1e-6, f1), 0.5), 5.0))
    w2 = float(min(max(0.5 / max(1e-6, f2), 0.5), 5.0))

    class_weights = torch.tensor([1.0, w1, w2, 1.0], device=logits.device)
    return F.cross_entropy(logits_m, labels_m, weight=class_weights)

def estimate_masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, obs_tokens: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = (obs_tokens == 3) & ((labels == 1) | (labels == 2))
        total = mask.sum().item()
        correct = ((preds == labels) & mask).sum().item()
        acc = (correct / total) if total > 0 else 0.0
        return {"masked_edge_acc": acc}


def train_one_epoch(model, loader, device, optimizer) -> Dict[str, float]:
    model.train()
    total_loss = 0.0; total_acc = 0.0; count = 0
    total_cls2_prec = total_cls2_rec = total_frac2 = 0.0
    for batch in loader:
        base_adj = batch['base_adj'].to(device)
        labels = batch['labels'].to(device)
        node_feats = batch['node_feats'].to(device)
        B = base_adj.shape[0]
        t = torch.rand((B,), device=device)
        m = t ** 2
        obs = torch.stack([mask_with_schedule(labels[b:b+1], float(m[b].item())).squeeze(0) for b in range(B)], dim=0)
        optimizer.zero_grad()
        logits = model(base_adj, node_feats, obs, t)
        m = cls2_metrics(logits.detach(), labels, obs)
        total_cls2_prec += m["cls2_prec"]
        total_cls2_rec  += m["cls2_rec"]
        total_frac2     += m["masked_frac_2"]
        loss = compute_loss(logits, labels, obs)
        loss.backward(); optimizer.step()
        total_loss += float(loss.item())
        total_acc += estimate_masked_accuracy(logits.detach(), labels, obs)['masked_edge_acc']
        count += 1
    return {"loss": total_loss / max(1, count), 
    		"masked_edge_acc": total_acc / max(1, count), 
    		"cls2_prec": total_cls2_prec / max(1, count),
		    "cls2_rec":  total_cls2_rec  / max(1, count),
		    "masked_frac_2": total_frac2 / max(1, count)}

def evaluate_masked_acc(model, loader, device) -> Dict[str, float]:
    model.eval(); total_acc = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            base_adj = batch['base_adj'].to(device)
            labels = batch['labels'].to(device)
            node_feats = batch['node_feats'].to(device)
            B = base_adj.shape[0]
            t = torch.full((B,), 0.5, device=device)
            obs = torch.stack([mask_with_schedule(labels[b:b+1], 0.5).squeeze(0) for b in range(B)], dim=0)
            logits = model(base_adj, node_feats, obs, t)
            total_acc += estimate_masked_accuracy(logits, labels, obs)['masked_edge_acc']
            count += 1
    return {"masked_edge_acc": total_acc / max(1, count)}