# File: sanity.py

from __future__ import annotations
import torch
import numpy as np

from utils import set_seed, assert_shapes_and_types, mask_with_schedule
from data import GraphPathDataset, collate, sanity_check_dataset
from model import SimpleGraphDenoiser
from train import compute_loss
from inference import sequential_unmask_unconstrained_sampling
from eval import extract_valid_path_component, evaluate_valid_path_rate
from torch.utils.data import DataLoader


def _assert_symmetric_and_diagonal_zero(mat: torch.Tensor, name: str):
    assert torch.allclose(mat, mat.transpose(-1, -2), atol=1e-6), f"{name} not symmetric"
    V = mat.shape[-1]
    assert torch.all(mat.diagonal(dim1=-2, dim2=-1) == 0), f"{name} diagonal must be zero"


def _check_inference_tokens(pred_tokens: np.ndarray, base_adj: np.ndarray):
    V = base_adj.shape[0]
    # diagonal zero
    assert np.all(np.diag(pred_tokens) == 0), "pred_tokens diagonal must be zero"
    # symmetry
    assert np.allclose(pred_tokens, pred_tokens.T), "pred_tokens not symmetric"
    # non-edges must be 0
    nz = (base_adj <= 0.5)
    assert np.all(pred_tokens[nz] == 0), "Non-edges should remain 0 after inference"
    # existing edges must be in {1,2}; no masks left
    ez = (base_adj > 0.5)
    assert np.all(np.isin(pred_tokens[ez], [1, 2])), "Existing edges must be 1 or 2 (no 0 or 3)"


def run_sanity():
    set_seed(0)
    device = torch.device('cpu')

    # --- 1) Dataset generation and invariants ---
    ds = GraphPathDataset(V=30, num_graphs=2, pairs_per_graph=3, extra_p=0.08, seed=0)
    sanity_check_dataset(ds)
    print("[OK] Dataset invariants passed (labels/base_adj consistency, presence of 2's)")

    train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    batch = next(iter(train_loader))
    base_adj, labels, node_feats = batch['base_adj'], batch['labels'], batch['node_feats']
    assert_shapes_and_types(base_adj, labels, node_feats)
    _assert_symmetric_and_diagonal_zero(base_adj, 'base_adj')
    _assert_symmetric_and_diagonal_zero(labels.float(), 'labels (float check)')
    # 0 wherever no edge
    assert torch.all(labels[base_adj == 0] == 0), "Labels must be 0 on non-edges"
    print("[OK] Batch shapes/types and structural checks passed")

    # --- 2) Forward pass + loss/backprop on a tiny model ---
    model = SimpleGraphDenoiser(node_in=1, hidden=32, layers=2, edge_hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = base_adj.shape[0]
    t = torch.full((B,), 0.5, device=device)
    obs = mask_with_schedule(labels, m=0.5)

    logits = model(base_adj.to(device), node_feats.to(device), obs.to(device), t)
    assert logits.shape == (B, node_feats.shape[1], node_feats.shape[1], 4), "Logits shape mismatch"
    loss = compute_loss(logits, labels.to(device), obs.to(device))
    assert torch.isfinite(loss).all(), "Loss not finite"
    loss.backward(); opt.step()
    print(f"[OK] Forward/backward step ran, loss={float(loss):.4f}")

    # --- 3) Inference checks (greedy edge selection + sampled token) ---
    one = ds[0]
    ba = one['base_adj'].to(device)
    nf = one['node_feats'].to(device)
    pred_tokens = sequential_unmask_unconstrained_sampling(
        model, ba, nf, device,
        edge_select='greedy', edge_select_temp=1.0, token_temp=1.2
    )
    _check_inference_tokens(pred_tokens, ba.cpu().numpy())
    ok, path_nodes = extract_valid_path_component(pred_tokens, ba.cpu().numpy())
    print(f"[OK] Inference (greedy edges, sampled tokens) produced valid_path={ok}, length={max(0, len(path_nodes)-1)}")

    # --- 4) Inference checks (sampled edge selection + sampled token) ---
    pred_tokens_2 = sequential_unmask_unconstrained_sampling(
        model, ba, nf, device,
        edge_select='sample', edge_select_temp=1.0, token_temp=1.0
    )
    _check_inference_tokens(pred_tokens_2, ba.cpu().numpy())
    ok2, path_nodes2 = extract_valid_path_component(pred_tokens_2, ba.cpu().numpy())
    print(f"[OK] Inference (sampled edges & tokens) produced valid_path={ok2}, length={max(0, len(path_nodes2)-1)}")

    # --- 5) Loader-level evaluation of valid path rate ---
    test_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    vpr = evaluate_valid_path_rate(model, test_loader, device,
                                   edge_select='greedy', edge_select_temp=1.0, token_temp=1.0,
                                   visualize_cb=None)
    print(f"[OK] Valid Path Rate over small dataset = {vpr['valid_path_rate']:.3f}")

    print("All sanity checks passed.")


if __name__ == '__main__':
    run_sanity()