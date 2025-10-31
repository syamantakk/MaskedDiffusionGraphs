# eval.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader
from inference import sequential_unmask_unconstrained_sampling

def extract_valid_path_component(pred_tokens: np.ndarray, base_adj: np.ndarray) -> Tuple[bool, List[int]]:
    V = pred_tokens.shape[0]
    Gp = nx.Graph(); Gp.add_nodes_from(range(V))
    for i in range(V):
        for j in range(i+1, V):
            if (pred_tokens[i, j] == 2 or pred_tokens[j, i] == 2) and base_adj[i, j] > 0.5:
                Gp.add_edge(i, j)
    if Gp.number_of_edges() == 0:
        return False, []
    best_path = []
    for comp in nx.connected_components(Gp):
        H = Gp.subgraph(comp)
        degs = dict(H.degree())
        if all(d <= 2 for d in degs.values()):
            endpoints = [n for n, d in degs.items() if d == 1]
            if len(endpoints) == 2:
                p = nx.shortest_path(H, endpoints[0], endpoints[1])
                if len(p) > len(best_path):
                    best_path = p
    return (len(best_path) >= 2), best_path

def evaluate_valid_path_rate(
    model,
    loader: DataLoader,
    device: torch.device,
    edge_select: str = 'greedy',
    edge_select_temp: float = 1.0,
    token_temp: float = 1.0,
    visualize_cb=None,
    num_plots: int = 1,       # <= NEW
) -> Dict[str, float]:
    valid_count = 0; total = 0; plotted = 0
    for batch in loader:
        base_adj = batch['base_adj'][0].to(device)
        node_feats = batch['node_feats'][0].to(device)
        pred_tokens = sequential_unmask_unconstrained_sampling(
            model, base_adj, node_feats, device,
            edge_select=edge_select, edge_select_temp=edge_select_temp, token_temp=token_temp)
        ok, path_nodes = extract_valid_path_component(pred_tokens, base_adj.cpu().numpy())
        valid_count += 1 if ok else 0; total += 1

        if (visualize_cb is not None) and (plotted < num_plots):
            title = f"Valid path: {ok} | length: {max(0, len(path_nodes)-1)}"
            visualize_cb(base_adj.cpu().numpy(), pred_tokens, path_nodes, title=title)
            plotted += 1

    return {"valid_path_rate": valid_count / max(1, total)}