# File: data.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx

@dataclass
class Sample:
    base_adj: np.ndarray   # (V,V) float32 0/1 undirected
    labels: np.ndarray     # (V,V) int64 tokens {0,1,2}
    node_feats: np.ndarray # (V,1) float32 degree_norm only


def generate_connected_graph(V: int = 100, extra_p: float = 0.04, seed: int = 0) -> nx.Graph:
    rng = random.Random(seed)
    T = nx.random_labeled_tree(V, seed=seed)
    G = nx.Graph(); G.add_nodes_from(range(V)); G.add_edges_from(T.edges())
    for i in range(V):
        for j in range(i+1, V):
            if not G.has_edge(i, j) and rng.random() < extra_p:
                G.add_edge(i, j)
    assert nx.is_connected(G)
    return G


def shortest_path_edges(G: nx.Graph, s: int, t: int) -> List[Tuple[int, int]]:
    path = nx.shortest_path(G, source=s, target=t)
    edges = []
    for a, b in zip(path[:-1], path[1:]):
        u, v = (a, b) if a < b else (b, a)
        edges.append((u, v))
    return edges


def build_sample_from_graph(G: nx.Graph, s: int, t: int) -> Sample:
    V = G.number_of_nodes()
    base_adj = np.zeros((V, V), dtype=np.float32)
    for u, v in G.edges():
        base_adj[u, v] = 1.0; base_adj[v, u] = 1.0
    labels = np.zeros((V, V), dtype=np.int64)
    labels[base_adj == 1.0] = 1
    for u, v in shortest_path_edges(G, s, t):
        labels[u, v] = 2; labels[v, u] = 2
    np.fill_diagonal(labels, 0)
    deg = np.array([G.degree(i) for i in range(V)], dtype=np.float32)
    deg_norm = deg / (deg.max() + 1e-6)
    node_feats = deg_norm[:, None]
    return Sample(base_adj, labels, node_feats)


class GraphPathDataset(Dataset):
    def __init__(self, V: int, num_graphs: int, pairs_per_graph: int, extra_p: float = 0.04, seed: int = 0):
        super().__init__()
        self.V = V
        self.samples: List[Sample] = []
        rng = random.Random(seed)
        for g_idx in range(num_graphs):
            G = generate_connected_graph(V, extra_p=extra_p, seed=seed + g_idx)
            for _ in range(pairs_per_graph):
                s, t = rng.sample(range(V), 2)
                self.samples.append(build_sample_from_graph(G, s, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'base_adj': torch.from_numpy(s.base_adj),
            'labels': torch.from_numpy(s.labels),
            'node_feats': torch.from_numpy(s.node_feats),
        }


def collate(batch: List[Dict[str, torch.Tensor]]):
    base_adj = torch.stack([b['base_adj'] for b in batch], dim=0)
    labels = torch.stack([b['labels'] for b in batch], dim=0)
    node_feats = torch.stack([b['node_feats'] for b in batch], dim=0)
    return {'base_adj': base_adj, 'labels': labels, 'node_feats': node_feats}


def sanity_check_dataset(ds: GraphPathDataset):
    # Check label <-> base_adj consistency and presence of some 2s
    cnt2 = 0
    for s in ds.samples[:min(len(ds), 20)]:
        assert np.all((s.base_adj == 0.0) | (s.base_adj == 1.0))
        # Non-edges must be label 0
        assert np.all(s.labels[s.base_adj == 0.0] == 0)
        # Path edges are subset of edges
        for i in range(ds.V):
            for j in range(ds.V):
                if s.labels[i, j] == 2:
                    assert s.base_adj[i, j] == 1.0
                    cnt2 += 1
    assert cnt2 > 0, "No path edges (2) found in first 20 samples; generation failed?"