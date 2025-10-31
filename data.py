from __future__ import annotations
"""
Manhattan grid + simple-path dataset.

Generates an R x C 4-neighbor grid (R*C = V). For each graph, picks two nodes
at random whose Manhattan distance is exactly L (i.e., shortest-path length L),
and labels the edges on one shortest (simple) path between them as class 2.
Other existing edges are class 1; non-edges are 0.

API (keeps your existing call signature; 'extra_p' is unused here):
- GraphPathDataset(V, num_graphs, pairs_per_graph, extra_p, L_walk=8, seed, rows=None, cols=None)
- __getitem__ -> dict of tensors:
    base_adj:  (V,V) float32 in {0,1}
    labels:    (V,V) int64   in {0,1,2}
    node_feats:(V,1) float32 (normalized degree)
- collate(batch) -> dict of batched tensors
- sanity_check_dataset(ds) -> raises on invariant violations
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
import torch


@dataclass
class Sample:
    base_adj: np.ndarray   # (V,V) int {0,1}
    labels:   np.ndarray   # (V,V) int {0,1,2}
    node_feats: np.ndarray # (V,1)   float32


# -----------------------------
# Grid helpers
# -----------------------------

def _factor_rows_cols(V: int) -> Tuple[int, int]:
    """Choose rows, cols with rows*cols=V and rows as close to sqrt(V) as possible."""
    r = int(math.isqrt(V))
    while V % r != 0:
        r -= 1
    c = V // r
    return r, c

def _make_grid(V: int, rows: Optional[int], cols: Optional[int]) -> Tuple[nx.Graph, int, int]:
    """Build an R x C Manhattan grid with nodes relabeled to 0..V-1."""
    if rows is None or cols is None:
        rows, cols = _factor_rows_cols(V)
    if rows * cols != V:
        raise ValueError(f"rows*cols must equal V; got rows={rows}, cols={cols}, rows*cols={rows*cols}, V={V}.")
    G2d = nx.grid_2d_graph(rows, cols, periodic=False)  # 4-neighborhood
    mapping = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    G = nx.relabel_nodes(G2d, mapping, copy=True)
    return G, rows, cols

def _pairs_at_exact_distance_L_grid(rows: int, cols: int, L: int) -> List[Tuple[int, int]]:
    """Unordered pairs (u,v) with Manhattan distance exactly L on an R x C grid."""
    pairs = set()
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            # All lattice points at L1 distance L from (r,c)
            for dr in range(-L, L + 1):
                dc_abs = L - abs(dr)
                for dc in ([0] if dc_abs == 0 else [-dc_abs, dc_abs]):
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < rows and 0 <= c2 < cols:
                        v = r2 * cols + c2
                        if u < v:
                            pairs.add((u, v))
    return list(pairs)


# -----------------------------
# Labels & feats
# -----------------------------

def _labels_from_path(V: int, base_adj01: np.ndarray, path_nodes: List[int]) -> np.ndarray:
    """0=non-edge, 1=edge-not-on-path, 2=edge-on-path (symmetric, diag=0)."""
    labels = np.zeros((V, V), dtype=np.int64)
    labels[base_adj01 > 0] = 1
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        labels[a, b] = 2
        labels[b, a] = 2
    np.fill_diagonal(labels, 0)
    return labels

def _node_feats_from_adj(base_adj01: np.ndarray) -> np.ndarray:
    """1-D node features: normalized degree in [0,1]."""
    deg = base_adj01.sum(axis=1)
    denom = float(max(1.0, deg.max()))
    nf = (deg / denom).astype(np.float32)[:, None]
    return nf


# -----------------------------
# Dataset
# -----------------------------

class GraphPathDataset(torch.utils.data.Dataset):
    """Manhattan grid graphs with simple paths of fixed length L (exactly L edges)."""

    def __init__(self, V: int = 100, num_graphs: int = 100, pairs_per_graph: int = 10,
                 extra_p: float = 0.0, L_walk: int = 8, seed: int = 42,
                 rows: Optional[int] = None, cols: Optional[int] = None):
        """
        Parameters
        ----------
        V : int
            Total number of nodes; must equal rows*cols (if rows/cols not provided, they are inferred).
        num_graphs : int
            Number of (identical-shape) grid graphs to generate.
        pairs_per_graph : int
            Samples per graph (choose endpoints with Manhattan distance L).
        extra_p : float
            Unused (kept for API compatibility).
        L_walk : int
            Desired simple-path length L (exact).
        seed : int
            RNG seed.
        rows, cols : Optional[int]
            Grid shape; if None, inferred to satisfy rows*cols = V.
        """
        super().__init__()
        self.V = V
        self.num_graphs = num_graphs
        self.samples_per_graph = pairs_per_graph
        self.L_walk = L_walk
        self.seed = seed
        self.rows = rows
        self.cols = cols

        self.samples: List[Sample] = []
        self._build()

    def _build(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_graphs):
            G, R, C = _make_grid(self.V, self.rows, self.cols)
            # Candidate endpoint pairs at exact distance L
            found_pairs = _pairs_at_exact_distance_L_grid(R, C, self.L_walk)
            if not found_pairs:
                raise ValueError(
                    f"No node pairs at distance L={self.L_walk} on a {R}x{C} grid. "
                    f"Ensure L <= (rows-1)+(cols-1)."
                )

            # Base tensors (same per graph)
            A = nx.to_numpy_array(G, dtype=np.float32)
            A[A > 0] = 1.0
            np.fill_diagonal(A, 0.0)
            A01 = (A > 0.5).astype(np.int64)
            nf = _node_feats_from_adj(A01)

            # Draw samples
            for _ in range(self.samples_per_graph):
                u, v = rng.choice(found_pairs)
                path_nodes = nx.shortest_path(G, source=u, target=v)  # unweighted -> Manhattan shortest path
                assert len(path_nodes) - 1 == self.L_walk, "shortest path length mismatch"
                labels = _labels_from_path(self.V, A01, path_nodes)
                self.samples.append(Sample(base_adj=A01.copy(), labels=labels, node_feats=nf.copy()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            'base_adj': torch.from_numpy(s.base_adj.astype(np.float32)),
            'labels': torch.from_numpy(s.labels.astype(np.int64)),
            'node_feats': torch.from_numpy(s.node_feats.astype(np.float32)),
        }


def collate(batch: List[dict]) -> dict:
    base = torch.stack([b['base_adj'] for b in batch], dim=0)
    labels = torch.stack([b['labels'] for b in batch], dim=0)
    feats = torch.stack([b['node_feats'] for b in batch], dim=0)
    return {'base_adj': base, 'labels': labels, 'node_feats': feats}


def sanity_check_dataset(ds: GraphPathDataset):
    """Basic invariants: symmetry, diag=0, label-2 implies edge, exactly L_walk path edges."""
    K = min(128, len(ds))
    for i in range(K):
        s = ds.samples[i]
        A = s.base_adj
        L = s.labels
        V = ds.V
        assert A.shape == (V, V), "bad shape for base_adj"
        assert L.shape == (V, V), "labels shape mismatch"
        assert np.allclose(A, A.T), "base_adj not symmetric"
        assert np.allclose(L, L.T), "labels not symmetric"
        assert np.all(np.diag(A) == 0), "base_adj diag must be 0"
        assert np.all(np.diag(L) == 0), "labels diag must be 0"
        assert np.all(L[A == 0] == 0), "labels must be 0 on non-edges"
        num_edge_twos = (L == 2).sum()
        assert num_edge_twos == 2 * ds.L_walk, f"expected 2*L_walk path entries, got {num_edge_twos}"