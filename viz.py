# File: viz.py

from __future__ import annotations
from typing import List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph_and_prediction(base_adj: np.ndarray, pred_tokens: np.ndarray, path_nodes: List[int], title: str = ""):
    V = base_adj.shape[0]
    G = nx.Graph(); G.add_nodes_from(range(V))
    edges = [(i, j) for i in range(V) for j in range(i+1, V) if base_adj[i, j] > 0.5]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='lightgray', width=1.0)
    pr_edges = []
    for i in range(V):
        for j in range(i+1, V):
            if base_adj[i, j] > 0.5 and (pred_tokens[i, j] == 2 or pred_tokens[j, i] == 2):
                pr_edges.append((i, j))
    if pr_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pr_edges, edge_color='orange',
                               width=2.5, style='dashed', label='All predicted 2-edges')

    if path_nodes and len(path_nodes) >= 2:
        path_edge_list = list(zip(path_nodes[:-1], path_nodes[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edge_list, edge_color='green',
                               width=4.0, label='Counted path')
        nx.draw_networkx_nodes(G, pos, nodelist=[path_nodes[0], path_nodes[-1]],
                               node_color='red', node_size=280)
    if path_nodes and len(path_nodes) >= 2:
        nx.draw_networkx_nodes(G, pos, nodelist=[path_nodes[0], path_nodes[-1]], node_color='red', node_size=280)
    print("Counted path nodes:", path_nodes, "=> length:", max(0, len(path_nodes)-1))
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout(); plt.show()