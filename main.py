# File: main.py

from __future__ import annotations
import argparse
import random
import torch
import math
from torch.utils.data import DataLoader, Subset
from utils import set_seed, assert_shapes_and_types
from data import GraphPathDataset, collate, sanity_check_dataset
from model import SimpleGraphDenoiser, EdgeTokenTransformer
from train import train_one_epoch, evaluate_masked_acc
from eval import evaluate_valid_path_rate
from viz import visualize_graph_and_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graphs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--edge_hidden', type=int, default=128)
    parser.add_argument('--V', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--L_walk', type=int, default=8, help='random-walk length (edges)')
    parser.add_argument('--extra_p', type=float, default=0.05, help='ER prob p in G(V,p)')
    parser.add_argument('--pairs_per_graph', type=int, default=10, help='walks per graph')
    parser.add_argument("--viz-train", type=int, default=2,
                    help="If > 0, visualize N random training samples (GT) and exit")
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--edge_select', type=str, default='greedy', choices=['greedy','sample'], help='How to choose which edge to unmask next')
    parser.add_argument('--edge_select_temp', type=float, default=1.0, help='Temperature for edge selection when sampling')
    parser.add_argument('--token_temp', type=float, default=1.0, help='Temperature for token sampling from {1,2}')
    parser.add_argument('--viz_n', type=int, default=5, help='Number of test samples to visualize at the end')
    parser.add_argument('--arch', type=str, default='transformer',
                                  choices=['gcn', 'transformer'],
                                  help='Backbone architecture')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer heads (if arch=transformer)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    print(f"Using device: {device}")

    dataset = GraphPathDataset(
        V=args.V,
        num_graphs=args.num_graphs,
        pairs_per_graph=args.pairs_per_graph,  # (= walks per graph)
        extra_p=args.extra_p,                  # p in G(V,p)
        L_walk=args.L_walk,
        seed=args.seed,
        rows=int(math.sqrt(args.V)),
        cols=int(math.sqrt(args.V)),
    )
    sanity_check_dataset(dataset)

    N = len(dataset); idxs = list(range(N)); random.shuffle(idxs)
    n_train = int(0.8*N); n_val = int(0.1*N)
    train_idx = idxs[:n_train]; val_idx = idxs[n_train:n_train+n_val]; test_idx = idxs[n_train+n_val:]

    subset = lambda d, idxs: torch.utils.data.Subset(d, idxs)
    train_loader = DataLoader(subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(subset(dataset, test_idx),  batch_size=1, shuffle=False, collate_fn=collate)

    if args.viz_train > 0:
        # unwrap to the underlying GraphPathDataset if you're using Subset/DataLoader
        base_ds = train_loader.dataset
        while isinstance(base_ds, Subset):
            base_ds = base_ds.dataset

        k = min(args.viz_train, len(base_ds))
        idxs = random.sample(range(len(base_ds)), k=k)

        for rank, idx in enumerate(idxs, 1):
            s = base_ds.samples[idx]  # dataclass: base_adj (np), labels (np), node_feats (np)
            # Use labels (0/1/2) as "pred_tokens" so class-2 edges are highlighted
            visualize_graph_and_prediction(
                base_adj=s.base_adj,
                pred_tokens=s.labels,
                path_nodes=[],  # no counted path overlay for GT
                title=f"Train sample #{rank} (idx={idx}) — GT path highlighted",
            )

    # Peek one batch to assert shapes
    peek = next(iter(train_loader))
    assert_shapes_and_types(peek['base_adj'], peek['labels'], peek['node_feats'])

    if args.arch == 'gcn':
        model = SimpleGraphDenoiser(node_in=1, hidden=args.hidden, layers=args.layers,
                                edge_hidden=args.edge_hidden).to(device)
    else:
        model = EdgeTokenTransformer(V=args.V, hidden=args.hidden, nhead=args.nhead,
                                     layers=args.layers, edge_hidden=args.edge_hidden,
                                     edge_only=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, device, opt)
        val_stats   = evaluate_masked_acc(model, val_loader, device)
        print(
          f"Epoch {epoch:02d} | loss={train_stats['loss']:.4f} "
          f"| masked_acc={train_stats['masked_edge_acc']:.3f} "
          f"| val_masked_acc={val_stats['masked_edge_acc']:.3f} "
          f"| cls2_prec={train_stats['cls2_prec']:.3f} "
          f"| cls2_rec={train_stats['cls2_rec']:.3f} "
          f"| masked_frac_2={train_stats['masked_frac_2']:.3f}"
        )

    # Evaluate valid-path rate and visualize one example
    def vis_cb(base_adj_np, pred_tokens_np, path_nodes, title):
        from viz import visualize_graph_and_prediction
        visualize_graph_and_prediction(base_adj_np, pred_tokens_np, path_nodes, title=title)

    vpr = evaluate_valid_path_rate(
        model, test_loader, device,
        edge_select=args.edge_select, edge_select_temp=args.edge_select_temp, token_temp=args.token_temp,
        visualize_cb=vis_cb,
        num_plots=args.viz_n,   # <= NEW
    )
    print(f"Test Valid Path Rate: {vpr['valid_path_rate']:.3f}")

if __name__ == '__main__':
    main()