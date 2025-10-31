# File: main.py

from __future__ import annotations
import argparse
import random
import torch
from torch.utils.data import DataLoader
from utils import set_seed, assert_shapes_and_types
from data import GraphPathDataset, collate, sanity_check_dataset
from model import SimpleGraphDenoiser
from train import train_one_epoch, evaluate_masked_acc
from eval import evaluate_valid_path_rate
from viz import visualize_graph_and_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graphs', type=int, default=100)
    parser.add_argument('--pairs_per_graph', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--edge_hidden', type=int, default=128)
    parser.add_argument('--V', type=int, default=20)
    parser.add_argument('--extra_p', type=float, default=0.04)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--edge_select', type=str, default='greedy', choices=['greedy','sample'], help='How to choose which edge to unmask next')
    parser.add_argument('--edge_select_temp', type=float, default=1.0, help='Temperature for edge selection when sampling')
    parser.add_argument('--token_temp', type=float, default=1.0, help='Temperature for token sampling from {1,2}')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    print(f"Using device: {device}")

    dataset = GraphPathDataset(V=args.V, num_graphs=args.num_graphs, pairs_per_graph=args.pairs_per_graph, extra_p=args.extra_p, seed=args.seed)
    sanity_check_dataset(dataset)

    N = len(dataset); idxs = list(range(N)); random.shuffle(idxs)
    n_train = int(0.8*N); n_val = int(0.1*N)
    train_idx = idxs[:n_train]; val_idx = idxs[n_train:n_train+n_val]; test_idx = idxs[n_train+n_val:]

    subset = lambda d, idxs: torch.utils.data.Subset(d, idxs)
    train_loader = DataLoader(subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(subset(dataset, test_idx),  batch_size=1, shuffle=False, collate_fn=collate)

    # Peek one batch to assert shapes
    peek = next(iter(train_loader))
    assert_shapes_and_types(peek['base_adj'], peek['labels'], peek['node_feats'])

    model = SimpleGraphDenoiser(node_in=1, hidden=args.hidden, layers=args.layers, edge_hidden=args.edge_hidden).to(device)
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
        visualize_graph_and_prediction(base_adj_np, pred_tokens_np, path_nodes, title=title)

    vpr = evaluate_valid_path_rate(model, test_loader, device,
                                   edge_select=args.edge_select, edge_select_temp=args.edge_select_temp, token_temp=args.token_temp,
                                   visualize_cb=vis_cb)
    print(f"Test Valid Path Rate: {vpr['valid_path_rate']:.3f}")

if __name__ == '__main__':
    main()