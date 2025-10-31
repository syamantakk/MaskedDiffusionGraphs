# stats.py
from __future__ import annotations
import argparse, random
import numpy as np
import matplotlib.pyplot as plt
from data import GraphPathDataset

def path_length_from_labels(labels: np.ndarray) -> int:
    """
    labels: (V,V) int in {0,1,2}
    Returns the number of undirected edges labeled 2, i.e., path length in edges.
    """
    # Count only the upper triangle to avoid double-counting (since labels are symmetric)
    return int(np.count_nonzero(np.triu(labels == 2, k=1)))

def summarize(x: list[int]) -> dict[str, float]:
    a = np.array(x, dtype=np.int32)
    return dict(
        count=int(a.size),
        min=int(a.min()) if a.size else 0,
        p10=float(np.percentile(a, 10)) if a.size else 0.0,
        median=float(np.median(a)) if a.size else 0.0,
        mean=float(a.mean()) if a.size else 0.0,
        p90=float(np.percentile(a, 90)) if a.size else 0.0,
        max=int(a.max()) if a.size else 0,
        std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V", type=int, default=100)
    p.add_argument("--num_graphs", type=int, default=100)
    p.add_argument("--pairs_per_graph", type=int, default=10)
    p.add_argument("--extra_p", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="train", choices=["all","train","val","test"])
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--save_png", type=str, default="")
    p.add_argument("--save_csv", type=str, default="")
    args = p.parse_args()

    # Build the same dataset as training
    ds = GraphPathDataset(V=args.V, num_graphs=args.num_graphs,
                          pairs_per_graph=args.pairs_per_graph,
                          extra_p=args.extra_p, seed=args.seed)

    # Reproduce the same split as main.py
    N = len(ds)
    idxs = list(range(N))
    random.seed(args.seed)
    random.shuffle(idxs)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train + n_val]
    test_idx  = idxs[n_train + n_val:]

    if args.split == "train":
        chosen = train_idx
    elif args.split == "val":
        chosen = val_idx
    elif args.split == "test":
        chosen = test_idx
    else:
        chosen = idxs

    lengths: list[int] = []
    for i in chosen:
        lengths.append(path_length_from_labels(ds.samples[i].labels))

    # Print summary stats
    s = summarize(lengths)
    print(f"Split={args.split} | count={s['count']}  min={s['min']}  p10={s['p10']:.1f}  "
          f"median={s['median']:.1f}  mean={s['mean']:.2f}  p90={s['p90']:.1f}  "
          f"max={s['max']}  std={s['std']:.2f}")

    # Optional CSV
    if args.save_csv:
        np.savetxt(args.save_csv, np.array(lengths, dtype=np.int32), fmt="%d")
        print(f"Saved raw lengths to {args.save_csv}")

    # Histogram
    plt.figure(figsize=(7,4.5))
    plt.hist(lengths, bins=args.bins, edgecolor="black")
    plt.xlabel("Path length (edges)")
    plt.ylabel("Count")
    plt.title(f"Path-length distribution ({args.split})")
    plt.tight_layout()
    if args.save_png:
        plt.savefig(args.save_png, dpi=150)
        print(f"Saved histogram to {args.save_png}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
