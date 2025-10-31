# Masked Diffusion for Paths on Graphs

Simple, modular code to train a **masked-diffusion-style** model that predicts path edges on random graphs. Training uses categorical tokens on edges (0,1,2,3) with masking; decoding is **unconstrained**, **one edge at a time**, and **samples** from {1,2} (never 0 for existing edges). Evaluation reports **Valid Path Rate**.

---

## TL;DR – Quick Start (CPU)

```bash
# 1) Create & activate a virtual environment (Unix/macOS)
python3 -m venv .venv
source .venv/bin/activate

# (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install requirements (CPU-only PyTorch)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install networkx matplotlib numpy

# 3) Train + evaluate + visualize one example
python main.py --device cpu \
  --num_graphs 100 --pairs_per_graph 10 --epochs 10 \
  --edge_select sample --token_temp 1.0
```

This will print training stats, compute **Valid Path Rate** on a test split, and pop up a NetworkX plot with one predicted sample.

> **Tip:** If you just want a tiny smoke test, run `python sanity.py`.

---

## Project Layout

```
main.py           # CLI entry point: train, test, single visualization
sanity.py         # Small, self-contained sanity checks
utils.py          # Seeding, time embeddings, masking, shape assertions
data.py           # Graph generation, labels, dataset+collate, data sanity
model.py          # GCN node encoder + Edge MLP classifier
train.py          # Loss, training loop, masked-accuracy eval
inference.py      # Unconstrained sequential decoder (sampling)
eval.py           # Valid Path Rate metric and evaluation loop
viz.py            # NetworkX visualization helpers
```

---

## Creating a Virtual Environment (alternatives)

### Python `venv` (recommended)

**Unix/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Conda (optional)

```bash
conda create -n graphdiff python=3.10 -y
conda activate graphdiff
```

---

## Installing Dependencies

### Minimal requirements

* `torch` (install the build appropriate for your platform)
* `networkx`
* `matplotlib`
* `numpy`

### CPU-only PyTorch

```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install networkx matplotlib numpy
```

### GPU PyTorch (example: CUDA 12.1 build)

> Pick the CUDA/ROCm build that matches your system. If unsure, check the official PyTorch install selector.

```bash
# Example for CUDA 12.1 wheels
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install networkx matplotlib numpy
```

> If you prefer, create a `requirements.txt` with:
>
> ```
> networkx
> matplotlib
> numpy
> ```
>
> and install PyTorch separately with the correct wheel for your system.

---

## Running Training & Evaluation

### CPU

```bash
python main.py --device cpu \
  --num_graphs 100 --pairs_per_graph 10 --epochs 10 \
  --edge_select sample --token_temp 1.0
```

### GPU

```bash
python main.py --device cuda \
  --num_graphs 100 --pairs_per_graph 10 --epochs 10 \
  --edge_select sample --token_temp 1.0
```

**Important flags**

* `--device {cpu|cuda}`: compute device.
* `--num_graphs`: how many base graphs to generate.
* `--pairs_per_graph`: number of (s,t) pairs per graph (each defines one datapoint with a GT path).
* `--epochs`: training epochs.
* `--edge_select {greedy|sample}`: how to pick **which masked edge** to reveal next during decoding.
* `--edge_select_temp`: temperature for **edge selection** when `--edge_select sample`.
* `--token_temp`: temperature when **sampling the token** for the chosen edge (from {1,2}).

**What happens when you run `main.py`**

1. Builds a dataset, runs basic sanity checks.
2. Trains the model and prints masked-edge accuracy.
3. Evaluates **Valid Path Rate** (fraction of test items whose predicted class-2 subgraph contains a simple path).
4. Visualizes **one** predicted test sample (base graph in grey, predicted path edges in orange dashed). Endpoints of the found path are highlighted.

---

## Visualizing More Samples

By default, `main.py` shows **one** example after evaluation. Quick options:

* **Rerun** `main.py` (it will visualize a different test item depending on shuffling/seed).
* **Change the seed**:

  ```bash
  python main.py --device cpu --seed 123 --edge_select sample --token_temp 0.9
  ```
* **(Optional) Edit `eval.py`**: set `did_vis = False` multiple times or add a counter to show more than one item.

---

## Sanity Checks (small, fast)

```bash
python sanity.py
```

This will:

* Validate dataset invariants (labels vs. adjacency, presence of 2’s).
* Run a forward/backward pass.
* Run decoding twice (greedy-edges+sampled-tokens, sampled-edges+sampled-tokens) and check structural properties.
* Compute a quick Valid Path Rate on a tiny loader.

---

## Common Variations

* **Graph density**: `--extra_p 0.02` (sparser) … `0.10` (denser).
* **Model capacity**: `--hidden`, `--layers`, `--edge_hidden`.
* **Sampling behaviour**: `--edge_select sample --edge_select_temp 0.7 --token_temp 0.8` for more randomness.
* **Tiny smoke test**: smaller graphs to verify everything runs:

  ```bash
  python main.py --device cpu --num_graphs 10 --pairs_per_graph 3 --epochs 2 --V 50
  ```

---

## What to Expect

* Training prints `loss`, `masked_acc`, and `val_masked_acc` per epoch.
* Evaluation prints `Test Valid Path Rate`.
* A plot window shows one sample with predicted path edges.

**Notes**

* During decoding, **non-edges stay 0**. We **never** predict 0 for an existing edge; tokens are sampled only from {1,2}.
* There are **no hard constraints** during decoding (no start/end, no degree caps). The metric only checks if **any** simple path exists in the predicted class-2 subgraph.

---

## Troubleshooting

* **PyTorch install issues**: choose the correct wheel for your OS/GPU (CPU-only build works everywhere).
* **Plots not showing** (headless servers): add `import matplotlib; matplotlib.use('Agg')` and save figures in `viz.py` instead of `plt.show()`.
* **Slow decode**: decoding touches each existing edge once. For much faster runs, try smaller graphs (`--V 50`) or denser GPU runs.

---

## License

MIT (feel free to adapt for your research demos).
