"""Generalizes the split-flip margin theory: does the margin (minimum
perturbation needed to flip a split) shrink as non-IID severity increases,
and how does it scale with node sample count? Tests the original thesis
empirically -- "the same heterogeneity that makes federated trees
efficient is what makes them cheap to attack" -- using the validated
compute_split_flip_margin machinery directly against real training-run
histograms, not a fresh derivation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import Client, Server, bin_features, compute_bin_edges
from theory.margin import compute_split_flip_margin

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#2a78d6", "red": "#e34948", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

N_CLIENTS, N_BINS, MAX_DEPTH, LAM, MIN_CHILD_HESS = 5, 32, 5, 1.0, 1.0
ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def dirichlet_partition(X, y, n_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    client_idx = [[] for _ in range(n_clients)]
    for idx in idx_by_class:
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, splits)):
            client_idx[cid].extend(part.tolist())
    return [np.array(idx, dtype=int) for idx in client_idx]


def collect_nodes(tree, max_nodes=12):
    collected = []
    frontier = [tree]
    while frontier and len(collected) < max_nodes:
        node = frontier.pop(0)
        if not node.is_leaf:
            collected.append(node)
            if node.left is not None:
                frontier.append(node.left)
            if node.right is not None:
                frontier.append(node.right)
    return collected


def margins_for_alpha(X_train, y_train, alpha, seed=0):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, alpha, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    X_all = np.vstack([Xc for Xc, _ in client_data.values()])
    bin_edges = compute_bin_edges(X_all, N_BINS)
    clients = [Client(client_id=cid, X=bin_features(Xc, bin_edges), y=yc, bin_edges=bin_edges)
               for cid, (Xc, yc) in client_data.items()]
    server = Server(clients, N_BINS, MAX_DEPTH, lam=LAM, min_child_hess=MIN_CHILD_HESS)

    tree = server.build_tree()
    nodes = collect_nodes(tree, max_nodes=12)

    results = []
    for node in nodes:
        agg_grad, agg_hess = server._aggregate_histograms(node.sample_idx)
        (gain, f1, b1), (gain2, f2, b2) = server._find_best_split(agg_grad, agg_hess)
        if f2 is None:
            continue
        try:
            margin, _ = compute_split_flip_margin(agg_grad, agg_hess, f2, b2,
                                                    lam=LAM, min_child_hess=MIN_CHILD_HESS)
        except ValueError:
            continue
        if not np.isfinite(margin):
            continue
        n_samples = sum(len(v) for v in node.sample_idx.values())
        results.append((n_samples, node.depth, margin))
    return results


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    alpha_median_margins = []
    all_alpha_results = {}
    for alpha in ALPHAS:
        results = margins_for_alpha(X_train, y_train, alpha, seed=0)
        all_alpha_results[alpha] = results
        margins = [m for n, d, m in results]
        median_margin = float(np.median(margins)) if margins else float("nan")
        alpha_median_margins.append(median_margin)
        print(f"alpha={alpha:.1f}  n_nodes={len(results)}  median_margin={median_margin:.2f}  "
              f"(range {min(margins) if margins else float('nan'):.1f}-"
              f"{max(margins) if margins else float('nan'):.1f})")

    # node-size relationship: pool nodes across all alphas, plot margin vs n_samples
    all_nodes = [(n, d, m) for results in all_alpha_results.values() for (n, d, m) in results]
    ns = np.array([n for n, d, m in all_nodes])
    margins_arr = np.array([m for n, d, m in all_nodes])
    valid = (ns > 0) & (margins_arr > 0)
    log_n, log_m = np.log(ns[valid]), np.log(margins_arr[valid])
    if len(log_n) > 2:
        slope, intercept = np.polyfit(log_n, log_m, 1)
        print(f"\nPower-law fit across all nodes/alphas: margin ~ n^{slope:.2f} "
              f"(intercept exp={np.exp(intercept):.3f})")
    else:
        slope, intercept = float("nan"), float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])

    ax0 = axes[0]
    ax0.set_facecolor(PALETTE["surface"])
    ax0.plot(ALPHAS, alpha_median_margins, color=PALETTE["blue"], linewidth=2, marker="o", markersize=6)
    ax0.set_xscale("log")
    ax0.set_xlabel("Dirichlet α (lower = more non-IID)", color=PALETTE["ink"])
    ax0.set_ylabel("Median split-flip margin", color=PALETTE["ink"])
    ax0.set_title("Margin vs. non-IID severity", color=PALETTE["ink"], fontsize=10)
    ax0.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.tick_params(colors=PALETTE["secondary"])

    ax1 = axes[1]
    ax1.set_facecolor(PALETTE["surface"])
    ax1.scatter(ns[valid], margins_arr[valid], color=PALETTE["red"], alpha=0.6, s=30)
    if len(log_n) > 2:
        n_fit = np.linspace(log_n.min(), log_n.max(), 50)
        ax1.plot(np.exp(n_fit), np.exp(slope * n_fit + intercept), color=PALETTE["ink"],
                  linewidth=1.5, linestyle="--", label=f"fit: margin ∝ n^{slope:.2f}")
        ax1.legend(frameon=False, fontsize=8, labelcolor=PALETTE["secondary"])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Node sample count (log scale)", color=PALETTE["ink"])
    ax1.set_ylabel("Split-flip margin (log scale)", color=PALETTE["ink"])
    ax1.set_title("Margin vs. node size (pooled across all α)", color=PALETTE["ink"], fontsize=10)
    ax1.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(colors=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "theory_margin_generalization.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
