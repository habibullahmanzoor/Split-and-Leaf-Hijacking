"""Validates the split-flip margin theory (src/theory/margin.py) against
the ACTUAL split-finding code (Server._find_best_split), not a re-derivation
of it -- the strongest form of check. For several real nodes from an honest
training run (root + deeper nodes at various depths), we compute the
closed-form margin needed to flip the winning split to that node's current
runner-up, then perturb the real histogram by slightly less and slightly
more than that margin and confirm the split-finder's actual winner changes
exactly where predicted.
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
from theory.margin import compute_split_flip_margin, score_at_delta

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#2a78d6", "red": "#e34948", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}


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
    return [np.array(idx) for idx in client_idx]


def collect_nodes(tree, max_nodes=6):
    """BFS collect internal (non-leaf) nodes, spread across depths."""
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


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    n_clients, alpha = 5, 0.5
    parts = dirichlet_partition(X_train, y_train, n_clients, alpha, seed=0)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    n_bins, max_depth, lam, min_child_hess = 32, 4, 1.0, 1.0
    X_all = np.vstack([Xc for Xc, _ in client_data.values()])
    bin_edges = compute_bin_edges(X_all, n_bins)
    clients = [Client(client_id=cid, X=bin_features(Xc, bin_edges), y=yc, bin_edges=bin_edges)
               for cid, (Xc, yc) in client_data.items()]
    server = Server(clients, n_bins, max_depth, lam=lam, min_child_hess=min_child_hess)

    print("Building one honest tree to collect real internal nodes ...")
    tree = server.build_tree()
    nodes = collect_nodes(tree, max_nodes=6)
    print(f"Collected {len(nodes)} internal nodes across the tree.\n")

    results = []
    for i, node in enumerate(nodes):
        agg_grad, agg_hess = server._aggregate_histograms(node.sample_idx)
        (gain_best, f_best, b_best), (gain_2nd, f_2nd, b_2nd) = server._find_best_split(agg_grad, agg_hess)
        if f_2nd is None:
            continue

        margin, binding = compute_split_flip_margin(agg_grad, agg_hess, f_2nd, b_2nd,
                                                      lam=lam, min_child_hess=min_child_hess)
        if not np.isfinite(margin) or margin <= 0:
            print(f"Node {i} (depth {node.depth}): margin is inf or 0 -- skipping (degenerate case)")
            continue

        def winner_at(delta):
            perturbed = agg_grad.copy()
            perturbed[f_2nd, b_2nd] += delta
            (_, f_w, b_w), _ = server._find_best_split(perturbed, agg_hess)
            return (f_w, b_w)

        below = winner_at(margin * 0.99)
        above = winner_at(margin * 1.01)
        flips_correctly = (below != (f_2nd, b_2nd)) and (above == (f_2nd, b_2nd))

        print(f"Node {i} (depth {node.depth}): honest winner=(f{f_best},b{b_best}) "
              f"runner-up=(f{f_2nd},b{b_2nd})  theoretical margin={margin:.2f}")
        print(f"  at 0.99*margin -> winner {below} {'(still honest, correct)' if below != (f_2nd,b_2nd) else '(ALREADY FLIPPED, unexpected)'}")
        print(f"  at 1.01*margin -> winner {above} {'(flipped to target, correct)' if above == (f_2nd,b_2nd) else '(NOT flipped, unexpected)'}")
        print(f"  theory prediction confirmed: {flips_correctly}\n")

        results.append(dict(node=i, depth=node.depth, margin=margin, confirmed=flips_correctly,
                             agg_grad=agg_grad, agg_hess=agg_hess, f_2nd=f_2nd, b_2nd=b_2nd,
                             f_best=f_best, b_best=b_best))

    n_confirmed = sum(r["confirmed"] for r in results)
    print(f"=== {n_confirmed}/{len(results)} nodes: theory exactly predicted the flip boundary ===")

    if not results:
        print("No valid nodes to plot.")
        return

    r = results[0]
    deltas = np.linspace(0, r["margin"] * 2.0, 200)
    score_target = [score_at_delta(r["agg_grad"], r["agg_hess"], r["f_2nd"], r["b_2nd"],
                                    r["f_2nd"], r["b_2nd"], d, lam=lam) for d in deltas]
    score_honest = [score_at_delta(r["agg_grad"], r["agg_hess"], r["f_best"], r["b_best"],
                                    r["f_2nd"], r["b_2nd"], d, lam=lam) for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(deltas, score_honest, color=PALETTE["blue"], linewidth=2, label="Honest winner's score")
    ax.plot(deltas, score_target, color=PALETTE["red"], linewidth=2, label="Attacker's target split score")
    ax.axvline(r["margin"], color=PALETTE["muted"], linewidth=1.5, linestyle="--",
               label=f"Theoretical margin = {r['margin']:.1f}")
    ax.set_xlabel("Gradient spike magnitude δ injected at target (feature, bin)", color=PALETTE["ink"])
    ax.set_ylabel("Candidate split score (↑ wins argmax)", color=PALETTE["ink"])
    ax.set_title("Split-flip margin theory validated against real training-run histograms\n"
                  "(Adult, HFL, node from an honest tree) -- crossover matches closed-form prediction",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "theory_margin_validation.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
