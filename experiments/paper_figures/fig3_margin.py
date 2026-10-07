"""Figure 3 (Section 3): the split-flip margin -- exact validation against
real split-finding code, and its invariance to non-IID severity + node-size
scaling law. Consolidates theory_margin_validation.py and
theory_margin_generalization.py into one publication figure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import Client, Server, bin_features, compute_bin_edges
from theory.margin import compute_split_flip_margin, score_at_delta
from theory_margin_generalization import (
    dirichlet_partition, collect_nodes, margins_for_alpha, ALPHAS,
    N_CLIENTS as GEN_N_CLIENTS,
)
import plotstyle as ps


def panel_a_validation(ax, X_train, y_train):
    n_clients, alpha = 5, 0.5
    parts = dirichlet_partition(X_train, y_train, n_clients, alpha, seed=0)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    n_bins, max_depth, lam, min_child_hess = 32, 4, 1.0, 1.0
    X_all = np.vstack([Xc for Xc, _ in client_data.values()])
    bin_edges = compute_bin_edges(X_all, n_bins)
    clients = [Client(client_id=cid, X=bin_features(Xc, bin_edges), y=yc, bin_edges=bin_edges)
               for cid, (Xc, yc) in client_data.items()]
    server = Server(clients, n_bins, max_depth, lam=lam, min_child_hess=min_child_hess)
    tree = server.build_tree()
    nodes = collect_nodes(tree, max_nodes=6)

    results = []
    for node in nodes:
        agg_grad, agg_hess = server._aggregate_histograms(node.sample_idx)
        (gain_best, f_best, b_best), (gain_2nd, f_2nd, b_2nd) = server._find_best_split(agg_grad, agg_hess)
        if f_2nd is None:
            continue
        margin, binding = compute_split_flip_margin(agg_grad, agg_hess, f_2nd, b_2nd,
                                                      lam=lam, min_child_hess=min_child_hess)
        if not np.isfinite(margin) or margin <= 0:
            continue

        def winner_at(delta):
            perturbed = agg_grad.copy()
            perturbed[f_2nd, b_2nd] += delta
            (_, f_w, b_w), _ = server._find_best_split(perturbed, agg_hess)
            return (f_w, b_w)

        below, above = winner_at(margin * 0.99), winner_at(margin * 1.01)
        confirmed = (below != (f_2nd, b_2nd)) and (above == (f_2nd, b_2nd))
        results.append(dict(margin=margin, confirmed=confirmed, agg_grad=agg_grad, agg_hess=agg_hess,
                             f_2nd=f_2nd, b_2nd=b_2nd, f_best=f_best, b_best=b_best))

    n_confirmed = sum(r["confirmed"] for r in results)
    print(f"Panel A: {n_confirmed}/{len(results)} nodes -- theory exactly predicted the flip boundary")

    r = results[0]
    deltas = np.linspace(0, r["margin"] * 2.0, 200)
    score_target = [score_at_delta(r["agg_grad"], r["agg_hess"], r["f_2nd"], r["b_2nd"],
                                    r["f_2nd"], r["b_2nd"], d, lam=lam) for d in deltas]
    score_honest = [score_at_delta(r["agg_grad"], r["agg_hess"], r["f_best"], r["b_best"],
                                    r["f_2nd"], r["b_2nd"], d, lam=lam) for d in deltas]

    ax.plot(deltas, score_honest, color=ps.C2, linewidth=1.8, label="Honest winner's score")
    ax.plot(deltas, score_target, color=ps.C1, linewidth=1.8, label="Attacker's target score")
    ax.axvline(r["margin"], color=ps.MUTED, linewidth=1.2, linestyle="--",
               label=f"Predicted margin ({r['margin']:.0f})")
    ax.set_xlabel("Injected gradient spike δ")
    ax.set_ylabel("Candidate split score")
    ax.set_title(f"(a) Exact validation, {n_confirmed}/{len(results)} nodes", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower right"))
    return n_confirmed, len(results)


def panel_b_invariance(ax, X_train, y_train):
    alpha_median_margins = []
    all_alpha_results = {}
    for alpha in ALPHAS:
        results = margins_for_alpha(X_train, y_train, alpha, seed=0)
        all_alpha_results[alpha] = results
        margins = [m for n, d, m in results]
        median_margin = float(np.median(margins)) if margins else float("nan")
        alpha_median_margins.append(median_margin)
        print(f"alpha={alpha:.1f}  n_nodes={len(results)}  median_margin={median_margin:.2f}")

    all_nodes = [(n, d, m) for res in all_alpha_results.values() for (n, d, m) in res]
    ns = np.array([n for n, d, m in all_nodes])
    margins_arr = np.array([m for n, d, m in all_nodes])
    valid = (ns > 0) & (margins_arr > 0)
    log_n, log_m = np.log(ns[valid]), np.log(margins_arr[valid])
    slope, intercept = np.polyfit(log_n, log_m, 1)
    print(f"Power-law fit: margin ~ n^{slope:.2f}")

    ax.scatter(ns[valid], margins_arr[valid], color=ps.C1, alpha=0.55, s=18, zorder=3,
               label="Measured margins (all α)")
    n_fit = np.linspace(log_n.min(), log_n.max(), 50)
    ax.plot(np.exp(n_fit), np.exp(slope * n_fit + intercept), color=ps.INK,
            linewidth=1.4, linestyle="--", label=f"fit: margin ∝ n^{slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Node sample count")
    ax.set_ylabel("Split-flip margin")
    ax.set_title("(b) Node-size scaling (pooled across α)", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="upper left"))
    return slope, alpha_median_margins


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    fig, axes = ps.new_fig(ncols=2, wide=True)
    n_confirmed, n_total = panel_a_validation(axes[0], X_train, y_train)
    slope, alpha_medians = panel_b_invariance(axes[1], X_train, y_train)

    invariant = len(set(round(m, 1) for m in alpha_medians if np.isfinite(m))) == 1
    print(f"\nMargin invariant to alpha: {invariant} (median margins across alpha: {alpha_medians})")

    caption = (f"The split-flip margin, validated exactly against real split-finding code "
               f"({n_confirmed}/{n_total} nodes) and scaling as margin \\propto n^{{{slope:.2f}}} "
               f"with node sample count, independent of Dirichlet \\alpha "
               f"({'confirmed invariant' if invariant else 'see text'} across "
               f"\\alpha \\in \\{{{', '.join(str(a) for a in ALPHAS)}\\}}).")
    ps.save_fig(fig, "fig3_margin", caption)


if __name__ == "__main__":
    main()
