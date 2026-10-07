"""Figure 12 (Discussion, "Attack node coverage in the horizontal setting"):
category 1's headline attack (margin_flip_attack) is restricted to
attack_max_depth=1 (root node only) throughout this paper. This tests
whether that restriction, not the fully-informed adaptive targeting logic
itself, is what limits damage: sweeping attack_max_depth from 1 to each
dataset's own max_depth, on Adult, Credit Default, and Heart Disease
(clinical), 5 seeds each. Heart Disease uses a shallower tree (max_depth=3)
than the other two (max_depth=4) elsewhere in this paper, so its own
attack-depth sweep stops at 3 rather than 4. Consolidates
verify_cat1_attack_depth.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from verify_cat1_attack_depth import run_once, DATASETS, N_TRIALS
from sklearn.model_selection import train_test_split
import plotstyle as ps


def main():
    fig, ax = ps.new_fig(wide=True, height=3.6)
    colors = {"adult": ps.C1, "credit_default": ps.C4, "heart_disease": ps.C2}
    labels = {"adult": "Adult (census)", "credit_default": "Credit Default (financial)",
              "heart_disease": "Heart Disease (clinical)"}

    summary = {}
    for name, cfg in DATASETS.items():
        X, y, _ = cfg["loader"]()
        X = X.to_numpy(dtype=float)
        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        depths = cfg["attack_depths"]

        def compute():
            means_, stds_, honest_mean_ = [], [], None
            for depth in depths:
                honest_t, attacked_t = [], []
                for t in range(N_TRIALS):
                    ah, aa = run_once(X_train, y_train, X_test, y_test, cfg["n_bins"], cfg["max_depth"],
                                       cfg["n_rounds"], depth, seed=t)
                    honest_t.append(ah)
                    attacked_t.append(aa)
                means_.append(float(np.mean(attacked_t)))
                stds_.append(float(np.std(attacked_t, ddof=1)))
                honest_mean_ = float(np.mean(honest_t))
            return {"means": means_, "stds": stds_, "honest_mean": honest_mean_}

        cached = ps.cached(f"fig12_{name}", compute)
        means, stds, honest_mean = cached["means"], cached["stds"], cached["honest_mean"]
        summary[name] = (honest_mean, means, stds, depths)
        print(f"{name}: honest={honest_mean:.4f}  " +
              "  ".join(f"depth{d}={m:.4f}+/-{s:.4f}" for d, m, s in zip(depths, means, stds)))

        ax.errorbar(depths, means, yerr=stds, color=colors[name], linewidth=1.8,
                    marker="o", markersize=6, capsize=3, label=f"{labels[name]} (honest={honest_mean:.3f})")
        ax.axhline(honest_mean, color=colors[name], linewidth=0.8, linestyle=":", alpha=0.6)

    ax.axhline(0.5, color=ps.MUTED, linewidth=1, linestyle="--")
    ax.annotate("random guessing", xy=(2.5, 0.5), xytext=(2.5, 0.525),
                fontsize=7.5, color=ps.SECONDARY, style="italic", ha="center")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Attack node coverage (attack_max_depth)")
    ax.set_ylabel("Test AUC")
    ps.style_axes(ax, legend_kwargs=dict(loc="center left", fontsize=7.5))

    a_h, a_m, a_s, a_d = summary["adult"]
    c_h, c_m, c_s, c_d = summary["credit_default"]
    h_h, h_m, h_s, h_d = summary["heart_disease"]
    caption = (f"The paper's headline category 1 magnitude (attack\\_max\\_depth=1, root node only) "
               f"is a measurement at a specific, conservative attack radius, not the mechanism's "
               f"damage ceiling, on any dataset: extending coverage to every node collapses Adult and "
               f"Credit Default to exactly random guessing (Adult {a_m[0]:.4f}$\\to${a_m[-1]:.4f}, "
               f"Credit Default {c_m[0]:.4f}$\\to${c_m[-1]:.4f}, zero variance across 5 seeds at full "
               f"coverage) and Heart Disease below it ({h_m[0]:.4f}$\\to${h_m[-1]:.4f}$\\pm${h_s[-1]:.4f}, "
               f"an overshoot we attribute to its roughly sixty row test set rather than a different "
               f"mechanism). Credit Default's apparent non-generalization at depth 1 is this same "
               f"scoping effect, not a targeting-quality failure, since margin\\_flip\\_attack already "
               f"uses the fully informed adaptive targeting of Section~\\ref{{sec:threatmodel}}.")
    ps.save_fig(fig, "fig12_attack_depth_scoping", caption)


if __name__ == "__main__":
    main()
