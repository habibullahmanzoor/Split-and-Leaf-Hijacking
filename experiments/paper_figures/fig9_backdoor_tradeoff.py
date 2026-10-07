"""Figure 9 (Section 5.3, label-inference-free backdoor): the attacker's
stealth-vs-success dial (num_plants, how many tree-root/node splits get the
same trigger re-planted), five seeds per point, as a mean +/- std Pareto
curve. Reuses verify_backdoor_tradeoff_multiseed.py's run_once so the
figure is generated from the same measured trials reported in the text,
not a re-derivation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fed_datasets.loaders import load_adult
from verify_backdoor_tradeoff_multiseed import run_once, PLANTS_SWEEP, N_TRIALS, N_ROUNDS
import plotstyle as ps


def main():
    X, y, groups = load_adult()

    def compute():
        means_cost_, stds_cost_, means_flip_, stds_flip_ = [], [], [], []
        for num_plants in PLANTS_SWEEP:
            honest_trials, backdoor_trials, flip_trials = [], [], []
            for t in range(N_TRIALS):
                auc_h, auc_b, fr = run_once(X, y, groups, num_plants, seed=t)
                honest_trials.append(auc_h)
                backdoor_trials.append(auc_b)
                flip_trials.append(fr)
            cost_trials = [h - b for h, b in zip(honest_trials, backdoor_trials)]
            means_cost_.append(float(np.mean(cost_trials)))
            stds_cost_.append(float(np.std(cost_trials, ddof=1)))
            means_flip_.append(float(np.mean(flip_trials)))
            stds_flip_.append(float(np.std(flip_trials, ddof=1)))
        return {"means_cost": means_cost_, "stds_cost": stds_cost_,
                "means_flip": means_flip_, "stds_flip": stds_flip_}

    cached = ps.cached("fig9_backdoor", compute)
    means_cost, stds_cost = cached["means_cost"], cached["stds_cost"]
    means_flip, stds_flip = cached["means_flip"], cached["stds_flip"]
    for num_plants, mc, sc, mf, sf in zip(PLANTS_SWEEP, means_cost, stds_cost, means_flip, stds_flip):
        print(f"num_plants={num_plants:>2d}  AUC_cost={mc:.4f}+/-{sc:.4f}  flip_rate={mf:.4f}+/-{sf:.4f}")

    fig, ax = ps.new_fig(wide=True, height=3.6)

    ax.errorbar(means_cost, means_flip, xerr=stds_cost, yerr=stds_flip,
                color=ps.C1, linewidth=1.6, marker="o", markersize=6,
                capsize=2.5, ecolor=ps.MUTED, elinewidth=0.9, zorder=4)

    # hand-placed label positions: points 1-3 sit in a cramped low-cost
    # cluster, so each gets its own offset plus a thin leader line rather
    # than a uniform offset that would stack the labels on top of each other
    label_pos = [(0.048, 0.100), (0.048, 0.060), (0.048, 0.020),
                 (0.078, 0.082), (0.128, 0.152), (0.300, 0.152)]
    for i, np_ in enumerate(PLANTS_SWEEP):
        lx, ly = label_pos[i]
        ax.annotate(f"{np_} plant" + ("s" if np_ > 1 else ""),
                    xy=(means_cost[i], means_flip[i]), xytext=(lx, ly),
                    fontsize=7, color=ps.SECONDARY,
                    arrowprops=dict(arrowstyle="-", color=ps.GRID, lw=0.7),
                    ha="left", va="center")

    ax.axhline(means_flip[-1], color=ps.MUTED, linewidth=0.9, linestyle=":")
    ax.annotate("8 and 13 plants land on the SAME flip rate --\nextra plants past 8 buy zero additional benefit,\nonly more collateral accuracy damage",
                xy=(means_cost[-1], means_flip[-1]), xytext=(0.16, means_flip[-1] - 0.085),
                fontsize=7.3, color=ps.SECONDARY, style="italic",
                arrowprops=dict(arrowstyle="->", color=ps.MUTED, lw=0.9))

    ax.set_xlabel("Accuracy cost (honest AUC minus backdoored AUC)")
    ax.set_ylabel("Flip rate")
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(-0.02, 0.32)
    ps.style_axes(ax, legend=False)

    caption = (f"The label inference free backdoor's stealth-vs-success dial: "
               f"re-planting the trigger at more tree nodes (num\\_plants, {N_TRIALS} seeds "
               f"each, n\\_rounds={N_ROUNDS}) buys a higher targeted flip rate only up to a "
               f"measured ceiling ({means_flip[-1]*100:.1f}\\%, reached at 8 plants and "
               f"unchanged at 13), beyond which additional plants cost accuracy with no "
               f"further targeted benefit.")
    ps.save_fig(fig, "fig9_backdoor_tradeoff", caption)


if __name__ == "__main__":
    main()
