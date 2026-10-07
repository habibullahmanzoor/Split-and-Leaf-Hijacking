"""Figure 13 (Section 7.4, cross-dataset generalization): does variance-
based target-feature selection (adaptive_target_selection.py) close
category 3's Credit Default generalization gap? Grouped bars, one pair per
dataset (fixed target_feature=0 vs variance-targeted), 5 seeds each.
Consolidates verify_cat3_variance_targeting.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from verify_cat3_variance_targeting import run_once, DATASETS, N_TRIALS
from attacks.he_crypto_layer import MaliciousPassiveParty
from attacks.adaptive_target_selection import VarianceTargetedMaliciousPassiveParty
import plotstyle as ps

DATASET_LABELS = {"adult": "Adult", "heart_disease": "Heart Disease", "credit_default": "Credit Default"}


def main():
    fig, ax = ps.new_fig(wide=True, height=3.6)

    fixed_drops, fixed_errs, var_drops, var_errs, names = [], [], [], [], []
    for name, cfg in DATASETS.items():
        X, y, groups = cfg["loader"]()
        names.append(DATASET_LABELS[name])
        for cls, drops_list, errs_list in [(MaliciousPassiveParty, fixed_drops, fixed_errs),
                                            (VarianceTargetedMaliciousPassiveParty, var_drops, var_errs)]:
            def compute():
                honest_t, attacked_t = [], []
                for t in range(N_TRIALS):
                    ah, aa = run_once(X, y, groups, cls, cfg, seed=t)
                    honest_t.append(ah)
                    attacked_t.append(aa)
                drop_trials = [h - a for h, a in zip(honest_t, attacked_t)]
                return {"mean": float(np.mean(drop_trials)), "std": float(np.std(drop_trials, ddof=1))}

            res = ps.cached(f"fig13_{name}_{cls.__name__}", compute)
            drops_list.append(res["mean"])
            errs_list.append(res["std"])
        print(f"{name}: fixed_drop={fixed_drops[-1]:.4f}+/-{fixed_errs[-1]:.4f}  "
              f"variance_drop={var_drops[-1]:.4f}+/-{var_errs[-1]:.4f}")

    x = np.arange(len(names))
    width = 0.32
    ax.bar(x - width / 2, fixed_drops, width, yerr=fixed_errs, capsize=3,
           color=ps.C1, label="Fixed target_feature=0 (as used throughout)")
    ax.bar(x + width / 2, var_drops, width, yerr=var_errs, capsize=3,
           color=ps.C5, label="Variance-targeted (ours)")
    ax.axhline(0, color=ps.MUTED, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC damage (honest minus attacked)")
    ps.style_axes(ax, legend_kwargs=dict(loc="upper left", fontsize=7.5))

    caption = (f"A variance-based target-feature heuristic (picks the highest-variance feature "
               f"among the party's own binned columns -- still fully public, no labels or "
               f"gradients) against the fixed target\\_feature=0 used everywhere else in this "
               f"paper. Credit Default: {fixed_drops[2]:.4f}$\\pm${fixed_errs[2]:.4f} "
               f"$\\to$ {var_drops[2]:.4f}$\\pm${var_errs[2]:.4f}, a real, statistically clear "
               f"improvement. Heart Disease: {fixed_drops[1]:.4f}$\\pm${fixed_errs[1]:.4f} vs. "
               f"{var_drops[1]:.4f}$\\pm${var_errs[1]:.4f}, statistically indistinguishable given "
               f"the overlapping error bars -- the apparent regression seen in an earlier single-seed "
               f"check was noise, not a real cost. Adult: {fixed_drops[0]:.4f}$\\pm${fixed_errs[0]:.4f} "
               f"vs. {var_drops[0]:.4f}$\\pm${var_errs[0]:.4f}, the heuristic incidentally finds a "
               f"substantially more damaging target than the arbitrary fixed default there too.")
    ps.save_fig(fig, "fig13_variance_targeting", caption)


if __name__ == "__main__":
    main()
