"""Does category-3's saturation finding (test AUC 0.855->0.777 at scale=2x,
flat through 200x) survive once the VFL model is actually trained to
convergence, instead of the paper's standard n_rounds=3?

Figure 0's extended convergence check (fig0_baseline_convergence.py) showed
Adult (VFL) test AUC keeps climbing from 0.855 at round 3 to ~0.890 by round
20-24 before plateauing -- i.e. every VFL experiment in the paper, including
this attack's own headline sweep, was run on an undertrained model. This
script reruns the exact same sweep at n_rounds=20 (captures the vast
majority of the convergence gain; round 20 test AUC 0.8899 vs. round-24 peak
0.8909, diminishing returns past that) to check whether the attack's
saturation shape is a property of the mechanism or an artifact of
undertraining.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fed_datasets.loaders import load_adult
from attack_category3_he_sweep import run_once

N_TRIALS = 5
SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
N_ROUNDS_CONVERGED = 20


def main():
    X, y, groups = load_adult()

    print(f"=== category-3 sweep at n_rounds=3 (paper's standard config) ===")
    means_3, stds_3 = [], []
    for sf in SCALE_FACTORS:
        aucs = [run_once(X, y, groups, sf, seed=t, n_rounds=3) for t in range(N_TRIALS)]
        means_3.append(float(np.mean(aucs)))
        stds_3.append(float(np.std(aucs, ddof=1)))
        print(f"  scale={sf:>6.1f}  AUC={means_3[-1]:.4f}+/-{stds_3[-1]:.4f}")

    print(f"\n=== category-3 sweep at n_rounds={N_ROUNDS_CONVERGED} (convergence-respecting) ===")
    means_c, stds_c = [], []
    for sf in SCALE_FACTORS:
        aucs = [run_once(X, y, groups, sf, seed=t, n_rounds=N_ROUNDS_CONVERGED) for t in range(N_TRIALS)]
        means_c.append(float(np.mean(aucs)))
        stds_c.append(float(np.std(aucs, ddof=1)))
        print(f"  scale={sf:>6.1f}  AUC={means_c[-1]:.4f}+/-{stds_c[-1]:.4f}")

    print("\n=== comparison ===")
    honest_drop_3 = means_3[0] - means_3[1]
    honest_drop_c = means_c[0] - means_c[1]
    print(f"n_rounds=3:  honest={means_3[0]:.4f}  at scale=2x={means_3[1]:.4f}  drop={honest_drop_3:.4f}")
    print(f"n_rounds={N_ROUNDS_CONVERGED}: honest={means_c[0]:.4f}  at scale=2x={means_c[1]:.4f}  drop={honest_drop_c:.4f}")
    saturates_3 = all(abs(means_3[i] - means_3[1]) < 2 * (stds_3[i] + stds_3[1]) for i in range(1, len(means_3)))
    saturates_c = all(abs(means_c[i] - means_c[1]) < 2 * (stds_c[i] + stds_c[1]) for i in range(1, len(means_c)))
    print(f"Saturates from scale=2x onward at n_rounds=3: {saturates_3}")
    print(f"Saturates from scale=2x onward at n_rounds={N_ROUNDS_CONVERGED}: {saturates_c}")


if __name__ == "__main__":
    main()
