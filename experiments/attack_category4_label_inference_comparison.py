"""Direct comparison: category 4's label-free backdoor vs the standard
label-inference-then-backdoor pattern from prior VFL literature.

Trains one backdoored model (num_plants=1, the stealthy headline config),
then asks: what would a passive party need to do to build the SAME kind of
attack the "standard" way (infer labels first, then target by class)?
Measures the inference accuracy/coverage that approach achieves given a
small auxiliary seed of known labels, and contrasts it with category 4's
approach, which needs neither labels nor a seed.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.passive_party_backdoor import BackdoorPassiveParty
from attacks.label_inference import infer_labels_via_leaf_comembership, inference_accuracy

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "magenta": "#e87ba4", "blue": "#2a78d6", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb", "yellow": "#eda100",
}

TRIGGER_FEATURE, TRIGGER_BIN, N_BINS = 1, 8, 16
N_ROUNDS = 10
SEED_FRACTIONS = [0.02, 0.05, 0.10, 0.20]


def main():
    X, y, groups = load_adult()
    party0_cols, party1_cols = groups["party0"], groups["party1"]

    rng = np.random.default_rng(0)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=0, stratify=y_sub
    )

    print("Training the headline backdoored model (num_plants=1) ...")
    model = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    factories = {"party1": partial(BackdoorPassiveParty, trigger_feature=TRIGGER_FEATURE,
                                    trigger_bin=TRIGGER_BIN, scale_factor=10.0, num_plants=1)}
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
    auc = roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))
    print(f"Backdoored model test AUC: {auc:.4f}\n")

    n_train = len(y_tr)
    accuracies, coverages = [], []
    for seed_frac in SEED_FRACTIONS:
        rng2 = np.random.default_rng(1)
        n_seed = int(seed_frac * n_train)
        seed_idx = rng2.choice(n_train, size=n_seed, replace=False)
        seed_labels = {int(i): int(y_tr[i]) for i in seed_idx}

        inferred = infer_labels_via_leaf_comembership(model.trees, seed_labels, n_train)
        acc, cov = inference_accuracy(inferred, y_tr, seed_labels)
        accuracies.append(acc)
        coverages.append(cov)
        print(f"seed={seed_frac:.0%} of train set ({n_seed} samples): "
              f"inference accuracy={acc:.1%}, coverage={cov:.1%} of remaining samples")

    print("\n=== Comparison ===")
    print(f"Category-4 (label-free): needs 0 labels, 0 auxiliary data. "
          f"Flip rate on feature-threshold-selected targets: 14.4% (from attack_category4_backdoor.py)")
    print(f"Label-inference baseline: needs an auxiliary seed of TRUE labels "
          f"({SEED_FRACTIONS[0]:.0%}-{SEED_FRACTIONS[-1]:.0%} of train set tried here) "
          f"to achieve {accuracies[0]:.0%}-{accuracies[-1]:.0%} inference accuracy on the rest, "
          f"with only {coverages[0]:.0%}-{coverages[-1]:.0%} coverage (samples that never share a leaf "
          f"with a seed sample are never labeled at all) -- and still needs THAT before it can even "
          f"attempt a targeted-by-class backdoor.")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax1.set_facecolor(PALETTE["surface"])

    x = [f"{s:.0%}" for s in SEED_FRACTIONS]
    ax1.bar([f"{s} seed" for s in x], accuracies, color=PALETTE["blue"], width=0.35,
            label="Inference accuracy")
    ax1.set_ylabel("Label-inference accuracy", color=PALETTE["blue"])
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax1.axhline(1.0, color=PALETTE["muted"], linewidth=1, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(range(len(SEED_FRACTIONS)), coverages, color=PALETTE["yellow"], linewidth=2,
             marker="o", markersize=7, label="Coverage of non-seed samples")
    ax2.set_ylabel("Coverage (fraction ever inferred)", color=PALETTE["yellow"])
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis="y", labelcolor=PALETTE["yellow"])

    ax1.set_title("Label-inference baseline's cost, vs. category 4's zero-label attack\n"
                  "(category 4 needs neither an auxiliary seed nor any inference step)",
                  color=PALETTE["ink"], fontsize=10)
    for spine in ["top"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right", fontsize=8,
               labelcolor=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "category4_label_inference_comparison.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
