"""Stealth-vs-success tradeoff for category 4: sweep num_plants (how many
tree roots get the same trigger re-planted) instead of reporting a single
point. More plants -> more of the ensemble routes through the trigger
(higher, more reliable flip rate) at the cost of more accuracy damage
(less stealthy). This characterizes the dial an attacker actually has,
rather than one lucky configuration.
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

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "magenta": "#e87ba4", "blue": "#2a78d6", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

TRIGGER_FEATURE, TRIGGER_BIN, N_BINS = 1, 8, 16
N_ROUNDS = 10


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

    print("Training honest baseline ...")
    honest = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, {"party1": Xp_te}))
    print(f"Honest baseline (n_rounds={N_ROUNDS}): test AUC = {auc_honest:.4f}\n")

    trigger_edges_cache = None
    plants_sweep = [1, 2, 3, 5, 8]
    aucs, flip_rates = [], []

    for num_plants in plants_sweep:
        model = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
        factories = {"party1": partial(BackdoorPassiveParty, trigger_feature=TRIGGER_FEATURE,
                                        trigger_bin=TRIGGER_BIN, scale_factor=10.0, num_plants=num_plants)}
        model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
        proba_clean = model.predict_proba(Xa_te, {"party1": Xp_te})
        auc = roc_auc_score(y_te, proba_clean)

        trigger_edges = model.passive_bin_edges["party1"][TRIGGER_FEATURE]
        threshold_value = trigger_edges[min(TRIGGER_BIN, len(trigger_edges) - 1)]
        below_mask = Xp_te[:, TRIGGER_FEATURE] <= threshold_value
        target_idx = np.where(below_mask)[0]
        preds_before = (proba_clean[target_idx] >= 0.5).astype(int)

        Xp_te_manip = Xp_te.copy()
        Xp_te_manip[target_idx, TRIGGER_FEATURE] = Xp_tr[:, TRIGGER_FEATURE].max() + 1.0
        proba_after = model.predict_proba(Xa_te, {"party1": Xp_te_manip})
        preds_after = (proba_after[target_idx] >= 0.5).astype(int)
        flip_rate = (preds_before != preds_after).mean()

        aucs.append(auc)
        flip_rates.append(flip_rate)
        print(f"num_plants={num_plants:>2d}  AUC={auc:.4f} (cost={auc_honest-auc:.4f})  "
              f"flip_rate={flip_rate:.1%}  ({len(target_idx)} targeted samples)")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax1.set_facecolor(PALETTE["surface"])

    ax1.plot(plants_sweep, [auc_honest - a for a in aucs], color=PALETTE["magenta"],
              linewidth=2, marker="o", markersize=6, label="Accuracy cost (AUC lost)")
    ax1.set_xlabel("Number of tree-root plants (attacker's dial)", color=PALETTE["ink"])
    ax1.set_ylabel("AUC cost (stealth ↓ is better)", color=PALETTE["magenta"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["magenta"])

    ax2 = ax1.twinx()
    ax2.plot(plants_sweep, flip_rates, color=PALETTE["blue"], linewidth=2, marker="s", markersize=6,
              label="Targeted flip rate")
    ax2.set_ylabel("Targeted flip rate (success ↑ is better)", color=PALETTE["blue"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["blue"])

    ax1.set_title("Category-4 backdoor: stealth-vs-success tradeoff\n"
                  "VFL SecureBoost-style, Adult dataset -- more plants trade stealth for reliability",
                  color=PALETTE["ink"], fontsize=10)
    ax1.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right", fontsize=8,
               labelcolor=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "category4_stealth_success_tradeoff.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
