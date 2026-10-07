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
    "blue": "#2a78d6", "magenta": "#e87ba4", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb", "good": "#0ca30c",
}

TRIGGER_FEATURE = 1  # fnlwgt -- continuous, high-cardinality (21648 unique values)
TRIGGER_BIN = 8
N_BINS = 16


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
    honest = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=3, lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, {"party1": Xp_te}))
    print(f"Honest baseline: test AUC = {auc_honest:.4f}")

    print("Training with backdoored passive party ...")
    backdoored = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=3, lr=0.3, key_size=512)
    factories = {"party1": partial(BackdoorPassiveParty, trigger_feature=TRIGGER_FEATURE,
                                    trigger_bin=TRIGGER_BIN, scale_factor=10.0)}
    backdoored.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
    proba_clean = backdoored.predict_proba(Xa_te, {"party1": Xp_te})
    auc_backdoored = roc_auc_score(y_te, proba_clean)
    print(f"Backdoored model, UNMANIPULATED test inputs: test AUC = {auc_backdoored:.4f} "
          f"(stealth check -- should be close to honest baseline)")

    # -- exploit the trigger at inference time --
    trigger_edges = backdoored.passive_bin_edges["party1"][TRIGGER_FEATURE]
    threshold_value = trigger_edges[min(TRIGGER_BIN, len(trigger_edges) - 1)]

    below_mask = Xp_te[:, TRIGGER_FEATURE] <= threshold_value
    target_idx = np.where(below_mask)[0]
    print(f"\n{len(target_idx)} of {len(Xp_te)} test samples start below the trigger threshold "
          f"({threshold_value:.1f}) and are targeted for the flip.")

    preds_before = (proba_clean[target_idx] >= 0.5).astype(int)

    Xp_te_manipulated = Xp_te.copy()
    Xp_te_manipulated[target_idx, TRIGGER_FEATURE] = Xp_tr[:, TRIGGER_FEATURE].max() + 1.0
    proba_after = backdoored.predict_proba(Xa_te, {"party1": Xp_te_manipulated})
    preds_after = (proba_after[target_idx] >= 0.5).astype(int)

    flipped = (preds_before != preds_after).sum()
    flip_rate = flipped / len(target_idx)
    print(f"Predictions flipped by manipulating ONLY the passive party's own feature "
          f"(no label access at any point): {flipped}/{len(target_idx)} = {flip_rate:.1%}")

    auc_after_manipulation = roc_auc_score(y_te, proba_after)

    # -- figure: stealth (AUC) vs attack success (flip rate) --
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])

    ax0 = axes[0]
    ax0.set_facecolor(PALETTE["surface"])
    bars = ax0.bar(["Honest\nbaseline", "Backdoored\n(unmanipulated inputs)"],
                    [auc_honest, auc_backdoored],
                    color=[PALETTE["blue"], PALETTE["magenta"]], width=0.55)
    ax0.set_ylim(0.5, 1.0)
    ax0.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax0.set_title(f"Stealth: modest accuracy cost\n({auc_honest - auc_backdoored:.3f} AUC, single planted split)",
                   fontsize=9, color=PALETTE["ink"])
    ax0.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.tick_params(colors=PALETTE["secondary"])

    ax1 = axes[1]
    ax1.set_facecolor(PALETTE["surface"])
    ax1.bar(["Targeted samples"], [flip_rate], color=PALETTE["good"], width=0.4)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Fraction of predictions flipped", color=PALETTE["ink"])
    ax1.set_title("Attack success: targeted flip rate\n(passive party manipulates only its own feature)",
                   fontsize=9, color=PALETTE["ink"])
    ax1.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(colors=PALETTE["secondary"])

    fig.suptitle("Category-4 passive-party role-abuse backdoor — VFL SecureBoost-style, Adult dataset",
                 fontsize=10, color=PALETTE["ink"])
    fig.tight_layout()
    out_path = FIG_DIR / "category4_backdoor_stealth_vs_success.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
