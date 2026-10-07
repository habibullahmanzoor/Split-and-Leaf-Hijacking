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
from attacks.leaf_routing import MisroutingPassiveParty

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "violet": "#4a3aa7", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}


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

    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    aucs = []
    for frac in fractions:
        model = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=3, lr=0.3, key_size=512)
        factories = None
        if frac > 0:
            factories = {"party1": partial(MisroutingPassiveParty, misroute_fraction=frac)}
        model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
        auc = roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))
        aucs.append(auc)
        print(f"misroute_fraction={frac:.2f}  test AUC = {auc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(fractions, aucs, color=PALETTE["violet"], linewidth=2, marker="o", markersize=6,
            label="Misrouted samples at passive party's winning splits")
    ax.axhline(aucs[0], color=PALETTE["muted"], linewidth=1, linestyle="--", label="No attack (fraction=0)")
    ax.set_xlabel("Fraction of samples misrouted", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-5 leaf-routing manipulation — VFL SecureBoost-style, Adult dataset\n"
                  "attacks post-split aggregation; the HE-protected split decision is never touched",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "category5_leaf_routing_sweep.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
