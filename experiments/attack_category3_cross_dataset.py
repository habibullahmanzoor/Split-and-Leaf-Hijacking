"""Cross-dataset validation for category 3 (HE/crypto-layer, VFL). Reruns
the ciphertext-scale sweep on Heart Disease and Credit Default.
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

from fed_datasets.loaders import load_heart_disease, load_credit_default
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.he_crypto_layer import MaliciousPassiveParty

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "grid": "#e1e0d9", "muted": "#898781", "ink": "#0b0b0b", "secondary": "#52514e",
    "surface": "#fcfcfb", "heart": "#4a3aa7", "credit": "#1baf7a",
}

SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 50.0]

DATASETS = {
    "heart_disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=5,
                           subsample=None, color=PALETTE["heart"]),
    "credit_default": dict(loader=load_credit_default, n_bins=16, max_depth=3, n_rounds=3,
                            subsample=1500, color=PALETTE["credit"]),
}


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        X, y, groups = cfg["loader"]()
        party0_cols, party1_cols = groups["party0"], groups["party1"]

        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]

        Xa = X[party0_cols].to_numpy(dtype=float)
        Xp = X[party1_cols].to_numpy(dtype=float)
        Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
            Xa, Xp, y, test_size=0.25, random_state=0, stratify=y
        )
        print(f"train={len(y_tr)}, test={len(y_te)}, party0_features={len(party0_cols)}, "
              f"party1_features={len(party1_cols)}")

        aucs = []
        for sf in SCALE_FACTORS:
            model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"],
                                      n_rounds=cfg["n_rounds"], lr=0.3, key_size=512)
            factories = None
            if sf != 1.0:
                factories = {"party1": partial(MaliciousPassiveParty, target_feature=0, scale_factor=sf)}
            model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
            auc = roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))
            aucs.append(auc)
            print(f"  scale_factor={sf:>5.1f}  test AUC = {auc:.4f}")

        ax.plot(SCALE_FACTORS, aucs, color=cfg["color"], linewidth=2, marker="o", markersize=6,
                label=f"{name.replace('_', ' ').title()} (honest={aucs[0]:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Ciphertext scale factor (log scale)", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-3 cross-dataset validation\n"
                  "does blind ciphertext rescaling generalize past Adult?",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "category3_cross_dataset.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
