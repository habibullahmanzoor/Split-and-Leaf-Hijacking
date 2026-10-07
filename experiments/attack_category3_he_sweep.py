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
from attacks.he_crypto_layer import MaliciousPassiveParty

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "orange": "#eb6834", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

N_TRIALS = 5


def run_once(X, y, groups, sf, seed, n_rounds=3):
    party0_cols, party1_cols = groups["party0"], groups["party1"]
    rng = np.random.default_rng(seed)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=seed, stratify=y_sub
    )

    model = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=n_rounds, lr=0.3, key_size=512)
    factories = None
    if sf != 1.0:
        factories = {"party1": partial(MaliciousPassiveParty, target_feature=0, scale_factor=sf)}
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
    proba = model.predict_proba(Xa_te, {"party1": Xp_te})
    return roc_auc_score(y_te, proba)


def main():
    X, y, groups = load_adult()

    scale_factors = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
    means, stds = [], []
    for sf in scale_factors:
        aucs = [run_once(X, y, groups, sf, seed=trial) for trial in range(N_TRIALS)]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1))
        means.append(mean_auc)
        stds.append(std_auc)
        print(f"scale_factor={sf:>6.1f}  test AUC = {mean_auc:.4f} +/- {std_auc:.4f}  "
              f"(n={N_TRIALS} seeds: {[f'{a:.3f}' for a in aucs]})")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.errorbar(scale_factors, means, yerr=stds, color=PALETTE["orange"], linewidth=2,
                marker="o", markersize=6, capsize=3,
                label=f"Malicious passive party (blind ciphertext rescaling), n={N_TRIALS} seeds")
    ax.axhline(means[0], color=PALETTE["muted"], linewidth=1, linestyle="--", label="No attack (scale=1)")
    ax.set_xscale("log")
    ax.set_xlabel("Ciphertext scale factor applied to target bin (log scale)", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-3 HE/crypto-layer attack — VFL SecureBoost-style, Adult dataset\n"
                  "passive party blindly rescales its own ciphertext, never sees plaintext gradients",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "category3_he_scale_sweep.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
