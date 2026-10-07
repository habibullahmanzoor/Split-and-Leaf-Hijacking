"""Cross-dataset validation for category 1 (histogram/split-integrity, HFL).
Reruns the malicious-fraction sweep on Heart Disease and Credit Default --
loaded since the project's start but never used in an attack experiment.
Tests whether the discrete-saturation finding is an Adult-specific artifact.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_heart_disease, load_credit_default
from harness.federated_gbdt import FederatedGBDT
from attacks.histogram_integrity import AttackServer, margin_flip_attack

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "red": "#e34948", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
    "heart": "#4a3aa7", "credit": "#1baf7a",
}

DIRICHLET_ALPHA = 0.5
N_CLIENTS = 5
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

DATASETS = {
    "heart_disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=15,
                           subsample=None, color=PALETTE["heart"]),
    "credit_default": dict(loader=load_credit_default, n_bins=32, max_depth=4, n_rounds=20,
                            subsample=3000, color=PALETTE["credit"]),
}


def dirichlet_partition(X, y, n_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    client_idx = [[] for _ in range(n_clients)]
    for idx in idx_by_class:
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, splits)):
            client_idx[cid].extend(part.tolist())
    return [np.array(idx) for idx in client_idx]


def run(X_train, y_train, X_test, y_test, n_bins, max_depth, n_rounds, malicious_fraction, attack_fn, seed):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    n_malicious = int(np.ceil(malicious_fraction * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))

    if n_malicious == 0:
        model = FederatedGBDT(n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(
                malicious_ids=malicious_ids, attack_fn=attack_fn,
                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                attack_max_depth=1,
            ),
        )
    model.fit(client_data)
    return roc_auc_score(y_test, model.predict_proba(X_test))


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        X, y, _ = cfg["loader"]()
        X = X.to_numpy(dtype=float)
        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        print(f"train={len(y_train)}, test={len(y_test)}, positive_rate={y.mean():.3f}")

        aucs = []
        for frac in FRACTIONS:
            af = margin_flip_attack if frac > 0 else None
            auc = run(X_train, y_train, X_test, y_test, cfg["n_bins"], cfg["max_depth"],
                      cfg["n_rounds"], frac, af, seed=0)
            aucs.append(auc)
            print(f"  malicious_fraction={frac:.1f}  test AUC = {auc:.4f}")

        ax.plot(FRACTIONS, aucs, color=cfg["color"], linewidth=2, marker="o", markersize=6,
                label=f"{name.replace('_', ' ').title()} (honest={aucs[0]:.3f})")

    ax.set_xlabel("Fraction of malicious clients", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-1 cross-dataset validation\n"
                  "does the split-flip attack's saturation pattern generalize past Adult?",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "category1_cross_dataset.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
