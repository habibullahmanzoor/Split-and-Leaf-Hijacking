"""Does the split-flip attack's "1 attacker suffices" finding hold as the
federation gets larger, or was it an artifact of testing with only 5
clients? Fixes n_malicious=1 (not a fraction) and sweeps total client
count, so this directly tests scale-invariance rather than re-testing the
already-known malicious-fraction axis.
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

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import FederatedGBDT
from attacks.histogram_integrity import AttackServer, margin_flip_attack

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "red": "#e34948", "blue": "#2a78d6", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

DIRICHLET_ALPHA = 0.5
CLIENT_COUNTS = [5, 10, 20, 40]


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


def run(X_train, y_train, X_test, y_test, n_clients, n_malicious, seed):
    parts = dirichlet_partition(X_train, y_train, n_clients, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    malicious_ids = set(range(n_malicious))

    if n_malicious == 0:
        model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=32, max_depth=4, n_rounds=20, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(
                malicious_ids=malicious_ids, attack_fn=margin_flip_attack,
                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                attack_max_depth=1,
            ),
        )
    model.fit(client_data)
    return roc_auc_score(y_test, model.predict_proba(X_test))


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    honest_aucs, mal1_aucs, mal2_aucs = [], [], []
    for n_clients in CLIENT_COUNTS:
        honest = run(X_train, y_train, X_test, y_test, n_clients, 0, seed=0)
        mal1 = run(X_train, y_train, X_test, y_test, n_clients, 1, seed=0)
        mal2 = run(X_train, y_train, X_test, y_test, n_clients, 2, seed=0)
        honest_aucs.append(honest)
        mal1_aucs.append(mal1)
        mal2_aucs.append(mal2)
        print(f"n_clients={n_clients:>3d}  honest={honest:.4f}  "
              f"1_malicious={mal1:.4f} (drop={honest-mal1:.4f})  "
              f"2_malicious={mal2:.4f} (drop={honest-mal2:.4f})")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(CLIENT_COUNTS, honest_aucs, color=PALETTE["blue"], linewidth=2, marker="o",
            markersize=6, label="No attack")
    ax.plot(CLIENT_COUNTS, mal1_aucs, color=PALETTE["red"], linewidth=2, marker="o",
            markersize=6, label="1 malicious client (fixed count, not fraction)")
    ax.plot(CLIENT_COUNTS, mal2_aucs, color=PALETTE["red"], linewidth=2, marker="s",
            markersize=6, linestyle="--", alpha=0.6, label="2 malicious clients")
    ax.set_xlabel("Total number of clients in the federation", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_xscale("log")
    ax.set_xticks(CLIENT_COUNTS)
    ax.set_xticklabels([str(c) for c in CLIENT_COUNTS])
    ax.set_title("Does 1 attacker still suffice as the federation grows?\n"
                  "HFL, Adult, split-flip attack, fixed at exactly 1 (or 2) malicious clients",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="best", fontsize=8, labelcolor=PALETTE["secondary"])
    fig.tight_layout()
    out_path = FIG_DIR / "category1_client_scaling.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
