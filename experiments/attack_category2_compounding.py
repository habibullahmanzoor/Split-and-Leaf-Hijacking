"""Category-2: boosting-compounding attack.

Isolates whether WHEN a single-round corruption happens matters, holding
attack strength fixed (1 malicious client, 1 boosting round, depth 0-1,
same margin_flip_attack as category 1). If boosting compounds damage, an
early corrupted tree's residuals propagate into everything trained after
it, while a late corrupted tree has almost no downstream rounds left to
poison. We track test AUC after every one of 20 rounds for three runs:
no attack, corrupt round 0 only, corrupt round 19 only (last round).
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
    "blue": "#2a78d6", "red": "#e34948", "violet": "#4a3aa7",
    "grid": "#e1e0d9", "muted": "#898781", "ink": "#0b0b0b",
    "secondary": "#52514e", "surface": "#fcfcfb",
}

N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_ROUNDS = 20


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


def run_trajectory(X_train, y_train, X_test, y_test, target_rounds, seed=0,
                    n_bins=32, max_depth=4, n_rounds=N_ROUNDS, n_clients=N_CLIENTS,
                    dirichlet_alpha=DIRICHLET_ALPHA):
    parts = dirichlet_partition(X_train, y_train, n_clients, dirichlet_alpha, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    aucs = []

    def eval_cb(t, model):
        aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)))

    if target_rounds is None:
        model = FederatedGBDT(n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(
                malicious_ids={0}, attack_fn=margin_flip_attack,
                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                attack_max_depth=1, target_rounds=set(target_rounds),
            ),
        )
    model.fit(client_data, eval_callback=eval_cb)
    return aucs


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    print("Running: no attack ...")
    aucs_clean = run_trajectory(X_train, y_train, X_test, y_test, target_rounds=None)
    print("Running: corrupt round 0 only ...")
    aucs_early = run_trajectory(X_train, y_train, X_test, y_test, target_rounds={0})
    print("Running: corrupt round 19 only ...")
    aucs_late = run_trajectory(X_train, y_train, X_test, y_test, target_rounds={N_ROUNDS - 1})

    for name, aucs in [("clean", aucs_clean), ("early(round0)", aucs_early), ("late(round19)", aucs_late)]:
        print(f"{name}: final AUC = {aucs[-1]:.4f}")

    rounds = list(range(1, N_ROUNDS + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.plot(rounds, aucs_clean, color=PALETTE["blue"], linewidth=2, label="No attack")
    ax.plot(rounds, aucs_early, color=PALETTE["red"], linewidth=2,
            label="1 corrupted round: round 0 (earliest)")
    ax.plot(rounds, aucs_late, color=PALETTE["violet"], linewidth=2,
            label="1 corrupted round: round 19 (latest)")
    ax.axvline(1, color=PALETTE["red"], linewidth=1, linestyle=":", alpha=0.5)
    ax.axvline(20, color=PALETTE["violet"], linewidth=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Boosting round", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-2 boosting-compounding attack — HFL, Adult dataset\n"
                  "(single malicious client, single corrupted round, same attack strength)",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "category2_compounding_trajectory.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
