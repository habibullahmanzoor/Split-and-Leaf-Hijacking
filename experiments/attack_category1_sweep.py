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
from attacks.histogram_integrity import AttackServer, naive_noise_attack, margin_flip_attack

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#2a78d6",
    "yellow": "#eda100",
    "red": "#e34948",
    "grid": "#e1e0d9",
    "muted": "#898781",
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "surface": "#fcfcfb",
}

N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_TRIALS = 5


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


def run_once(X_train, y_train, X_test, y_test, malicious_fraction, attack_fn, attack_kwargs, seed):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    n_malicious = int(np.ceil(malicious_fraction * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))  # always compromise the first N clients

    if attack_fn is None:
        model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=32, max_depth=4, n_rounds=20, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(
                malicious_ids=malicious_ids, attack_fn=attack_fn,
                attack_kwargs=attack_kwargs, attack_max_depth=1,
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

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {"naive": [], "margin_aware": []}
    stds = {"naive": [], "margin_aware": []}
    raw_trials = {"naive": [], "margin_aware": []}

    attacks = {
        "naive": (naive_noise_attack, dict(noise_scale=5.0)),
        "margin_aware": (margin_flip_attack, dict(lam=1.0, gamma=0.0, budget_multiplier=3.0)),
    }

    for name, (fn, kwargs) in attacks.items():
        for frac in fractions:
            aucs = []
            for trial in range(N_TRIALS):
                af = fn if frac > 0 else None
                auc = run_once(X_train, y_train, X_test, y_test, frac, af, kwargs, seed=trial)
                aucs.append(auc)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1))
            results[name].append(mean_auc)
            stds[name].append(std_auc)
            raw_trials[name].append(aucs)
            print(f"[{name}] malicious_fraction={frac:.1f}  test AUC = {mean_auc:.4f} +/- {std_auc:.4f} "
                  f"(n={N_TRIALS} seeds: {[f'{a:.3f}' for a in aucs]})")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.errorbar(fractions, results["naive"], yerr=stds["naive"], color=PALETTE["yellow"],
                linewidth=2, marker="o", markersize=6, capsize=3,
                label=f"Naive noise (generic FL baseline), n={N_TRIALS} seeds")
    ax.errorbar(fractions, results["margin_aware"], yerr=stds["margin_aware"], color=PALETTE["red"],
                linewidth=2, marker="o", markersize=6, capsize=3,
                label=f"Margin-aware split-flip (ours, category 1), n={N_TRIALS} seeds")
    ax.axhline(results["naive"][0], color=PALETTE["muted"], linewidth=1, linestyle="--",
               label="No attack (baseline)")

    ax.set_xlabel("Fraction of malicious clients", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Category-1 histogram/split-integrity attack — HFL, Adult dataset\n"
                  "(5 clients, Dirichlet non-IID α=0.5, attack on root of tree 0)",
                  color=PALETTE["ink"], fontsize=10)
    y_min = min(m - s for m, s in zip(results["naive"] + results["margin_aware"],
                                       stds["naive"] + stds["margin_aware"])) - 0.005
    ax.set_ylim(y_min, results["naive"][0] + 0.01)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "category1_attack_sweep.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
