"""Control experiment: federated bagging Random Forest, same malicious-fraction
sweep as category-1's histogram attack, to test whether the histogram
attack's behavior (saturate at 1 attacker) is specific to boosting +
histogram-sharing, or just a generic property of "trees."

Bagging RF has no cross-client histogram exchange and no sequential
residual fitting: each client independently trains local trees on its own
partition; the server just averages predicted probabilities across all
clients' forests. A malicious client here can only corrupt its OWN trees
(we simulate this by training that client's forest on label-shuffled data)
— there is no shared aggregation step to manipulate, so if this control
shows a smooth, gradually-worsening curve instead of category-1's sharp
saturation, that confirms the saturation effect tracks the discrete-argmax
histogram-sharing mechanism specifically, not "any tree ensemble."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "green": "#008300", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_TRIALS = 3
TREES_PER_CLIENT = 20


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


def run_once(X_train, y_train, X_test, y_test, malicious_fraction, seed):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    n_malicious = int(np.ceil(malicious_fraction * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))

    rng = np.random.default_rng(seed + 100)
    all_proba = []
    for cid, idx in enumerate(parts):
        Xc, yc = X_train[idx], y_train[idx]
        if cid in malicious_ids:
            yc = rng.permutation(yc)  # corrupt this client's own trees only
        rf = RandomForestClassifier(n_estimators=TREES_PER_CLIENT, max_depth=4, random_state=seed)
        rf.fit(Xc, yc)
        all_proba.append(rf.predict_proba(X_test)[:, 1])

    avg_proba = np.mean(all_proba, axis=0)  # bagging: server averages client forests
    return roc_auc_score(y_test, avg_proba)


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    means = []
    for frac in fractions:
        aucs = [run_once(X_train, y_train, X_test, y_test, frac, seed=t) for t in range(N_TRIALS)]
        mean_auc = float(np.mean(aucs))
        means.append(mean_auc)
        print(f"[bagging RF] malicious_fraction={frac:.1f}  test AUC = {mean_auc:.4f} "
              f"(trials: {[f'{a:.3f}' for a in aucs]})")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.plot(fractions, means, color=PALETTE["green"], linewidth=2, marker="o",
            markersize=6, label="Bagging Random Forest (label-shuffle attack)")
    ax.axhline(means[0], color=PALETTE["muted"], linewidth=1, linestyle="--",
               label="No attack (baseline)")

    ax.set_xlabel("Fraction of malicious clients", color=PALETTE["ink"])
    ax.set_ylabel("Test AUC", color=PALETTE["ink"])
    ax.set_title("Control: bagging Random Forest (no shared histogram, no boosting)\n"
                  "HFL, Adult dataset, 5 clients, Dirichlet non-IID α=0.5",
                  color=PALETTE["ink"], fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["muted"])
    ax.tick_params(colors=PALETTE["secondary"])
    ax.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=PALETTE["secondary"])

    fig.tight_layout()
    out_path = FIG_DIR / "rf_bagging_control.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
