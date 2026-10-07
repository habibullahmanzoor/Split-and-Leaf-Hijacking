"""Category 1's headline attack (margin_flip_attack) is configured with
attack_max_depth=1 (root node only) throughout this paper, including the
cross-dataset rerun on Credit Default, where it produces no damage despite
being the fully informed, adaptive attacker -- not because its targeting
is blind (it isn't; it uses the true honest gradient aggregate to find the
worst-gain candidate), but because Credit Default's deeper, wider trees
(depth 4, 20 rounds, 32 bins) leave enough uncorrupted capacity at depth
>1 to route around root-level corruption. This script tests that directly:
does extending attack_max_depth restore saturating damage on Credit
Default, and does the SAME extension change anything on Adult, where
depth=1 already saturates? If Adult is already flat across attack_max_depth
while Credit Default climbs sharply, that confirms this is a per-dataset
attack-surface calibration issue, not a per-dataset targeting-quality
issue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult, load_credit_default, load_heart_disease
from harness.federated_gbdt import FederatedGBDT
from attacks.histogram_integrity import AttackServer, margin_flip_attack

N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_TRIALS = 5
MALICIOUS_FRACTION = 0.4
ATTACK_DEPTHS = [1, 2, 3, 4]

# attack_depths is capped at each dataset's own max_depth (Heart Disease uses
# shallower trees than Adult/Credit Default elsewhere in this paper, so depth 4
# is not a valid node depth for it).
DATASETS = {
    "adult": dict(loader=load_adult, n_bins=32, max_depth=4, n_rounds=20, subsample=None,
                   attack_depths=ATTACK_DEPTHS),
    "credit_default": dict(loader=load_credit_default, n_bins=32, max_depth=4, n_rounds=20, subsample=3000,
                            attack_depths=ATTACK_DEPTHS),
    "heart_disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=15, subsample=None,
                           attack_depths=[1, 2, 3]),
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


def run_once(X_train, y_train, X_test, y_test, n_bins, max_depth, n_rounds,
             attack_max_depth, seed):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    n_malicious = int(np.ceil(MALICIOUS_FRACTION * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))

    honest = FederatedGBDT(n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3)
    honest.fit(client_data)
    auc_honest = roc_auc_score(y_test, honest.predict_proba(X_test))

    model = FederatedGBDT(
        n_bins=n_bins, max_depth=max_depth, n_rounds=n_rounds, lr=0.3,
        server_cls=AttackServer,
        server_kwargs=dict(malicious_ids=malicious_ids, attack_fn=margin_flip_attack,
                            attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                            attack_max_depth=attack_max_depth),
    )
    model.fit(client_data)
    auc_attacked = roc_auc_score(y_test, model.predict_proba(X_test))
    return auc_honest, auc_attacked


def main():
    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        X, y, _ = cfg["loader"]()
        X = X.to_numpy(dtype=float)
        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        for depth in cfg["attack_depths"]:
            honest_t, attacked_t = [], []
            for t in range(N_TRIALS):
                ah, aa = run_once(X_train, y_train, X_test, y_test, cfg["n_bins"], cfg["max_depth"],
                                   cfg["n_rounds"], depth, seed=t)
                honest_t.append(ah)
                attacked_t.append(aa)
            h_mean = float(np.mean(honest_t))
            a_mean, a_std = float(np.mean(attacked_t)), float(np.std(attacked_t, ddof=1))
            print(f"  attack_max_depth={depth}  honest={h_mean:.4f}  "
                  f"attacked={a_mean:.4f}+/-{a_std:.4f}  drop={h_mean-a_mean:+.4f}")


if __name__ == "__main__":
    main()
