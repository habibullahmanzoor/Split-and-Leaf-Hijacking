"""Multi-seed version of attack_category1_client_scaling.py, which tested
this at a single seed only (no error bars). Does the split-flip attack's
"1 attacker suffices" finding hold as the federation gets larger, or was
it an artifact of testing with only 5 clients? Fixes n_malicious=1 (not a
fraction) and sweeps total client count, at the same 5-seed statistical
care applied elsewhere in this paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import FederatedGBDT
from attacks.histogram_integrity import AttackServer, margin_flip_attack

DIRICHLET_ALPHA = 0.5
CLIENT_COUNTS = [5, 10, 20, 40]
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
    return [np.array(idx, dtype=int) for idx in client_idx]


def run_once(X_train, y_train, X_test, y_test, n_clients, n_malicious, seed):
    parts = dirichlet_partition(X_train, y_train, n_clients, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    malicious_ids = set(range(n_malicious))

    if n_malicious == 0:
        model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=32, max_depth=4, n_rounds=20, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(malicious_ids=malicious_ids, attack_fn=margin_flip_attack,
                                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                                attack_max_depth=1),
        )
    model.fit(client_data)
    return roc_auc_score(y_test, model.predict_proba(X_test))


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    for n_clients in CLIENT_COUNTS:
        honest_t, mal1_t = [], []
        for t in range(N_TRIALS):
            honest_t.append(run_once(X_train, y_train, X_test, y_test, n_clients, 0, seed=t))
            mal1_t.append(run_once(X_train, y_train, X_test, y_test, n_clients, 1, seed=t))
        h_mean = float(np.mean(honest_t))
        m_mean, m_std = float(np.mean(mal1_t)), float(np.std(mal1_t, ddof=1))
        drop_trials = [h - m for h, m in zip(honest_t, mal1_t)]
        d_mean, d_std = float(np.mean(drop_trials)), float(np.std(drop_trials, ddof=1))
        print(f"n_clients={n_clients:>3d}  honest={h_mean:.4f}  1_malicious={m_mean:.4f}+/-{m_std:.4f}  "
              f"drop={d_mean:.4f}+/-{d_std:.4f}")


if __name__ == "__main__":
    main()
