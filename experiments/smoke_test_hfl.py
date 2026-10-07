import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import FederatedGBDT


def dirichlet_partition(X, y, n_clients, alpha, seed=0):
    """Split rows across clients with label-skew controlled by alpha
    (small alpha = more non-IID)."""
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


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    n_clients = 5
    parts = dirichlet_partition(X_train, y_train, n_clients, alpha=0.5)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    print("Client partition sizes:", [len(v[1]) for v in client_data.values()])
    print("Client positive rates:", [f"{v[1].mean():.2f}" for v in client_data.values()])

    model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    model.fit(client_data)

    proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"\nFederated GBDT (HFL, 5 clients, non-IID alpha=0.5): test AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
