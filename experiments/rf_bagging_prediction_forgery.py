"""Second bagging control: forge the REPORTED PREDICTION rather than the
local training data.

rf_bagging_control.py corrupts a malicious bagging client by label-shuffling
its own local training data before it fits its forest -- a data-poisoning
attack. Every other attack in this paper (histogram forgery under HFL
boosting, ciphertext rescaling under VFL) is instead a report-poisoning
attack: the attacker's own data/model is untouched, and it forges the
aggregate CONTRIBUTION it sends to the server. Label shuffling under bagging
is therefore not a clean analog of those attacks -- it corrupts the wrong
stage of the pipeline.

This script fixes that: the malicious client trains an honest forest on its
own honest, uncorrupted local partition (exactly as it would if benign),
then reports a forged prediction vector -- the complement 1 - p of its own
honest predicted probability -- instead of its true predict_proba output,
before the server averages. This is the direct bagging analog of "forging a
reported histogram contribution": the corruption happens at the report,
not the data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult

N_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
N_TRIALS = 5
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

    all_proba = []
    for cid, idx in enumerate(parts):
        Xc, yc = X_train[idx], y_train[idx]
        rf = RandomForestClassifier(n_estimators=TREES_PER_CLIENT, max_depth=4, random_state=seed)
        rf.fit(Xc, yc)  # always honest data -- no label shuffling
        proba = rf.predict_proba(X_test)[:, 1]
        if cid in malicious_ids:
            proba = 1.0 - proba  # forge the reported contribution, not the data
        all_proba.append(proba)

    avg_proba = np.mean(all_proba, axis=0)
    return roc_auc_score(y_test, avg_proba)


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for frac in fractions:
        aucs = [run_once(X_train, y_train, X_test, y_test, frac, seed=t) for t in range(N_TRIALS)]
        mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        print(f"[bagging, prediction forgery] malicious_fraction={frac:.1f}  "
              f"test AUC = {mean_auc:.4f} +/- {std_auc:.4f} (trials: {[f'{a:.3f}' for a in aucs]})")


if __name__ == "__main__":
    main()
