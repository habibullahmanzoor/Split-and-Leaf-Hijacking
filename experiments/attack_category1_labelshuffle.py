"""Category-1 control: label-shuffle attack under HFL boosting (shared
histogram, discrete argmax), rather than histogram forgery. Isolates the
aggregation-mechanism variable from the attack-type variable in the bagging
control comparison: rf_bagging_control.py corrupts its malicious client via
label-shuffle under BAGGING (no shared histogram); attack_category1_sweep.py
corrupts via histogram FORGERY under BOOSTING. Those two differ on two axes
at once (mechanism AND attack type), so the smooth-vs-saturating contrast
between them does not by itself isolate the aggregation mechanism as the
cause. This script holds the attack type fixed (label-shuffle) and varies
only the aggregation mechanism, by running the identical label-shuffle
corruption through the boosting harness instead of bagging: a malicious
client's local labels are permuted before it ever computes a histogram, and
the corrupted histogram it reports is aggregated and boosted exactly as an
honest one would be.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from harness.federated_gbdt import FederatedGBDT
from attack_category1_sweep import dirichlet_partition, N_CLIENTS, DIRICHLET_ALPHA

N_TRIALS = 5


def run_once(X_train, y_train, X_test, y_test, malicious_fraction, seed):
    from sklearn.metrics import roc_auc_score
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    n_malicious = int(np.ceil(malicious_fraction * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))

    rng = np.random.default_rng(seed + 100)
    client_data = {}
    for cid, idx in enumerate(parts):
        Xc, yc = X_train[idx], y_train[idx]
        if cid in malicious_ids:
            yc = rng.permutation(yc)  # identical corruption to rf_bagging_control.py
        client_data[cid] = (Xc, yc)

    model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    model.fit(client_data)
    return roc_auc_score(y_test, model.predict_proba(X_test))


def main():
    from sklearn.model_selection import train_test_split
    from fed_datasets.loaders import load_adult

    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for frac in fractions:
        aucs = [run_once(X_train, y_train, X_test, y_test, frac, seed=t) for t in range(N_TRIALS)]
        mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        print(f"[label-shuffle under boosting] malicious_fraction={frac:.1f}  "
              f"test AUC = {mean_auc:.4f} +/- {std_auc:.4f} (trials: {[f'{a:.3f}' for a in aucs]})")


if __name__ == "__main__":
    main()
