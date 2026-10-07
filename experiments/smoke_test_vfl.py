import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT


def main():
    X, y, groups = load_adult()
    party0_cols = groups["party0"]  # active party (holds labels)
    party1_cols = groups["party1"]  # passive party (features only)

    # subsample for a fast CPU smoke test -- HE ops scale with rows x features x bins x nodes
    rng = np.random.default_rng(0)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)

    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=0, stratify=y_sub
    )

    print(f"Active party features: {party0_cols}")
    print(f"Passive party features: {party1_cols}")
    print(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")

    model = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=3, lr=0.3, key_size=512)

    t0 = time.time()
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    fit_time = time.time() - t0

    proba = model.predict_proba(Xa_te, {"party1": Xp_te})
    auc = roc_auc_score(y_te, proba)
    print(f"\nVFL SecureBoost-style (1 active + 1 passive party): "
          f"test AUC = {auc:.4f}, fit time = {fit_time:.1f}s")


if __name__ == "__main__":
    main()
