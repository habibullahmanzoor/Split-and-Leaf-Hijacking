"""Tree-depth robustness check (Tier C1): does the single-attacker saturation
finding hold at production-realistic tree depths, not just this paper's
standard depth=4? Fixes attack_max_depth=1 (the root node only, the paper's
disclosed attack radius throughout) and instead varies the OVERALL tree depth
the honest and attacked models are allowed to grow to: 4 (this paper's
standard), 6, 8, 10. The attack itself never reaches deeper than the root
regardless of how deep the rest of the tree is allowed to grow; this isolates
whether a deeper, more production-realistic tree gives the model enough extra
uncorrupted capacity at depth > 1 to dilute the root-level attack, the same
question already answered for attack RADIUS (verify_cat1_attack_depth.py) but
here asked of the tree's own maximum depth instead.
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
from attack_category1_sweep import dirichlet_partition, N_CLIENTS, DIRICHLET_ALPHA

N_TRIALS = 5
TREE_DEPTHS = [4, 6, 8, 10]
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_BINS, N_ROUNDS = 32, 20


def run_once(X_train, y_train, X_test, y_test, tree_depth, malicious_fraction, seed):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    n_malicious = int(np.ceil(malicious_fraction * N_CLIENTS - 1e-9))
    malicious_ids = set(range(n_malicious))

    if n_malicious == 0:
        model = FederatedGBDT(n_bins=N_BINS, max_depth=tree_depth, n_rounds=N_ROUNDS, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=N_BINS, max_depth=tree_depth, n_rounds=N_ROUNDS, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(malicious_ids=malicious_ids, attack_fn=margin_flip_attack,
                                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                                attack_max_depth=1),  # root node only, regardless of tree_depth
        )
    model.fit(client_data)
    return roc_auc_score(y_test, model.predict_proba(X_test))


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    for depth in TREE_DEPTHS:
        print(f"\n=== tree max_depth={depth} (attack_max_depth=1 throughout) ===")
        means = []
        for frac in FRACTIONS:
            aucs = [run_once(X_train, y_train, X_test, y_test, depth, frac, seed=t) for t in range(N_TRIALS)]
            mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
            means.append(mean_auc)
            print(f"  malicious_fraction={frac:.1f}  test AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
        drop_at_one = means[0] - means[1]
        drop_at_half = means[0] - means[-1]
        print(f"  honest={means[0]:.4f}  drop@1_malicious={drop_at_one:.4f}  "
              f"drop@half_federation={drop_at_half:.4f}  "
              f"saturated={'yes' if abs(drop_at_one - drop_at_half) < 0.01 else 'no'}")


if __name__ == "__main__":
    main()
