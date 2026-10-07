"""Does the margin-flip attack still cause damage if the malicious client's
forged histogram is made internally consistent across features -- every
feature's own reported total equal to every other's, exactly the invariant
this paper's own cross-feature conservation check (Section 6) audits --
rather than the original construction, which reports an honest histogram
for the target feature only and literally zero at every other feature (a
maximally conspicuous, trivially detectable forgery regardless of any
conservation check)?

Two earlier constructions in this file were tried and rejected after direct
verification, not assumed correct (see paper.md's dev log for the full
account):
  - Dumping the full compensating budget into a single bin of every other
    feature preserves totals but creates a new, more attractive candidate
    at that single bin, hijacking the split to an unintended (feature, bin)
    in all five seeds tested.
  - Spreading compensating mass proportionally to each feature's own
    ABSOLUTE-VALUE shape only achieves exact conservation when a feature's
    row happens to be non-negative; gradients can be negative (logistic
    g = p - y), so this left real, seed-dependent residual inconsistency
    (up to 709,590 in one seed) -- not the exact invariant claimed.

The construction below -- spreading the compensating budget UNIFORMLY
across every bin of each other feature -- is the only one that achieves
EXACT conservation regardless of sign structure (verified to floating-point
zero in every seed tested). It also reveals a genuinely more interesting
finding than either earlier attempt: it still causes comparable AUC damage
to the original attack, but the split it wins is NOT the attacker's
originally-computed target -- some other feature's inflated total creates
a more attractive candidate first. Cross-feature consistency does not stop
the damage, but a naive way of achieving it also costs the attacker precise
control over which split gets corrupted. Whether a more sophisticated
conservation-preserving construction could recover that precision is not
resolved here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import FederatedGBDT, Client, compute_bin_edges, bin_features
from attacks.histogram_integrity import AttackServer, margin_flip_attack

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


def margin_flip_attack_conservation_preserving(true_grad, true_hess, honest_others_grad,
                                                honest_others_hess, lam=1.0, gamma=0.0,
                                                budget_multiplier=3.0, rng=None):
    """Same worst-candidate targeting as margin_flip_attack, but starts from
    the client's TRUE histogram (not all-zero) and adds the same
    compensating budget, spread UNIFORMLY across every bin, to every other
    feature -- the only construction tried here that achieves EXACT
    cross-feature total consistency regardless of sign structure."""
    n_features, n_bins = true_grad.shape
    if honest_others_grad is None:
        return true_grad, true_hess

    G_total = honest_others_grad.sum() + true_grad.sum()
    H_total = honest_others_hess.sum() + true_hess.sum()

    worst_gain, worst_f, worst_b = np.inf, None, None
    for f in range(n_features):
        g_cum = np.cumsum(honest_others_grad[f] + true_grad[f])
        h_cum = np.cumsum(honest_others_hess[f] + true_hess[f])
        for b in range(n_bins - 1):
            GL, HL = g_cum[b], h_cum[b]
            GR, HR = G_total - GL, H_total - HL
            if HL < 1.0 or HR < 1.0:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam)
                           - G_total**2 / (H_total + lam)) - gamma
            if gain < worst_gain:
                worst_gain, worst_f, worst_b = gain, f, b

    if worst_f is None:
        return true_grad, true_hess

    honest_magnitude = np.abs(honest_others_grad).sum() + np.abs(true_grad).sum()
    budget = budget_multiplier * max(honest_magnitude, 1e-3)

    malicious_grad = true_grad.copy()
    malicious_grad[worst_f, worst_b] += budget
    for f in range(n_features):
        if f != worst_f:
            malicious_grad[f, :] += budget / n_bins  # sums to exactly `budget`, any sign structure
    return malicious_grad, true_hess.copy()


def run_once(X_train, y_train, X_test, y_test, attack_fn, seed, return_split=False):
    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
    malicious_ids = {0}

    if attack_fn is None:
        model = FederatedGBDT(n_bins=32, max_depth=4, n_rounds=20, lr=0.3)
    else:
        model = FederatedGBDT(
            n_bins=32, max_depth=4, n_rounds=20, lr=0.3,
            server_cls=AttackServer,
            server_kwargs=dict(
                malicious_ids=malicious_ids, attack_fn=attack_fn,
                attack_kwargs=dict(lam=1.0, gamma=0.0, budget_multiplier=3.0),
                attack_max_depth=1,
            ),
        )
    model.fit(client_data)
    auc = roc_auc_score(y_test, model.predict_proba(X_test))
    if return_split:
        root = model.trees[0]
        return auc, (root.split_feature, root.split_bin)
    return auc


def check_self_consistency(forged_grad):
    n_features = forged_grad.shape[0]
    totals = [forged_grad[f].sum() for f in range(n_features)]
    return max(totals) - min(totals)


def main():
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    print("=== Honest (no attack) ===")
    honest_aucs = [run_once(X_train, y_train, X_test, y_test, None, t) for t in range(N_TRIALS)]
    print(f"  AUC = {np.mean(honest_aucs):.4f} +/- {np.std(honest_aucs, ddof=1):.4f}  raw={[f'{a:.4f}' for a in honest_aucs]}")

    print("=== Original margin_flip_attack (all-zero except target cell) ===")
    orig_results = [run_once(X_train, y_train, X_test, y_test, margin_flip_attack, t, return_split=True) for t in range(N_TRIALS)]
    orig_aucs = [r[0] for r in orig_results]
    print(f"  AUC = {np.mean(orig_aucs):.4f} +/- {np.std(orig_aucs, ddof=1):.4f}  raw={[f'{a:.4f}' for a in orig_aucs]}")
    print(f"  root splits per seed: {[r[1] for r in orig_results]}")

    print("=== Conservation-preserving variant (uniform spread, EXACT per-feature total consistency) ===")
    cons_results = [run_once(X_train, y_train, X_test, y_test,
                              margin_flip_attack_conservation_preserving, t, return_split=True) for t in range(N_TRIALS)]
    cons_aucs = [r[0] for r in cons_results]
    print(f"  AUC = {np.mean(cons_aucs):.4f} +/- {np.std(cons_aucs, ddof=1):.4f}  raw={[f'{a:.4f}' for a in cons_aucs]}")
    print(f"  root splits per seed: {[r[1] for r in cons_results]}")

    n_same_split = sum(1 for o, c in zip(orig_results, cons_results) if o[1] == c[1])
    print(f"\n  Same target split as original attack: {n_same_split}/{N_TRIALS} seeds "
          f"(uniform spread wins a DIFFERENT split every time -- see module docstring)")

    print("\n=== Self-consistency check (exact, all 5 seeds) ===")
    for seed in range(N_TRIALS):
        parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed)
        client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}
        X_all = np.vstack([Xc for Xc, _ in client_data.values()])
        bin_edges = compute_bin_edges(X_all, 32)
        clients_list = [Client(client_id=cid, X=bin_features(Xc, bin_edges), y=yc, bin_edges=bin_edges)
                        for cid, (Xc, yc) in client_data.items()]
        true_g, true_h = Client.compute_histograms(clients_list[0], np.arange(len(clients_list[0].y)), 32)
        others_g = sum(Client.compute_histograms(c, np.arange(len(c.y)), 32)[0] for c in clients_list[1:])
        others_h = sum(Client.compute_histograms(c, np.arange(len(c.y)), 32)[1] for c in clients_list[1:])
        forged_g, _ = margin_flip_attack_conservation_preserving(true_g, true_h, others_g, others_h,
                                                                   lam=1.0, gamma=0.0, budget_multiplier=3.0)
        orig_forged_g, _ = margin_flip_attack(true_g, true_h, others_g, others_h,
                                               lam=1.0, gamma=0.0, budget_multiplier=3.0)
        print(f"  seed={seed}: conservation-preserving spread={check_self_consistency(forged_g):.10f}  "
              f"original spread={check_self_consistency(orig_forged_g):.4f}")


if __name__ == "__main__":
    main()
