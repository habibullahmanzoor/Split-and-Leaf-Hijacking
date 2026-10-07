"""Confidence-filtered label inference (Tier C2): the fuller answer to the
external review's point 6 -- rather than only noting in prose that an
inference-based attacker could filter to high-confidence predictions,
trading coverage for reliability, this measures what that filtering
actually buys. Uses the paper's most generous auxiliary seed (20% of the
training set) and sweeps a confidence threshold on the per-sample vote
agreement already latent in leaf-comembership inference (attacks/
label_inference.py's infer_labels_with_confidence): accuracy and remaining
coverage at each threshold, 5 seeds each (over the train/test split and the
random auxiliary seed draw).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.label_inference import infer_labels_with_confidence

TRIGGER_FEATURE, TRIGGER_BIN, N_BINS = 1, 8, 16
N_ROUNDS = 10
SEED_FRACTION = 0.20  # this paper's most generous auxiliary seed, tested elsewhere
CONFIDENCE_THRESHOLDS = [0.5, 0.7, 0.85, 1.0]  # 0.5 = no filtering (every inferred sample)
N_TRIALS = 5


def run_once(X, y, groups, seed):
    party0_cols, party1_cols = groups["party0"], groups["party1"]
    rng = np.random.default_rng(seed)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=seed, stratify=y_sub
    )

    model = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr})  # honest model; inference only needs the tree structure

    n_train = len(y_tr)
    rng2 = np.random.default_rng(seed + 1000)
    n_seed = int(SEED_FRACTION * n_train)
    seed_idx = rng2.choice(n_train, size=n_seed, replace=False)
    seed_labels = {int(i): int(y_tr[i]) for i in seed_idx}

    inferred, confidence = infer_labels_with_confidence(model.trees, seed_labels, n_train)
    eval_idx = [i for i in inferred if i not in seed_labels]

    results = {}
    for thresh in CONFIDENCE_THRESHOLDS:
        kept = [i for i in eval_idx if confidence[i] >= thresh]
        if not kept:
            results[thresh] = (float("nan"), 0.0)
            continue
        correct = sum(1 for i in kept if inferred[i] == y_tr[i])
        acc = correct / len(kept)
        cov = len(kept) / (n_train - n_seed)
        results[thresh] = (acc, cov)
    return results


def main():
    X, y, groups = load_adult()
    print(f"=== Confidence-filtered label inference, seed_fraction={SEED_FRACTION:.0%}, {N_TRIALS} trials ===")

    all_trials = {thresh: {"acc": [], "cov": []} for thresh in CONFIDENCE_THRESHOLDS}
    for t in range(N_TRIALS):
        results = run_once(X, y, groups, seed=t)
        for thresh, (acc, cov) in results.items():
            if not np.isnan(acc):
                all_trials[thresh]["acc"].append(acc)
            all_trials[thresh]["cov"].append(cov)

    for thresh in CONFIDENCE_THRESHOLDS:
        accs, covs = all_trials[thresh]["acc"], all_trials[thresh]["cov"]
        acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        cov_mean, cov_std = float(np.mean(covs)), float(np.std(covs, ddof=1))
        label = "no filtering" if thresh == 0.5 else f">={thresh:.0%} vote agreement"
        print(f"  confidence>={thresh:.2f} ({label}): "
              f"accuracy={acc_mean:.4f}+/-{acc_std:.4f}  coverage_of_nonseed={cov_mean:.4f}+/-{cov_std:.4f}")


if __name__ == "__main__":
    main()
