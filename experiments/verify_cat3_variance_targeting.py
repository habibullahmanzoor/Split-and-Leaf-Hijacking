"""Does variance-based target-feature selection (adaptive_target_selection.py's
VarianceTargetedMaliciousPassiveParty) fix category 3's Credit Default
non-generalization? he_crypto_layer.py's MaliciousPassiveParty has always
used a hardcoded target_feature=0, never itself chosen by any heuristic;
this compares that fixed choice against picking the feature with the
largest sample variance among the party's own (binned) columns -- still
fully public information, no labels, no gradients, no cross-client
knowledge, just a different, real heuristic in the feature-choice slot
that was previously left unfilled.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.he_crypto_layer import MaliciousPassiveParty
from attacks.adaptive_target_selection import VarianceTargetedMaliciousPassiveParty

N_TRIALS = 5
SCALE_FACTOR = 10.0

DATASETS = {
    "adult": dict(loader=load_adult, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
    "heart_disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=5, subsample=None),
    "credit_default": dict(loader=load_credit_default, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
}


def run_once(X, y, groups, attacker_cls, cfg, seed):
    party0_cols, party1_cols = groups["party0"], groups["party1"]
    X_sub, y_sub = X, y
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=seed, stratify=y_sub
    )

    honest = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"],
                               n_rounds=cfg["n_rounds"], lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, {"party1": Xp_te}))

    factory = partial(attacker_cls, target_feature=0, scale_factor=SCALE_FACTOR)
    model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"],
                              n_rounds=cfg["n_rounds"], lr=0.3, key_size=512)
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories={"party1": factory})
    auc_attacked = roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))

    return auc_honest, auc_attacked


def main():
    for name, cfg in DATASETS.items():
        X, y, groups = cfg["loader"]()
        print(f"\n=== {name} ===")
        for label, cls in [("fixed(feat=0)", MaliciousPassiveParty),
                            ("variance-targeted", VarianceTargetedMaliciousPassiveParty)]:
            honest_t, attacked_t = [], []
            for t in range(N_TRIALS):
                ah, aa = run_once(X, y, groups, cls, cfg, seed=t)
                honest_t.append(ah)
                attacked_t.append(aa)
            h_mean = float(np.mean(honest_t))
            a_mean, a_std = float(np.mean(attacked_t)), float(np.std(attacked_t, ddof=1))
            print(f"  {label:>20s}  honest={h_mean:.4f}  attacked={a_mean:.4f}+/-{a_std:.4f}  "
                  f"drop={h_mean-a_mean:+.4f}")


if __name__ == "__main__":
    main()
