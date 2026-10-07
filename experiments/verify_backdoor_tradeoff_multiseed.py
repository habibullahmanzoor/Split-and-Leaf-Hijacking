"""Multi-seed version of the category-4 stealth-vs-success tradeoff
(attack_category4_tradeoff.py ran this at a single seed). The paper's
headline backdoor number, 0.023 AUC cost / 14.4% flip rate, is the single
num_plants=1 point from that single-seed run. This script reruns the full
num_plants dial at the paper's five-seed statistical standard (same pattern
as defense_bounded_verification_vfl.py's run_once) so the paper can present
the attacker's actual stealth-vs-success curve, with error bars, instead of
one cherry-pickable point -- feeds paper_figures/fig9_backdoor_tradeoff.py.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.passive_party_backdoor import BackdoorPassiveParty

TRIGGER_FEATURE, TRIGGER_BIN, N_BINS = 1, 8, 16
N_ROUNDS = 3  # paper's standard vertical config
N_TRIALS = 5
PLANTS_SWEEP = [1, 2, 3, 5, 8, 13]


def run_once(X, y, groups, num_plants, seed, n_rounds=N_ROUNDS):
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

    honest = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=n_rounds, lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, {"party1": Xp_te}))

    model = FederatedVFLGBDT(n_bins=N_BINS, max_depth=3, n_rounds=n_rounds, lr=0.3, key_size=512)
    factories = {"party1": partial(BackdoorPassiveParty, trigger_feature=TRIGGER_FEATURE,
                                    trigger_bin=TRIGGER_BIN, scale_factor=10.0, num_plants=num_plants)}
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
    proba_clean = model.predict_proba(Xa_te, {"party1": Xp_te})
    auc_backdoored = roc_auc_score(y_te, proba_clean)

    trigger_edges = model.passive_bin_edges["party1"][TRIGGER_FEATURE]
    threshold_value = trigger_edges[min(TRIGGER_BIN, len(trigger_edges) - 1)]
    below_mask = Xp_te[:, TRIGGER_FEATURE] <= threshold_value
    target_idx = np.where(below_mask)[0]
    preds_before = (proba_clean[target_idx] >= 0.5).astype(int)

    Xp_te_manip = Xp_te.copy()
    Xp_te_manip[target_idx, TRIGGER_FEATURE] = Xp_tr[:, TRIGGER_FEATURE].max() + 1.0
    proba_after = model.predict_proba(Xa_te, {"party1": Xp_te_manip})
    preds_after = (proba_after[target_idx] >= 0.5).astype(int)
    flip_rate = (preds_before != preds_after).mean()

    return auc_honest, auc_backdoored, flip_rate


def main():
    X, y, groups = load_adult()

    print(f"=== Category-4 backdoor stealth-vs-success, n_rounds={N_ROUNDS}, {N_TRIALS} seeds ===")
    for num_plants in PLANTS_SWEEP:
        honest_trials, backdoor_trials, flip_trials = [], [], []
        for t in range(N_TRIALS):
            auc_h, auc_b, fr = run_once(X, y, groups, num_plants, seed=t)
            honest_trials.append(auc_h)
            backdoor_trials.append(auc_b)
            flip_trials.append(fr)
        cost_trials = [h - b for h, b in zip(honest_trials, backdoor_trials)]
        h_mean = float(np.mean(honest_trials))
        c_mean, c_std = float(np.mean(cost_trials)), float(np.std(cost_trials, ddof=1))
        f_mean, f_std = float(np.mean(flip_trials)), float(np.std(flip_trials, ddof=1))
        print(f"  num_plants={num_plants:>2d}  honest={h_mean:.4f}  "
              f"AUC_cost={c_mean:.4f}+/-{c_std:.4f}  flip_rate={f_mean:.4f}+/-{f_std:.4f}")


if __name__ == "__main__":
    main()
