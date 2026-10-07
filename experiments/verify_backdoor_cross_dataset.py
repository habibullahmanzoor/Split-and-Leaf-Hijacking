"""Cross-dataset generalization check for the label-free backdoor (category
4), extending verify_backdoor_tradeoff_multiseed.py (Adult only) to Heart
Disease and Credit Default, five seeds each, same fixed target feature
convention used for every other cross-dataset check in this paper (index 1
within the passive party's own feature block, chosen once in advance, never
tuned per dataset).
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_heart_disease, load_credit_default
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.passive_party_backdoor import BackdoorPassiveParty

TRIGGER_FEATURE, TRIGGER_BIN = 1, 8
N_TRIALS = 5
PLANTS_SWEEP = [1, 5, 8]  # stealthy end, midpoint, and this paper's measured ceiling on Adult

DATASETS = {
    "Heart Disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=5, subsample=None),
    "Credit Default": dict(loader=load_credit_default, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
}


def run_once(X, y, groups, cfg, num_plants, seed):
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

    honest = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"], n_rounds=cfg["n_rounds"],
                               lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, {"party1": Xp_te}))

    model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"], n_rounds=cfg["n_rounds"],
                              lr=0.3, key_size=512)
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
    flip_rate = (preds_before != preds_after).mean() if len(target_idx) else float("nan")

    return auc_honest, auc_backdoored, flip_rate, len(target_idx)


def main():
    for name, cfg in DATASETS.items():
        X, y, groups = cfg["loader"]()
        print(f"=== {name}, n_rounds={cfg['n_rounds']}, {N_TRIALS} seeds ===")
        for num_plants in PLANTS_SWEEP:
            honest_t, backdoor_t, flip_t = [], [], []
            for t in range(N_TRIALS):
                auc_h, auc_b, fr, n_target = run_once(X, y, groups, cfg, num_plants, seed=t)
                honest_t.append(auc_h)
                backdoor_t.append(auc_b)
                if not np.isnan(fr):
                    flip_t.append(fr)
            cost_t = [h - b for h, b in zip(honest_t, backdoor_t)]
            h_mean = float(np.mean(honest_t))
            c_mean, c_std = float(np.mean(cost_t)), float(np.std(cost_t, ddof=1))
            if flip_t:
                f_mean, f_std = float(np.mean(flip_t)), float(np.std(flip_t, ddof=1))
            else:
                f_mean, f_std = float("nan"), float("nan")
            print(f"  num_plants={num_plants:>2d}  honest={h_mean:.4f}  "
                  f"AUC_cost={c_mean:+.4f}+/-{c_std:.4f}  flip_rate={f_mean:.4f}+/-{f_std:.4f}  "
                  f"(n_target last trial={n_target})")


if __name__ == "__main__":
    main()
