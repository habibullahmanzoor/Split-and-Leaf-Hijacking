"""Does category 3's ciphertext rescaling attack's damage depend on how
many total passive parties exist in the deployment, or only on the
malicious party's own feature? Every VFL experiment in this paper uses
exactly one active and one passive party; FederatedVFLGBDT.fit() already
takes X_passives as a dict keyed by party id and loops over every entry in
_candidate_splits, so more passive parties work with zero harness changes.
This splits Adult's existing party1 feature block into 1, 2, or 3 separate
passive parties (holding the same total features, just partitioned across
more institutions), attacks the SAME underlying feature (the first column
of the first passive party, "workclass") in every configuration, and holds
every other passive party honest.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.he_crypto_layer import MaliciousPassiveParty

N_TRIALS = 5
SCALE_FACTOR = 10.0
PARTY_SPLITS = {
    1: [slice(0, 8)],
    2: [slice(0, 4), slice(4, 8)],
    3: [slice(0, 3), slice(3, 6), slice(6, 8)],
}


def run_once(X, y, groups, n_passive, seed, n_rounds=3):
    party0_cols = groups["party0"]
    party1_cols = list(groups["party1"])
    rng = np.random.default_rng(seed)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    groups_cols = [party1_cols[s] for s in PARTY_SPLITS[n_passive]]
    Xp_list = [X_sub[cols].to_numpy(dtype=float) for cols in groups_cols]

    split_arrays = train_test_split(Xa, *Xp_list, y_sub, test_size=0.25, random_state=seed, stratify=y_sub)
    Xa_tr, Xa_te = split_arrays[0], split_arrays[1]
    Xp_tr_list = [split_arrays[2 + 2 * i] for i in range(len(Xp_list))]
    Xp_te_list = [split_arrays[2 + 2 * i + 1] for i in range(len(Xp_list))]
    y_tr, y_te = split_arrays[-2], split_arrays[-1]

    party_ids = [f"party1_{i}" for i in range(n_passive)]
    Xp_tr_dict = dict(zip(party_ids, Xp_tr_list))
    Xp_te_dict = dict(zip(party_ids, Xp_te_list))

    honest = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=n_rounds, lr=0.3, key_size=512)
    honest.fit(Xa_tr, y_tr, Xp_tr_dict)
    auc_honest = roc_auc_score(y_te, honest.predict_proba(Xa_te, Xp_te_dict))

    factory = partial(MaliciousPassiveParty, target_feature=0, scale_factor=SCALE_FACTOR)
    model = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=n_rounds, lr=0.3, key_size=512)
    model.fit(Xa_tr, y_tr, Xp_tr_dict, passive_party_factories={party_ids[0]: factory})
    auc_attacked = roc_auc_score(y_te, model.predict_proba(Xa_te, Xp_te_dict))

    return auc_honest, auc_attacked


def main():
    X, y, groups = load_adult()
    for n_passive in [1, 2, 3]:
        honest_t, attacked_t = [], []
        for t in range(N_TRIALS):
            ah, aa = run_once(X, y, groups, n_passive, seed=t)
            honest_t.append(ah)
            attacked_t.append(aa)
        h_mean = float(np.mean(honest_t))
        a_mean, a_std = float(np.mean(attacked_t)), float(np.std(attacked_t, ddof=1))
        drop_trials = [h - a for h, a in zip(honest_t, attacked_t)]
        d_mean, d_std = float(np.mean(drop_trials)), float(np.std(drop_trials, ddof=1))
        print(f"n_passive_parties={n_passive}  honest={h_mean:.4f}  attacked={a_mean:.4f}+/-{a_std:.4f}  "
              f"drop={d_mean:.4f}+/-{d_std:.4f}")


if __name__ == "__main__":
    main()
