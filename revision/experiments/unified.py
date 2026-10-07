"""One shared data protocol for EVERY experiment in the revision (reviewer 4,
point 10). Before, each script drew its own split (a fixed random_state=0 in
some, per-seed in others), its own subsample (a fixed one reused across seeds
in some) and its own rounds/depth per dataset, so the honest baselines differed
between figures. Here:

  HFL : per seed, subsample (if configured) with rng(seed), stratified 80/20
        split with random_state=seed, Dirichlet(0.5) partition with seed,
        K=5 clients. One configuration per dataset (HC).
  VFL : per seed, subsample 1500 (Adult, Credit) with rng(seed), stratified
        75/25 split with random_state=seed, two parties (party0 active,
        party1 passive). One configuration per dataset (VC), 20 rounds.

A clean run and every attacked run of the same (dataset, seed) share all of
these draws, so every effect is a paired difference.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
from attack_category1_sweep import dirichlet_partition
import rev_common as rc

HC = {
    "adult": dict(loader=load_adult, bins=32, depth=4, rounds=20, sub=None, name="Adult"),
    "heart": dict(loader=load_heart_disease, bins=16, depth=3, rounds=15, sub=None, name="Heart Disease"),
    "credit": dict(loader=load_credit_default, bins=32, depth=4, rounds=20, sub=3000, name="Credit Default"),
}
VC = {
    "adult": dict(loader=load_adult, bins=16, depth=3, rounds=20, sub=1500, name="Adult"),
    "heart": dict(loader=load_heart_disease, bins=16, depth=3, rounds=20, sub=None, name="Heart Disease"),
    "credit": dict(loader=load_credit_default, bins=16, depth=3, rounds=20, sub=1500, name="Credit Default"),
}
N_CLIENTS, ALPHA, LR = 5, 0.5, 0.3
_CACHE = {}


def _raw(ds):
    if ds not in _CACHE:
        X, y, g = HC[ds]["loader"]()
        _CACHE[ds] = (X, np.asarray(y), g)
    return _CACHE[ds]


# ------------------------------------------------------------------ HFL
def hfl_split(ds, seed):
    X, y, _ = _raw(ds)
    X = X.to_numpy(dtype=float)
    cfg = HC[ds]
    if cfg["sub"] and len(y) > cfg["sub"]:
        idx = np.random.default_rng(seed).choice(len(y), size=cfg["sub"], replace=False)
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def hfl_clients(Xtr, ytr, seed, n_clients=N_CLIENTS):
    parts = dirichlet_partition(Xtr, ytr, n_clients, ALPHA, seed)
    parts = [np.asarray(p, dtype=int) for p in parts]
    return {c: (Xtr[p], ytr[p]) for c, p in enumerate(parts) if len(p) > 0}


def hfl_fit(ds, seed, attack=None, malicious=(0,), attack_depth=2, rounds=None,
            n_clients=N_CLIENTS, depth=None, curve=False, client_hook=None, log=None):
    """Train one HFL model. attack_depth = r + 1 (see paper, radius definition).
    Returns (test AUC, model) or (test AUC per round, model) if curve."""
    cfg = HC[ds]
    Xtr, Xte, ytr, yte = hfl_split(ds, seed)
    cd = hfl_clients(Xtr, ytr, seed, n_clients)
    if client_hook is not None:
        cd = client_hook(cd)
    kw = dict(malicious=malicious, attack=attack, attack_depth=attack_depth, rounds=rounds, log=log)
    aucs = []
    cb = (lambda t, m: aucs.append(roc_auc_score(yte, m.predict_proba(Xte)))) if curve else None
    m = rc.RevGBDT(n_bins=cfg["bins"], max_depth=depth or cfg["depth"], n_rounds=cfg["rounds"],
                   lr=LR, server_cls=rc.RevServer, server_kwargs=kw)
    m.fit(cd, eval_callback=cb)
    if curve:
        return aucs, m
    return roc_auc_score(yte, m.predict_proba(Xte)), m


# ------------------------------------------------------------------ VFL
def vfl_split(ds, seed):
    X, y, g = _raw(ds)
    cfg = VC[ds]
    Xs, ys = X, y
    if cfg["sub"] and len(y) > cfg["sub"]:
        idx = np.random.default_rng(seed).choice(len(y), size=cfg["sub"], replace=False)
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    Xa, Xp = Xs[g["party0"]].to_numpy(dtype=float), Xs[g["party1"]].to_numpy(dtype=float)
    return train_test_split(Xa, Xp, ys, test_size=0.25, random_state=seed, stratify=ys)


def rf_p1(rf, X):
    """P(class 1) from a forest that may have seen only one class (extreme label skew)."""
    pr = rf.predict_proba(X)
    cls = list(rf.classes_)
    return pr[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
