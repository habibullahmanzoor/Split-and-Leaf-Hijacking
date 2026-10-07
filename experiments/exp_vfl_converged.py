"""Revision VFL experiments at (near-)converged training (n_rounds=20; Fig. 0
shows the Adult VFL test AUC plateaus by round ~20). Covers reviewer point 6
(converged reruns of ciphertext scale, prediction disruption, target selection)
and the extra backdoor metrics requested under point 5 (directional success
rates, false-trigger rate, clean behaviour). Tasks run in a process pool and
are cached one JSON per task, so the run is resumable."""
import sys
import json
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parents[1] / "results" / "vfl_cache"
CACHE.mkdir(parents=True, exist_ok=True)
N_ROUNDS = 20
SEEDS = range(5)
SCALES = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
PLANTS = [1, 3, 5, 8, 13]
TRIG_F, TRIG_B = 1, 8


def split(X, y, groups, seed, subsample=1500):
    p0, p1 = groups["party0"], groups["party1"]
    Xs, ys = X, y
    if subsample and len(y) > subsample:
        idx = np.random.default_rng(seed).choice(len(y), size=subsample, replace=False)
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    Xa, Xp = Xs[p0].to_numpy(dtype=float), Xs[p1].to_numpy(dtype=float)
    return train_test_split(Xa, Xp, ys, test_size=0.25, random_state=seed, stratify=ys)


def task_scale(sf, seed):
    from fed_datasets.loaders import load_adult
    from harness.vfl_secureboost import FederatedVFLGBDT
    from attacks.he_crypto_layer import MaliciousPassiveParty
    X, y, g = load_adult()
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = split(X, y, g, seed)
    m = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    fac = None if sf == 1.0 else {"party1": partial(MaliciousPassiveParty, target_feature=0, scale_factor=sf)}
    m.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=fac)
    return dict(auc=roc_auc_score(y_te, m.predict_proba(Xa_te, {"party1": Xp_te})))


def task_backdoor(plants, seed):
    from fed_datasets.loaders import load_adult
    from harness.vfl_secureboost import FederatedVFLGBDT
    from attacks.passive_party_backdoor import BackdoorPassiveParty
    X, y, g = load_adult()
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = split(X, y, g, seed)
    hon = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    hon.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    p_h = hon.predict_proba(Xa_te, {"party1": Xp_te})
    bd = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
    fac = {"party1": partial(BackdoorPassiveParty, trigger_feature=TRIG_F, trigger_bin=TRIG_B,
                              scale_factor=10.0, num_plants=plants)}
    bd.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=fac)
    p_b = bd.predict_proba(Xa_te, {"party1": Xp_te})
    edges = bd.passive_bin_edges["party1"][TRIG_F]
    thr = edges[min(TRIG_B, len(edges) - 1)]
    tgt = np.where(Xp_te[:, TRIG_F] <= thr)[0]
    Xm = Xp_te.copy()
    Xm[tgt, TRIG_F] = Xp_tr[:, TRIG_F].max() + 1.0
    p_t = bd.predict_proba(Xa_te, {"party1": Xm})
    before = (p_b[tgt] >= 0.5).astype(int)
    after = (p_t[tgt] >= 0.5).astype(int)
    hon_pred = (p_h >= 0.5).astype(int)
    bd_pred = (p_b >= 0.5).astype(int)
    n0, n1 = (before == 0).sum(), (before == 1).sum()
    nontgt = np.setdiff1d(np.arange(len(y_te)), tgt)
    return dict(
        auc_h=roc_auc_score(y_te, p_h), auc_b=roc_auc_score(y_te, p_b),
        flip=float((before != after).mean()) if len(tgt) else None,
        asr_0to1=float(((before == 0) & (after == 1)).sum() / n0) if n0 else None,
        asr_1to0=float(((before == 1) & (after == 0)).sum() / n1) if n1 else None,
        n_target=int(len(tgt)),
        # false triggers: disagreement between honest and backdoored models on
        # untouched inputs (clean behaviour), overall and on non-target inputs
        clean_disagree=float((hon_pred != bd_pred).mean()),
        clean_disagree_nontarget=float((hon_pred[nontgt] != bd_pred[nontgt]).mean()) if len(nontgt) else None,
        # accuracy on non-triggered samples
        acc_clean_h=float((hon_pred == y_te).mean()), acc_clean_b=float((bd_pred == y_te).mean()),
        # trigger applied to NON-target inputs does nothing by construction; verify empirically
        nontarget_flip=float(((bd.predict_proba(Xa_te, {"party1": Xm})[nontgt] >= 0.5).astype(int)
                              != bd_pred[nontgt]).mean()) if len(nontgt) else None,
    )


def task_target(name, cls_name, seed):
    from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
    from harness.vfl_secureboost import FederatedVFLGBDT
    from attacks.he_crypto_layer import MaliciousPassiveParty
    from attacks.adaptive_target_selection import VarianceTargetedMaliciousPassiveParty
    loader = dict(adult=load_adult, heart_disease=load_heart_disease, credit_default=load_credit_default)[name]
    X, y, g = loader()
    sub = None if name == "heart_disease" else 1500
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = split(X, y, g, seed, subsample=sub)
    cls = dict(fixed=MaliciousPassiveParty, variance=VarianceTargetedMaliciousPassiveParty)[cls_name]
    out = {}
    for label, fac in (("honest", None),
                       ("attacked", {"party1": partial(cls, target_feature=0, scale_factor=10.0)})):
        m = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=N_ROUNDS, lr=0.3, key_size=512)
        m.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=fac)
        out[label] = roc_auc_score(y_te, m.predict_proba(Xa_te, {"party1": Xp_te}))
    return out


def run_task(spec):
    kind, args = spec[0], spec[1:]
    key = CACHE / ("_".join(map(str, spec)) + ".json")
    if key.exists():
        return spec, json.loads(key.read_text())
    try:
        res = dict(scale=task_scale, bd=task_backdoor, tgt=task_target)[kind](*args)
    except OverflowError as e:
        res = dict(error='paillier_overflow', detail=str(e))
    key.write_text(json.dumps(res))
    return spec, res


if __name__ == "__main__":
    specs = [("scale", sf, s) for sf in SCALES for s in SEEDS]
    specs += [("bd", p, s) for p in PLANTS for s in SEEDS]
    specs += [("tgt", d, c, s) for d in ("adult", "heart_disease", "credit_default")
              for c in ("fixed", "variance") for s in SEEDS]
    done = 0
    with ProcessPoolExecutor(max_workers=9) as ex:
        futs = [ex.submit(run_task, sp) for sp in specs]
        for f in as_completed(futs):
            spec, _ = f.result()
            done += 1
            print(f"[{done}/{len(specs)}] {spec}", flush=True)
    print("ALL DONE")
