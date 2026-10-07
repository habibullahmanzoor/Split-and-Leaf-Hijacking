"""Unified VFL experiments (shared protocol of unified.py, 20 rounds, 5 matched
seeds), run in a process pool; one JSON per task so the run is resumable.
Tasks (all paired against the honest model of the same dataset and seed):
  scale  (ds, sf, seed, rounds)   ciphertext rescaling of one cell, sf=1 is the honest run
  leaf   (ds, p, seed)            misrouting probability p
  comb   (he, mis, seed)          Adult: uniform HE rescaling x misrouting
  bd     (ds, plants, seed)       prediction-disruption mechanism with directional / false-trigger metrics
  party  (n_passive, seed)        Adult: scale 10 with the attacker among n passive parties
Adult scale / bd tasks at 20 rounds reuse the earlier run's cache (identical protocol)."""
import sys
import json
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = HERE.parent / "results" / "vfl_u_cache"
OLD = HERE.parent / "results" / "vfl_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SEEDS = range(5)
DSS = ("adult", "heart", "credit")
OLDN = OLDNAME = {"adult": "adult", "heart": "heart_disease", "credit": "credit_default"}
TRIG_F, TRIG_B = 1, 8


def model(ds, rounds):
    from harness.vfl_secureboost import FederatedVFLGBDT
    import unified as u
    return FederatedVFLGBDT(n_bins=u.VC[ds]["bins"], max_depth=u.VC[ds]["depth"], n_rounds=rounds,
                            lr=0.3, key_size=512)


def fit_auc(ds, seed, factory=None, rounds=20):
    import unified as u
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = u.vfl_split(ds, seed)
    m = model(ds, rounds)
    m.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=({"party1": factory} if factory else None))
    return roc_auc_score(y_te, m.predict_proba(Xa_te, {"party1": Xp_te}))


def t_scale(ds, sf, seed, rounds=20):
    from attacks.he_crypto_layer import MaliciousPassiveParty
    fac = None if sf == 1.0 else partial(MaliciousPassiveParty, target_feature=0, scale_factor=sf)
    return dict(auc=fit_auc(ds, seed, fac, rounds))


def t_leaf(ds, p, seed):
    from attacks.leaf_routing import MisroutingPassiveParty
    fac = partial(MisroutingPassiveParty, misroute_fraction=p, _rng=np.random.default_rng(seed))
    return dict(auc=fit_auc(ds, seed, fac))


def t_comb(he, mis, seed):
    from attacks.multi_vector import CombinedPassiveParty
    fac = partial(CombinedPassiveParty, he_uniform_scale=he, misroute_fraction=mis,
                  _rng=np.random.default_rng(seed))
    return dict(auc=fit_auc("adult", seed, fac))


def t_party(n, seed):
    """Adult, scale-10 attack on the first of n passive parties sharing party1's columns."""
    from fed_datasets.loaders import load_adult
    from harness.vfl_secureboost import FederatedVFLGBDT
    from attacks.he_crypto_layer import MaliciousPassiveParty
    from sklearn.model_selection import train_test_split
    splits = {1: [slice(0, 8)], 2: [slice(0, 4), slice(4, 8)], 3: [slice(0, 3), slice(3, 6), slice(6, 8)]}
    X, y, g = load_adult()
    y = np.asarray(y)
    idx = np.random.default_rng(seed).choice(len(y), size=1500, replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    Xa = Xs[g["party0"]].to_numpy(dtype=float)
    cols = list(g["party1"])
    Xps = [Xs[cols[s]].to_numpy(dtype=float) for s in splits[n]]
    sp = train_test_split(Xa, *Xps, ys, test_size=0.25, random_state=seed, stratify=ys)
    Xa_tr, Xa_te = sp[0], sp[1]
    Xp_tr = {f"p{i}": sp[2 + 2 * i] for i in range(n)}
    Xp_te = {f"p{i}": sp[3 + 2 * i] for i in range(n)}
    y_tr, y_te = sp[-2], sp[-1]
    out = {}
    for lab, fac in (("honest", None),
                     ("attacked", {"p0": partial(MaliciousPassiveParty, target_feature=0, scale_factor=10.0)})):
        m = FederatedVFLGBDT(n_bins=16, max_depth=3, n_rounds=20, lr=0.3, key_size=512)
        m.fit(Xa_tr, y_tr, Xp_tr, passive_party_factories=fac)
        out[lab] = roc_auc_score(y_te, m.predict_proba(Xa_te, Xp_te))
    return out


def t_bd(ds, plants, seed):
    from attacks.passive_party_backdoor import BackdoorPassiveParty
    import unified as u
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = u.vfl_split(ds, seed)
    hon = model(ds, 20)
    hon.fit(Xa_tr, y_tr, {"party1": Xp_tr})
    p_h = hon.predict_proba(Xa_te, {"party1": Xp_te})
    bd = model(ds, 20)
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
    before, after = (p_b[tgt] >= 0.5).astype(int), (p_t[tgt] >= 0.5).astype(int)
    hp, bp = (p_h >= 0.5).astype(int), (p_b >= 0.5).astype(int)
    nt = np.setdiff1d(np.arange(len(y_te)), tgt)
    n0, n1 = int((before == 0).sum()), int((before == 1).sum())
    return dict(auc_h=roc_auc_score(y_te, p_h), auc_b=roc_auc_score(y_te, p_b),
                flip=float((before != after).mean()) if len(tgt) else None,
                asr_0to1=float(((before == 0) & (after == 1)).sum() / n0) if n0 else None,
                asr_1to0=float(((before == 1) & (after == 0)).sum() / n1) if n1 else None,
                n_target=int(len(tgt)),
                clean_disagree=float((hp != bp).mean()),
                clean_disagree_nontarget=float((hp[nt] != bp[nt]).mean()) if len(nt) else None,
                acc_clean_h=float((hp == y_te).mean()), acc_clean_b=float((bp == y_te).mean()))


def t_tgt(ds, cls_name, seed):
    import exp_vfl_converged as E
    return E.task_target(OLDN[ds], cls_name, seed)


def key(spec):
    return "_".join(str(x) for x in spec) + ".json"


def run(spec):
    kind, a = spec[0], spec[1:]
    f = CACHE / key(spec)
    if f.exists() and f.stat().st_size > 2:
        return spec, json.loads(f.read_text())
    # reuse the earlier 20-round Adult run (identical protocol)
    if kind == "scale" and a[0] == "adult" and a[3] == 20 and (OLD / f"scale_{a[1]}_{a[2]}.json").exists():
        res = json.loads((OLD / f"scale_{a[1]}_{a[2]}.json").read_text())
    elif kind == "bd" and a[0] == "adult" and (OLD / f"bd_{a[1]}_{a[2]}.json").exists():
        res = json.loads((OLD / f"bd_{a[1]}_{a[2]}.json").read_text())
    elif kind == "tgt" and (OLD / f"tgt_{OLDN[a[0]]}_{a[1]}_{a[2]}.json").exists():
        res = json.loads((OLD / f"tgt_{OLDN[a[0]]}_{a[1]}_{a[2]}.json").read_text())
    else:
        try:
            res = dict(scale=t_scale, leaf=t_leaf, comb=t_comb, party=t_party, bd=t_bd, tgt=t_tgt)[kind](*a)
        except OverflowError as e:
            res = dict(error="paillier_overflow", detail=str(e))
    f.write_text(json.dumps(res))
    return spec, res


def all_specs():
    sp = []
    for ds in ("heart", "credit"):
        sp += [("scale", ds, sf, s, 20) for sf in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0) for s in SEEDS]
    sp += [("scale", "adult", sf, s, 3) for sf in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0) for s in SEEDS]
    sp += [("leaf", ds, p, s) for ds in DSS for p in (0.1, 0.25, 0.5, 0.75, 1.0) for s in SEEDS]
    sp += [("comb", he, mis, s) for he in (1.0, 1.5, 3.0) for mis in (0.0, 0.15, 0.30)
           for s in SEEDS if not (he == 1.0 and mis == 0.0)]
    sp += [("bd", ds, pl, s) for ds in ("heart", "credit") for pl in (1, 5, 8) for s in SEEDS]
    sp += [("party", n, s) for n in (1, 2, 3) for s in SEEDS]
    sp += [("scale", "adult", 1.0, s, 20) for s in SEEDS]
    return sp


def ext_specs():
    """Adult headline VFL experiments extended from 5 to 10 seeds (seeds 5..9)."""
    ES = range(5, 10)
    sp = [("scale", "adult", sf, s, 20) for sf in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0) for s in ES]
    sp += [("leaf", "adult", p, s) for p in (0.1, 0.25, 0.5, 0.75, 1.0) for s in ES]
    sp += [("bd", "adult", pl, s) for pl in (1, 3, 5, 8, 13) for s in ES]
    sp += [("tgt", "adult", c, s) for c in ("fixed", "variance") for s in ES]
    return sp


if __name__ == "__main__":
    import os
    workers = int(os.environ.get("VFL_WORKERS", "8"))
    specs = ext_specs() if os.environ.get("VFL_EXT") else all_specs()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, sp) for sp in specs]
        for f in as_completed(futs):
            spec, _ = f.result()
            done += 1
            print(f"[{done}/{len(specs)}] {spec}", flush=True)
    print("ALL DONE")
