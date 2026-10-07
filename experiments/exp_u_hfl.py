"""Unified HFL experiments (10 matched seeds, shared protocol of unified.py).
Sections of u_hfl.json:
  clean[ds]            per-seed honest AUC
  count[ds][k]         zero attack, k = 1..3 malicious of 5
  count_adult          Adult: gauss5 / zero / one-cell for k = 1..3
  radius[ds][r]        zero attack, one client, r = 0..D
  federation[K]        Adult, zero attack, K = 5, 10, 20, 40
  transient[ds]        per-round test-AUC curves: clean / first / last (+ all)
  depth[D]             Adult, one client, r = 1, D = 4, 6, 8, 10
  bagging / boosting   label-shuffle and prediction-forgery controls, Adult
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import rev_common as rc
import rev_attacks as ra
import unified as u

S = range(10)
res = {}


def summ(clean, att):
    m, lo, hi = rc.paired(clean, att)
    return dict(clean=float(np.mean(clean)), attacked=float(np.mean(att)), loss=m, ci=[lo, hi],
                per_seed=[float(c - a) for c, a in zip(clean, att)])


def clean_of(ds):
    if ds not in res["clean"]:
        res["clean"][ds] = [u.hfl_fit(ds, s)[0] for s in S]
    return res["clean"][ds]


res["clean"] = {}


def checkpoint():
    rc.save("u_hfl", res)


for ds in ("adult", "heart", "credit"):
    clean_of(ds)
    print("clean", ds, round(float(np.mean(res["clean"][ds])), 4), flush=True)

checkpoint()
# ---- attacker count ----
res["count"] = {ds: {} for ds in u.HC}
for ds in u.HC:
    for k in (1, 2, 3):
        att = [u.hfl_fit(ds, s, attack=ra.FullKnowledgeZero(), malicious=tuple(range(k)))[0] for s in S]
        res["count"][ds][k] = summ(clean_of(ds), att)
        print("count", ds, k, round(res["count"][ds][k]["loss"], 4), flush=True)

res["count_adult"] = {}
for name, mk in (("gauss5", lambda s: ra.UnboundedNoise(s)), ("zero", lambda s: ra.FullKnowledgeZero()),
                 ("one_cell", lambda s: ra.FullKnowledgeOneCell())):
    res["count_adult"][name] = {}
    for k in (1, 2, 3):
        att = [u.hfl_fit("adult", s, attack=mk(s), malicious=tuple(range(k)))[0] for s in S]
        res["count_adult"][name][k] = summ(clean_of("adult"), att)
        print("count_adult", name, k, round(res["count_adult"][name][k]["loss"], 4), flush=True)

checkpoint()
# ---- radius ----
res["radius"] = {ds: {} for ds in u.HC}
for ds in u.HC:
    D = u.HC[ds]["depth"]
    for r in range(0, D + 1):
        att = [u.hfl_fit(ds, s, attack=ra.FullKnowledgeZero(), attack_depth=r + 1)[0] for s in S]
        res["radius"][ds][r] = summ(clean_of(ds), att)
        print("radius", ds, r, round(res["radius"][ds][r]["attacked"], 4), flush=True)

checkpoint()
# ---- federation size ----
res["federation"] = {}
for K in (5, 10, 20, 40):
    cl = [u.hfl_fit("adult", s, n_clients=K)[0] for s in S]
    att = [u.hfl_fit("adult", s, attack=ra.FullKnowledgeZero(), n_clients=K)[0] for s in S]
    res["federation"][K] = summ(cl, att)
    print("federation", K, round(res["federation"][K]["loss"], 4), flush=True)

checkpoint()
# ---- transient rounds, per-round curves ----
res["transient"] = {}
for ds in u.HC:
    R = u.HC[ds]["rounds"]
    conds = {"clean": (None, None), "first": (ra.FullKnowledgeZero, {0}), "last": (ra.FullKnowledgeZero, {R - 1}),
             "all": (ra.FullKnowledgeZero, None)}
    out = {}
    for name, (cls, rounds) in conds.items():
        curves = []
        for s in S:
            att = cls() if cls else None
            c, _ = u.hfl_fit(ds, s, attack=att, rounds=rounds, curve=True)
            curves.append([float(x) for x in c])
        out[name] = curves
    res["transient"][ds] = out
    fin = {k: np.array(v)[:, -1] for k, v in out.items()}
    print("transient", ds, {k: round(float(v.mean()), 4) for k, v in fin.items()}, flush=True)

checkpoint()
# ---- tree depth sensitivity ----
res["depth"] = {}
for D in (4, 6, 8, 10):
    cl = [u.hfl_fit("adult", s, depth=D)[0] for s in S]
    att = [u.hfl_fit("adult", s, attack=ra.FullKnowledgeZero(), attack_depth=2, depth=D)[0] for s in S]
    res["depth"][D] = summ(cl, att)
    print("depth", D, round(res["depth"][D]["loss"], 4), flush=True)

checkpoint()
# ---- aggregation controls (Adult) ----
def shuffle_hook(k, seed):
    def hook(cd):
        rng = np.random.default_rng(seed + 100)
        out = {}
        for c, (X, y) in cd.items():
            out[c] = (X, rng.permutation(y) if c < k else y)
        return out
    return hook


res["boosting_labelshuffle"] = {}
for k in (1, 2, 3):
    att = [u.hfl_fit("adult", s, client_hook=shuffle_hook(k, s))[0] for s in S]
    res["boosting_labelshuffle"][k] = summ(clean_of("adult"), att)
    print("boost label shuffle", k, round(res["boosting_labelshuffle"][k]["loss"], 4), flush=True)


def bagging(seed, k, mode):
    Xtr, Xte, ytr, yte = u.hfl_split("adult", seed)
    cd = u.hfl_clients(Xtr, ytr, seed)
    rng = np.random.default_rng(seed + 100)
    probs = []
    for c, (X, y) in cd.items():
        yy = rng.permutation(y) if (mode == "shuffle" and c < k) else y
        rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=seed).fit(X, yy)
        p = u.rf_p1(rf, Xte)
        if mode == "forge" and c < k:
            p = 1.0 - p
        probs.append(p)
    return roc_auc_score(yte, np.mean(probs, axis=0))


res["bagging"] = {"clean": [bagging(s, 0, "shuffle") for s in S], "shuffle": {}, "forge": {}}
print("bagging clean", round(float(np.mean(res["bagging"]["clean"])), 4), flush=True)
for mode in ("shuffle", "forge"):
    for k in (1, 2, 3):
        att = [bagging(s, k, mode) for s in S]
        res["bagging"][mode][k] = summ(res["bagging"]["clean"], att)
        print("bagging", mode, k, round(res["bagging"][mode][k]["loss"], 4), flush=True)

rc.save("u_hfl", res)
print("saved")
