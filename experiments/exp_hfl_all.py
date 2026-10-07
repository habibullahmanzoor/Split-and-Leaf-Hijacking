"""Revision HFL experiments (Adult, 5 clients, 10 matched seeds, paired vs clean).
Covers reviewer points 1 (limited knowledge), 2 (bounded reports), 3
(conservation + detectors), 7 (persistence schedules), 9 (matched budgets)."""
import numpy as np
import rev_common as rc
import rev_attacks as ra
from sklearn.metrics import roc_auc_score

import os
AD = int(os.environ.get("ATTACK_DEPTH", "2"))   # depth indices 0..AD-1 attacked; 2 = root + children (original default)
S = range(rc.N_SEEDS)
DATA = {s: rc.get_data(s) for s in S}


def run(att, seed, **kw):
    log = []
    Xtr, Xte, ytr, yte = DATA[seed]
    auc, m = rc.fit_eval(Xtr, ytr, Xte, yte, seed, attack=att, log=log, attack_depth=AD, **kw)
    node = [r for r in log]
    hit = [r["target"] is not None and tuple(r["target"]) == r["winner"] for r in node]
    det = {k: [] for k in ("cons", "mag", "l1", "any")}
    fpr = {k: [] for k in det}
    for r in node:
        for d in r["mal"]:
            for k in det:
                det[k].append(d[k])
        for d in r["hon"]:
            for k in fpr:
                fpr[k].append(d[k])
    return dict(auc=auc, hit=float(np.mean(hit)) if hit else None, n_nodes=len(node),
                det={k: float(np.mean(v)) if v else None for k, v in det.items()},
                fpr={k: float(np.mean(v)) if v else None for k, v in fpr.items()})


def summarize(runs, clean):
    a = [r["auc"] for r in runs]
    mean, lo, hi = rc.paired(clean, a)
    hits = [r["hit"] for r in runs if r["hit"] is not None]
    out = dict(auc_mean=float(np.mean(a)), loss=mean, ci=[lo, hi],
               hit=float(np.mean(hits)) if hits else None)
    for k in ("cons", "mag", "l1", "any"):
        v = [r["det"][k] for r in runs if r["det"][k] is not None]
        f = [r["fpr"][k] for r in runs if r["fpr"][k] is not None]
        out[f"tpr_{k}"] = float(np.mean(v)) if v else None
        out[f"fpr_{k}"] = float(np.mean(f)) if f else None
    return out


clean = [rc.fit_eval(*[DATA[s][i] for i in (0, 2, 1, 3)], s)[0] for s in S]
res = {"clean_mean": float(np.mean(clean)), "clean": clean}
print("clean", np.mean(clean))

# ---- E1/E3/E9: attacker variants (k=1 malicious, root only) ----
variants = {
    "full_zero": lambda s: ra.FullKnowledgeZero(),
    "full_one_cell": lambda s: ra.FullKnowledgeOneCell(),
    "limited_x1.0": lambda s: ra.LimitedKnowledge(1.0),
    "limited_x1.5": lambda s: ra.LimitedKnowledge(1.5),
    "limited_x3.0": lambda s: ra.LimitedKnowledge(3.0),
    "limited_localonly_x1.5": lambda s: ra.LimitedKnowledge(1.5, mode="local"),
    "cons_pair": lambda s: ra.ConservationPair(),
    "matched_noise": lambda s: ra.MatchedNoise(s),
    "matched_random_cell": lambda s: ra.MatchedRandomCell(s),
    "gauss5": lambda s: ra.UnboundedNoise(s),
}
res["variants"] = {}
for name, mk in variants.items():
    runs = [run(mk(s), s) for s in S]
    res["variants"][name] = summarize(runs, clean)
    v = res["variants"][name]
    print(f"{name:26s} loss={v['loss']:.4f} CI={v['ci'][0]:.4f},{v['ci'][1]:.4f} hit={v['hit']} "
          f"tpr_cons={v['tpr_cons']} tpr_any={v['tpr_any']} fpr_any={v['fpr_any']}")

# ---- label-flip data poisoning (client-level), L1 deviation measured at root ----
lf = []
for s in S:
    Xtr, Xte, ytr, yte = DATA[s]
    cd = rc.make_clients(Xtr, ytr, s)
    X0, y0 = cd[0]
    cd[0] = (X0, 1 - y0)
    m = rc.RevGBDT(n_bins=rc.N_BINS, max_depth=rc.DEPTH, n_rounds=rc.ROUNDS, lr=rc.LR,
                   server_cls=rc.RevServer, server_kwargs={}).fit(cd)
    lf.append(roc_auc_score(yte, m.predict_proba(Xte)))
res["variants"]["label_flip_client0"] = dict(auc_mean=float(np.mean(lf)),
                                             loss=rc.paired(clean, lf)[0],
                                             ci=list(rc.paired(clean, lf)[1:]))
print("label_flip", res["variants"]["label_flip_client0"])

# ---- E2: bounded reports x colluders ----
res["bounded"] = {}
for rho in (0.05, 0.1, 0.25, 0.5, 1.0, None):
    for k in (1, 2, 3, 4):
        runs = [run(ra.FullKnowledgeOneCell(bound_rho=rho), s, malicious=tuple(range(k))) for s in S]
        key = f"rho={rho}|k={k}"
        res["bounded"][key] = summarize(runs, clean)
        v = res["bounded"][key]
        print(f"{key:16s} loss={v['loss']:.4f} CI={v['ci'][0]:.4f},{v['ci'][1]:.4f} hit={v['hit']}")

# ---- E7: persistence schedules ----
R = rc.ROUNDS
res["persistence"] = {}
for m in (2, 5, 10, 15, 20):
    tau = m / R
    scheds = {
        "contig_early": set(range(m)),
        "contig_late": set(range(R - m, R)),
        "contig_middle": set(range((R - m) // 2, (R - m) // 2 + m)),
        "intermittent": set(int(round(i * (R - 1) / max(m - 1, 1))) for i in range(m)) if m > 1 else {0},
    }
    for sname, rounds in scheds.items():
        runs = [run(ra.FullKnowledgeZero(), s, rounds=rounds) for s in S]
        res["persistence"][f"tau={tau}|{sname}"] = summarize(runs, clean)
    runs = []
    for s in S:
        rr = set(np.random.default_rng(1000 + s).choice(R, m, replace=False).tolist())
        runs.append(run(ra.FullKnowledgeZero(), s, rounds=rr))
    res["persistence"][f"tau={tau}|random"] = summarize(runs, clean)
    for sname in list(scheds) + ["random"]:
        v = res["persistence"][f"tau={tau}|{sname}"]
        print(f"tau={tau:.2f} {sname:14s} loss={v['loss']:.4f} CI={v['ci'][0]:.4f},{v['ci'][1]:.4f}")

rc.save("hfl_default" if AD == 2 else f"hfl_depth{AD}", res)
print("saved")
