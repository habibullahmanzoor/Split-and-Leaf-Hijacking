"""Adult, 10 matched seeds, default radius r=1, one malicious client.
 (1) Released-tree limited-knowledge attacker (reviewer 4, point 1).
 (2) Budget-matched baselines (reviewer 4, point 9): client-level label flipping and
     report-level bagging forgery, each sized to the same L1 deviation of the
     malicious client's root histogram (tree 0) as the clean one-cell injection.
Output: u_extra.json"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import rev_common as rc
import rev_attacks as ra
import unified as u
from harness.federated_gbdt import compute_bin_edges, bin_features

S = range(10)
res = {"variants": {}, "matched": {}}


def hit_rate(log):
    h = [r["target"] is not None and tuple(r["target"]) == r["winner"] for r in log]
    return float(np.mean(h)) if h else None


clean = [u.hfl_fit("adult", s)[0] for s in S]
res["clean"] = clean
print("clean", round(float(np.mean(clean)), 4), flush=True)

# ---------- (1) released-tree attacker ----------
for name, mk in (("released_x1.0", lambda: ra.ReleasedTreeAttack(1.0)),
                 ("released_x1.5", lambda: ra.ReleasedTreeAttack(1.5)),
                 ("released_x3.0", lambda: ra.ReleasedTreeAttack(3.0))):
    aucs, hits, used = [], [], []
    for s in S:
        att, log = mk(), []
        a, _ = u.hfl_fit("adult", s, attack=att, log=log)
        aucs.append(a)
        hits.append(hit_rate(log))
        used.append(float(np.mean(att.used_release)) if att.used_release else None)
    m, lo, hi = rc.paired(clean, aucs)
    res["variants"][name] = dict(auc=float(np.mean(aucs)), loss=m, ci=[lo, hi],
                                 hit=float(np.mean([h for h in hits if h is not None])),
                                 release_used=float(np.mean([x for x in used if x is not None])))
    print(name, res["variants"][name], flush=True)

# ---------- (2) budget-matched baselines ----------
F = u.hfl_split("adult", 0)[0].shape[1]


def l1_hist(X_binned, g, n_bins):
    tot = 0.0
    for f in range(X_binned.shape[1]):
        tot += np.abs(np.bincount(X_binned[:, f], weights=g, minlength=n_bins)).sum()
    return tot


def l1_dev_of_flip(Xb, y, perm, m, n_bins):
    y2 = y.copy()
    y2[perm[:m]] = 1 - y2[perm[:m]]
    g0, g1 = 0.5 - y, 0.5 - y2
    dev = 0.0
    for f in range(Xb.shape[1]):
        dev += np.abs(np.bincount(Xb[:, f], weights=g1 - g0, minlength=n_bins)).sum()
    return dev


flip_aucs, forge_aucs, info = [], [], []
for s in S:
    # budget: L1 deviation of the clean one-cell injection at the root of tree 0
    att = ra.FullKnowledgeOneCell()
    u.hfl_fit("adult", s, attack=att)
    delta0 = 1.01 * att.margins[0]

    Xtr, Xte, ytr, yte = u.hfl_split("adult", s)
    cd = u.hfl_clients(Xtr, ytr, s)
    edges = compute_bin_edges(np.vstack([X for X, _ in cd.values()]), u.HC["adult"]["bins"])
    nb = u.HC["adult"]["bins"]
    X0, y0 = cd[0]
    Xb0 = bin_features(X0, edges)
    g0_l1 = l1_hist(Xb0, 0.5 - y0, nb)

    # (a) label flip of m random rows of client 0, m chosen so the root-histogram deviation equals delta0
    perm = np.random.default_rng(s + 7).permutation(len(y0))
    lo_m, hi_m = 0, len(y0)
    while lo_m < hi_m:
        mid = (lo_m + hi_m) // 2
        if l1_dev_of_flip(Xb0, y0, perm, mid, nb) < delta0:
            lo_m = mid + 1
        else:
            hi_m = mid
    m_flip = lo_m
    dev_real = l1_dev_of_flip(Xb0, y0, perm, m_flip, nb)

    def hook(cd_, m_flip=m_flip, perm=perm):
        out = dict(cd_)
        X, y = out[0]
        y2 = y.copy()
        y2[perm[:m_flip]] = 1 - y2[perm[:m_flip]]
        out[0] = (X, y2)
        return out

    a, _ = u.hfl_fit("adult", s, client_hook=hook)
    flip_aucs.append(a)

    # (b) bagging prediction forgery with the same relative deviation
    b_rel = delta0 / g0_l1
    probs = []
    for c, (X, y) in cd.items():
        rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=s).fit(X, y)
        probs.append(u.rf_p1(rf, Xte))
    p0 = probs[0]
    lam_full = b_rel * np.abs(p0).sum() / np.abs(1 - 2 * p0).sum()
    lam = float(min(1.0, lam_full))
    forged = list(probs)
    forged[0] = (1 - lam) * p0 + lam * (1 - p0)
    forge_aucs.append(roc_auc_score(yte, np.mean(forged, axis=0)))
    base_bag = roc_auc_score(yte, np.mean(probs, axis=0))
    info.append(dict(seed=s, delta0=float(delta0), rel_budget=float(b_rel), m_flip=int(m_flip), n0=int(len(y0)),
                     flip_dev=float(dev_real), lam_needed=float(lam_full), lam_used=lam, bag_clean=float(base_bag)))
    print(info[-1], flush=True)

res["matched"]["label_flip"] = dict(auc=float(np.mean(flip_aucs)), loss=rc.paired(clean, flip_aucs)[0],
                                    ci=list(rc.paired(clean, flip_aucs)[1:]))
bag_clean = [i["bag_clean"] for i in info]
res["matched"]["bagging_forgery"] = dict(auc=float(np.mean(forge_aucs)), clean=float(np.mean(bag_clean)),
                                         loss=rc.paired(bag_clean, forge_aucs)[0],
                                         ci=list(rc.paired(bag_clean, forge_aucs)[1:]))
res["matched"]["info"] = info
print(res["matched"]["label_flip"], res["matched"]["bagging_forgery"], flush=True)
rc.save("u_extra", res)
print("saved")
