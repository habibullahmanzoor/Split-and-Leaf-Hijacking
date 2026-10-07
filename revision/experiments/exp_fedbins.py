"""Reviewer point 4: does the heterogeneity invariance (Prop. 1) and the attack
survive realistic federated binning (merged local quantile sketches) instead of
centralized bin edges?"""
import numpy as np
import rev_common as rc
import rev_attacks as ra
from harness.federated_gbdt import Client, bin_features, compute_bin_edges, Server
from theory.margin import compute_split_flip_margin

S = range(rc.N_SEEDS)
res = {}

# --- (a) invariance of the root split-flip margin across partitions (fixed split) ---
Xtr, Xte, ytr, yte = rc.get_data(0)
margins = {"central": [], "fed": []}
winners = {"central": [], "fed": []}
for mode in ("central", "fed"):
    for part_seed in range(6):
        cd = rc.make_clients(Xtr, ytr, 100 + part_seed)
        Xs = [X for X, _ in cd.values()]
        edges = (rc.federated_bin_edges(Xs, rc.N_BINS) if mode == "fed"
                 else compute_bin_edges(np.vstack(Xs), rc.N_BINS))
        clients = [Client(client_id=c, X=bin_features(X, edges), y=y, bin_edges=edges)
                   for c, (X, y) in cd.items()]
        srv = Server(clients, rc.N_BINS, rc.DEPTH)
        idx = {c.client_id: np.arange(len(c.y)) for c in clients}
        ag, ah = srv._aggregate_histograms(idx)
        (g1, f1, b1), (g2, f2, b2) = srv._find_best_split(ag, ah)
        m, _ = compute_split_flip_margin(ag, ah, f2, b2)
        margins[mode].append(float(m))
        winners[mode].append([int(f1), int(b1)])
res["margin_across_partitions"] = {
    k: dict(values=v, rel_range=float((max(v) - min(v)) / np.mean(v)), winners=winners[k])
    for k, v in margins.items()}
for k, v in margins.items():
    print(k, "margins", [round(x, 3) for x in v], "relative range", (max(v) - min(v)) / np.mean(v))

# --- (b) honest AUC and attack under both binning schemes, paired over matched seeds ---
DATA = {s: rc.get_data(s) for s in S}
for mode in ("central", "fed"):
    fb = mode == "fed"
    clean, one, zero = [], [], []
    for s in S:
        Xtr, Xte, ytr, yte = DATA[s]
        clean.append(rc.fit_eval(Xtr, ytr, Xte, yte, s, fed_bins=fb)[0])
        one.append(rc.fit_eval(Xtr, ytr, Xte, yte, s, attack=ra.FullKnowledgeOneCell(), fed_bins=fb)[0])
        zero.append(rc.fit_eval(Xtr, ytr, Xte, yte, s, attack=ra.FullKnowledgeZero(), fed_bins=fb)[0])
    res[mode] = dict(clean_mean=float(np.mean(clean)), clean=clean,
                     one_cell=dict(auc=float(np.mean(one)), loss=rc.paired(clean, one)),
                     zero=dict(auc=float(np.mean(zero)), loss=rc.paired(clean, zero)))
    print(mode, "clean", np.mean(clean), "one-cell loss", rc.paired(clean, one), "zero loss", rc.paired(clean, zero))

d = [a - b for a, b in zip(res["central"]["clean"], res["fed"]["clean"])]
res["central_minus_fed_clean_auc"] = dict(mean=float(np.mean(d)), ci=list(rc.paired(res["central"]["clean"], res["fed"]["clean"])[1:]))
rc.save("fedbins", res)
print("saved")
