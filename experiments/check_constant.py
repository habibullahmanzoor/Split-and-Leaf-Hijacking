"""At full radius, is the attacked model a constant predictor? (answers a claim in the paper)"""
import json
import numpy as np
import rev_common as rc
import rev_attacks as ra
import unified as u

res = {}
for ds in ("adult", "heart", "credit"):
    D = u.HC[ds]["depth"]
    nuniq, constant = [], 0
    for s in range(10):
        auc, m = u.hfl_fit(ds, s, attack=ra.FullKnowledgeZero(), attack_depth=D + 1)
        Xtr, Xte, ytr, yte = u.hfl_split(ds, s)
        p = m.predict_proba(Xte)
        n = len(np.unique(np.round(p, 12)))
        nuniq.append(n)
        constant += int(n == 1)
    res[ds] = dict(n_constant=constant, unique_probs=nuniq)
    print(ds, res[ds], flush=True)
rc.save("constant_check", res)
