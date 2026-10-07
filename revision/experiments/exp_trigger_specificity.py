"""Trigger specificity (reviewer 4, point 5): apply the SAME test-time manipulation used by the
prediction-disruption mechanism to an HONEST model, i.e. one with no planted trigger. The flip rate
of the honest model is the rate at which the manipulation changes predictions by itself (a false
trigger). Sequential, one training per (dataset, seed)."""
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import unified as u
from exp_u_vfl import model, TRIG_F, TRIG_B

OUT = {}
for ds, seeds in (("adult", range(10)), ("heart", range(5)), ("credit", range(5))):
    rows = []
    for s in seeds:
        Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = u.vfl_split(ds, s)
        m = model(ds, 20)
        m.fit(Xa_tr, y_tr, {"party1": Xp_tr})
        p0 = m.predict_proba(Xa_te, {"party1": Xp_te})
        edges = m.passive_bin_edges["party1"][TRIG_F]
        thr = edges[min(TRIG_B, len(edges) - 1)]
        tgt = np.where(Xp_te[:, TRIG_F] <= thr)[0]
        Xm = Xp_te.copy()
        Xm[tgt, TRIG_F] = Xp_tr[:, TRIG_F].max() + 1.0
        p1 = m.predict_proba(Xa_te, {"party1": Xm})
        b, a = (p0[tgt] >= 0.5).astype(int), (p1[tgt] >= 0.5).astype(int)
        rows.append(dict(seed=s, n_target=int(len(tgt)), flip=float((b != a).mean()) if len(tgt) else None,
                         a01=float(((b == 0) & (a == 1)).sum() / max((b == 0).sum(), 1)),
                         a10=float(((b == 1) & (a == 0)).sum() / max((b == 1).sum(), 1))))
        print(ds, s, rows[-1], flush=True)
    OUT[ds] = rows
    (HERE.parent / "results" / "trigger_specificity.json").write_text(json.dumps(OUT, indent=1))
print("done")
