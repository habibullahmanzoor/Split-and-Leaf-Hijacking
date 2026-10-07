"""Excess flip rate of the planted trigger over the honest-model baseline (paired by seed)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
C, OLD = R / "vfl_u_cache", R / "vfl_cache"
T = json.load(open(R / "trigger_specificity.json"))
base = {ds: {r["seed"]: r for r in rows} for ds, rows in T.items()}


def load(ds, pl, s):
    f = (OLD / f"bd_{pl}_{s}.json") if (ds == "adult" and s < 5) else (C / f"bd_{ds}_{pl}_{s}.json")
    return json.loads(f.read_text())


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


out = {}
for ds, seeds, plants in (("adult", range(10), (1, 3, 5, 8, 13)), ("heart", range(5), (1, 5, 8)), ("credit", range(5), (1, 5, 8))):
    out[ds] = {"honest_flip": float(np.mean([base[ds][s]["flip"] for s in seeds])), "plants": {}}
    for pl in plants:
        ex, ex01, ex10 = [], [], []
        for s in seeds:
            r = load(ds, pl, s)
            b = base[ds][s]
            ex.append(r["flip"] - b["flip"])
            ex01.append(r["asr_0to1"] - b["a01"])
            ex10.append(r["asr_1to0"] - b["a10"])
        out[ds]["plants"][pl] = dict(excess=ci(ex), excess_0to1=ci(ex01), excess_1to0=ci(ex10),
                                     flip=float(np.mean([load(ds, pl, s)["flip"] for s in seeds])))
        print(ds, pl, "flip", round(out[ds]["plants"][pl]["flip"], 4), "honest", round(out[ds]["honest_flip"], 4),
              "excess", [round(x, 4) for x in out[ds]["plants"][pl]["excess"]])
(R / "excess_flip.json").write_text(json.dumps(out, indent=1))
