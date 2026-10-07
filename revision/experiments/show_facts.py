import json
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
H = json.load(open(R / "u_hfl.json"))
f = lambda v: f"{v['loss']:.4f} [{v['ci'][0]:.4f},{v['ci'][1]:.4f}] att={v['attacked']:.4f} clean={v['clean']:.4f}"
print("CLEAN", {k: round(sum(v) / len(v), 4) for k, v in H["clean"].items()})
for ds in H["count"]:
    for k, v in H["count"][ds].items():
        print("count", ds, k, f(v))
for n, d in H.get("count_adult", {}).items():
    for k, v in d.items():
        print("count_adult", n, k, f(v))
for ds in H["radius"]:
    for r, v in H["radius"][ds].items():
        print("radius", ds, r, f(v))
for k, v in H.get("federation", {}).items():
    print("fed", k, f(v))
for ds, d in H.get("transient", {}).items():
    import numpy as np
    fin = {k: np.array(v)[:, -1] for k, v in d.items()}
    clean = fin["clean"]
    for k in ("first", "last", "all"):
        diff = clean - fin[k]
        from scipy import stats
        h = stats.t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / len(diff) ** .5
        print("transient", ds, k, f"final={fin[k].mean():.4f} clean={clean.mean():.4f} loss={diff.mean():.4f} [{diff.mean()-h:.4f},{diff.mean()+h:.4f}]")
for k, v in H.get("depth", {}).items():
    print("depth", k, f(v))
for k, v in H.get("boosting_labelshuffle", {}).items():
    print("boost_ls", k, f(v))
if "bagging" in H:
    print("bag clean", sum(H["bagging"]["clean"]) / 10)
    for m in ("shuffle", "forge"):
        for k, v in H["bagging"].get(m, {}).items():
            print("bag", m, k, f(v))
