"""Rerun the three tasks that raised a Paillier OverflowError, three times each, one after the other.
Each repeat generates a fresh random Paillier key. Outputs overflow_repro2.json."""
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import exp_u_vfl as E

TASKS = [("leaf", "adult", 1.0, 7), ("bd", "credit", 8, 1), ("party", 3, 0)]
fn = dict(leaf=E.t_leaf, bd=E.t_bd, party=E.t_party)
out = {}
for spec in TASKS:
    key = "_".join(str(x) for x in spec)
    out[key] = []
    for rep in range(3):
        try:
            r = fn[spec[0]](*spec[1:])
            out[key].append(dict(status="ok", result=r))
        except OverflowError as e:
            out[key].append(dict(status="OVERFLOW", detail=str(e)))
        print(key, rep, out[key][-1]["status"], flush=True)
        (HERE.parent / "results" / "overflow_repro2.json").write_text(json.dumps(out, indent=1))
print("done")
