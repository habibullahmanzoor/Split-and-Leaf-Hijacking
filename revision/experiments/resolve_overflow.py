import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
C = R / "vfl_u_cache"
rep = json.loads((R / "overflow_repro2.json").read_text())
events = []
for key, runs in rep.items():
    f = C / f"{key}.json"
    old = json.loads(f.read_text())
    assert old.get("error") == "paillier_overflow", (key, old)
    oks = [r["result"] for r in runs if r["status"] == "ok"]
    assert len(oks) == 3 and all(o == oks[0] for o in oks), key       # identical across fresh keys
    events.append(dict(task=key, first_execution="paillier_overflow", reruns=3, reruns_ok=3, reruns_identical=True))
    f.write_text(json.dumps(oks[0]))
(R / "overflow_events.json").write_text(json.dumps(
    dict(events=events,
         note="Three tasks of the unified VFL batch raised a Paillier OverflowError on first execution; each was rerun three times with fresh keys, "
              "all nine reruns succeeded with identical results. A fourth overflow occurred earlier in the first converged batch (task not identified; "
              "its unfinished tasks were repeated eight times without overflow)."), indent=1))
print("resolved", [e["task"] for e in events])
