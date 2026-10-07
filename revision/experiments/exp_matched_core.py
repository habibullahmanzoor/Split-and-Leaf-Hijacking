"""Reviewer 4 point 10: re-establish the older HFL figures' claims under the
matched design (ten seeds; split and partition redrawn per seed and shared by
the clean and attacked runs; paired differences with 95% CIs).
  (a) attacker count k = 1..3 of 5            (Fig. 1a claim: saturation)
  (b) radius r = 0..4 on Adult and Credit      (Fig. 12 claim: coverage)
  (c) federation size K = 5, 10, 20, 40        (Fig. 14a claim)
  (d) first-round / last-round corruption      (Fig. 2 claim: transient recovery)
Resumable: finished sections are read from results/matched_cache.json, which is
seeded from the log of the first (crashed) run."""
import ast
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import rev_common as rc
import rev_attacks as ra
from fed_datasets.loaders import load_adult, load_credit_default
from attack_category1_sweep import dirichlet_partition

LOAD = {"adult": load_adult, "credit_default": load_credit_default}
CACHE_DATA = {}
CACHE_FILE = rc.OUT / "matched_cache.json"
LOG = rc.OUT / "matched_core_run1.log"


def data(name, seed):
    if name not in CACHE_DATA:
        X, y, _ = LOAD[name]()
        CACHE_DATA[name] = (X.to_numpy(dtype=float), np.asarray(y))
    X, y = CACHE_DATA[name]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def fit(name, seed, n_clients=5, attack=None, malicious=(0,), attack_depth=2, rounds=None):
    Xtr, Xte, ytr, yte = data(name, seed)
    parts = dirichlet_partition(Xtr, ytr, n_clients, rc.ALPHA, seed)
    # a client can receive no rows when K is large; keep integer dtype so indexing works
    cd = {c: (Xtr[np.asarray(i, dtype=int)], ytr[np.asarray(i, dtype=int)])
          for c, i in enumerate(parts) if len(i) > 0}
    kw = dict(malicious=malicious, attack=attack, attack_depth=attack_depth, rounds=rounds)
    m = rc.RevGBDT(n_bins=rc.N_BINS, max_depth=rc.DEPTH, n_rounds=rc.ROUNDS, lr=rc.LR,
                   server_cls=rc.RevServer, server_kwargs=kw).fit(cd)
    return roc_auc_score(yte, m.predict_proba(Xte))


def summ(clean, att):
    m, lo, hi = rc.paired(clean, att)
    return dict(clean=float(np.mean(clean)), attacked=float(np.mean(att)), loss=m, ci=[lo, hi])


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    cache = {}
    if LOG.exists():     # seed from the first run's log
        for ln in LOG.read_text().splitlines():
            parts = ln.split(" ", 3)
            if parts[0] in ("count", "radius", "federation", "transient") and "{" in ln:
                head, body = ln.split(" {", 1)
                toks = head.split(" ")
                key = "|".join(toks)
                cache[key] = ast.literal_eval("{" + body)
    return cache


def get(cache, key, fn):
    if key not in cache:
        cache[key] = fn()
        CACHE_FILE.write_text(json.dumps(cache, indent=1))
    print(key, cache[key], flush=True)
    return cache[key]


if __name__ == "__main__":
    S = range(rc.N_SEEDS)
    cache = load_cache()
    clean_adult = [fit("adult", s) for s in S]
    res = {"count": {}, "radius": {}, "federation": {}, "transient": {}}

    for k in (1, 2, 3):
        res["count"][k] = get(cache, f"count|{k}", lambda k=k: summ(
            clean_adult, [fit("adult", s, attack=ra.FullKnowledgeZero(), malicious=tuple(range(k))) for s in S]))

    clean_cd = [fit("credit_default", s) for s in S]
    for name, clean in (("adult", clean_adult), ("credit_default", clean_cd)):
        for r in range(0, 5):
            res["radius"][f"{name}|r={r}"] = get(cache, f"radius|{name}|{r}", lambda name=name, r=r, clean=clean: summ(
                clean, [fit(name, s, attack=ra.FullKnowledgeZero(), attack_depth=r + 1) for s in S]))

    for K in (5, 10, 20, 40):
        def run(K=K):
            clean = clean_adult if K == 5 else [fit("adult", s, n_clients=K) for s in S]
            return summ(clean, [fit("adult", s, n_clients=K, attack=ra.FullKnowledgeZero()) for s in S])
        res["federation"][K] = get(cache, f"federation|{K}", run)

    for lab, rr in (("first", {0}), ("last", {rc.ROUNDS - 1}), ("all", None)):
        res["transient"][lab] = get(cache, f"transient|{lab}", lambda rr=rr: summ(
            clean_adult, [fit("adult", s, attack=ra.FullKnowledgeZero(), rounds=rr) for s in S]))

    rc.save("matched_core", res)
    print("saved")
