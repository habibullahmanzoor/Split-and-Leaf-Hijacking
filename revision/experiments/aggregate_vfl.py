"""Aggregate the unified VFL task caches into u_vfl_summary.json (paired differences, 95% CIs)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
C = R / "vfl_u_cache"
OLD = R / "vfl_cache"
SEEDS = range(5)
SA = range(10)      # Adult headline experiments run with ten seeds
sd_ = lambda ds: SA if ds == 'adult' else SEEDS
DS = ("adult", "heart", "credit")
OLDN = {"adult": "adult", "heart": "heart_disease", "credit": "credit_default"}


def L(name, base=C):
    return json.loads((base / f"{name}.json").read_text())


def ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return [float(d.mean()), float(d.mean()), float(d.mean())]
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


out = {"scale": {}, "scale3": {}, "leaf": {}, "comb": {}, "bd": {}, "party": {}, "tgt": {}, "honest": {}}
SF = (1.0, 2.0, 5.0, 10.0, 50.0, 200.0)

for ds in DS:
    hon = [L(f"scale_{ds}_1.0_{s}_20")["auc"] for s in sd_(ds)]
    out["honest"][ds] = dict(mean=float(np.mean(hon)), sd=float(np.std(hon, ddof=1)), per_seed=hon)
    out["scale"][ds] = {}
    for sf in SF[1:]:
        a = [L(f"scale_{ds}_{sf}_{s}_20")["auc"] for s in sd_(ds)]
        out["scale"][ds][sf] = dict(auc=float(np.mean(a)), sd=float(np.std(a, ddof=1)),
                                    loss=ci(np.array(hon) - np.array(a)))

hon3 = [L(f"scale_adult_1.0_{s}_3")["auc"] for s in SEEDS]
out["honest"]["adult_3rounds"] = dict(mean=float(np.mean(hon3)), sd=float(np.std(hon3, ddof=1)))
for sf in SF[1:]:
    a = [L(f"scale_adult_{sf}_{s}_3")["auc"] for s in SEEDS]
    out["scale3"][sf] = dict(auc=float(np.mean(a)), loss=ci(np.array(hon3) - np.array(a)))

PS = (0.1, 0.25, 0.5, 0.75, 1.0)
for ds in DS:
    hon = out["honest"][ds]["per_seed"]
    out["leaf"][ds] = {0.0: dict(auc=float(np.mean(hon)), sd=float(np.std(hon, ddof=1)), loss=[0.0, 0.0, 0.0])}
    for p in PS:
        a = [L(f"leaf_{ds}_{p}_{s}")["auc"] for s in sd_(ds)]
        out["leaf"][ds][p] = dict(auc=float(np.mean(a)), sd=float(np.std(a, ddof=1)),
                                  loss=ci(np.array(hon) - np.array(a)), per_seed=a)

hon = np.array(out["honest"]["adult"]["per_seed"])[:5]
cell = {(1.0, 0.0): hon}
for he in (1.0, 1.5, 3.0):
    for mis in (0.0, 0.15, 0.30):
        if (he, mis) != (1.0, 0.0):
            cell[(he, mis)] = np.array([L(f"comb_{he}_{mis}_{s}")["auc"] for s in SEEDS])
out["comb"]["cells"] = {f"{he}|{mis}": dict(auc=float(v.mean()), sd=float(v.std(ddof=1))) for (he, mis), v in cell.items()}
out["comb"]["synergy"] = {}
for he in (1.5, 3.0):
    for mis in (0.15, 0.30):
        syn = cell[(he, 0.0)] + cell[(1.0, mis)] - cell[(1.0, 0.0)] - cell[(he, mis)]
        out["comb"]["synergy"][f"{he}|{mis}"] = dict(
            split_loss=float((hon - cell[(he, 0.0)]).mean()), leaf_loss=float((hon - cell[(1.0, mis)]).mean()),
            comb_loss=float((hon - cell[(he, mis)]).mean()), syn=ci(syn))

for ds in ("heart", "credit"):
    out["bd"][ds] = {}
    for pl in (1, 5, 8):
        R_ = [L(f"bd_{ds}_{pl}_{s}") for s in SEEDS]
        g = lambda k: [r[k] for r in R_ if r[k] is not None]
        out["bd"][ds][pl] = dict(
            cost=ci([r["auc_h"] - r["auc_b"] for r in R_]), flip=float(np.mean(g("flip"))),
            flip_sd=float(np.std(g("flip"), ddof=1)), a01=float(np.mean(g("asr_0to1"))) if g("asr_0to1") else None,
            a10=float(np.mean(g("asr_1to0"))) if g("asr_1to0") else None,
            dis_nt=float(np.mean(g("clean_disagree_nontarget"))), n_target=float(np.mean(g("n_target"))),
            acc_h=float(np.mean(g("acc_clean_h"))), acc_b=float(np.mean(g("acc_clean_b"))))
out["bd"]["adult"] = {}
for pl in (1, 3, 5, 8, 13):
    R_ = [L(f"bd_{pl}_{s}", OLD) for s in SEEDS] + [L(f"bd_adult_{pl}_{s}") for s in range(5, 10)]
    g = lambda k: [r[k] for r in R_ if r[k] is not None]
    out["bd"]["adult"][pl] = dict(
        cost=ci([r["auc_h"] - r["auc_b"] for r in R_]), flip=float(np.mean(g("flip"))),
        flip_sd=float(np.std(g("flip"), ddof=1)), a01=float(np.mean(g("asr_0to1"))), a10=float(np.mean(g("asr_1to0"))),
        dis_nt=float(np.mean(g("clean_disagree_nontarget"))), n_target=float(np.mean(g("n_target"))),
        acc_h=float(np.mean(g("acc_clean_h"))), acc_b=float(np.mean(g("acc_clean_b"))))

for n in (1, 2, 3):
    R_ = [L(f"party_{n}_{s}") for s in SEEDS]
    out["party"][n] = dict(honest=float(np.mean([r["honest"] for r in R_])), attacked=float(np.mean([r["attacked"] for r in R_])),
                           loss=ci([r["honest"] - r["attacked"] for r in R_]))

for ds in DS:
    for c in ("fixed", "variance"):
        R_ = [L(f"tgt_{OLDN[ds]}_{c}_{s}", OLD) for s in SEEDS]
        if ds == "adult":
            R_ += [L(f"tgt_adult_{c}_{s}") for s in range(5, 10)]
        out["tgt"][f"{ds}|{c}"] = dict(honest=float(np.mean([r["honest"] for r in R_])),
                                       loss=ci([r["honest"] - r["attacked"] for r in R_]))

(R / "u_vfl_summary.json").write_text(json.dumps(out, indent=1))
print("written")
