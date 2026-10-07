"""Honest-training conformance check against an established implementation.
No attack runs here. For each dataset and each of ten seeds, our honest HFL model
(histogram summation over five clients) is compared with XGBoost's histogram
tree method trained on the pooled training rows, with matching hyperparameters
(learning rate 0.3, lambda 1, gamma 0, min child weight 1, same bins, depth and
rounds, base score 0.5). The comparison is: paired test-AUC difference, and the
fraction of seeds in which the first tree's root split uses the same feature."""
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import rev_common as rc
import unified as u

S = range(10)
res = {}
for ds in ("adult", "heart", "credit"):
    cfg = u.HC[ds]
    ours, ref, same_root, same_root3 = [], [], [], []
    for s in S:
        a, model = u.hfl_fit(ds, s)
        Xtr, Xte, ytr, yte = u.hfl_split(ds, s)
        clf = xgb.XGBClassifier(n_estimators=cfg["rounds"], max_depth=cfg["depth"], learning_rate=0.3,
                                reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, max_bin=cfg["bins"],
                                tree_method="hist", base_score=0.5, subsample=1.0, colsample_bytree=1.0,
                                n_jobs=4, random_state=s, eval_metric="logloss")
        clf.fit(Xtr, ytr)
        b = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        ours.append(a)
        ref.append(b)
        dump = json.loads(clf.get_booster().get_dump(dump_format="json")[0])
        root_feat = int(dump["split"].lstrip("f")) if "split" in dump else None
        same_root.append(root_feat == model.trees[0].split_feature)
        # agreement of the three nodes at depth <= 1 of the first tree
        def feats(d):
            out = [int(d["split"].lstrip("f")) if "split" in d else None]
            for ch in d.get("children", []):
                out.append(int(ch["split"].lstrip("f")) if "split" in ch else None)
            return out
        t0 = model.trees[0]
        mine = [t0.split_feature, t0.left.split_feature if t0.left else None, t0.right.split_feature if t0.right else None]
        # xgboost children are ordered yes/no (left/right); compare as sets to avoid orientation issues
        same_root3.append(sorted(x for x in mine if x is not None) == sorted(x for x in feats(dump) if x is not None))
    d = np.array(ours) - np.array(ref)
    m, lo, hi = rc.paired(ref, ours)   # reference minus ours
    res[ds] = dict(ours=float(np.mean(ours)), xgb=float(np.mean(ref)),
                   xgb_minus_ours=m, ci=[lo, hi],
                   root_feature_agree=float(np.mean(same_root)), top3_features_agree=float(np.mean(same_root3)))
    print(ds, res[ds], flush=True)
rc.save("xgb_conformance", res)
print("saved")
