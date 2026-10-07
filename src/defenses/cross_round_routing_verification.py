"""Cross-round routing monotonicity verification: closes routing
verification's own named limitation (deterministic misrouting evades a
same-node repeated query, since answering the same wrong question
identically twice is self-consistent) using genuine cross-node
information, exactly the "cross round or cross query consistency beyond
simple repetition" direction routing_verification.py's docstring and
paper.tex's Discussion section name as the open direction.

The invariant: an honest passive party's routing decision for sample x at
(feature f, threshold b) is `x_f <= b`, a fixed, unchanging function of x's
own feature value. If the SAME feature wins splits at two different nodes
anywhere in the training run, at thresholds b1 < b2, and a given sample x
is present at both nodes, its two decisions are constrained by ordinary
transitivity: if x routed left at b1 (x_f <= b1), it must ALSO route left
at b2 (since b1 < b2), and if it routed right at b2 (x_f > b2), it must
ALSO have routed right at b1 (since b1 < b2). The other two combinations
(left at the larger threshold, right at the smaller one) are uninformative
on their own and are not checked.

A party whose misrouting rule is a fixed, deterministic function of
something OTHER than the true feature value (e.g. a hash of the sample id)
answers the SAME query identically every time -- passing
routing_verification.py's repeated-query check -- but has no reason to
respect this cross-threshold transitivity, since its reported decisions are
not actually a function of a single underlying x_f at all. We track, per
(party, feature) pair, every sample's reported side at every threshold seen
so far across the ENTIRE training run (all nodes, all rounds -- the server
object already persists across rounds within one `fit()` call), and
permanently exclude any party whose reports ever violate this transitivity
for a sample seen at two different thresholds.
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import TreeNode
from harness.vfl_secureboost import logistic_grad_hess
from defenses.routing_verification import RoutingVerifiedVFLServer


class CrossRoundRoutingVerifiedVFLServer(RoutingVerifiedVFLServer):
    def __init__(self, active, passives, n_bins, max_depth, lam=1.0, gamma=0.0,
                 min_child_hess=1.0):
        super().__init__(active, passives, n_bins, max_depth, lam, gamma, min_child_hess)
        self.feature_history: dict[tuple[str, int], dict[int, list[tuple[int, bool]]]] = {}
        self.n_monotonicity_checks = 0
        self.n_monotonicity_violations = 0

    def _check_and_record_monotonicity(self, owner: str, feat: int, b: int,
                                        sample_idx: np.ndarray, is_left_mask: np.ndarray) -> bool:
        history = self.feature_history.setdefault((owner, feat), {})
        violation = False
        for sid, is_left in zip(sample_idx.tolist(), is_left_mask.tolist()):
            record = history.get(sid)
            if record is not None:
                for old_b, old_is_left in record:
                    self.n_monotonicity_checks += 1
                    if (b > old_b and old_is_left and not is_left) or \
                       (b < old_b and (not old_is_left) and is_left):
                        violation = True
                        break
            if violation:
                break
        if not violation:
            for sid, is_left in zip(sample_idx.tolist(), is_left_mask.tolist()):
                history.setdefault(sid, []).append((b, bool(is_left)))
        return violation

    def _build_node(self, sample_idx: np.ndarray, depth: int, enc_grad, enc_hess) -> TreeNode:
        node = TreeNode(depth=depth, sample_idx={"all": sample_idx})

        if depth >= self.max_depth or len(sample_idx) < 2:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        candidates = self._candidate_splits(sample_idx, enc_grad, enc_hess)
        candidates = [c for c in candidates if c[0] not in self.banned_parties]
        if not candidates:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        G_total, H_total = grad[sample_idx].sum(), hess[sample_idx].sum()
        (gain, owner, feat, b), _second = self._find_best_split(candidates, G_total, H_total)

        if owner is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        node.split_feature = feat
        node.split_bin = b
        node.split_threshold = owner

        if owner == "active":
            mask = self.active.X[sample_idx, feat] <= b
            left_idx, right_idx = sample_idx[mask], sample_idx[~mask]
        else:
            passive = self.passives[owner]
            left1, right1 = passive.route(sample_idx, feat, b)
            left2, right2 = passive.route(sample_idx, feat, b)
            if set(left1.tolist()) != set(left2.tolist()):
                self.banned_parties.add(owner)
                self.n_banned_events += 1
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

            left_set = set(left1.tolist())
            is_left_mask = np.array([sid in left_set for sid in sample_idx.tolist()])
            if self._check_and_record_monotonicity(owner, feat, b, sample_idx, is_left_mask):
                self.banned_parties.add(owner)
                self.n_banned_events += 1
                self.n_monotonicity_violations += 1
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

            self.n_verified += 1
            left_idx, right_idx = left1, right1

        node.left = self._build_node(left_idx, depth + 1, enc_grad, enc_hess)
        node.right = self._build_node(right_idx, depth + 1, enc_grad, enc_hess)
        return node
