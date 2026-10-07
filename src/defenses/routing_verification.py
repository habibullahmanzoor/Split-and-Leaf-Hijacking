"""Routing-verification defense against category 5 (Leaf Misdirection).

Different strategy from the margin-certified defense family (which spent 4
iterations struggling with a fundamentally ambiguous question: is this
contribution unusual because of malice, or because of real non-IID
heterogeneity?). Routing verification sidesteps that ambiguity entirely by
exploiting a PROTOCOL-level property instead of a statistical one: an
honest passive party's routing decision is deterministic -- it's just
comparing real, unchanging feature values against a threshold, so asking
the same routing question twice must always produce the same answer. A
malicious party using fresh randomness to misroute (as our own
MisroutingPassiveParty does) will very likely disagree with itself on a
second, independent query.

Once a party is caught disagreeing with itself even once, it is
PERMANENTLY excluded from all future splits for the rest of training
(reputation-based) -- no per-decision statistical judgment call needed.

Honest limitation, stated plainly: this catches attackers whose lie is
NON-deterministic (which is how MisroutingPassiveParty is implemented). A
more sophisticated attacker using a DETERMINISTIC but still-wrong routing
rule (e.g., hash the sample id and flip based on that) would answer the
same both times and evade a pure repeated-query check. This defense closes
the specific attack we built, not every conceivable misrouting attacker.
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import TreeNode
from harness.vfl_secureboost import VFLServer, logistic_grad_hess


class RoutingVerifiedVFLServer(VFLServer):
    def __init__(self, active, passives, n_bins, max_depth, lam=1.0, gamma=0.0,
                 min_child_hess=1.0):
        super().__init__(active, passives, n_bins, max_depth, lam, gamma, min_child_hess)
        self.banned_parties = set()
        self.n_verified = 0
        self.n_banned_events = 0

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
                # Can't trust this split's routing -- make this node a leaf
                # rather than use a possibly-corrupted partition.
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node
            self.n_verified += 1
            left_idx, right_idx = left1, right1

        node.left = self._build_node(left_idx, depth + 1, enc_grad, enc_hess)
        node.right = self._build_node(right_idx, depth + 1, enc_grad, enc_hess)
        return node
