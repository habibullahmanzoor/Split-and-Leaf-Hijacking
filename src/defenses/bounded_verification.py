"""Magnitude-bounded verification defense against category 3 (HE/crypto-
layer blind ciphertext rescaling).

This is the specific experiment flagged as needed during the novelty
gap-check: existing verifiable-HE schemes (VerifyNet-style) are built to
catch exactly this kind of malleability abuse, but for SIMPLE SUM
aggregation. SecureBoost's aggregation feeds a DISCRETE ARGMAX over splits,
which is structurally different. This tests whether the most natural such
bound -- ported directly to our setting -- actually stops the attack, or
whether it's too loose to matter at the scale that causes real damage.

The bound: for logistic loss, any single sample's gradient satisfies
|grad| <= 1. The active party always knows exactly how many samples are at
a node (it holds labels; sample_idx is plaintext) -- call it n. No PARTIAL
sum over any subset of those samples can exceed n in absolute value. This
requires no extra protocol machinery (no range proofs, no extra rounds) --
it's the natural, always-available sanity check, exactly the kind of thing
a VerifyNet-style scheme would enforce.
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import TreeNode
from harness.vfl_secureboost import VFLServer, logistic_grad_hess


class BoundedVerificationVFLServer(VFLServer):
    def __init__(self, active, passives, n_bins, max_depth, lam=1.0, gamma=0.0,
                 min_child_hess=1.0, max_grad_per_sample: float = 1.0):
        super().__init__(active, passives, n_bins, max_depth, lam, gamma, min_child_hess)
        self.max_grad_per_sample = max_grad_per_sample
        self.n_candidates_seen = 0
        self.n_candidates_rejected = 0

    def _build_node(self, sample_idx: np.ndarray, depth: int, enc_grad, enc_hess) -> TreeNode:
        node = TreeNode(depth=depth, sample_idx={"all": sample_idx})

        if depth >= self.max_depth or len(sample_idx) < 2:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        bound = self.max_grad_per_sample * len(sample_idx)
        candidates = self._candidate_splits(sample_idx, enc_grad, enc_hess)

        filtered = []
        for owner, g_hist, h_hist in candidates:
            self.n_candidates_seen += 1
            if owner == "active":
                # active party's own histogram is plaintext and self-
                # reported truthfully in this simulation -- only encrypted
                # passive-party contributions need the bound check.
                filtered.append((owner, g_hist, h_hist))
                continue
            max_abs_cell = float(np.max(np.abs(g_hist))) if g_hist.size else 0.0
            if max_abs_cell > bound:
                self.n_candidates_rejected += 1
                continue
            filtered.append((owner, g_hist, h_hist))

        if not filtered:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        G_total, H_total = grad[sample_idx].sum(), hess[sample_idx].sum()
        (gain, owner, feat, b), _second = self._find_best_split(filtered, G_total, H_total)

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
            left_idx, right_idx = self.passives[owner].route(sample_idx, feat, b)

        node.left = self._build_node(left_idx, depth + 1, enc_grad, enc_hess)
        node.right = self._build_node(right_idx, depth + 1, enc_grad, enc_hess)
        return node
