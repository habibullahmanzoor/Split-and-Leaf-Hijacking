"""Margin-certified split selection.

Instead of distrusting every client's contribution everywhere (what median/
trimmed-mean do, and what made them actively harmful under non-IID data --
see defense_robust_aggregation_hfl.py), this defense only intervenes on the
SPECIFIC decision at risk: at each node, after finding the winning split,
check whether it would still win with the single largest contributor to
its winning histogram cell removed entirely (leave-one-out). If yes, no
single client's contribution was decisive -- the split is CERTIFIED. If
the winner changes once that one contributor is removed, the split was
effectively decided by one party -- REJECT it rather than trust it.

This is surgical: it only ever recomputes and possibly discards ONE split
decision, using the actual data (not a blanket coordinate-wise statistic),
so it shouldn't pay the same honest-case accuracy cost median/trimmed-mean
did -- but that has to be measured, not assumed, same as everything else
in this project.
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import Client, Server, TreeNode
from theory.margin import compute_split_flip_margin


class MarginCertifiedServer(Server):
    def __init__(self, clients, n_bins, max_depth, lam=1.0, gamma=0.0, min_child_hess=1.0,
                 malicious_ids=(), attack_fn=None, attack_kwargs=None, attack_max_depth=1,
                 fallback: str = "leaf"):
        super().__init__(clients, n_bins, max_depth, lam, gamma, min_child_hess)
        self.malicious_ids = set(malicious_ids)
        self.attack_fn = attack_fn
        self.attack_kwargs = attack_kwargs or {}
        self.attack_max_depth = attack_max_depth
        self.fallback = fallback  # "leaf" or "runner_up"
        self._current_depth = 0
        self.n_certified = 0
        self.n_rejected = 0

    def _per_client_histograms(self, sample_idx: dict):
        truthful = {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            truthful[c.client_id] = Client.compute_histograms(c, idx, self.n_bins)

        attack_here = self._current_depth <= self.attack_max_depth
        result = {}
        for cid, (g, h) in truthful.items():
            if attack_here and cid in self.malicious_ids and self.attack_fn is not None:
                others_g = others_h = None
                for ocid, (og, oh) in truthful.items():
                    if ocid == cid:
                        continue
                    others_g = og if others_g is None else others_g + og
                    others_h = oh if others_h is None else others_h + oh
                g, h = self.attack_fn(g, h, others_g, others_h, **self.attack_kwargs)
            result[cid] = (g, h)
        return result

    def _aggregate_histograms(self, sample_idx: dict):
        per_client = self._per_client_histograms(sample_idx)
        if not per_client:
            return None, None
        agg_grad = sum(g for g, h in per_client.values())
        agg_hess = sum(h for g, h in per_client.values())
        return agg_grad, agg_hess

    def _build_node(self, sample_idx: dict, depth: int) -> TreeNode:
        self._current_depth = depth
        node = TreeNode(depth=depth, sample_idx=sample_idx)
        total_n = sum(len(v) for v in sample_idx.values())

        per_client = self._per_client_histograms(sample_idx)
        if depth >= self.max_depth or total_n < 2 or not per_client:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        agg_grad = sum(g for g, h in per_client.values())
        agg_hess = sum(h for g, h in per_client.values())

        (gain, feat, b), (gain2, feat2, b2) = self._find_best_split(agg_grad, agg_hess)
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        # Leave-one-out certification: remove the single largest
        # contributor to the WINNING cell and check the winner is stable.
        contributions = {cid: g[feat, b] for cid, (g, h) in per_client.items()}
        top_client = max(contributions, key=contributions.get)
        g_top, h_top = per_client[top_client]
        reduced_grad = agg_grad - g_top
        reduced_hess = agg_hess - h_top
        (gain_wo, feat_wo, b_wo), _ = self._find_best_split(reduced_grad, reduced_hess)

        certified = (feat_wo == feat and b_wo == b)
        if certified:
            self.n_certified += 1
        else:
            self.n_rejected += 1
            if self.fallback == "runner_up" and feat2 is not None:
                feat, b = feat2, b2
            else:
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

        node.split_feature, node.split_bin = feat, b
        left_idx, right_idx = {}, {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            mask = c.X[idx, feat] <= b
            left_idx[c.client_id] = idx[mask]
            right_idx[c.client_id] = idx[~mask]

        node.left = self._build_node(left_idx, depth + 1)
        node.right = self._build_node(right_idx, depth + 1)
        return node


class CalibratedMarginCertifiedServer(MarginCertifiedServer):
    """v2: fixes v1's failure mode. Instead of removing the winning cell's
    top contributor ENTIRELY (which flags normal non-IID heterogeneity as
    suspicious -- a client can legitimately hold a large, but not
    anomalous, share of the signal for a good split), this caps the top
    contributor down to a TYPICAL contribution level (the median of the
    OTHER clients' contributions to that same cell) and only tests whether
    the EXCESS above that typical level was decisive. A client contributing
    a normal-range amount is never touched at all; only genuine outliers
    (contributions far beyond what peers show) get scrutinized.

    NOTE: measured to fix the honest-case cost but ALSO stop catching the
    attack (too lenient) -- see TheoryGroundedMarginCertifiedServer (v3)
    below, which replaces the ad-hoc tolerance_multiplier with the actual
    validated margin theorem."""

    def __init__(self, *args, tolerance_multiplier: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance_multiplier = tolerance_multiplier

    def _build_node(self, sample_idx: dict, depth: int) -> TreeNode:
        self._current_depth = depth
        node = TreeNode(depth=depth, sample_idx=sample_idx)
        total_n = sum(len(v) for v in sample_idx.values())

        per_client = self._per_client_histograms(sample_idx)
        if depth >= self.max_depth or total_n < 2 or not per_client:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        agg_grad = sum(g for g, h in per_client.values())
        agg_hess = sum(h for g, h in per_client.values())

        (gain, feat, b), (gain2, feat2, b2) = self._find_best_split(agg_grad, agg_hess)
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        contributions = {cid: g[feat, b] for cid, (g, h) in per_client.items()}
        top_client = max(contributions, key=contributions.get)
        top_value = contributions[top_client]
        # Reference "typical single-client size" from the OTHER clients'
        # TOTAL gradient magnitude across the WHOLE feature (all bins), not
        # this one specific (feature, bin) cell -- a single cell is sparse
        # enough under non-IID partitioning that most clients contribute
        # ~0 to it, which would make a per-cell reference collapse to ~0
        # and defeat the calibration entirely (this was v1's hidden bug,
        # not just v1's design flaw).
        others = [cid for cid in per_client if cid != top_client]
        other_feature_totals = [np.abs(per_client[cid][0][feat]).sum() for cid in others]
        typical = float(np.median(other_feature_totals)) if other_feature_totals else 0.0
        excess = max(0.0, top_value - self.tolerance_multiplier * max(typical, 1e-6))

        if excess <= 0.0:
            # top contributor isn't unusual relative to peers -- no need to
            # even test removal, this is normal heterogeneity
            certified = True
        else:
            g_top, h_top = per_client[top_client]
            reduced_grad = agg_grad.copy()
            reduced_grad[feat, b] = reduced_grad[feat, b] - excess
            (gain_wo, feat_wo, b_wo), _ = self._find_best_split(reduced_grad, agg_hess)
            certified = (feat_wo == feat and b_wo == b)

        if certified:
            self.n_certified += 1
        else:
            self.n_rejected += 1
            if self.fallback == "runner_up" and feat2 is not None:
                feat, b = feat2, b2
            else:
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

        node.split_feature, node.split_bin = feat, b
        left_idx, right_idx = {}, {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            mask = c.X[idx, feat] <= b
            left_idx[c.client_id] = idx[mask]
            right_idx[c.client_id] = idx[~mask]

        node.left = self._build_node(left_idx, depth + 1)
        node.right = self._build_node(right_idx, depth + 1)
        return node


class CombinedMarginCertifiedServer(MarginCertifiedServer):
    """v4: combines v2's size-plausibility check with v3's margin-awareness
    check. v2's failure was ignoring the margin (too lenient towards
    size-plausible-but-still-attacking values); v3's failure was ignoring
    plausible size (too strict on margin-exceeding-but-perfectly-
    normal-sized values). Certifies if EITHER signal independently says
    the contribution looks plausible -- rejects only when BOTH the size
    check and the margin check agree it's suspicious."""

    def __init__(self, *args, size_tolerance: float = 3.0, overshoot_tolerance: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_tolerance = size_tolerance
        self.overshoot_tolerance = overshoot_tolerance

    def _build_node(self, sample_idx: dict, depth: int) -> TreeNode:
        self._current_depth = depth
        node = TreeNode(depth=depth, sample_idx=sample_idx)
        total_n = sum(len(v) for v in sample_idx.values())

        per_client = self._per_client_histograms(sample_idx)
        if depth >= self.max_depth or total_n < 2 or not per_client:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        agg_grad = sum(g for g, h in per_client.values())
        agg_hess = sum(h for g, h in per_client.values())

        (gain, feat, b), (gain2, feat2, b2) = self._find_best_split(agg_grad, agg_hess)
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        contributions = {cid: g[feat, b] for cid, (g, h) in per_client.items()}
        top_client = max(contributions, key=contributions.get)
        top_value = contributions[top_client]
        g_top, h_top = per_client[top_client]
        reduced_grad = agg_grad - g_top
        reduced_hess = agg_hess - h_top

        (gain_wo, feat_wo, b_wo), _ = self._find_best_split(reduced_grad, reduced_hess)

        if feat_wo == feat and b_wo == b:
            certified = True
        else:
            others = [cid for cid in per_client if cid != top_client]
            other_feature_totals = [np.abs(per_client[cid][0][feat]).sum() for cid in others]
            typical = float(np.median(other_feature_totals)) if other_feature_totals else 0.0
            size_plausible = top_value <= self.size_tolerance * max(typical, 1e-6)

            try:
                exact_margin, _ = compute_split_flip_margin(
                    reduced_grad, reduced_hess, target_feature=feat, target_bin=b,
                    lam=self.lam, min_child_hess=self.min_child_hess,
                )
            except ValueError:
                exact_margin = np.inf
            margin_plausible = np.isfinite(exact_margin) and top_value <= self.overshoot_tolerance * max(exact_margin, 1e-6)

            certified = size_plausible or margin_plausible

        if certified:
            self.n_certified += 1
        else:
            self.n_rejected += 1
            if self.fallback == "runner_up" and feat2 is not None:
                feat, b = feat2, b2
            else:
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

        node.split_feature, node.split_bin = feat, b
        left_idx, right_idx = {}, {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            mask = c.X[idx, feat] <= b
            left_idx[c.client_id] = idx[mask]
            right_idx[c.client_id] = idx[~mask]

        node.left = self._build_node(left_idx, depth + 1)
        node.right = self._build_node(right_idx, depth + 1)
        return node


class TheoryGroundedMarginCertifiedServer(MarginCertifiedServer):
    """v3: fixes v2's failure mode using the validated split-flip margin
    theorem (src/theory/margin.py) directly, instead of an ad-hoc
    size-based tolerance. Key insight: an attacker who doesn't know the
    exact theoretical minimum perturbation needed to flip a split tends to
    OVERSHOOT it for safety margin (our own margin_flip_attack uses
    budget_multiplier=3.0, well beyond the exact minimum). An honest value
    that happens to be pivotal by coincidence should sit CLOSE to the
    theoretical minimum; a deliberately forged one should vastly exceed
    it. That ratio -- actual contribution vs. the exact theoretical
    minimum needed -- is the real signal, not an arbitrary size-based
    multiplier (which is what made v2 too lenient)."""

    def __init__(self, *args, overshoot_tolerance: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.overshoot_tolerance = overshoot_tolerance

    def _build_node(self, sample_idx: dict, depth: int) -> TreeNode:
        self._current_depth = depth
        node = TreeNode(depth=depth, sample_idx=sample_idx)
        total_n = sum(len(v) for v in sample_idx.values())

        per_client = self._per_client_histograms(sample_idx)
        if depth >= self.max_depth or total_n < 2 or not per_client:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        agg_grad = sum(g for g, h in per_client.values())
        agg_hess = sum(h for g, h in per_client.values())

        (gain, feat, b), (gain2, feat2, b2) = self._find_best_split(agg_grad, agg_hess)
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        contributions = {cid: g[feat, b] for cid, (g, h) in per_client.items()}
        top_client = max(contributions, key=contributions.get)
        top_value = contributions[top_client]
        g_top, h_top = per_client[top_client]
        reduced_grad = agg_grad - g_top
        reduced_hess = agg_hess - h_top

        (gain_wo, feat_wo, b_wo), _ = self._find_best_split(reduced_grad, reduced_hess)

        if feat_wo == feat and b_wo == b:
            # Winner doesn't depend on top_client's contribution to this
            # cell at all -- certify immediately, no theory check needed.
            certified = True
        else:
            # Winner WOULD change without top_client -- use the theorem to
            # find the EXACT minimum contribution needed to flip it back.
            try:
                exact_margin, _ = compute_split_flip_margin(
                    reduced_grad, reduced_hess, target_feature=feat, target_bin=b,
                    lam=self.lam, min_child_hess=self.min_child_hess,
                )
            except ValueError:
                exact_margin = np.inf  # target cell no longer even a valid split candidate

            if not np.isfinite(exact_margin):
                certified = False
            else:
                # Actual contribution close to the exact minimum needed ->
                # plausibly an honest, coincidentally-pivotal value.
                # Vastly exceeding it -> the attacker-overshoot signature.
                certified = top_value <= self.overshoot_tolerance * max(exact_margin, 1e-6)

        if certified:
            self.n_certified += 1
        else:
            self.n_rejected += 1
            if self.fallback == "runner_up" and feat2 is not None:
                feat, b = feat2, b2
            else:
                node.is_leaf = True
                node.leaf_value = self._leaf_value(sample_idx)
                return node

        node.split_feature, node.split_bin = feat, b
        left_idx, right_idx = {}, {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            mask = c.X[idx, feat] <= b
            left_idx[c.client_id] = idx[mask]
            right_idx[c.client_id] = idx[~mask]

        node.left = self._build_node(left_idx, depth + 1)
        node.right = self._build_node(right_idx, depth + 1)
        return node
