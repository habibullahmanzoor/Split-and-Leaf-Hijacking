"""Category-1 attack: histogram/split-integrity.

A malicious client forges its reported gradient histogram instead of
computing it truthfully from local data. Two variants:

  naive_noise_attack   - adds Gaussian noise to the client's true gradient
                          histogram (the generic FL poisoning baseline —
                          works the same way regardless of aggregation
                          mechanism, included only for comparison).

  margin_flip_attack   - tree-specific: the malicious client dumps its
                          entire gradient budget into a single (feature, bin)
                          cell to force that cell to win the argmax split,
                          starving its true contribution everywhere else.
                          Targets the shallowest, earliest node available
                          (root of tree 0) since that has the highest
                          leverage per the boosting-compounding theory.

  clean_one_cell_attack - the literal one-cell primitive that theory/margin.py's
                          split-flip margin analyzes: adds exactly the closed-
                          form margin (times a small safety factor) to a single
                          cell of the client's TRUE histogram, leaving every
                          other cell -- including this client's own other
                          features -- untouched. margin_flip_attack above is a
                          stronger, more conspicuous construction; this
                          function checks the same saturating collapse holds
                          under the exact, minimal perturbation the theory
                          models.

Both attacks are given "honest_others" — the sum of every other client's
truthful histogram for the current node — modeling an adaptive attacker
with knowledge of the aggregate distribution (a standard, conservative
assumption in the poisoning literature; a weaker attacker only degrades
these results, doesn't invalidate them).
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import Client, Server, TreeNode
from theory.margin import compute_split_flip_margin


def naive_noise_attack(true_grad, true_hess, honest_others_grad, honest_others_hess,
                        noise_scale=5.0, rng=None):
    rng = rng or np.random.default_rng()
    noisy_grad = true_grad + rng.normal(0, noise_scale, true_grad.shape)
    return noisy_grad, true_hess


def _find_worst_gain_candidate(true_grad, true_hess, honest_others_grad, honest_others_hess,
                                lam=1.0, gamma=0.0, min_child_hess=1.0):
    """Search every valid (feature, bin) candidate over the combined
    (honest_others + true) histogram and return the one with the lowest
    gain under margin_flip_attack's own gain estimate -- the split furthest
    from winning the argmax under that estimate, and so the cheapest target
    for that attack to promote. This scores every candidate against one
    G_total/H_total taken over the WHOLE histogram (all features), not the
    per-feature total Server._find_best_split and compute_split_flip_margin
    both use (see theory/margin.py's docstring). That mismatch is a
    pre-existing property of margin_flip_attack's own target selection, kept
    here unchanged rather than "corrected" to avoid shifting its existing,
    already-published results. clean_one_cell_attack below does not reuse
    this helper for exactly this reason -- it needs a target that is valid
    under the per-feature convention its margin computation requires."""
    n_features, n_bins = true_grad.shape
    G_total = honest_others_grad.sum() + true_grad.sum()
    H_total = honest_others_hess.sum() + true_hess.sum()

    worst_gain, worst_f, worst_b = np.inf, None, None
    for f in range(n_features):
        g_cum = np.cumsum(honest_others_grad[f] + true_grad[f])
        h_cum = np.cumsum(honest_others_hess[f] + true_hess[f])
        for b in range(n_bins - 1):
            GL, HL = g_cum[b], h_cum[b]
            GR, HR = G_total - GL, H_total - HL
            if HL < min_child_hess or HR < min_child_hess:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam)
                           - G_total**2 / (H_total + lam)) - gamma
            if gain < worst_gain:
                worst_gain, worst_f, worst_b = gain, f, b
    return worst_f, worst_b


def _find_worst_gain_candidate_per_feature(agg_grad, agg_hess, lam=1.0, min_child_hess=1.0):
    """Same idea as _find_worst_gain_candidate above, but scores each
    candidate against ITS OWN feature's total (matching
    Server._find_best_split and compute_split_flip_margin's convention) so
    that whatever candidate is returned is always a legal input to
    compute_split_flip_margin. Operates on the already-aggregated histogram
    directly, since clean_one_cell_attack needs the aggregate either way."""
    n_features, n_bins = agg_grad.shape
    worst_gain, worst_f, worst_b = np.inf, None, None
    for f in range(n_features):
        G_total_f = agg_grad[f].sum()
        H_total_f = agg_hess[f].sum()
        g_cum = np.cumsum(agg_grad[f])
        h_cum = np.cumsum(agg_hess[f])
        for b in range(n_bins - 1):
            GL, HL = g_cum[b], h_cum[b]
            GR, HR = G_total_f - GL, H_total_f - HL
            if HL < min_child_hess or HR < min_child_hess:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam)
                           - G_total_f**2 / (H_total_f + lam))
            if gain < worst_gain:
                worst_gain, worst_f, worst_b = gain, f, b
    return worst_f, worst_b


def margin_flip_attack(true_grad, true_hess, honest_others_grad, honest_others_hess,
                        lam=1.0, gamma=0.0, budget_multiplier=3.0, rng=None):
    """Pick the worst-gain candidate split (from the honest aggregate) and
    concentrate this client's entire gradient budget there, so that after
    summation with honest clients it becomes the argmax.

    Budget is sized *relative to the honest aggregate's own gradient
    magnitude* at this node (not a fixed constant) — a fixed absolute budget
    would be swamped once thousands of honest samples land at a node, and
    would be needlessly huge at a node with only a handful of samples. This
    mirrors the margin concept: the attacker only needs to out-weigh
    whatever magnitude the honest clients actually contributed here."""
    if honest_others_grad is None:
        return true_grad, true_hess

    worst_f, worst_b = _find_worst_gain_candidate(true_grad, true_hess,
                                                   honest_others_grad, honest_others_hess,
                                                   lam=lam, gamma=gamma)
    if worst_f is None:
        return true_grad, true_hess

    honest_magnitude = np.abs(honest_others_grad).sum() + np.abs(true_grad).sum()
    budget = budget_multiplier * max(honest_magnitude, 1e-3)
    malicious_grad = np.zeros_like(true_grad)
    malicious_grad[worst_f, worst_b] = budget
    return malicious_grad, true_hess.copy()


def clean_one_cell_attack(true_grad, true_hess, honest_others_grad, honest_others_hess,
                           lam=1.0, min_child_hess=1.0, margin_multiplier=1.01, rng=None):
    """The literal one-cell primitive Section 4.2's split-flip margin models:
    add exactly (margin_multiplier * margin) to a single histogram cell of
    the client's TRUE report, leaving every other cell -- including this
    client's own other features -- untouched. Target selection uses
    _find_worst_gain_candidate_per_feature, not the helper margin_flip_attack
    uses above, because compute_split_flip_margin requires its target
    candidate to already be valid under the PER-FEATURE min_child_hess
    convention -- the whole-histogram convention margin_flip_attack's search
    uses can pick a candidate compute_split_flip_margin would reject. If the
    chosen candidate cannot be flipped by any single-cell spike (margin is 0
    or infinite), the client reports honestly."""
    if honest_others_grad is None:
        return true_grad, true_hess

    agg_grad = honest_others_grad + true_grad
    agg_hess = honest_others_hess + true_hess

    worst_f, worst_b = _find_worst_gain_candidate_per_feature(agg_grad, agg_hess,
                                                               lam=lam, min_child_hess=min_child_hess)
    if worst_f is None:
        return true_grad, true_hess

    margin, _ = compute_split_flip_margin(agg_grad, agg_hess, worst_f, worst_b,
                                           lam=lam, min_child_hess=min_child_hess)
    if not np.isfinite(margin) or margin <= 0:
        return true_grad, true_hess

    forged_grad = true_grad.copy()
    forged_grad[worst_f, worst_b] += margin_multiplier * margin
    return forged_grad, true_hess.copy()


class AttackServer(Server):
    """Same aggregation/split-finding as Server, but malicious clients get to
    craft their histogram contribution using an `attack_fn(true_grad,
    true_hess, honest_others_grad, honest_others_hess, **kwargs)` before
    summation. Restricting `attack_rounds`/`attack_max_depth` lets us target
    only early, shallow nodes -- the highest-leverage points per the theory."""

    def __init__(self, clients, n_bins, max_depth, lam=1.0, gamma=0.0,
                 min_child_hess=1.0, malicious_ids=(), attack_fn=None,
                 attack_kwargs=None, attack_max_depth=1, target_rounds=None):
        super().__init__(clients, n_bins, max_depth, lam, gamma, min_child_hess)
        self.malicious_ids = set(malicious_ids)
        self.attack_fn = attack_fn
        self.attack_kwargs = attack_kwargs or {}
        self.attack_max_depth = attack_max_depth
        # target_rounds=None means attack every round (persistent malicious
        # client); a set of round indices confines the attack to just those
        # boosting rounds, used to isolate early- vs late-round corruption.
        self.target_rounds = target_rounds
        self._current_depth = 0
        self._round = -1

    def build_tree(self):
        self._round += 1
        return super().build_tree()

    def _aggregate_histograms(self, sample_idx: dict):
        truthful = {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            truthful[c.client_id] = Client.compute_histograms(c, idx, self.n_bins)

        round_ok = self.target_rounds is None or self._round in self.target_rounds
        attack_here = round_ok and self._current_depth <= self.attack_max_depth

        agg_grad = agg_hess = None
        for cid, (g, h) in truthful.items():
            if attack_here and cid in self.malicious_ids:
                others_g = None
                others_h = None
                for ocid, (og, oh) in truthful.items():
                    if ocid == cid:
                        continue
                    others_g = og if others_g is None else others_g + og
                    others_h = oh if others_h is None else others_h + oh
                g, h = self.attack_fn(g, h, others_g, others_h, **self.attack_kwargs)
            agg_grad = g if agg_grad is None else agg_grad + g
            agg_hess = h if agg_hess is None else agg_hess + h
        return agg_grad, agg_hess

    def _build_node(self, sample_idx, depth):
        self._current_depth = depth
        return super()._build_node(sample_idx, depth)
