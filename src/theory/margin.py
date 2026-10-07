"""Split-flip margin: the minimum gradient-histogram perturbation needed to
change which split wins the argmax at a node.

Attack shape considered: a single-cell spike of magnitude delta >= 0 added
to the aggregated gradient histogram at (target_feature, target_bin) --
exactly the perturbation shape category-1's margin_flip_attack and
category-3's ciphertext rescaling both use. Hessian is never perturbed
(established convention across the whole attack family -- see memory notes
on why: perturbing Hessian trips the min_child_hess guard).

Derivation matches Server._find_best_split's PER-FEATURE total (G_total,
H_total computed from that feature's own row, not a fixed reference
feature -- see the fix in federated_gbdt.py and the memory note on why a
fixed-feature total silently ignores attacks on any other feature).
Injecting delta at (f0, b0):
  - A rival in a DIFFERENT feature (f != f0) is completely unaffected --
    its own feature's row total never sees the injected mass, so its score
    is a CONSTANT in delta.
  - A rival in the SAME feature (f == f0) sees that feature's total shift
    by delta. If its bin b >= b0, its cumulative GL already includes the
    spike (GL -> GL+delta) while GR stays put (both G_total and GL shift by
    the same delta and cancel). If b < b0, GL is untouched but GR ->
    GR+delta (only G_total moved).

So every candidate's Score(f,b) = GL^2/(HL+lam) + GR^2/(HR+lam) is either a
true constant (different feature) or a quadratic A*delta^2+B*delta+C (same
feature, with A,B set by whichever of GL/GR shifts). The margin is the
smallest delta >= 0 that makes the target's quadratic exceed every
competing candidate's score -- found by solving, per rival, the resulting
inequality and taking the binding (max) requirement across all rivals.
"""
from __future__ import annotations

import numpy as np


def _delta_min_for_rival(A_t, B_t, C_t, A_r, B_r, C_r, tol=1e-9):
    """Smallest delta >= 0 with (A_t-A_r)d^2+(B_t-B_r)d+(C_t-C_r) > 0.
    Returns np.inf if no such delta exists (target can never beat this
    rival with a single-cell spike, however large)."""
    a, b, c = A_t - A_r, B_t - B_r, C_t - C_r

    if abs(a) < tol:
        if abs(b) < tol:
            return 0.0 if c > 0 else np.inf
        if b > 0:
            return max(0.0, -c / b)
        return 0.0 if c > 0 else np.inf

    disc = b * b - 4 * a * c
    if a > 0:
        if disc < 0:
            return 0.0  # upward parabola, never touches zero -> always positive
        sq = np.sqrt(disc)
        r1, r2 = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
        if 0.0 < r1 or 0.0 > r2:
            return 0.0  # already outside the negative interval at delta=0
        return max(0.0, r2)
    else:  # a < 0: downward parabola, positive only inside a bounded window
        if disc < 0:
            return np.inf
        sq = np.sqrt(disc)
        r1, r2 = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
        if r2 <= 0:
            return np.inf
        lo = max(0.0, r1)
        return lo if lo < r2 else np.inf


def compute_split_flip_margin(agg_grad: np.ndarray, agg_hess: np.ndarray,
                               target_feature: int, target_bin: int,
                               lam: float = 1.0, min_child_hess: float = 1.0):
    """Returns (margin, binding_rival) where margin is the minimum
    single-cell gradient spike at (target_feature, target_bin) needed for
    that split to become the unique argmax over all valid candidates, and
    binding_rival = (feature, bin) of whichever candidate was hardest to
    beat (np.inf margin means no rival blocks it, e.g. it's already best)."""
    n_features, n_bins = agg_grad.shape
    GT0, HT0 = agg_grad[target_feature].sum(), agg_hess[target_feature].sum()

    g_cum_t = np.cumsum(agg_grad[target_feature])
    h_cum_t = np.cumsum(agg_hess[target_feature])
    GLt, HLt = g_cum_t[target_bin], h_cum_t[target_bin]
    GRt, HRt = GT0 - GLt, HT0 - HLt
    if HLt < min_child_hess or HRt < min_child_hess:
        raise ValueError("target split violates min_child_hess before any perturbation")

    # _find_best_split's actual gain subtracts a per-FEATURE "parent score"
    # G_total_f^2/(H_total_f+lam) -- since the total is now computed per
    # feature (not a fixed reference feature), this term differs between
    # features and so DOES affect cross-feature ranking; it is not a
    # uniform constant we can drop. It's also delta-dependent for the
    # target itself, since the target's own feature total shifts by delta.
    # For a SAME-feature rival this term is identical to the target's (same
    # feature, same total) and cancels exactly in the A_t-A_r/B_t-B_r/C_t-C_r
    # differences below -- for a DIFFERENT-feature rival it's an unaffected
    # constant using that rival's own feature total.
    A_t = 1.0 / (HLt + lam) - 1.0 / (HT0 + lam)
    B_t = 2 * GLt / (HLt + lam) - 2 * GT0 / (HT0 + lam)
    C_t = GLt**2 / (HLt + lam) + GRt**2 / (HRt + lam) - GT0**2 / (HT0 + lam)

    max_delta, binding = 0.0, None
    for f in range(n_features):
        G_total_f, H_total_f = agg_grad[f].sum(), agg_hess[f].sum()
        g_cum = np.cumsum(agg_grad[f])
        h_cum = np.cumsum(agg_hess[f])
        for b in range(n_bins - 1):
            if f == target_feature and b == target_bin:
                continue
            GLr, HLr = g_cum[b], h_cum[b]
            GRr, HRr = G_total_f - GLr, H_total_f - HLr
            if HLr < min_child_hess or HRr < min_child_hess:
                continue

            C_r = GLr**2 / (HLr + lam) + GRr**2 / (HRr + lam) - G_total_f**2 / (H_total_f + lam)
            if f != target_feature:
                # different feature: its own total is unaffected by the
                # spike, so this candidate's score is a true constant
                A_r, B_r = 0.0, 0.0
            elif b >= target_bin:
                A_r = 1.0 / (HLr + lam) - 1.0 / (HT0 + lam)
                B_r = 2 * GLr / (HLr + lam) - 2 * GT0 / (HT0 + lam)
            else:
                A_r = 1.0 / (HRr + lam) - 1.0 / (HT0 + lam)
                B_r = 2 * GRr / (HRr + lam) - 2 * GT0 / (HT0 + lam)

            d = _delta_min_for_rival(A_t, B_t, C_t, A_r, B_r, C_r)
            if d == np.inf:
                return np.inf, (f, b)
            if d > max_delta:
                max_delta, binding = d, (f, b)
    return max_delta, binding


def score_at_delta(agg_grad: np.ndarray, agg_hess: np.ndarray, feature: int, bin_: int,
                    target_feature: int, target_bin: int, delta: float,
                    lam: float = 1.0) -> float:
    """Full gain-equivalent score of candidate (feature, bin_) after
    injecting `delta` at (target_feature, target_bin) -- used to numerically
    double-check the closed-form margin against direct evaluation. Matches
    Server._find_best_split's per-feature total exactly, including the
    subtracted per-feature parent-score term (needed for cross-feature
    comparisons to be meaningful, not just same-feature ones)."""
    G_total = agg_grad[feature].sum() + (delta if feature == target_feature else 0.0)
    H_total = agg_hess[feature].sum()
    g_cum = np.cumsum(agg_grad[feature])
    h_cum = np.cumsum(agg_hess[feature])
    GL, HL = g_cum[bin_], h_cum[bin_]
    if feature == target_feature and bin_ >= target_bin:
        GL = GL + delta
    GR = G_total - GL
    return (GL**2 / (HL + lam) + GR**2 / (H_total - HL + lam)
            - G_total**2 / (H_total + lam))
