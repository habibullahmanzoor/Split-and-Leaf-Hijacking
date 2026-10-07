"""Exact computations for the contribution-level indistinguishability
result in THEORY_IMPOSSIBILITY.md.

Lemma 1: for p ~ Dir(alpha,...,alpha) (K-dimensional, symmetric), each
coordinate p_i ~ Beta(alpha, (K-1)*alpha). This module wraps scipy's exact
Beta distribution for the two quantities the Proposition needs:
  - Pr[p_i >= t]                          (Corollary 1)
  - tau(epsilon) such that Pr[p_i >= tau(epsilon)] = epsilon   (Proposition)
"""
from __future__ import annotations

from scipy.stats import beta as beta_dist


def beta_params(alpha: float, n_clients: int) -> tuple[float, float]:
    """Beta(alpha, (K-1)*alpha) parameters for a single coordinate's
    marginal of a symmetric K-dim Dirichlet(alpha)."""
    return alpha, (n_clients - 1) * alpha


def prob_concentration_at_least(t: float, alpha: float, n_clients: int) -> float:
    """Pr[p_i >= t] -- Corollary 1's lower bound on Pr[some client holds
    share >= t of a cell]. Exact, via the Beta survival function."""
    a, b = beta_params(alpha, n_clients)
    return float(beta_dist.sf(t, a, b))


def tau_for_epsilon(epsilon: float, alpha: float, n_clients: int) -> float:
    """The (1-epsilon)-quantile threshold tau(epsilon) such that
    Pr[p_i >= tau(epsilon)] = epsilon -- the honest-concentration ceiling
    a false-positive-rate-epsilon detector is forced to tolerate."""
    a, b = beta_params(alpha, n_clients)
    return float(beta_dist.isf(epsilon, a, b))


def implied_epsilon_for_margin(margin: float, m_samples: int, alpha: float,
                                n_clients: int, max_grad_per_sample: float = 1.0) -> float:
    """Inverts the Proposition: given a measured margin at a cell with
    m_samples total samples, what is the SMALLEST false-positive rate
    epsilon a detector would need to tolerate for that margin to fall at
    or below the honest-concentration ceiling tau(epsilon)*m*G_max? A
    SMALL returned epsilon means the margin is easily explained by honest
    concentration -- the indistinguishability bites hard. A LARGE
    (near-1) epsilon means the margin is implausibly large for any honest
    client to produce -- contribution-level detection could plausibly work.

    This is exactly epsilon*(delta) in the single-coordinate detection-rate
    theorem below: the honest-case probability that some client's true
    share alone reaches the tested margin.
    """
    if m_samples <= 0:
        return float("nan")
    tau = margin / (m_samples * max_grad_per_sample)
    tau = min(max(tau, 0.0), 1.0)
    return prob_concentration_at_least(tau, alpha, n_clients)


def detection_rate_bound(epsilon_budget: float, margin: float, m_samples: int,
                          alpha: float, n_clients: int,
                          max_grad_per_sample: float = 1.0) -> float:
    """The rigorous, quantitative single-coordinate detection-rate theorem.

    For any magnitude-only detector (accept/reject depends only on the
    flagged client's own reported value, e.g. a threshold or margin-ratio
    rule -- this exactly covers the "size-calibrated" and "theory-grounded"
    margin-certified designs) with honest-case false-positive rate at most
    epsilon_budget, consider the adaptive attacker (already licensed by the
    threat model) who reports a value drawn from the conditional law of an
    honest client's contribution given its share exceeds margin/(m*G_max).
    That report is >= margin almost surely (so it still flips the split),
    and its marginal distribution is EXACTLY the honest tail distribution
    the detector must tolerate, by construction -- no approximation.

    Writing epsilon*(margin) = implied_epsilon_for_margin(...) for the
    honest-case probability of that tail event, conditional-probability
    bookkeeping (P_honest(reject) >= epsilon*(margin) * P_honest(reject |
    tail)) gives the detector's best possible detection rate against this
    attacker as at most epsilon_budget / epsilon*(margin), capped at 1.
    """
    eps_star = implied_epsilon_for_margin(margin, m_samples, alpha, n_clients, max_grad_per_sample)
    if eps_star <= 0:
        return 1.0
    return float(min(1.0, epsilon_budget / eps_star))
