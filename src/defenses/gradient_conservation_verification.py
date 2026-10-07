"""Gradient-conservation verification: a SECOND protocol-invariant defense,
orthogonal to routing_verification.py's repeated-query check.

Different invariant, same underlying principle (check something the
protocol structurally guarantees, not a statistic of one reported value):
in SecureBoost-style VFL, the active party computes G_total/H_total for a
node's sample set directly from its own labels and current predictions --
this total has zero dependence on any passive party's histogram. An HONEST
passive party's own per-feature histogram is just a regrouping of the same
underlying gradient/Hessian values into bins by its own feature, so for any
feature f it reports, sum(grad_hist[f]) must equal G_total exactly (an
algebraic identity, true for any dataset, any heterogeneity, any node size
-- not a statistical judgment call the way every contribution-level
detector in Section 5.4 of the paper is).

Category 3's ciphertext-rescaling attack (he_crypto_layer.py) inflates one
bin without compensating elsewhere, so it breaks this identity outright and
is caught for free. Section discusses the anticipated limitation: an
attacker who ALSO deflates a decoy bin by the same amount it inflated the
target bin (gradient_conservation_evasion.py) preserves the identity while
still forcing its split, and evades this check by construction -- exactly
analogous to routing_verification.py's deterministic-misrouting escape
hatch, just for a different invariant.
"""
from __future__ import annotations

import numpy as np

from harness.vfl_secureboost import VFLServer, logistic_grad_hess


class ConservationVerifiedVFLServer(VFLServer):
    def __init__(self, active, passives, n_bins, max_depth, lam=1.0, gamma=0.0,
                 min_child_hess=1.0, tol=1e-4):
        super().__init__(active, passives, n_bins, max_depth, lam, gamma, min_child_hess)
        self.tol = tol
        self.banned_parties = set()
        self.n_verified = 0
        self.n_banned_events = 0

    def _candidate_splits(self, sample_idx: np.ndarray, enc_grad, enc_hess):
        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        G_total, H_total = grad[sample_idx].sum(), hess[sample_idx].sum()
        a_grad_hist, a_hess_hist = self.active.local_histograms(sample_idx, grad, hess, self.n_bins)
        candidates = [("active", a_grad_hist, a_hess_hist)]

        for pid, passive in self.passives.items():
            if pid in self.banned_parties:
                continue
            eg, eh = passive.compute_encrypted_histograms(sample_idx, enc_grad, enc_hess, self.n_bins)
            g_hist, h_hist = self.active.decrypt_histogram(eg, eh, self.n_bins)
            n_features = g_hist.shape[0]
            tol_g = self.tol * max(1.0, abs(G_total))
            tol_h = self.tol * max(1.0, abs(H_total))
            violated = any(
                abs(g_hist[f].sum() - G_total) > tol_g or abs(h_hist[f].sum() - H_total) > tol_h
                for f in range(n_features)
            )
            if violated:
                self.banned_parties.add(pid)
                self.n_banned_events += 1
                continue
            self.n_verified += 1
            candidates.append((pid, g_hist, h_hist))
        return candidates
