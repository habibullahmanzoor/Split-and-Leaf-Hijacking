"""Category-6 (multi-vector): combines two mechanistically distinct VFL
attacks that touch different parts of the pipeline -- HE/crypto-layer
manipulation (attacks SPLIT SELECTION) and leaf-routing manipulation
(attacks POST-SPLIT aggregation) -- and asks whether a jointly
sub-threshold combination beats either alone.

Unlike category 3's single-cell ciphertext spike (an obvious outlier once
decrypted), this uses a UNIFORM, modest amplification across the entire
histogram -- no single cell stands out, but the party's features become
more likely to win splits overall. That increases how often
`PassiveParty.route` gets invoked for this party, which is the only place
the misrouting half of the attack can do anything. Each half is tuned to
be below the range that caused visible damage alone in categories 3 and 5
respectively; the question is whether the combination is synergistic
(> sum of the two individual costs) or merely additive.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from harness.vfl_secureboost import PassiveParty


@dataclass
class CombinedPassiveParty(PassiveParty):
    he_uniform_scale: float = 1.0  # 1.0 = no HE manipulation
    misroute_fraction: float = 0.0  # 0.0 = no misrouting
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0), repr=False)
    splits_won: int = field(default=0, init=False, repr=False)

    def compute_encrypted_histograms(self, sample_idx: np.ndarray,
                                      enc_grad: list, enc_hess: list, n_bins: int):
        grad_hist, hess_hist = super().compute_encrypted_histograms(
            sample_idx, enc_grad, enc_hess, n_bins
        )
        if self.he_uniform_scale != 1.0:
            n_features = len(grad_hist)
            for f in range(n_features):
                for b in range(n_bins):
                    if grad_hist[f][b] is not None:
                        grad_hist[f][b] = grad_hist[f][b] * self.he_uniform_scale
        return grad_hist, hess_hist

    def route(self, sample_idx: np.ndarray, feature: int, split_bin: int):
        self.splits_won += 1
        true_mask = self.X[sample_idx, feature] <= split_bin
        if self.misroute_fraction > 0:
            flip = self._rng.random(len(sample_idx)) < self.misroute_fraction
            reported_mask = np.where(flip, ~true_mask, true_mask)
        else:
            reported_mask = true_mask
        return sample_idx[reported_mask], sample_idx[~reported_mask]
