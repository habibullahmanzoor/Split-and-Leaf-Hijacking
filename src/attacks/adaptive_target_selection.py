"""Category-3 attack variant: adaptive, still fully blind, target FEATURE
selection. he_crypto_layer.py's MaliciousPassiveParty requires
target_feature to be fixed in advance -- every experiment in this paper
hardcodes it to 0, an arbitrary choice never itself selected by any
heuristic. This variant picks its own target feature using only
information it already has, its own local (binned) feature values, no
labels, no gradients, no cross-client aggregate: the feature with the
largest sample variance among its own columns. Bin selection within that
feature is unchanged from MaliciousPassiveParty (still the most-populated
-bin heuristic).

Motivation: on the credit default dataset, feature 0 (a low-cardinality
repayment-status code) produces no damage at any scale, while every
high-cardinality dollar-amount feature the same party holds produces
substantial damage, confirmed by a direct sweep over every feature this
party owns. Variance among a party's own binned values is a cheap, public,
label-free proxy for exactly this cardinality distinction, since a
low-cardinality feature collapses into a small number of repeated bin
indices under quantile binning while a high-cardinality continuous feature
spreads roughly evenly across all bins.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from attacks.he_crypto_layer import MaliciousPassiveParty


@dataclass
class VarianceTargetedMaliciousPassiveParty(MaliciousPassiveParty):
    def compute_encrypted_histograms(self, sample_idx: np.ndarray,
                                      enc_grad: list, enc_hess: list, n_bins: int):
        variances = self.X[sample_idx].astype(float).var(axis=0)
        self.target_feature = int(np.argmax(variances))
        return super().compute_encrypted_histograms(sample_idx, enc_grad, enc_hess, n_bins)
