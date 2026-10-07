"""Category-4 attack: passive-party role-abuse.

Distinct from categories 1 and 3, whose goal is to maximize accuracy
damage, this attacker wants something quieter: get the tree to split on a
(feature, bin) the passive party controls, then exploit that trigger at
INFERENCE time by manipulating its own submitted feature value for chosen
samples -- flipping their prediction on demand, in production, without ever
needing to see a label or infer one first (unlike the label-inference ->
backdoor attack chain documented for SecureBoost in prior work, which does
require inferring labels as a prerequisite).

The passive party gets this power purely from its STRUCTURAL role as a
feature-holder whose statistics the active party must trust to build the
tree -- it doesn't need label inference, doesn't need to see plaintext
gradients (same blind ciphertext-rescaling mechanism as category 3), it
just needs to consistently win the split at an attacker-chosen (feature,
bin) of its own choosing.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from harness.vfl_secureboost import PassiveParty


@dataclass
class BackdoorPassiveParty(PassiveParty):
    trigger_feature: int = 0
    trigger_bin: int = 8
    scale_factor: float = 10.0
    num_plants: int = 1
    _fires_remaining: int = field(default=None, init=False, repr=False)

    def compute_encrypted_histograms(self, sample_idx: np.ndarray,
                                      enc_grad: list, enc_hess: list, n_bins: int):
        grad_hist, hess_hist = super().compute_encrypted_histograms(
            sample_idx, enc_grad, enc_hess, n_bins
        )
        if self._fires_remaining is None:
            self._fires_remaining = self.num_plants
        # Fire at the first `num_plants` calls -- each call is the root of
        # successive trees (tree 0, tree 1, ...), the highest-leverage nodes
        # available. num_plants=1 is maximally stealthy (one plant, minimal
        # accuracy cost); increasing it re-plants the SAME trigger at
        # multiple tree roots, broadening the fraction of the ensemble that
        # routes through it -- more reliable trigger coverage at the cost of
        # more accuracy damage. This is the stealth-vs-success dial.
        if self._fires_remaining > 0:
            f, b = self.trigger_feature, self.trigger_bin
            if grad_hist[f][b] is not None:
                grad_hist[f][b] = grad_hist[f][b] * self.scale_factor
                self._fires_remaining -= 1
        return grad_hist, hess_hist
