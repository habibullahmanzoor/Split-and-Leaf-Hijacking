"""Category-5 attack: leaf-routing manipulation.

SecureBoost's HE protects the gradient/Hessian histogram exchange used to
pick a split -- that's the part with cryptographic backing. But once a
passive party's feature wins a split, the active party has no way to
verify the passive party's ROUTING of samples to the left/right child; it
is simply trusted. `PassiveParty.route` in vfl_secureboost.py is called
directly with zero verification.

A malicious passive party can misroute a fraction of samples independent
of their true feature value, corrupting the leaf-value gradient statistics
computed downstream -- without ever touching a ciphertext or a split
decision. This is mechanistically distinct from categories 1/3/4: it
attacks the aggregation step AFTER the split is already chosen honestly.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from harness.vfl_secureboost import PassiveParty


@dataclass
class MisroutingPassiveParty(PassiveParty):
    misroute_fraction: float = 0.5
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0), repr=False)

    def route(self, sample_idx: np.ndarray, feature: int, split_bin: int):
        true_mask = self.X[sample_idx, feature] <= split_bin
        flip = self._rng.random(len(sample_idx)) < self.misroute_fraction
        reported_mask = np.where(flip, ~true_mask, true_mask)
        return sample_idx[reported_mask], sample_idx[~reported_mask]
