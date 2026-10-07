"""The anticipated adaptive counter-attacker for gradient conservation
verification (defenses/gradient_conservation_verification.py): rather than
just inflating one bin's ciphertext (he_crypto_layer.py's
MaliciousPassiveParty, which breaks the per-feature sum-to-total identity
outright), also deflate a decoy bin of the SAME feature by the exact same
amount.

The attacker never decrypts anything and never learns any plaintext
gradient value -- it only uses the fact that Paillier ciphertexts support
addition and subtraction homomorphically: E(g*k) - E(g) = E(g*(k-1)) is
computable directly from the target bin's own ciphertext, with no
decryption oracle required, so subtracting that exact quantity from a
decoy bin's ciphertext exactly cancels the inflation's effect on the
feature's own bin sum, evading the conservation check by construction.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from harness.vfl_secureboost import PassiveParty


@dataclass
class CompensatingMaliciousPassiveParty(PassiveParty):
    target_feature: int = 0
    scale_factor: float = 10.0

    def compute_encrypted_histograms(self, sample_idx: np.ndarray,
                                      enc_grad: list, enc_hess: list, n_bins: int):
        grad_hist, hess_hist = super().compute_encrypted_histograms(
            sample_idx, enc_grad, enc_hess, n_bins
        )
        f = self.target_feature
        bins_at_f = self.X[sample_idx, f]
        counts = np.bincount(bins_at_f, minlength=n_bins)
        target_bin = int(np.argmax(counts[:-1]))

        if grad_hist[f][target_bin] is None or self.scale_factor == 1.0:
            return grad_hist, hess_hist

        decoy_bin = None
        for b in range(n_bins - 1, -1, -1):
            if b != target_bin and grad_hist[f][b] is not None:
                decoy_bin = b
                break
        if decoy_bin is None:
            # No distinct non-empty bin left to absorb the compensation at
            # this particular node (can happen at small, deep nodes) --  an
            # attacker that cannot hide the inflation here simply does not
            # attack here, rather than attacking and getting caught. This is
            # the strongest, most honest version of the adaptive attacker,
            # not a weaker one: it only ever corrupts where it can also
            # evade detection.
            return grad_hist, hess_hist

        original = grad_hist[f][target_bin]
        inflated = original * self.scale_factor
        delta = inflated - original  # E(g*(scale_factor-1)) -- pure ciphertext arithmetic
        grad_hist[f][target_bin] = inflated
        grad_hist[f][decoy_bin] = grad_hist[f][decoy_bin] - delta

        return grad_hist, hess_hist
