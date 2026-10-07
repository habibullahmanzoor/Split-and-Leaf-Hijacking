"""Category-3 attack: HE/crypto-layer.

Unlike category 1's malicious client, a malicious passive party in
SecureBoost-style VFL never sees plaintext gradients -- that opacity is the
entire point of the HE layer. So this attack cannot use category 1's
"look at the honest aggregate, target the worst-gain split" strategy. It
instead exploits a structural property of unauthenticated Paillier
ciphertexts: they are malleable. Anyone holding the public key (which the
passive party legitimately receives, to encrypt nothing itself but to
combine received ciphertexts) can homomorphically scale a ciphertext by a
known plaintext scalar -- E(g) * k = E(gk) -- without ever decrypting it or
learning g. The active party has no way to bounds-check a ciphertext before
decryption, so a rescaled contribution is indistinguishable from a truthful
one until it's already been used to pick a split.

The attacker doesn't need to know which bin has the largest true gradient
sum -- it only needs public information (which bin has the most samples,
visible to the passive party since it owns its own feature values) to pick
a plausible high-leverage target, then blindly inflates that bin's
ciphertext so gain's quadratic term guarantees it wins the argmax.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from harness.vfl_secureboost import PassiveParty


@dataclass
class MaliciousPassiveParty(PassiveParty):
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
        # highest-sample-count bin among valid split positions (never the last bin)
        target_bin = int(np.argmax(counts[:-1]))

        if grad_hist[f][target_bin] is not None and self.scale_factor != 1.0:
            # Scale gradient only, leave Hessian truthful -- scaling Hessian
            # too trips the tree-builder's min_child_hess sanity check (a
            # standard regularization guard, not a security defense) once
            # the scale factor is large, causing the forced split to be
            # silently rejected and an honest split chosen instead. Matches
            # category 1's margin_flip_attack, which forges gradient only.
            grad_hist[f][target_bin] = grad_hist[f][target_bin] * self.scale_factor
        return grad_hist, hess_hist
