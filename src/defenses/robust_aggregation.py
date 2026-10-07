"""Baseline defenses: coordinate-wise median and trimmed-mean, the standard
Byzantine-robust aggregation rules from the NN-style FL literature (Krum's
family), adapted to histogram aggregation. Tests whether they transfer to
the discrete-argmax setting our attacks exploit, rather than assuming they
do or don't.

Only meaningful for HFL, where multiple clients redundantly contribute a
histogram for the SAME features -- there's something to compare across. In
VFL, each passive party owns disjoint, non-overlapping features, so there
is no redundant multi-party contribution to take a median/trimmed-mean
over; these defenses are structurally inapplicable there, not merely
untested.
"""
from __future__ import annotations

import numpy as np

from harness.federated_gbdt import Client
from attacks.histogram_integrity import AttackServer


def median_aggregate(client_grads, client_hess, n_clients_total):
    stacked_g = np.stack(client_grads, axis=0)
    stacked_h = np.stack(client_hess, axis=0)
    med_g = np.median(stacked_g, axis=0) * n_clients_total
    med_h = np.median(stacked_h, axis=0) * n_clients_total
    return med_g, med_h


def trimmed_mean_aggregate(client_grads, client_hess, n_clients_total, trim_count):
    stacked_g = np.stack(client_grads, axis=0)
    stacked_h = np.stack(client_hess, axis=0)
    n = stacked_g.shape[0]
    k = min(trim_count, (n - 1) // 2)
    sorted_g = np.sort(stacked_g, axis=0)
    sorted_h = np.sort(stacked_h, axis=0)
    if k > 0:
        sorted_g, sorted_h = sorted_g[k:n - k], sorted_h[k:n - k]
    mean_g = sorted_g.mean(axis=0) * n_clients_total
    mean_h = sorted_h.mean(axis=0) * n_clients_total
    return mean_g, mean_h


class RobustAttackServer(AttackServer):
    """Same attack-injection logic as AttackServer, but the final
    combination step is a pluggable robust-aggregation rule instead of a
    naive sum -- lets the SAME attack be tested against SUM/MEDIAN/
    TRIMMED_MEAN aggregation directly, apples to apples."""

    def __init__(self, *args, aggregation_rule: str = "sum", trim_count: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_rule = aggregation_rule
        self.trim_count = trim_count

    def _aggregate_histograms(self, sample_idx: dict):
        truthful = {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            truthful[c.client_id] = Client.compute_histograms(c, idx, self.n_bins)

        round_ok = self.target_rounds is None or self._round in self.target_rounds
        attack_here = round_ok and self._current_depth <= self.attack_max_depth

        client_grads, client_hess = [], []
        for cid, (g, h) in truthful.items():
            if attack_here and cid in self.malicious_ids:
                others_g = others_h = None
                for ocid, (og, oh) in truthful.items():
                    if ocid == cid:
                        continue
                    others_g = og if others_g is None else others_g + og
                    others_h = oh if others_h is None else others_h + oh
                g, h = self.attack_fn(g, h, others_g, others_h, **self.attack_kwargs)
            client_grads.append(g)
            client_hess.append(h)

        if not client_grads:
            return None, None

        n_total = len(self.clients)
        if self.aggregation_rule == "sum":
            agg_grad = sum(client_grads)
            agg_hess = sum(client_hess)
        elif self.aggregation_rule == "median":
            agg_grad, agg_hess = median_aggregate(client_grads, client_hess, n_total)
        elif self.aggregation_rule == "trimmed_mean":
            agg_grad, agg_hess = trimmed_mean_aggregate(client_grads, client_hess, n_total, self.trim_count)
        else:
            raise ValueError(f"unknown aggregation_rule: {self.aggregation_rule}")
        return agg_grad, agg_hess
