"""Label-inference baseline, for direct comparison against category 4's
label-free backdoor.

Reproduces the standard pattern documented in prior VFL attack literature
(the "instance clustering" style attack against SecureBoost, and the
label-inference-as-prerequisite step in VFL backdoor papers): a passive
party observes which samples land in the same leaf across the ensemble
(this is leaked by the protocol -- every party sees sample_idx at every
node it's asked to contribute a histogram for, regardless of whose feature
won the split). Combined with a small auxiliary seed of known labels, the
attacker assigns majority-vote pseudo-labels to every sample that
co-occurs in a leaf with at least one seed-labeled sample.

This lets a passive party build a TARGETED backdoor (flip only samples it
believes belong to a chosen true class) -- something category 4's
label-free attack cannot do by design, since it selects targets purely by
feature-threshold, blind to label. The comparison quantifies exactly what
that blindness costs (inference accuracy, coverage, extra auxiliary-data
assumption) against what category 4 gives up by not paying it.
"""
from __future__ import annotations

import numpy as np


def collect_leaves(tree):
    leaves = []
    frontier = [tree]
    while frontier:
        node = frontier.pop()
        if node.is_leaf:
            leaves.append(node)
        else:
            if node.left is not None:
                frontier.append(node.left)
            if node.right is not None:
                frontier.append(node.right)
    return leaves


def infer_labels_via_leaf_comembership(trees: list, seed_labels: dict[int, int], n_samples: int):
    """Returns {sample_idx: inferred_label (0/1)} for every sample that
    co-occurs in a leaf with at least one seed-labeled sample, in any tree.
    Samples that never share a leaf with a seed sample are left uninferred
    (not in the returned dict) -- this "coverage gap" is itself part of the
    cost of the label-inference approach, reported alongside accuracy."""
    votes = _collect_votes(trees, seed_labels, n_samples)
    return {i: (1 if np.mean(v) >= 0.5 else 0) for i, v in votes.items() if v}


def _collect_votes(trees: list, seed_labels: dict[int, int], n_samples: int):
    votes = {i: [] for i in range(n_samples)}
    for tree in trees:
        for leaf in collect_leaves(tree):
            idx = leaf.sample_idx["all"]
            seed_in_leaf = [i for i in idx if i in seed_labels]
            if not seed_in_leaf:
                continue
            vote = 1 if np.mean([seed_labels[i] for i in seed_in_leaf]) >= 0.5 else 0
            for i in idx:
                votes[i].append(vote)
    return votes


def infer_labels_with_confidence(trees: list, seed_labels: dict[int, int], n_samples: int):
    """Same majority-vote inference as infer_labels_via_leaf_comembership,
    but also returns a per-sample confidence score: the fraction of votes
    agreeing with the majority (0.5 = a coin flip across the trees this
    sample appeared in with a seed neighbor, 1.0 = every tree agreed). This
    is the same information the binary version discards, and is what an
    attacker filtering for reliability rather than coverage would actually
    condition on: a sample with unanimous votes across many trees is a much
    more trustworthy inference than one that narrowly won a single tree's
    leaf majority."""
    votes = _collect_votes(trees, seed_labels, n_samples)
    inferred, confidence = {}, {}
    for i, v in votes.items():
        if not v:
            continue
        label = 1 if np.mean(v) >= 0.5 else 0
        agree = np.mean([1 if vote == label else 0 for vote in v])
        inferred[i] = label
        confidence[i] = float(agree)
    return inferred, confidence


def inference_accuracy(inferred: dict[int, int], y_true: np.ndarray, seed_labels: dict[int, int]):
    """Accuracy and coverage EXCLUDING the seed samples themselves (which
    would trivially be 100% correct -- they're given, not inferred)."""
    eval_idx = [i for i in inferred if i not in seed_labels]
    if not eval_idx:
        return 0.0, 0.0
    correct = sum(1 for i in eval_idx if inferred[i] == y_true[i])
    accuracy = correct / len(eval_idx)
    coverage = len(eval_idx) / (len(y_true) - len(seed_labels))
    return accuracy, coverage
