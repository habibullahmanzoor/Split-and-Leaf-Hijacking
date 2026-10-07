"""Horizontal federated GBDT via histogram sharing.

Implements the standard protocol used by FedTree / federated-XGBoost papers:
clients hold row-partitions of data; for every tree node, each client bins its
local gradient/Hessian sums per (feature, bin) and reports the histogram to
the server; the server aggregates histograms across clients and picks the
best split by the usual GBDT gain formula. This repeats per node until trees
reach max_depth, and trees are boosted sequentially.

Global bin edges are assumed pre-agreed (e.g. via a secure quantile sketch,
out of scope here) and are computed once from the full feature range.

Every client -> server exchange goes through `Client.compute_histograms`,
and every split decision goes through `Server._find_best_split`. Those two
methods are the injection points for attacks (histogram manipulation,
malicious split influence) and defenses (margin-certified split selection).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logistic_grad_hess(y_true, raw_score):
    p = sigmoid(raw_score)
    grad = p - y_true
    hess = p * (1 - p)
    return grad, hess


@dataclass
class TreeNode:
    depth: int
    sample_idx: dict  # client_id -> np.ndarray of local row indices at this node
    is_leaf: bool = False
    leaf_value: float = 0.0
    split_feature: int | None = None
    split_bin: int | None = None
    split_threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class Client:
    client_id: int
    X: np.ndarray  # (n_i, n_features), pre-binned to bin indices (int)
    y: np.ndarray
    bin_edges: list  # per-feature array of edges, shared across clients
    raw_score: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.raw_score is None:
            self.raw_score = np.zeros(len(self.y))

    def gradients(self):
        return logistic_grad_hess(self.y, self.raw_score)

    def compute_histograms(self, sample_idx: np.ndarray, n_bins: int):
        """Return per-feature (n_bins,) grad_sum / hess_sum arrays for the
        given local sample indices. This is the message sent to the server —
        the exact quantity an attacker controlling this client can forge."""
        grad, hess = self.gradients()
        g, h = grad[sample_idx], hess[sample_idx]
        Xb = self.X[sample_idx]
        n_features = Xb.shape[1]
        grad_hist = np.zeros((n_features, n_bins))
        hess_hist = np.zeros((n_features, n_bins))
        for f in range(n_features):
            np.add.at(grad_hist[f], Xb[:, f], g)
            np.add.at(hess_hist[f], Xb[:, f], h)
        return grad_hist, hess_hist


class Server:
    def __init__(self, clients: list[Client], n_bins: int, max_depth: int,
                 lam: float = 1.0, gamma: float = 0.0, min_child_hess: float = 1.0):
        self.clients = clients
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_hess = min_child_hess

    def _aggregate_histograms(self, sample_idx: dict):
        """Collect histograms from every client for the current node and sum
        them. This is the aggregation step a histogram-manipulation attack
        (category 1) targets, and where a defense would audit contributions."""
        agg_grad = agg_hess = None
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            g, h = c.compute_histograms(idx, self.n_bins)
            agg_grad = g if agg_grad is None else agg_grad + g
            agg_hess = h if agg_hess is None else agg_hess + h
        return agg_grad, agg_hess

    def _find_best_split(self, agg_grad, agg_hess):
        """Standard GBDT gain formula over aggregated histograms. Returns
        (feature, bin, gain, runner_up_gain) — the runner-up is kept around
        for margin-certified defenses to compare against."""
        n_features, n_bins = agg_grad.shape
        best = (-np.inf, None, None)
        second = (-np.inf, None, None)
        for f in range(n_features):
            # Per-feature total, not a fixed reference feature (e.g. "feature
            # 0"): for an honest histogram every feature's row sums to the
            # same true total, so this is equivalent in the honest case. But
            # a malicious client's forged histogram only touches its target
            # feature's row -- anchoring to a fixed feature silently ignores
            # attacks on every OTHER feature (G_total never reflects them).
            # Using each feature's own row keeps every feature's candidates
            # internally consistent regardless of which feature is attacked.
            G_total, H_total = agg_grad[f].sum(), agg_hess[f].sum()
            g_cum = np.cumsum(agg_grad[f])
            h_cum = np.cumsum(agg_hess[f])
            for b in range(n_bins - 1):
                GL, HL = g_cum[b], h_cum[b]
                GR, HR = G_total - GL, H_total - HL
                if HL < self.min_child_hess or HR < self.min_child_hess:
                    continue
                gain = 0.5 * (GL**2 / (HL + self.lam) + GR**2 / (HR + self.lam)
                               - G_total**2 / (H_total + self.lam)) - self.gamma
                if gain > best[0]:
                    second = best
                    best = (gain, f, b)
                elif gain > second[0]:
                    second = (gain, f, b)
        return best, second  # ((gain, feature, bin), (gain2, feature2, bin2))

    def _build_node(self, sample_idx: dict, depth: int) -> TreeNode:
        node = TreeNode(depth=depth, sample_idx=sample_idx)
        total_n = sum(len(v) for v in sample_idx.values())
        agg_grad, agg_hess = self._aggregate_histograms(sample_idx)

        if depth >= self.max_depth or total_n < 2 or agg_grad is None:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        (gain, feat, b), (gain2, feat2, b2) = self._find_best_split(agg_grad, agg_hess)
        if feat is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        node.split_feature, node.split_bin = feat, b
        left_idx, right_idx = {}, {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            mask = c.X[idx, feat] <= b
            left_idx[c.client_id] = idx[mask]
            right_idx[c.client_id] = idx[~mask]

        node.left = self._build_node(left_idx, depth + 1)
        node.right = self._build_node(right_idx, depth + 1)
        return node

    def _leaf_value(self, sample_idx: dict):
        agg_grad, agg_hess = self._aggregate_histograms(sample_idx)
        if agg_grad is None:
            return 0.0
        G, H = agg_grad.sum(), agg_hess.sum()
        return -G / (H + self.lam)

    def build_tree(self) -> TreeNode:
        root_idx = {c.client_id: np.arange(len(c.y)) for c in self.clients}
        return self._build_node(root_idx, depth=0)


def predict_tree(tree: TreeNode, X_row_binned: np.ndarray) -> float:
    node = tree
    while not node.is_leaf:
        node = node.left if X_row_binned[node.split_feature] <= node.split_bin else node.right
    return node.leaf_value


def bin_features(X: np.ndarray, bin_edges: list[np.ndarray]) -> np.ndarray:
    Xb = np.zeros_like(X, dtype=int)
    for f in range(X.shape[1]):
        Xb[:, f] = np.clip(np.searchsorted(bin_edges[f], X[:, f], side="right") - 1,
                            0, len(bin_edges[f]) - 2)
    return Xb


def compute_bin_edges(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
    edges = []
    for f in range(X.shape[1]):
        qs = np.quantile(X[:, f], np.linspace(0, 1, n_bins))
        edges.append(np.unique(qs))
    return edges


class FederatedGBDT:
    def __init__(self, n_bins: int = 32, max_depth: int = 4, n_rounds: int = 20,
                 lam: float = 1.0, gamma: float = 0.0, lr: float = 0.3,
                 server_cls=Server, server_kwargs: dict | None = None):
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.server_cls = server_cls
        self.server_kwargs = server_kwargs or {}
        self.trees: list[TreeNode] = []
        self.bin_edges: list[np.ndarray] | None = None

    def fit(self, client_data: dict[int, tuple[np.ndarray, np.ndarray]], eval_callback=None):
        X_all = np.vstack([X for X, _ in client_data.values()])
        self.bin_edges = compute_bin_edges(X_all, self.n_bins)

        clients = [
            Client(client_id=cid, X=bin_features(X, self.bin_edges), y=y,
                   bin_edges=self.bin_edges)
            for cid, (X, y) in client_data.items()
        ]
        server = self.server_cls(clients, self.n_bins, self.max_depth, self.lam, self.gamma,
                                  **self.server_kwargs)

        for t in range(self.n_rounds):
            tree = server.build_tree()
            self.trees.append(tree)
            for c in clients:
                updates = np.array([predict_tree(tree, row) for row in c.X])
                c.raw_score = c.raw_score + self.lr * updates
            if eval_callback is not None:
                eval_callback(t, self)
        self._clients = clients
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = bin_features(X, self.bin_edges)
        raw = np.zeros(len(X))
        for tree in self.trees:
            raw += self.lr * np.array([predict_tree(tree, row) for row in Xb])
        return sigmoid(raw)
