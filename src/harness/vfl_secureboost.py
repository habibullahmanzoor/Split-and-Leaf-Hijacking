"""Vertical federated GBDT following SecureBoost's published protocol
(Cheng et al., 2021): one active party holds labels (and its own feature
block); one or more passive parties hold only features.

Per node: the active party encrypts its current gradients/Hessians with its
own Paillier keypair and sends the ciphertexts to each passive party. Each
passive party bins its local features and homomorphically sums the received
ciphertexts per (feature, bin) — it never sees plaintext gradients — and
returns the encrypted histogram. The active party decrypts, merges these
candidate splits with its own plaintext local splits, and picks the global
best by the standard gain formula. As in real SecureBoost, the winning
(feature, threshold) is revealed to the active party even when it belongs to
a passive party — a known, documented leakage of the protocol, not a bug we
introduced.

`PassiveParty.compute_encrypted_histograms` and `VFLServer._find_best_split`
are the injection points for category-3 (HE/crypto-layer) and category-4
(passive-party role-abuse) attacks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from phe import paillier

from harness.federated_gbdt import (
    sigmoid, logistic_grad_hess, bin_features, compute_bin_edges, TreeNode,
)


@dataclass
class PassiveParty:
    party_id: str
    X: np.ndarray  # already bin-indexed
    bin_edges: list

    def compute_encrypted_histograms(self, sample_idx: np.ndarray,
                                      enc_grad: list, enc_hess: list, n_bins: int):
        """Homomorphically sum the active party's encrypted gradients/Hessians
        into this party's own (feature, bin) histogram. `enc_grad`/`enc_hess`
        are full-dataset ciphertext lists; only entries at `sample_idx` are
        used. Never decrypts, never sees plaintext gradient values."""
        n_features = self.X.shape[1]
        grad_hist = [[None] * n_bins for _ in range(n_features)]
        hess_hist = [[None] * n_bins for _ in range(n_features)]
        for f in range(n_features):
            bins_at_f = self.X[sample_idx, f]
            for b in range(n_bins):
                mask = bins_at_f == b
                idx_in_bin = sample_idx[mask]
                if len(idx_in_bin) == 0:
                    continue
                g_sum = enc_grad[idx_in_bin[0]]
                h_sum = enc_hess[idx_in_bin[0]]
                for i in idx_in_bin[1:]:
                    g_sum = g_sum + enc_grad[i]
                    h_sum = h_sum + enc_hess[i]
                grad_hist[f][b] = g_sum
                hess_hist[f][b] = h_sum
        return grad_hist, hess_hist

    def route(self, sample_idx: np.ndarray, feature: int, split_bin: int):
        mask = self.X[sample_idx, feature] <= split_bin
        return sample_idx[mask], sample_idx[~mask]


@dataclass
class ActiveParty:
    X: np.ndarray  # already bin-indexed, active party's own features
    y: np.ndarray
    bin_edges: list
    key_size: int = 512
    raw_score: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.raw_score is None:
            self.raw_score = np.zeros(len(self.y))
        self.pub, self.priv = paillier.generate_paillier_keypair(n_length=self.key_size)

    def encrypt_gradients(self):
        grad, hess = logistic_grad_hess(self.y, self.raw_score)
        enc_grad = [self.pub.encrypt(float(g)) for g in grad]
        enc_hess = [self.pub.encrypt(float(h)) for h in hess]
        return enc_grad, enc_hess

    def local_histograms(self, sample_idx: np.ndarray, grad: np.ndarray,
                          hess: np.ndarray, n_bins: int):
        n_features = self.X.shape[1]
        grad_hist = np.zeros((n_features, n_bins))
        hess_hist = np.zeros((n_features, n_bins))
        Xb = self.X[sample_idx]
        g, h = grad[sample_idx], hess[sample_idx]
        for f in range(n_features):
            np.add.at(grad_hist[f], Xb[:, f], g)
            np.add.at(hess_hist[f], Xb[:, f], h)
        return grad_hist, hess_hist

    def decrypt_histogram(self, enc_grad_hist, enc_hess_hist, n_bins):
        n_features = len(enc_grad_hist)
        grad_hist = np.zeros((n_features, n_bins))
        hess_hist = np.zeros((n_features, n_bins))
        for f in range(n_features):
            for b in range(n_bins):
                if enc_grad_hist[f][b] is not None:
                    grad_hist[f][b] = self.priv.decrypt(enc_grad_hist[f][b])
                    hess_hist[f][b] = self.priv.decrypt(enc_hess_hist[f][b])
        return grad_hist, hess_hist


class VFLServer:
    """Orchestrates the active party + passive parties. In a real deployment
    the active party plays this role itself; kept separate here for clarity
    and so attacks/defenses can sit at the orchestration boundary too."""

    def __init__(self, active: ActiveParty, passives: dict[str, PassiveParty],
                 n_bins: int, max_depth: int, lam: float = 1.0, gamma: float = 0.0,
                 min_child_hess: float = 1.0):
        self.active = active
        self.passives = passives
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_hess = min_child_hess

    def _candidate_splits(self, sample_idx: np.ndarray, enc_grad, enc_hess):
        """Returns list of (owner, feature, bin, grad_hist_col, hess_hist_col)
        candidates pooled from the active party and every passive party."""
        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        a_grad_hist, a_hess_hist = self.active.local_histograms(sample_idx, grad, hess, self.n_bins)
        candidates = [("active", a_grad_hist, a_hess_hist)]

        for pid, passive in self.passives.items():
            eg, eh = passive.compute_encrypted_histograms(sample_idx, enc_grad, enc_hess, self.n_bins)
            g_hist, h_hist = self.active.decrypt_histogram(eg, eh, self.n_bins)
            candidates.append((pid, g_hist, h_hist))
        return candidates

    def _find_best_split(self, candidates, G_total, H_total):
        best = (-np.inf, None, None, None)
        second = (-np.inf, None, None, None)
        for owner, g_hist, h_hist in candidates:
            n_features, n_bins = g_hist.shape
            for f in range(n_features):
                g_cum = np.cumsum(g_hist[f])
                h_cum = np.cumsum(h_hist[f])
                for b in range(n_bins - 1):
                    GL, HL = g_cum[b], h_cum[b]
                    GR, HR = G_total - GL, H_total - HL
                    if HL < self.min_child_hess or HR < self.min_child_hess:
                        continue
                    gain = 0.5 * (GL**2 / (HL + self.lam) + GR**2 / (HR + self.lam)
                                   - G_total**2 / (H_total + self.lam)) - self.gamma
                    if gain > best[0]:
                        second = best
                        best = (gain, owner, f, b)
                    elif gain > second[0]:
                        second = (gain, owner, f, b)
        return best, second

    def _build_node(self, sample_idx: np.ndarray, depth: int, enc_grad, enc_hess) -> TreeNode:
        node = TreeNode(depth=depth, sample_idx={"all": sample_idx})

        if depth >= self.max_depth or len(sample_idx) < 2:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        candidates = self._candidate_splits(sample_idx, enc_grad, enc_hess)
        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        G_total, H_total = grad[sample_idx].sum(), hess[sample_idx].sum()
        (gain, owner, feat, b), _second = self._find_best_split(candidates, G_total, H_total)

        if owner is None or gain <= 0:
            node.is_leaf = True
            node.leaf_value = self._leaf_value(sample_idx)
            return node

        node.split_feature = feat
        node.split_bin = b
        node.split_threshold = owner  # reuse field to record which party owns this split

        if owner == "active":
            mask = self.active.X[sample_idx, feat] <= b
            left_idx, right_idx = sample_idx[mask], sample_idx[~mask]
        else:
            left_idx, right_idx = self.passives[owner].route(sample_idx, feat, b)

        node.left = self._build_node(left_idx, depth + 1, enc_grad, enc_hess)
        node.right = self._build_node(right_idx, depth + 1, enc_grad, enc_hess)
        return node

    def _leaf_value(self, sample_idx: np.ndarray):
        grad, hess = logistic_grad_hess(self.active.y, self.active.raw_score)
        G, H = grad[sample_idx].sum(), hess[sample_idx].sum()
        return -G / (H + self.lam)

    def build_tree(self) -> TreeNode:
        enc_grad, enc_hess = self.active.encrypt_gradients()
        root_idx = np.arange(len(self.active.y))
        return self._build_node(root_idx, depth=0, enc_grad=enc_grad, enc_hess=enc_hess)


def predict_tree_vfl(tree: TreeNode, active_row, passive_rows: dict) -> float:
    node = tree
    while not node.is_leaf:
        owner = node.split_threshold
        row = active_row if owner == "active" else passive_rows[owner]
        node = node.left if row[node.split_feature] <= node.split_bin else node.right
    return node.leaf_value


class FederatedVFLGBDT:
    def __init__(self, n_bins: int = 16, max_depth: int = 3, n_rounds: int = 3,
                 lam: float = 1.0, gamma: float = 0.0, lr: float = 0.3, key_size: int = 512,
                 server_cls=None, server_kwargs: dict | None = None):
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.key_size = key_size
        self.server_cls = server_cls or VFLServer
        self.server_kwargs = server_kwargs or {}
        self.trees: list[TreeNode] = []
        self.active_bin_edges = None
        self.passive_bin_edges: dict[str, list] = {}

    def fit(self, X_active: np.ndarray, y: np.ndarray, X_passives: dict[str, np.ndarray],
            passive_party_factories: dict[str, callable] | None = None, eval_callback=None):
        """`passive_party_factories[pid]`, if given, is called as
        `factory(party_id=pid, X=binned_X, bin_edges=edges)` instead of the
        default PassiveParty constructor -- used to swap in a malicious
        passive party for a specific party id."""
        passive_party_factories = passive_party_factories or {}
        self.active_bin_edges = compute_bin_edges(X_active, self.n_bins)
        self.active = ActiveParty(
            X=bin_features(X_active, self.active_bin_edges), y=y,
            bin_edges=self.active_bin_edges, key_size=self.key_size,
        )
        self.passives = {}
        for pid, Xp in X_passives.items():
            edges = compute_bin_edges(Xp, self.n_bins)
            self.passive_bin_edges[pid] = edges
            factory = passive_party_factories.get(pid, PassiveParty)
            self.passives[pid] = factory(party_id=pid, X=bin_features(Xp, edges), bin_edges=edges)

        server = self.server_cls(self.active, self.passives, self.n_bins, self.max_depth,
                                  self.lam, self.gamma, **self.server_kwargs)
        self.server = server

        for t in range(self.n_rounds):
            tree = server.build_tree()
            self.trees.append(tree)
            n = len(y)
            updates = np.zeros(n)
            for i in range(n):
                active_row = self.active.X[i]
                passive_rows = {pid: p.X[i] for pid, p in self.passives.items()}
                updates[i] = predict_tree_vfl(tree, active_row, passive_rows)
            self.active.raw_score = self.active.raw_score + self.lr * updates
            if eval_callback is not None:
                eval_callback(t, self)
        return self

    def predict_proba(self, X_active: np.ndarray, X_passives: dict[str, np.ndarray]) -> np.ndarray:
        Xa = bin_features(X_active, self.active_bin_edges)
        Xp = {pid: bin_features(X, self.passive_bin_edges[pid]) for pid, X in X_passives.items()}
        n = len(X_active)
        raw = np.zeros(n)
        for tree in self.trees:
            for i in range(n):
                active_row = Xa[i]
                passive_rows = {pid: Xp[pid][i] for pid in Xp}
                raw[i] += self.lr * predict_tree_vfl(tree, active_row, passive_rows)
        return sigmoid(raw)
