"""Shared revision infrastructure: one data split + one partition per seed,
used by BOTH the clean and the attacked run (paired design), a generalized
attack server (knowledge modes, per-client deviation bounds, round schedules,
detector logging), federated binning, and paired-difference statistics."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import json
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.federated_gbdt import (FederatedGBDT, Server, Client, bin_features,
                                    compute_bin_edges, predict_tree, TreeNode)
from theory.margin import compute_split_flip_margin
from attack_category1_sweep import dirichlet_partition

N_CLIENTS, ALPHA, N_SEEDS = 5, 0.5, 10          # 10 matched seeds (was 5)
N_BINS, DEPTH, ROUNDS, LR = 32, 4, 20, 0.3
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def get_data(seed=0):
    """Train/test split is redrawn per seed (matched between clean and attacked)."""
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte


def make_clients(Xtr, ytr, seed):
    parts = dirichlet_partition(Xtr, ytr, N_CLIENTS, ALPHA, seed)
    return {cid: (Xtr[i], ytr[i]) for cid, i in enumerate(parts)}


# ---------------- federated binning ----------------
def federated_bin_edges(client_X, n_bins, sketch=64):
    """Each client sends `sketch` local quantile points per feature; the server
    merges them by a count-weighted quantile (a standard approximate sketch
    merge). No party sees another party's raw feature values."""
    n_feat = client_X[0].shape[1]
    sizes = np.array([len(x) for x in client_X], float)
    edges = []
    for f in range(n_feat):
        pts, w = [], []
        for x, n in zip(client_X, sizes):
            q = np.quantile(x[:, f], np.linspace(0, 1, sketch))
            pts.append(q)
            w.append(np.full(sketch, n / sketch))
        pts, w = np.concatenate(pts), np.concatenate(w)
        o = np.argsort(pts)
        pts, w = pts[o], w[o]
        cw = (np.cumsum(w) - 0.5 * w) / w.sum()
        e = np.interp(np.linspace(0, 1, n_bins), cw, pts)
        e[0] = min(x[:, f].min() for x in client_X)
        e[-1] = max(x[:, f].max() for x in client_X)
        edges.append(np.unique(e))
    return edges


class RevGBDT(FederatedGBDT):
    def __init__(self, *a, fed_bins=False, **k):
        super().__init__(*a, **k)
        self.fed_bins = fed_bins

    def fit(self, client_data, eval_callback=None):
        Xs = [X for X, _ in client_data.values()]
        self.bin_edges = (federated_bin_edges(Xs, self.n_bins) if self.fed_bins
                          else compute_bin_edges(np.vstack(Xs), self.n_bins))
        clients = [Client(client_id=c, X=bin_features(X, self.bin_edges), y=y,
                          bin_edges=self.bin_edges) for c, (X, y) in client_data.items()]
        self.server = self.server_cls(clients, self.n_bins, self.max_depth, self.lam,
                                      self.gamma, **self.server_kwargs)
        self.server.released = self.trees   # completed trees, visible to every client
        for t in range(self.n_rounds):
            self.server.round_idx = t
            tree = self.server.build_tree()
            self.trees.append(tree)
            for c in clients:
                c.raw_score = c.raw_score + self.lr * np.array([predict_tree(tree, r) for r in c.X])
            if eval_callback is not None:
                eval_callback(t, self)
        return self


# ---------------- generalized attack server ----------------
class RevServer(Server):
    """malicious: set of client ids. attack(ctx) -> (g,h) forged report, where
    ctx exposes only what the chosen knowledge model allows. Logs detector
    statistics and the node argmax winner."""

    def __init__(self, clients, n_bins, max_depth, lam=1.0, gamma=0.0, min_child_hess=1.0,
                 malicious=(), attack=None, attack_depth=1, rounds=None, log=None, **_):
        super().__init__(clients, n_bins, max_depth, lam, gamma, min_child_hess)
        self.malicious, self.attack, self.attack_depth = set(malicious), attack, attack_depth
        self.rounds = rounds
        self.log = log
        self.round_idx, self._depth = 0, 0

    def build_tree(self):
        self._depth1_seen = 0
        self._node_id = None
        return super().build_tree()

    def _build_node(self, sample_idx, depth):
        self._depth = depth
        # node identity for depth <= 1 follows the depth-first call order (root, left, right)
        if depth == 0:
            self._node_id = "R"
        elif depth == 1:
            self._node_id = "L" if self._depth1_seen == 0 else "Rt"
            self._depth1_seen += 1
        else:
            self._node_id = None
        return super()._build_node(sample_idx, depth)

    def _aggregate_histograms(self, sample_idx):
        truth = {}
        for c in self.clients:
            idx = sample_idx.get(c.client_id, np.array([], dtype=int))
            if len(idx):
                truth[c.client_id] = c.compute_histograms(idx, self.n_bins)
        active = (self.attack is not None and self._depth < self.attack_depth
                  and (self.rounds is None or self.round_idx in self.rounds))
        reports = dict(truth)
        mal_here = [c for c in truth if c in self.malicious] if active else []
        for k, cid in enumerate(mal_here):
            others = [truth[o] for o in truth if o != cid]
            osum = (sum(o[0] for o in others), sum(o[1] for o in others)) if others else None
            ctx = dict(cid=cid, own=truth[cid], others_sum=osum, n_clients=len(truth),
                       n_mal=len(mal_here), k=k, lam=self.lam, mch=self.min_child_hess,
                       round=self.round_idx, depth=self._depth, node_id=self._node_id,
                       released=getattr(self, 'released', []))
            reports[cid] = self.attack(ctx)
        ag = sum(r[0] for r in reports.values())
        ah = sum(r[1] for r in reports.values())
        if mal_here and self.log is not None:
            self.log.append(self._detect(reports, mal_here, ag, ah))
        return ag, ah

    def _detect(self, reports, mal, ag, ah):
        """Per-node detector outcomes (conservation, magnitude, L1 norm)."""
        def feats(g):
            tot = g.sum(axis=1)
            return dict(cons=float(tot.max() - tot.min()), mag=float(np.abs(g).max()),
                        l1=float(np.abs(g).sum()))
        H = {c: feats(reports[c][0]) for c in reports}
        hon = [c for c in reports if c not in mal]
        med_mag = np.median([H[c]["mag"] for c in hon]) if hon else 1.0
        med_l1 = np.median([H[c]["l1"] for c in hon]) if hon else 1.0
        row = {"mal": [], "hon": []}
        for c in reports:
            f = H[c]
            tol = 1e-6 * max(1.0, f["l1"])
            d = dict(cons=f["cons"] > tol, mag=f["mag"] > 3 * med_mag, l1=f["l1"] > 3 * med_l1)
            d["any"] = bool(d["cons"] or d["mag"] or d["l1"])
            d = {k: bool(v) for k, v in d.items()}
            (row["mal"] if c in mal else row["hon"]).append(d)
        best, _ = self._find_best_split(ag, ah)
        row["winner"] = (best[1], best[2])
        row["depth"] = self._depth
        row["target"] = getattr(self.attack, "last_target", None)
        return row


def fit_eval(Xtr, ytr, Xte, yte, seed, attack=None, malicious=(0,), rounds=None,
             fed_bins=False, attack_depth=1, log=None):
    cd = make_clients(Xtr, ytr, seed)
    kw = dict(malicious=malicious, attack=attack, attack_depth=attack_depth,
              rounds=rounds, log=log)
    m = RevGBDT(n_bins=N_BINS, max_depth=DEPTH, n_rounds=ROUNDS, lr=LR,
                server_cls=RevServer, server_kwargs=kw, fed_bins=fed_bins).fit(cd)
    return roc_auc_score(yte, m.predict_proba(Xte)), m


def paired(clean, attacked):
    d = np.array(clean) - np.array(attacked)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
