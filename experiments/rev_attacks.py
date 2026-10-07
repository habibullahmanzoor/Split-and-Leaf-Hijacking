"""Revision attack library. Every attack is a callable object taking a context
dict (see rev_common.RevServer) and returning a forged (grad_hist, hess_hist).
`last_target` records the intended (feature, bin) so the server log can report
target control."""
import numpy as np
from rev_common import compute_split_flip_margin


def best_split(g, h, lam=1.0, mch=1.0):
    """Same per-feature-total argmax as Server._find_best_split."""
    nf, nb = g.shape
    best, arg = -np.inf, None
    for f in range(nf):
        G, H = g[f].sum(), h[f].sum()
        gc, hc = np.cumsum(g[f]), np.cumsum(h[f])
        for b in range(nb - 1):
            GL, HL = gc[b], hc[b]
            GR, HR = G - GL, H - HL
            if HL < mch or HR < mch:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))
            if gain > best:
                best, arg = gain, (f, b)
    return arg


def worst_candidate(g, h, lam=1.0, mch=1.0):
    nf, nb = g.shape
    worst, arg = np.inf, None
    for f in range(nf):
        G, H = g[f].sum(), h[f].sum()
        gc, hc = np.cumsum(g[f]), np.cumsum(h[f])
        for b in range(nb - 1):
            GL, HL = gc[b], hc[b]
            GR, HR = G - GL, H - HL
            if HL < mch or HR < mch:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))
            if gain < worst:
                worst, arg = gain, (f, b)
    return arg


class Attack:
    last_target = None
    name = "attack"

    def __call__(self, ctx):
        raise NotImplementedError


def _margin(agg_g, agg_h, lam, mch, target=None):
    t = target or worst_candidate(agg_g, agg_h, lam, mch)
    if t is None:
        return None, np.nan
    m, _ = compute_split_flip_margin(agg_g, agg_h, t[0], t[1], lam=lam, min_child_hess=mch)
    return t, m


class FullKnowledgeOneCell(Attack):
    """Clean one-cell injection of exactly 1.01 x margin; attacker sees the
    honest aggregate (worst-case knowledge). `bound_rho`: optional per-report
    cell cap B = rho * max|own honest cell| (a verifiable range limit); colluders
    add greedily until the margin is met."""
    name = "full_one_cell"

    def __init__(self, mult=1.01, bound_rho=None):
        self.mult, self.bound_rho = mult, bound_rho
        self.margins = []

    def __call__(self, ctx):
        og, oh = ctx["others_sum"]
        g, h = ctx["own"]
        ag, ah = og + g, oh + h
        t, m = _margin(ag, ah, ctx["lam"], ctx["mch"])
        self.last_target = t
        if t is None or not np.isfinite(m) or m <= 0:
            return g, h
        self.margins.append(float(m))
        need = self.mult * m
        if self.bound_rho is not None:
            B = self.bound_rho * np.abs(g).max()
            # colluders k=0.. add B each until the residual need is covered;
            # (the margin is computed on the honest aggregate, so the residual
            # for colluder k is need - k*B)
            add = min(B, max(need - ctx["k"] * B, 0.0))
        else:
            add = need if ctx["k"] == 0 else 0.0
        f = g.copy()
        f[t[0], t[1]] += add
        return f, h.copy()


class FullKnowledgeZero(Attack):
    """Original (implemented) margin-flip attack: zero everything but the
    target cell, which receives 3 x the honest L1 mass."""
    name = "full_zero"

    def __call__(self, ctx):
        og, oh = ctx["others_sum"]
        g, h = ctx["own"]
        t = worst_candidate(og + g, oh + h, ctx["lam"], ctx["mch"])
        self.last_target = t
        if t is None:
            return g, h
        z = np.zeros_like(g)
        z[t] = 3.0 * (np.abs(og).sum() + np.abs(g).sum())
        return z, h.copy()


class LimitedKnowledge(Attack):
    """Attacker sees ONLY its own local histogram. It extrapolates the
    aggregate as n_clients x own report (the best a client can do without any
    other report), targets the worst candidate under that estimate, and
    injects `mult` x the estimated margin."""
    name = "limited"

    def __init__(self, mult=1.5, mode="extrapolate"):
        self.mult, self.mode = mult, mode

    def __call__(self, ctx):
        g, h = ctx["own"]
        K = ctx["n_clients"]
        eg, eh = (g * K, h * K) if self.mode == "extrapolate" else (g, h)
        t, m = _margin(eg, eh, ctx["lam"], ctx["mch"])
        self.last_target = t
        if t is None or not np.isfinite(m) or m <= 0:
            return g, h
        f = g.copy()
        f[t] += self.mult * m
        return f, h.copy()


class ConservationPair(Attack):
    """Targeted, conservation-preserving attack: add +d to the target cell and
    -d to a cell of the SAME feature on the other side of the split, so every
    feature's row total (and hence the cross-feature conservation check) is
    unchanged. d is the smallest value on a geometric grid for which the
    target wins the full argmax; if none does, the client reports honestly
    (and the attack is recorded as failed)."""
    name = "cons_pair"

    def __init__(self, grid=(1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)):
        self.grid = grid

    def __call__(self, ctx):
        og, oh = ctx["others_sum"]
        g, h = ctx["own"]
        ag, ah = og + g, oh + h
        t, m = _margin(ag, ah, ctx["lam"], ctx["mch"])
        self.last_target = t
        if t is None or not np.isfinite(m) or m <= 0:
            return g, h
        f_, b_ = t
        nb = g.shape[1]
        best = None
        for dec in sorted({nb - 1, min(b_ + 1, nb - 1), min(b_ + 2, nb - 1)}):
            if dec <= b_:
                continue
            for s in self.grid:
                d = s * m
                tg = ag.copy()
                tg[f_, b_] += d
                tg[f_, dec] -= d
                if best is not None and d >= best[0]:
                    break
                if best_split(tg, ah, ctx["lam"], ctx["mch"]) == t:
                    best = (d, dec)
                    break
        if best is None:
            self.last_target = None
            return g, h
        d, dec = best
        fg = g.copy()
        fg[f_, b_] += d
        fg[f_, dec] -= d
        return fg, h.copy()


class MatchedNoise(Attack):
    """Random-direction histogram corruption with the SAME L1 deviation as the
    clean one-cell attack at that node (budget-matched baseline)."""
    name = "matched_noise"

    def __init__(self, rng_seed=0):
        self.rng = np.random.default_rng(rng_seed)

    def __call__(self, ctx):
        og, oh = ctx["others_sum"]
        g, h = ctx["own"]
        t, m = _margin(og + g, oh + h, ctx["lam"], ctx["mch"])
        self.last_target = None
        if t is None or not np.isfinite(m) or m <= 0:
            return g, h
        v = self.rng.normal(size=g.shape)
        v *= 1.01 * m / np.abs(v).sum()
        return g + v, h.copy()


class MatchedRandomCell(Attack):
    """Same deviation magnitude as the one-cell attack, but on a RANDOM cell
    (no split awareness)."""
    name = "matched_random_cell"

    def __init__(self, rng_seed=0):
        self.rng = np.random.default_rng(rng_seed)

    def __call__(self, ctx):
        og, oh = ctx["others_sum"]
        g, h = ctx["own"]
        t, m = _margin(og + g, oh + h, ctx["lam"], ctx["mch"])
        self.last_target = None
        if t is None or not np.isfinite(m) or m <= 0:
            return g, h
        f = g.copy()
        f[self.rng.integers(g.shape[0]), self.rng.integers(g.shape[1] - 1)] += 1.01 * m
        return f, h.copy()


class UnboundedNoise(Attack):
    name = "gauss5"

    def __init__(self, rng_seed=0, scale=5.0):
        self.rng, self.scale = np.random.default_rng(rng_seed), scale

    def __call__(self, ctx):
        g, h = ctx["own"]
        self.last_target = None
        return g + self.rng.normal(0, self.scale, g.shape), h


def pairwise_margin(ag, ah, target, rival, lam=1.0, mch=1.0):
    """Smallest single-cell injection at `target` that makes it beat ONE named rival
    (same algebra as theory.margin, restricted to that rival)."""
    from theory.margin import _delta_min_for_rival
    tf, tb = target
    GT0, HT0 = ag[tf].sum(), ah[tf].sum()
    gc, hc = np.cumsum(ag[tf]), np.cumsum(ah[tf])
    GLt, HLt = gc[tb], hc[tb]
    GRt, HRt = GT0 - GLt, HT0 - HLt
    if HLt < mch or HRt < mch:
        return np.nan
    A_t = 1.0 / (HLt + lam) - 1.0 / (HT0 + lam)
    B_t = 2 * GLt / (HLt + lam) - 2 * GT0 / (HT0 + lam)
    C_t = GLt**2 / (HLt + lam) + GRt**2 / (HRt + lam) - GT0**2 / (HT0 + lam)
    f, b = rival
    G_f, H_f = ag[f].sum(), ah[f].sum()
    g2, h2 = np.cumsum(ag[f]), np.cumsum(ah[f])
    GLr, HLr = g2[b], h2[b]
    GRr, HRr = G_f - GLr, H_f - HLr
    if HLr < mch or HRr < mch:
        return np.nan
    C_r = GLr**2 / (HLr + lam) + GRr**2 / (HRr + lam) - G_f**2 / (H_f + lam)
    if f != tf:
        A_r = B_r = 0.0
    elif b >= tb:
        A_r = 1.0 / (HLr + lam) - 1.0 / (HT0 + lam)
        B_r = 2 * GLr / (HLr + lam) - 2 * GT0 / (HT0 + lam)
    else:
        A_r = 1.0 / (HRr + lam) - 1.0 / (HT0 + lam)
        B_r = 2 * GRr / (HRr + lam) - 2 * GT0 / (HT0 + lam)
    return _delta_min_for_rival(A_t, B_t, C_t, A_r, B_r, C_r)


class ReleasedTreeAttack(Attack):
    """Limited-knowledge attacker that also uses the trees released after each
    round (public to every client). It sees only its own histogram, estimates the
    aggregate as K x its own report, takes the winner of the same node in the
    previous round's released tree as the split it must beat, targets the weakest
    candidate under its estimate, and injects `mult` x the margin against that
    one observed winner. In round 0 nothing has been released, so it falls back
    to the own-report estimate."""
    name = "released_tree"

    def __init__(self, mult=1.5):
        self.mult = mult
        self.used_release = []

    def _prev_winner(self, ctx):
        rel = ctx.get("released") or []
        nid = ctx.get("node_id")
        if not rel or nid is None:
            return None
        node = rel[-1]
        node = {"R": node, "L": node.left, "Rt": node.right}.get(nid)
        if node is None or node.is_leaf or node.split_feature is None:
            return None
        return (int(node.split_feature), int(node.split_bin))

    def __call__(self, ctx):
        g, h = ctx["own"]
        K = ctx["n_clients"]
        eg, eh = g * K, h * K
        t = worst_candidate(eg, eh, ctx["lam"], ctx["mch"])
        self.last_target = t
        if t is None:
            return g, h
        w = self._prev_winner(ctx)
        self.used_release.append(w is not None)
        if w is not None and w != t:
            m = pairwise_margin(eg, eh, t, w, ctx["lam"], ctx["mch"])
        else:
            _, m = _margin(eg, eh, ctx["lam"], ctx["mch"], target=t)
        if m is None or not np.isfinite(m) or m <= 0:
            return g, h
        f = g.copy()
        f[t] += self.mult * m
        return f, h.copy()
