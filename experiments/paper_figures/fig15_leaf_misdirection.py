"""Figure 15 (Section 6.4): leaf misdirection degrades continuously rather
than saturating. Mean +/- one standard deviation across five seeds, on all
three datasets -- previously single-seed, single-dataset (Adult only), and
with no main-text figure at all (Section 6.4's prose cited none). Extends
attack_category5_leaf_routing.py.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.leaf_routing import MisroutingPassiveParty
import plotstyle as ps

N_TRIALS = 5
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
DATASETS = {
    "Adult (Census)": dict(loader=load_adult, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
    "Heart Disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=5, subsample=None),
    "Credit Default": dict(loader=load_credit_default, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
}


def run_once(Xa, Xp, y, cfg, frac, seed):
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"],
                              n_rounds=cfg["n_rounds"], lr=0.3, key_size=512)
    factories = None
    if frac > 0:
        factories = {"party1": partial(MisroutingPassiveParty, misroute_fraction=frac,
                                        _rng=np.random.default_rng(seed))}
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
    return roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))


def panel(ax, name, cfg):
    X, y, groups = cfg["loader"]()
    party0_cols, party1_cols = groups["party0"], groups["party1"]
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    Xa = X[party0_cols].to_numpy(dtype=float)
    Xp = X[party1_cols].to_numpy(dtype=float)

    def compute():
        means_, stds_ = [], []
        for frac in FRACTIONS:
            aucs = [run_once(Xa, Xp, y, cfg, frac, seed=t) for t in range(N_TRIALS)]
            means_.append(float(np.mean(aucs)))
            stds_.append(float(np.std(aucs, ddof=1)))
        return {"means": means_, "stds": stds_}

    cached = ps.cached(f"fig15_{name}", compute)
    means, stds = np.array(cached["means"]), np.array(cached["stds"])

    ax.plot(FRACTIONS, means, color=ps.C4, linewidth=1.8, marker=ps.MARKERS[2], markersize=4)
    ax.fill_between(FRACTIONS, means - stds, means + stds, color=ps.C4, alpha=0.2)
    ax.axhline(means[0], color=ps.MUTED, linewidth=1, linestyle="--", label="No attack")
    ax.axhline(0.5, color=ps.C1, linewidth=1, linestyle=":", label="Random guessing")
    ax.set_ylabel("Test AUC")
    ax.set_title(name, fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="best", fontsize=6.5))

    print(f"  {name}: " + "  ".join(f"frac={f:.2f} auc={m:.4f}+/-{s:.4f}"
                                     for f, m, s in zip(FRACTIONS, means, stds)))
    return means, stds


def main():
    fig, axes = ps.new_fig(ncols=3, wide=True)
    results = {}
    for ax, (name, cfg) in zip(axes, DATASETS.items()):
        print(f"Running leaf misdirection (5 seeds): {name} ...")
        results[name] = panel(ax, name, cfg)

    fig.supxlabel("Fraction of samples misrouted")

    parts = []
    for name, (means, stds) in results.items():
        parts.append(f"{name}: honest={means[0]:.4f}, full={means[-1]:.4f}+/-{stds[-1]:.4f}")
    caption = ("Leaf misdirection degrades continuously with the misrouted fraction rather than "
               "saturating immediately, the opposite shape from split-flip injection (Figure 1), "
               "mean +/- one standard deviation across five seeds, on all three datasets (" +
               "; ".join(parts) + ").")
    ps.save_fig(fig, "fig15_leaf_misdirection", caption)


if __name__ == "__main__":
    main()
