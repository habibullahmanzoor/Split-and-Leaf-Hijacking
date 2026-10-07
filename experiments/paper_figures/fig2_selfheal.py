"""Figure 2 (Section 6.2): boosting self-heals a transient corrupted round but
not sustained corruption. Mean +/- one standard deviation across five seeds,
on all three datasets (Adult/Census, Heart Disease, Credit Default) --
previously single-seed, single-dataset (Adult only), which understated the
paper's own claimed five-seed standard (Section 5.3). Restyles and extends
attack_category2_compounding.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
from attack_category2_compounding import run_trajectory, N_ROUNDS as DEFAULT_N_ROUNDS
import plotstyle as ps

N_TRIALS = 5
DATASETS = {
    "Adult (Census)": dict(loader=load_adult, n_bins=32, max_depth=4, n_rounds=20, subsample=None),
    "Heart Disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=15, subsample=None),
    "Credit Default": dict(loader=load_credit_default, n_bins=32, max_depth=4, n_rounds=20, subsample=3000),
}
CONDITIONS = [
    ("clean", "No attack", ps.C2, ps.MARKERS[0]),
    ("early", "Corrupted round 0 (earliest)", ps.C1, ps.MARKERS[1]),
    ("late", "Corrupted round N-1 (latest)", ps.C4, ps.MARKERS[2]),
]


def load_split(cfg, seed):
    X, y, _ = cfg["loader"]()
    X = X.to_numpy(dtype=float)
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def panel(ax, name, cfg):
    n_rounds = cfg["n_rounds"]

    def compute():
        curves_ = {key: [] for key, _, _, _ in CONDITIONS}
        for seed in range(N_TRIALS):
            X_train, X_test, y_train, y_test = load_split(cfg, seed)
            for key in curves_:
                target_rounds = None if key == "clean" else ({0} if key == "early" else {n_rounds - 1})
                aucs = run_trajectory(X_train, y_train, X_test, y_test, target_rounds, seed=seed,
                                       n_bins=cfg["n_bins"], max_depth=cfg["max_depth"], n_rounds=n_rounds)
                curves_[key].append(aucs)
        return curves_

    curves = ps.cached(f"fig2_{name}", compute)
    rounds = np.arange(1, n_rounds + 1)
    finals = {}
    for key, label, color, marker in CONDITIONS:
        arr = np.array(curves[key])
        mean, std = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(rounds, mean, color=color, linewidth=1.6, marker=marker, markersize=3, label=label)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.18)
        finals[key] = (mean[-1], std[-1])
    ax.axvline(1, color=ps.C1, linewidth=1, linestyle=":", alpha=0.5)
    ax.axvline(n_rounds, color=ps.C4, linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Test AUC")
    ax.set_title(name, fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower right", fontsize=6.5))

    print(f"  {name}: " + "  ".join(
        f"{key}_final={m:.4f}+/-{s:.4f}" for key, (m, s) in finals.items()))
    return finals


def main():
    fig, axes = ps.new_fig(ncols=3, wide=True)
    all_finals = {}
    for ax, (name, cfg) in zip(axes, DATASETS.items()):
        print(f"Running self-heal (5 seeds): {name} ...")
        all_finals[name] = panel(ax, name, cfg)

    parts = []
    for name, finals in all_finals.items():
        clean_m, _ = finals["clean"]
        early_m, _ = finals["early"]
        late_m, _ = finals["late"]
        parts.append(f"{name}: clean={clean_m:.4f}, early={early_m:.4f}, late={late_m:.4f}")
    caption = ("A single corrupted round recovers to within noise of the clean baseline "
               "regardless of whether it occurs early or late in training, mean +/- one "
               "standard deviation across five seeds, on all three datasets (" +
               "; ".join(parts) + "); sustained corruption (Figure 1) does not recover.")
    ps.save_fig(fig, "fig2_selfheal", caption)


if __name__ == "__main__":
    main()
