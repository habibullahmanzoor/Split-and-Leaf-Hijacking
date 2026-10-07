"""Figure 7 (Section 7): cross-dataset generalization boundary condition.
Both headline attacks (category 1 HFL, category 3 VFL) rerun on Heart
Disease and Credit Default, 3 seeds each (up from the original single-trial
runs). Consolidates attack_category1_cross_dataset.py and
attack_category3_cross_dataset.py; Adult itself is already in Figure 1.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_heart_disease, load_credit_default
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.he_crypto_layer import MaliciousPassiveParty
from attack_category1_cross_dataset import run as run_cat1, DATASETS as CAT1_DATASETS, FRACTIONS as CAT1_FRACS
import plotstyle as ps

N_TRIALS = 3
CAT3_SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 50.0]
CAT3_DATASETS = {
    "heart_disease": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=5, subsample=None),
    "credit_default": dict(loader=load_credit_default, n_bins=16, max_depth=3, n_rounds=3, subsample=1500),
}
DATASET_STYLE = {
    "heart_disease": dict(color=ps.C4, marker=ps.MARKERS[2], label="Heart Disease"),
    "credit_default": dict(color=ps.C5, marker=ps.MARKERS[3], label="Credit Default"),
}


def panel_a_cat1(ax):
    for name, cfg in CAT1_DATASETS.items():
        X, y, _ = cfg["loader"]()
        X = X.to_numpy(dtype=float)
        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        from attacks.histogram_integrity import margin_flip_attack

        def compute():
            means_ = []
            for frac in CAT1_FRACS:
                af = margin_flip_attack if frac > 0 else None
                aucs = [run_cat1(X_train, y_train, X_test, y_test, cfg["n_bins"], cfg["max_depth"],
                                  cfg["n_rounds"], frac, af, seed=t) for t in range(N_TRIALS)]
                means_.append(float(np.mean(aucs)))
            return {"means": means_}

        means = ps.cached(f"fig7_cat1_{name}", compute)["means"]
        print(f"[cat1 cross-dataset] {name}: {list(zip(CAT1_FRACS, [f'{m:.4f}' for m in means]))}")
        style = DATASET_STYLE[name]
        ax.plot(CAT1_FRACS, means, color=style["color"], marker=style["marker"], markersize=4,
                linewidth=1.8, label=f"{style['label']} (honest={means[0]:.3f})")

    ax.set_xlabel("Fraction malicious clients")
    ax.set_ylabel("Test AUC")
    ax.set_title("(a) Category 1 (HFL) cross-dataset", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="best", fontsize=7))


def panel_b_cat3(ax):
    for name, cfg in CAT3_DATASETS.items():
        X, y, groups = cfg["loader"]()
        party0_cols, party1_cols = groups["party0"], groups["party1"]
        if cfg["subsample"] and len(y) > cfg["subsample"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]

        Xa = X[party0_cols].to_numpy(dtype=float)
        Xp = X[party1_cols].to_numpy(dtype=float)

        def compute():
            means_ = []
            for sf in CAT3_SCALE_FACTORS:
                aucs = []
                for t in range(N_TRIALS):
                    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
                        Xa, Xp, y, test_size=0.25, random_state=t, stratify=y
                    )
                    model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"],
                                              n_rounds=cfg["n_rounds"], lr=0.3, key_size=512)
                    factories = None
                    if sf != 1.0:
                        factories = {"party1": partial(MaliciousPassiveParty, target_feature=0, scale_factor=sf)}
                    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories=factories)
                    aucs.append(roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te})))
                means_.append(float(np.mean(aucs)))
            return {"means": means_}

        means = ps.cached(f"fig7_cat3_{name}", compute)["means"]
        print(f"[cat3 cross-dataset] {name}: {list(zip(CAT3_SCALE_FACTORS, [f'{m:.4f}' for m in means]))}")
        style = DATASET_STYLE[name]
        ax.plot(CAT3_SCALE_FACTORS, means, color=style["color"], marker=style["marker"], markersize=4,
                linewidth=1.8, label=f"{style['label']} (honest={means[0]:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Ciphertext scale factor")
    ax.set_ylabel("Test AUC")
    ax.set_title("(b) Category 3 (VFL) cross-dataset", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="best", fontsize=7))


def main():
    fig, axes = ps.new_fig(ncols=2, wide=True)
    panel_a_cat1(axes[0])
    panel_b_cat3(axes[1])

    caption = ("Both headline attacks generalize to Heart Disease (confirmed, and amplified for "
               "category 3) but not to Credit Default, where a blind, public-information-only "
               "target-selection heuristic happens not to land on a damaging split -- a boundary "
               "condition on the heuristic, not the underlying split-flip mechanism (Section 7.4).")
    ps.save_fig(fig, "fig7_cross_dataset", caption)


if __name__ == "__main__":
    main()
