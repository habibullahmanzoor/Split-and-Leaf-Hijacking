"""Figure 0 (Section 7, evaluation hygiene): train vs. test AUC over
boosting rounds for the HONEST (undefended, no attackers) baseline model,
for every protocol-dataset combination used elsewhere in the paper, mean
plus or minus one standard deviation shaded band across five seeds. Not a
security result -- a sanity check that the underlying models are sensibly
fit (no pathological overfitting) before any attack or defense is layered
on top. 2x3 grid: row 1 is HFL, row 2 is VFL, columns are the three
datasets. VFL is run at an extended round count (30) well past this
paper's native per-dataset VFL round count, to see where train/test AUC
actually plateaus.

This was single seed until a single-seed anomaly was caught and corrected:
seed 0's Heart Disease (VFL) trajectory happened to peak early (round 3)
and decline toward round 30, which read as genuine overfitting, but a
five seed check showed the peak round is essentially random across seeds
(3, 9, 15, 25, 3) and round 5 (0.8526 +/- 0.0126) is statistically
indistinguishable from round 30 (0.8504 +/- 0.0199) once averaged -- the
single-seed "decline" was noise from an approximately 60-row test set, not
a real effect. Every panel is now multi-seed for exactly this reason.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult, load_heart_disease, load_credit_default
from harness.federated_gbdt import FederatedGBDT
from harness.vfl_secureboost import FederatedVFLGBDT
from attack_category1_sweep import dirichlet_partition
import plotstyle as ps

HFL_DATASETS = {
    "Adult (HFL)": dict(loader=load_adult, n_bins=32, max_depth=4, n_rounds=20, subsample=None),
    "Heart Disease (HFL)": dict(loader=load_heart_disease, n_bins=16, max_depth=3, n_rounds=15, subsample=None),
    "Credit Default (HFL)": dict(loader=load_credit_default, n_bins=32, max_depth=4, n_rounds=20, subsample=3000),
}
VFL_DATASETS = {
    "Adult (VFL)": dict(loader=load_adult, n_bins=16, max_depth=3, subsample=1500),
    "Heart Disease (VFL)": dict(loader=load_heart_disease, n_bins=16, max_depth=3, subsample=None),
    "Credit Default (VFL)": dict(loader=load_credit_default, n_bins=16, max_depth=3, subsample=1500),
}
N_CLIENTS, DIRICHLET_ALPHA = 5, 0.5
VFL_EXTENDED_ROUNDS = 30
N_TRIALS = 5


def hfl_curve(cfg, seed):
    X, y, _ = cfg["loader"]()
    X = X.to_numpy(dtype=float)
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    parts = dirichlet_partition(X_train, y_train, N_CLIENTS, DIRICHLET_ALPHA, seed=seed)
    client_data = {cid: (X_train[idx], y_train[idx]) for cid, idx in enumerate(parts)}

    train_aucs, test_aucs = [], []

    def eval_cb(t, model):
        train_aucs.append(roc_auc_score(y_train, model.predict_proba(X_train)))
        test_aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)))

    model = FederatedGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"], n_rounds=cfg["n_rounds"], lr=0.3)
    model.fit(client_data, eval_callback=eval_cb)
    return train_aucs, test_aucs


def vfl_curve(cfg, seed, n_rounds=VFL_EXTENDED_ROUNDS):
    X, y, groups = cfg["loader"]()
    party0_cols, party1_cols = groups["party0"], groups["party1"]
    X_sub, y_sub = X, y
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]
    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y_sub, test_size=0.25, random_state=seed, stratify=y_sub
    )

    train_aucs, test_aucs = [], []

    def eval_cb(t, model):
        train_aucs.append(roc_auc_score(y_tr, model.predict_proba(Xa_tr, {"party1": Xp_tr})))
        test_aucs.append(roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te})))

    model = FederatedVFLGBDT(n_bins=cfg["n_bins"], max_depth=cfg["max_depth"], n_rounds=n_rounds,
                              lr=0.3, key_size=512)
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, eval_callback=eval_cb)
    return train_aucs, test_aucs


def plot_panel(ax, curve_fn, cfg, title):
    def compute():
        all_train, all_test = [], []
        for seed in range(N_TRIALS):
            train_aucs, test_aucs = curve_fn(cfg, seed)
            all_train.append(train_aucs)
            all_test.append(test_aucs)
        return {"train": all_train, "test": all_test}

    data = ps.cached(f"fig0_{title}", compute)
    train_arr = np.array(data["train"])
    test_arr = np.array(data["test"])
    rounds = np.arange(1, train_arr.shape[1] + 1)
    train_mean, train_std = train_arr.mean(axis=0), train_arr.std(axis=0, ddof=1)
    test_mean, test_std = test_arr.mean(axis=0), test_arr.std(axis=0, ddof=1)

    ax.plot(rounds, train_mean, color=ps.C2, linewidth=1.8, marker=ps.MARKERS[0], markersize=3.5, label="Train")
    ax.fill_between(rounds, train_mean - train_std, train_mean + train_std, color=ps.C2, alpha=0.2)
    ax.plot(rounds, test_mean, color=ps.C1, linewidth=1.8, marker=ps.MARKERS[1], markersize=3.5, label="Test")
    ax.fill_between(rounds, test_mean - test_std, test_mean + test_std, color=ps.C1, alpha=0.2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("AUC")
    ax.set_title(title, fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower right", fontsize=7))

    gap = train_mean[-1] - test_mean[-1]
    peak_round = int(np.argmax(test_mean)) + 1
    decline = test_mean[peak_round - 1] - test_mean[-1]
    print(f"  {title}: final train={train_mean[-1]:.4f}+/-{train_std[-1]:.4f}  "
          f"test={test_mean[-1]:.4f}+/-{test_std[-1]:.4f}  gap={gap:.4f}  "
          f"peak_test={test_mean[peak_round-1]:.4f} (round {peak_round})  decline_to_final={decline:+.4f}")
    return gap, peak_round, decline


def main():
    fig, axes = ps.new_fig(ncols=3, nrows=2, wide=True, height=5.4)
    stats = {}

    for ax, (name, cfg) in zip(axes[0], HFL_DATASETS.items()):
        print(f"Running HFL baseline (5 seeds): {name} ...")
        stats[name] = plot_panel(ax, hfl_curve, cfg, name)

    for ax, (name, cfg) in zip(axes[1], VFL_DATASETS.items()):
        print(f"Running VFL baseline (5 seeds): {name} ...")
        stats[name] = plot_panel(ax, vfl_curve, cfg, name)

    max_gap_name = max(stats, key=lambda k: stats[k][0])
    caption = ("Train vs. test AUC over boosting rounds for the honest baseline model, mean "
               "+/- one standard deviation across five seeds, for every protocol/dataset "
               "combination used elsewhere in the paper (top: HFL, bottom: VFL) -- a sanity "
               f"check, not a security result. Largest train-test gap: {stats[max_gap_name][0]:.4f} "
               f"({max_gap_name}), consistent with sensible fitting rather than pathological "
               "overfitting on every dataset once averaged across seeds; VFL panels are shown at "
               f"an extended, {VFL_EXTENDED_ROUNDS} round configuration, well past each dataset's "
               "native round count used elsewhere in the paper.")
    ps.save_fig(fig, "fig0_baseline_convergence", caption)


if __name__ == "__main__":
    main()
