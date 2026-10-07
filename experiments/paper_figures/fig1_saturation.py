"""Figure 1 (Section 4): damage saturates under discrete argmax, not under
continuous/smooth aggregation. Four panels: HFL histogram-forgery attack
(category 1), VFL ciphertext-rescaling attack (category 3), a bagging-RF
control with no shared histogram at all, and a fourth panel that isolates the
aggregation mechanism from the attack type -- the same label-shuffle
corruption used in panel (c), but run through the boosting harness instead of
bagging. Panels (a) vs (c) differ on two axes at once (mechanism AND attack
type: forgery vs shuffle), so that contrast alone cannot isolate which one
causes the saturating-vs-smooth shape. Panel (d) holds the attack type fixed
at label-shuffle and varies only the mechanism, closing that gap. Consolidates
attack_category1_sweep.py, attack_category3_he_sweep.py,
rf_bagging_control.py, and attack_category1_labelshuffle.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from attack_category1_sweep import dirichlet_partition as dp_hfl, run_once as run_cat1
from attack_category1_sweep import N_CLIENTS as N1, DIRICHLET_ALPHA as A1, N_TRIALS as T1
from attacks.histogram_integrity import naive_noise_attack, margin_flip_attack, clean_one_cell_attack
from attack_category3_he_sweep import run_once as run_cat3
from rf_bagging_control import run_once as run_rf
from attack_category1_labelshuffle import run_once as run_labelshuffle_boost
import plotstyle as ps

FRACTIONS = [0.0, 0.2, 0.4, 0.6]  # exact malicious fractions for N_CLIENTS=5: 0/1/2/3 clients,
# no duplicate points. The previous [0.0,0.1,0.2,0.3,0.4,0.5] rounded via ceil(frac*N_CLIENTS) to
# client counts [0,1,1,2,2,3] -- 0.1 and 0.2 both realized 1 client, 0.3 and 0.4 both realized 2,
# and the point plotted at 0.5 actually realized 3/5=0.6, not 0.5. Downstream code indexes this
# list positionally ([0]=zero clients, [1]=one, [-1]=three); that indexing is unchanged by this fix.
SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0]


def panel_a_hfl(ax):
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    attacks = {
        "naive": (naive_noise_attack, dict(noise_scale=5.0)),
        "margin_aware": (margin_flip_attack, dict(lam=1.0, gamma=0.0, budget_multiplier=3.0)),
        "clean_one_cell": (clean_one_cell_attack, dict(lam=1.0, min_child_hess=1.0, margin_multiplier=1.01)),
    }

    def compute():
        results_, stds_out = {}, {}
        for name, (fn, kwargs) in attacks.items():
            means_, stds_ = [], []
            for frac in FRACTIONS:
                aucs = [run_cat1(X_train, y_train, X_test, y_test, frac, fn if frac > 0 else None, kwargs, seed=t)
                        for t in range(T1)]
                means_.append(float(np.mean(aucs)))
                stds_.append(float(np.std(aucs, ddof=1)))
            results_[name], stds_out[name] = means_, stds_
        return {"results": results_, "stds": stds_out}

    cached = ps.cached("fig1_panel_a_hfl", compute)
    results, stds = cached["results"], cached["stds"]
    for name in attacks:
        print(f"[HFL cat1 {name}] {list(zip(FRACTIONS, [f'{m:.4f}' for m in results[name]]))}")

    ax.errorbar(FRACTIONS, results["naive"], yerr=stds["naive"], color=ps.C3, linewidth=1.6,
                marker=ps.MARKERS[2], markersize=4, capsize=2, label="Naive noise")
    ax.errorbar(FRACTIONS, results["margin_aware"], yerr=stds["margin_aware"], color=ps.C1,
                linewidth=1.8, marker=ps.MARKERS[0], markersize=4, capsize=2, label="Margin-aware (ours)")
    ax.errorbar(FRACTIONS, results["clean_one_cell"], yerr=stds["clean_one_cell"], color=ps.C4,
                linewidth=1.6, marker=ps.MARKERS[3], markersize=4, capsize=2,
                label="One-cell injection (exact margin)")
    ax.axhline(results["naive"][0], color=ps.MUTED, linewidth=1, linestyle="--", label="No attack")
    ax.set_xlabel("Fraction malicious clients")
    ax.set_ylabel("Test AUC")
    ax.set_title("(a) HFL histogram attack", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower left", fontsize=6.5))
    return results["margin_aware"][0], results["margin_aware"][1]


def panel_b_vfl(ax):
    X, y, groups = load_adult()

    def compute():
        means_, stds_ = [], []
        for sf in SCALE_FACTORS:
            aucs = [run_cat3(X, y, groups, sf, seed=t) for t in range(5)]
            means_.append(float(np.mean(aucs)))
            stds_.append(float(np.std(aucs, ddof=1)))
        return {"means": means_, "stds": stds_}

    cached = ps.cached("fig1_panel_b_vfl", compute)
    means, stds = cached["means"], cached["stds"]
    print(f"[VFL cat3] {list(zip(SCALE_FACTORS, [f'{m:.4f}' for m in means]))}")

    ax.errorbar(SCALE_FACTORS, means, yerr=stds, color=ps.C1, linewidth=1.8,
                marker=ps.MARKERS[0], markersize=4, capsize=2, label="Passive party (blind)")
    ax.axhline(means[0], color=ps.MUTED, linewidth=1, linestyle="--", label="No attack (scale=1)")
    ax.set_xscale("log")
    ax.set_xlabel("Ciphertext scale factor")
    ax.set_ylabel("Test AUC")
    ax.set_title("(b) VFL ciphertext attack", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="best", fontsize=6.5))
    return means[0], means[1]


def panel_c_rf(ax):
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    def compute():
        means_ = []
        for frac in FRACTIONS:
            aucs = [run_rf(X_train, y_train, X_test, y_test, frac, seed=t) for t in range(3)]
            means_.append(float(np.mean(aucs)))
        return {"means": means_}

    means = ps.cached("fig1_panel_c_rf", compute)["means"]
    print(f"[RF control] {list(zip(FRACTIONS, [f'{m:.4f}' for m in means]))}")

    ax.plot(FRACTIONS, means, color=ps.C5, linewidth=1.8, marker=ps.MARKERS[4], markersize=4,
            label="Bagging RF (label-shuffle)")
    ax.axhline(means[0], color=ps.MUTED, linewidth=1, linestyle="--", label="No attack")
    ax.set_xlabel("Fraction malicious clients")
    ax.set_ylabel("Test AUC")
    ax.set_title("(c) Bagging + label-shuffle", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower left", fontsize=6.5))
    return means


def panel_d_labelshuffle_boosting(ax):
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    def compute():
        means_, stds_ = [], []
        for frac in FRACTIONS:
            aucs = [run_labelshuffle_boost(X_train, y_train, X_test, y_test, frac, seed=t) for t in range(5)]
            means_.append(float(np.mean(aucs)))
            stds_.append(float(np.std(aucs, ddof=1)))
        return {"means": means_, "stds": stds_}

    cached = ps.cached("fig1_panel_d_boost", compute)
    means, stds = cached["means"], cached["stds"]
    print(f"[label-shuffle+boosting] {list(zip(FRACTIONS, [f'{m:.4f}' for m in means]))}")

    ax.errorbar(FRACTIONS, means, yerr=stds, color=ps.C4, linewidth=1.8, marker=ps.MARKERS[3],
                markersize=4, capsize=2, label="Boosting (label-shuffle)")
    ax.axhline(means[0], color=ps.MUTED, linewidth=1, linestyle="--", label="No attack")
    ax.set_xlabel("Fraction malicious clients")
    ax.set_ylabel("Test AUC")
    ax.set_title("(d) Boosting + label-shuffle", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower left", fontsize=6.5))
    return means


def main():
    # A 1x4 row at the paper's fixed page width leaves too little room per
    # panel for axis labels at readable font sizes without colliding into
    # the next panel; a 2x2 grid gives each panel roughly double the width.
    fig, axes = ps.new_fig(ncols=2, nrows=2, wide=True, height=5.6)
    axes = axes.flatten()
    hfl0, hfl1 = panel_a_hfl(axes[0])
    vfl0, vfl1 = panel_b_vfl(axes[1])
    bag_means = panel_c_rf(axes[2])
    boost_means = panel_d_labelshuffle_boosting(axes[3])

    bag_drop_1 = bag_means[0] - bag_means[1]
    bag_drop_3 = bag_means[0] - bag_means[-1]
    boost_drop_1 = boost_means[0] - boost_means[1]
    boost_drop_3 = boost_means[0] - boost_means[-1]

    caption = (f"Damage saturates at a single attacker under discrete-argmax aggregation "
               f"in both HFL ((a): {hfl0:.4f}\\to{hfl1:.4f} at one malicious client of five) "
               f"and VFL ((b): {vfl0:.4f}\\to{vfl1:.4f} at the first tested scale factor). "
               f"Panel (a) also shows a clean one-cell margin injection, which adds only the "
               f"closed-form margin to a single histogram cell rather than zeroing the rest of "
               f"the client's report, confirming the same collapse under the exact primitive "
               f"Section~4.2's margin models (Section~6.2). "
               f"The same label-shuffle attack, held fixed across (c) and (d), isolates the "
               f"aggregation mechanism from the attack type: under bagging (c), damage grows "
               f"steadily with each added colluder ({bag_drop_1:.4f} drop at one malicious client "
               f"to {bag_drop_3:.4f} at three); under boosting (d), a second colluder adds almost "
               f"nothing beyond the first ({boost_drop_1:.4f} at one, versus {boost_drop_1:.4f} to "
               f"{boost_drop_3:.4f} through three), the same flattening tendency as (a) and (b) but "
               f"far smaller in magnitude, since label-shuffle is a far blunter corruption than "
               f"targeted histogram forgery.")
    ps.save_fig(fig, "fig1_saturation", caption)


if __name__ == "__main__":
    main()
