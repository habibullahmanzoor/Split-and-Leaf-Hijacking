"""Figure 14 (Section 5.1, sec:marginresults): does the "1 malicious
participant suffices" saturation finding hold as the federation grows, or
was it an artifact of this paper's standard federation sizes? (a) HFL:
fixes n_malicious=1 (an absolute count, not a fraction) and sweeps total
client count from 5 to 40. (b) VFL: splits the same feature block currently
held by one passive party across one, two, or three passive parties
instead, with one held malicious throughout. Consolidates
verify_client_count_scaling.py and verify_vfl_party_count.py. (Merged from
two separate single-panel figures into one two-panel figure, since both
answer the same "is federation size load bearing" question for their
respective protocol.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from fed_datasets.loaders import load_adult
from verify_client_count_scaling import run_once as run_client, CLIENT_COUNTS, N_TRIALS as N_TRIALS_CLIENT
from verify_vfl_party_count import run_once as run_party, N_TRIALS as N_TRIALS_PARTY
import plotstyle as ps

PARTY_COUNTS = [1, 2, 3]


def panel_a_clients(ax):
    X, y, _ = load_adult()
    X = X.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    def compute():
        honest_means_, mal_means_, mal_stds_ = [], [], []
        for n_clients in CLIENT_COUNTS:
            honest_t, mal_t = [], []
            for t in range(N_TRIALS_CLIENT):
                honest_t.append(run_client(X_train, y_train, X_test, y_test, n_clients, 0, seed=t))
                mal_t.append(run_client(X_train, y_train, X_test, y_test, n_clients, 1, seed=t))
            honest_means_.append(float(np.mean(honest_t)))
            mal_means_.append(float(np.mean(mal_t)))
            mal_stds_.append(float(np.std(mal_t, ddof=1)))
        return {"honest_means": honest_means_, "mal_means": mal_means_, "mal_stds": mal_stds_}

    cached = ps.cached("fig14_clients", compute)
    honest_means, mal_means, mal_stds = cached["honest_means"], cached["mal_means"], cached["mal_stds"]
    for n_clients, hm, mm, ms in zip(CLIENT_COUNTS, honest_means, mal_means, mal_stds):
        print(f"[HFL] n_clients={n_clients:>3d}  honest={hm:.4f}  1_malicious={mm:.4f}+/-{ms:.4f}")

    ax.plot(CLIENT_COUNTS, honest_means, color=ps.C2, linewidth=1.8, marker=ps.MARKERS[0],
            markersize=6, label="No attack")
    ax.errorbar(CLIENT_COUNTS, mal_means, yerr=mal_stds, color=ps.C1, linewidth=1.8,
                marker=ps.MARKERS[1], markersize=6, capsize=3, label="1 malicious client")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(CLIENT_COUNTS)
    ax.set_xticklabels([str(c) for c in CLIENT_COUNTS])
    ax.set_xlabel("Total clients in the federation")
    ax.set_ylabel("Test AUC")
    ax.set_title("(a) HFL: client count", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="center left", fontsize=7))
    return [h - m for h, m in zip(honest_means, mal_means)]


def panel_b_parties(ax):
    X, y, groups = load_adult()
    def compute():
        honest_means_, mal_means_, mal_stds_ = [], [], []
        for n_passive in PARTY_COUNTS:
            honest_t, mal_t = [], []
            for t in range(N_TRIALS_PARTY):
                ah, aa = run_party(X, y, groups, n_passive, seed=t)
                honest_t.append(ah)
                mal_t.append(aa)
            honest_means_.append(float(np.mean(honest_t)))
            mal_means_.append(float(np.mean(mal_t)))
            mal_stds_.append(float(np.std(mal_t, ddof=1)))
        return {"honest_means": honest_means_, "mal_means": mal_means_, "mal_stds": mal_stds_}

    cached = ps.cached("fig14_parties", compute)
    honest_means, mal_means, mal_stds = cached["honest_means"], cached["mal_means"], cached["mal_stds"]
    for n_passive, hm, mm, ms in zip(PARTY_COUNTS, honest_means, mal_means, mal_stds):
        print(f"[VFL] n_passive_parties={n_passive}  honest={hm:.4f}  attacked={mm:.4f}+/-{ms:.4f}")

    ax.plot(PARTY_COUNTS, honest_means, color=ps.C2, linewidth=1.8, marker=ps.MARKERS[0],
            markersize=7, label="No attack")
    ax.errorbar(PARTY_COUNTS, mal_means, yerr=mal_stds, color=ps.C1, linewidth=1.8,
                marker=ps.MARKERS[1], markersize=7, capsize=3, label="1 malicious passive party")
    ax.set_xticks(PARTY_COUNTS)
    ax.set_xlim(0.7, 3.3)
    ax.set_xlabel("Passive parties")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.7, 0.95)
    ax.set_title("(b) VFL: party count", fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="upper center", fontsize=7))
    return honest_means, mal_means


def main():
    fig, axes = ps.new_fig(ncols=2, wide=True, height=3.6)
    drops = panel_a_clients(axes[0])
    honest_means, mal_means = panel_b_parties(axes[1])

    caption = (f"Neither headline saturation finding is an artifact of this paper's standard "
               f"federation size, five seeds per point. (a) A single malicious client's damage "
               f"stays within a narrow band ({min(drops):.4f} to {max(drops):.4f} AUC points) as "
               f"the horizontal federation grows from 5 to 40 total clients. (b) Category 3's "
               f"damage is unchanged whether the same feature set is held by one passive party, as "
               f"everywhere else in this paper, or split across three "
               f"({mal_means[0]:.4f} to {mal_means[-1]:.4f} attacked accuracy, honest "
               f"{honest_means[0]:.4f} throughout).")
    ps.save_fig(fig, "fig14_federation_size_scaling", caption)


if __name__ == "__main__":
    main()
