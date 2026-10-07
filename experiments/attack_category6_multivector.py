"""Category-6 multi-vector attack: HE uniform amplification (sub-threshold,
attacks split selection) x leaf misrouting (sub-threshold, attacks
post-split aggregation), swept as a 2D grid to test for synergy vs a
purely additive interaction.
"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fed_datasets.loaders import load_adult
from harness.vfl_secureboost import FederatedVFLGBDT
from attacks.multi_vector import CombinedPassiveParty

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
PALETTE = {"ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb", "muted": "#898781"}

N_BINS, MAX_DEPTH, N_ROUNDS = 16, 3, 10
HE_SCALES = [1.0, 1.5, 3.0]
MISROUTE_FRACS = [0.0, 0.15, 0.30]
N_TRIALS = 5


def run_once(Xa, Xp, y, he_scale, mis_frac, seed):
    Xa_tr, Xa_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
        Xa, Xp, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = FederatedVFLGBDT(n_bins=N_BINS, max_depth=MAX_DEPTH, n_rounds=N_ROUNDS,
                              lr=0.3, key_size=512)
    factory = partial(CombinedPassiveParty, he_uniform_scale=he_scale,
                       misroute_fraction=mis_frac, _rng=np.random.default_rng(seed))
    model.fit(Xa_tr, y_tr, {"party1": Xp_tr}, passive_party_factories={"party1": factory})
    return roc_auc_score(y_te, model.predict_proba(Xa_te, {"party1": Xp_te}))


def main():
    X, y, groups = load_adult()
    party0_cols, party1_cols = groups["party0"], groups["party1"]

    rng = np.random.default_rng(0)
    n_sub = 1500
    idx = rng.choice(len(y), size=n_sub, replace=False)
    X_sub, y_sub = X.iloc[idx].reset_index(drop=True), y[idx]

    Xa = X_sub[party0_cols].to_numpy(dtype=float)
    Xp = X_sub[party1_cols].to_numpy(dtype=float)

    # raw_grid[i, j] holds all N_TRIALS per-seed AUCs for that cell, seed-aligned
    # across cells (every cell uses the identical seed -> identical train/test
    # split for that seed), so cross-cell differences at a fixed seed are paired,
    # not independent draws.
    raw_grid = [[None] * len(MISROUTE_FRACS) for _ in HE_SCALES]
    auc_grid = np.zeros((len(HE_SCALES), len(MISROUTE_FRACS)))
    std_grid = np.zeros_like(auc_grid)

    for i, he_scale in enumerate(HE_SCALES):
        for j, mis_frac in enumerate(MISROUTE_FRACS):
            aucs = np.array([run_once(Xa, Xp, y_sub, he_scale, mis_frac, seed=t) for t in range(N_TRIALS)])
            raw_grid[i][j] = aucs
            auc_grid[i, j] = aucs.mean()
            std_grid[i, j] = aucs.std(ddof=1)
            print(f"he_scale={he_scale:.1f}  misroute={mis_frac:.2f}  "
                  f"AUC={auc_grid[i,j]:.4f}+/-{std_grid[i,j]:.4f}  ({N_TRIALS} seeds)  "
                  f"raw={np.round(aucs, 4).tolist()}")

    honest_auc = auc_grid[0, 0]
    cost_he_only = honest_auc - auc_grid[:, 0]
    cost_mis_only = honest_auc - auc_grid[0, :]
    print("\n=== Synergy check, unpaired (means across 5 seeds, original method) ===")
    for i, he_scale in enumerate(HE_SCALES):
        for j, mis_frac in enumerate(MISROUTE_FRACS):
            if he_scale == 1.0 or mis_frac == 0.0:
                continue
            combined_cost = honest_auc - auc_grid[i, j]
            additive_prediction = cost_he_only[i] + cost_mis_only[j]
            synergy = combined_cost - additive_prediction
            print(f"he={he_scale:.1f}, misroute={mis_frac:.2f}: combined cost={combined_cost:.4f}, "
                  f"additive prediction={additive_prediction:.4f}, "
                  f"synergy={synergy:+.4f} ({'SUPER-additive' if synergy > 0.002 else 'roughly additive' if abs(synergy) <= 0.002 else 'SUB-additive'})")

    print("\n=== Synergy check, PAIRED per seed (same train/test split shared across cells) ===")
    # T-critical for a two-sided 95% CI at df = N_TRIALS - 1 (paired t-test).
    from scipy import stats as _stats
    t_crit = _stats.t.ppf(0.975, df=N_TRIALS - 1)
    honest_raw = raw_grid[0][0]
    for i, he_scale in enumerate(HE_SCALES):
        for j, mis_frac in enumerate(MISROUTE_FRACS):
            if he_scale == 1.0 or mis_frac == 0.0:
                continue
            heonly_raw = raw_grid[i][0]
            misonly_raw = raw_grid[0][j]
            combined_raw = raw_grid[i][j]
            # synergy_t = heonly_t + misonly_t - honest_t - combined_t, matching the
            # unpaired formula's algebra but evaluated per seed before averaging.
            synergy_per_seed = heonly_raw + misonly_raw - honest_raw - combined_raw
            mean_syn = synergy_per_seed.mean()
            sem_syn = synergy_per_seed.std(ddof=1) / np.sqrt(N_TRIALS)
            ci_half = t_crit * sem_syn
            significant = abs(mean_syn) > ci_half
            print(f"he={he_scale:.1f}, misroute={mis_frac:.2f}: per-seed synergy={np.round(synergy_per_seed, 4).tolist()}, "
                  f"mean={mean_syn:+.4f}, SEM={sem_syn:.4f}, 95% CI=[{mean_syn - ci_half:+.4f}, {mean_syn + ci_half:+.4f}]  "
                  f"({'SIGNIFICANT, ' + ('SUPER' if mean_syn > 0 else 'SUB') + '-additive' if significant else 'not significant, CI contains zero'})")

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", PALETTE_SEQ[::-1])
    im = ax.imshow(auc_grid, cmap=cmap, aspect="auto", vmin=auc_grid.min(), vmax=auc_grid.max())
    ax.set_xticks(range(len(MISROUTE_FRACS)))
    ax.set_xticklabels([f"{m:.0%}" for m in MISROUTE_FRACS])
    ax.set_yticks(range(len(HE_SCALES)))
    ax.set_yticklabels([f"{s:.1f}x" for s in HE_SCALES])
    ax.set_xlabel("Misrouting fraction (sub-threshold, alone)", color=PALETTE["ink"])
    ax.set_ylabel("HE uniform scale (sub-threshold, alone)", color=PALETTE["ink"])
    for i in range(len(HE_SCALES)):
        for j in range(len(MISROUTE_FRACS)):
            ax.text(j, i, f"{auc_grid[i,j]:.3f}", ha="center", va="center",
                    color=PALETTE["ink"] if auc_grid[i,j] > auc_grid.mean() else "white", fontsize=10)
    ax.set_title("Category-6 multi-vector attack: test AUC\n"
                  "(HE split-selection bias × leaf-routing corruption, each sub-threshold alone)",
                  color=PALETTE["ink"], fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test AUC", color=PALETTE["ink"])
    fig.tight_layout()
    out_path = FIG_DIR / "category6_multivector_grid.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
