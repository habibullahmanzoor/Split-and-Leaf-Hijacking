"""Revision figures (read from results/*.json; no recomputation)."""
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import plotstyle as ps

RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
H = json.load(open(RES / "hfl_default.json"))


def out(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", facecolor=ps.SURFACE)
    fig.savefig(FIG / f"{name}.png", facecolor=ps.SURFACE)
    print("saved", name)


# ---- R2: bounded reports x colluders ----
fig, axes = ps.new_fig(ncols=2, wide=True, height=3.2)
rhos = [0.05, 0.1, 0.25, 0.5, 1.0, None]
cmap = [ps.C2, ps.C4, ps.C3, ps.C5, ps.C1, ps.MUTED]
for rho, c, mk in zip(rhos, cmap, ps.MARKERS + ["x"]):
    ks = [1, 2, 3, 4]
    key = lambda k: H["bounded"][f"rho={rho}|k={k}"]
    lab = "no bound" if rho is None else rf"$\rho$={rho}"
    axes[0].errorbar(ks, [key(k)["loss"] for k in ks],
                     yerr=[[key(k)["loss"] - key(k)["ci"][0] for k in ks],
                           [key(k)["ci"][1] - key(k)["loss"] for k in ks]],
                     color=c, marker=mk, markersize=4, capsize=2, linewidth=1.4, label=lab)
    axes[1].plot(ks, [key(k)["hit"] for k in ks], color=c, marker=mk, markersize=4, linewidth=1.4, label=lab)
axes[0].set_xlabel("Colluding clients $k$")
axes[0].set_ylabel("Paired AUC loss")
axes[0].set_title("(a) Damage", fontsize=9)
axes[1].set_xlabel("Colluding clients $k$")
axes[1].set_ylabel("Fraction of attacked nodes won")
axes[1].set_title("(b) Target control", fontsize=9)
ps.style_axes(axes[0], legend_kwargs=dict(fontsize=6, loc="upper left"))
ps.style_axes(axes[1], legend=False)
for a in axes:
    a.set_xticks([1, 2, 3, 4])
out(fig, "figR2_bounded_reports")

# ---- R3: detectors ----
names = [("full_zero", "Zeroing"), ("full_one_cell", "One cell"), ("cons_pair", "Conservation-\npreserving"),
         ("limited_x1.5", "Limited\nknowledge"), ("matched_noise", "Matched\nnoise")]
dets = [("cons", "Conservation", ps.C1), ("mag", "Cell magnitude", ps.C3), ("l1", "Report L1 norm", ps.C2),
        ("any", "Any check", ps.C4)]
fig, ax = ps.new_fig(wide=True, height=3.4)
w = 0.2
for i, (d, lab, c) in enumerate(dets):
    ax.bar(np.arange(len(names)) + (i - 1.5) * w,
           [H["variants"][k][f"tpr_{d}"] for k, _ in names], w, color=c, label=lab)
fpr = H["variants"]["full_zero"]["fpr_any"]
ax.axhline(fpr, color="black", linestyle="--", linewidth=1, label=f"Honest false-alarm rate (any check) = {fpr:.2f}")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([l for _, l in names], fontsize=7)
ax.set_ylabel("Fraction of malicious reports flagged")
ax.set_ylim(0, 1.15)
ps.style_axes(ax, legend_kwargs=dict(fontsize=6, loc="upper right", ncol=2))
out(fig, "figR3_detectors")

# ---- R4: persistence ----
fig, ax = ps.new_fig(wide=True, height=3.4)
taus = [0.1, 0.25, 0.5, 0.75, 1.0]
for sname, c, mk, lab in [("contig_early", ps.C1, "o", "Contiguous, early"), ("contig_middle", ps.C3, "s", "Contiguous, middle"),
                          ("contig_late", ps.C2, "^", "Contiguous, late"), ("intermittent", ps.C4, "D", "Intermittent"),
                          ("random", ps.C5, "v", "Random")]:
    P = [H["persistence"][f"tau={t}|{sname}"] for t in taus]
    ax.errorbar(taus, [p["loss"] for p in P], yerr=[[p["loss"] - p["ci"][0] for p in P], [p["ci"][1] - p["loss"] for p in P]],
                color=c, marker=mk, markersize=4, capsize=2, linewidth=1.3, label=lab)
ax.set_xlabel(r"Persistence $\tau$ (fraction of corrupted rounds)")
ax.set_ylabel("Paired AUC loss (95% CI)")
ax.set_xticks(taus)
ps.style_axes(ax, legend_kwargs=dict(fontsize=6.5, loc="upper left"))
out(fig, "figR4_persistence")
