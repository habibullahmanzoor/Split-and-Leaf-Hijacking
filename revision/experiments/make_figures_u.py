"""Figures under the unified protocol. Reads u_hfl.json, u_vfl_summary.json,
u_extra.json and hfl_default.json; writes PDF+PNG into revision/figures."""
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import plotstyle as ps

RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
H = json.load(open(RES / "u_hfl.json"))
V = json.load(open(RES / "u_vfl_summary.json"))
X = json.load(open(RES / "u_extra.json"))
D = json.load(open(RES / "hfl_default.json"))
NAMES = {"adult": "Adult", "heart": "Heart Disease", "credit": "Credit Default"}
COL = {"adult": ps.C1, "heart": ps.C2, "credit": ps.C3}


def out(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", facecolor=ps.SURFACE)
    fig.savefig(FIG / f"{name}.png", facecolor=ps.SURFACE)
    print("saved", name)


def eb(ax, xs, ys, lo, hi, **kw):
    ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], capsize=2, markersize=4, linewidth=1.5, **kw)


def lossci(v):
    return v["loss"], v["ci"][0], v["ci"][1]


# ============ Fig. 1: saturation ============
fig, axes = ps.new_fig(ncols=2, nrows=2, wide=True, height=5.6)
axes = axes.flatten()
fr = [0.2, 0.4, 0.6]
ax = axes[0]
for key, lab, c, mk in (("gauss5", "Naive noise", ps.C3, "^"), ("zero", "Margin-aware (zeroing)", ps.C1, "o"),
                        ("one_cell", "One-cell injection", ps.C4, "D")):
    L = [lossci(H["count_adult"][key][str(k)]) for k in (1, 2, 3)]
    eb(ax, [0] + fr, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L], color=c, marker=mk, label=lab)
ax.set_xlabel("Fraction malicious clients")
ax.set_ylabel("Paired AUC loss")
ax.set_title("(a) HFL histogram attacks", fontsize=9)
ax.set_ylim(-0.0003, 0.0145)
ps.style_axes(ax, legend_kwargs=dict(loc="upper right", fontsize=6.5))

ax = axes[1]
sfs = [2.0, 5.0, 10.0, 50.0, 200.0]
L = [V["scale"]["adult"][str(s)]["loss"] for s in sfs]
eb(ax, [1.0] + sfs, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L], color=ps.C1, marker="o",
   label="20 rounds (converged)")
L3 = [V["scale3"][str(s)]["loss"] for s in sfs]
eb(ax, [1.0] + sfs, [0] + [x[0] for x in L3], [0] + [x[1] for x in L3], [0] + [x[2] for x in L3], color=ps.C3, marker="s",
   linestyle="--", label="3 rounds (undertrained)")
ax.set_xscale("log")
ax.set_xlabel("Ciphertext scale factor")
ax.set_ylabel("Paired AUC loss")
ax.set_title("(b) VFL ciphertext attack", fontsize=9)
ax.set_ylim(-0.004, 0.15)
ps.style_axes(ax, legend_kwargs=dict(loc="upper right", fontsize=6.5))

ax = axes[2]
for mode, lab, c, mk in (("shuffle", "Label shuffle", ps.C5, "o"), ("forge", "Report forgery ($1-p$)", ps.C4, "s")):
    L = [lossci(H["bagging"][mode][str(k)]) for k in (1, 2, 3)]
    eb(ax, [0] + fr, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L], color=c, marker=mk, label=lab)
ax.set_xlabel("Fraction malicious clients")
ax.set_ylabel("Paired AUC loss")
ax.set_title("(c) Bagging controls", fontsize=9)
ps.style_axes(ax, legend_kwargs=dict(loc="upper left", fontsize=6.5))

ax = axes[3]
L = [lossci(H["boosting_labelshuffle"][str(k)]) for k in (1, 2, 3)]
eb(ax, [0] + fr, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L], color=ps.C4, marker="D",
   label="Boosting, label shuffle")
L = [lossci(H["bagging"]["shuffle"][str(k)]) for k in (1, 2, 3)]
eb(ax, [0] + fr, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L], color=ps.C5, marker="o",
   linestyle=":", label="Bagging, label shuffle")
ax.set_xlabel("Fraction malicious clients")
ax.set_ylabel("Paired AUC loss")
ax.set_title("(d) Same attack, two mechanisms", fontsize=9)
ps.style_axes(ax, legend_kwargs=dict(loc="upper left", fontsize=6.5))
out(fig, "fig1_saturation")

# ============ Fig. 2: transient recovery ============
fig, axes = ps.new_fig(ncols=3, wide=True)
for ax, ds in zip(axes, ("adult", "heart", "credit")):
    T = H["transient"][ds]
    R = np.arange(1, len(T["clean"][0]) + 1)
    for key, lab, c, mk in (("clean", "No attack", ps.C2, "o"), ("first", "Round 1 corrupted", ps.C1, "s"),
                            ("last", "Last round corrupted", ps.C4, "^")):
        a = np.array(T[key])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.plot(R, m, color=c, linewidth=1.5, marker=mk, markersize=3, label=lab)
        ax.fill_between(R, m - sd, m + sd, color=c, alpha=0.15)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Test AUC")
    ax.set_title(NAMES[ds], fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower right", fontsize=6.5))
out(fig, "fig2_selfheal")

# ============ Fig. 7: cross-dataset ============
fig, axes = ps.new_fig(ncols=2, wide=True, height=3.4)
w = 0.25
for i, ds in enumerate(("adult", "heart", "credit")):
    L = [lossci(H["count"][ds][str(k)]) for k in (1, 2, 3)]
    ys = np.array([x[0] for x in L])
    axes[0].bar(np.arange(3) + (i - 1) * w, ys, w, color=COL[ds], label=NAMES[ds],
                yerr=[ys - [x[1] for x in L], [x[2] for x in L] - ys], capsize=2)
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(["1 of 5", "2 of 5", "3 of 5"])
axes[0].set_xlabel("Malicious clients")
axes[0].set_ylabel("Paired AUC loss")
axes[0].set_title("(a) HFL", fontsize=9)
ps.style_axes(axes[0], legend_kwargs=dict(loc="upper right", fontsize=6.5))
for ds in ("adult", "heart", "credit"):
    L = [V["scale"][ds][str(s)]["loss"] for s in sfs]
    eb(axes[1], [1.0] + sfs, [0] + [x[0] for x in L], [0] + [x[1] for x in L], [0] + [x[2] for x in L],
       color=COL[ds], marker="o", label=NAMES[ds])
axes[1].set_xscale("log")
axes[1].set_xlabel("Ciphertext scale factor")
axes[1].set_ylabel("Paired AUC loss")
axes[1].set_title("(b) VFL (20 rounds)", fontsize=9)
ps.style_axes(axes[1], legend_kwargs=dict(loc="upper left", fontsize=6.5))
out(fig, "fig7_cross_dataset")

# ============ Fig. 9: prediction disruption ============
fig, axes = ps.new_fig(ncols=2, wide=True, height=3.4)
pl = [1, 3, 5, 8, 13]
B = [V["bd"]["adult"][str(p)] for p in pl]
cost = np.array([b["cost"][0] for b in B]) * 1000       # plot in units of 1e-3 so tick labels stay short
cost_lo = np.array([b["cost"][1] for b in B]) * 1000
cost_hi = np.array([b["cost"][2] for b in B]) * 1000
axes[0].errorbar(cost, [b["flip"] for b in B], xerr=[cost - cost_lo, cost_hi - cost],
                 yerr=[b["flip_sd"] for b in B], color=ps.C1, marker="o", markersize=4, capsize=2, linewidth=1.3)
LABEL_OFFSET = {1: (-14, -10, "right", "top"), 3: (0, -14, "center", "top"), 5: (-15, 8, "right", "bottom"),
                8: (11, 7, "left", "bottom"), 13: (7, 11, "left", "bottom")}
for p, cx, b in zip(pl, cost, B):
    dx, dy, ha, va = LABEL_OFFSET[p]
    axes[0].annotate(f"{p}", (cx, b["flip"]), textcoords="offset points", xytext=(dx, dy),
                      fontsize=7, ha=ha, va=va)
axes[0].axvline(0, color=ps.MUTED, linewidth=0.8)
axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
axes[0].set_xlabel(r"Paired AUC cost ($\times10^{-3}$, 95% CI); labels = plants")
axes[0].set_ylabel("Flip rate on selected inputs")
axes[0].set_title("(a) Cost versus effect", fontsize=9)
ps.style_axes(axes[0], legend=False)
axes[1].plot(pl, [b["a01"] for b in B], color=ps.C2, marker="o", markersize=4, linewidth=1.4, label=r"$0\to1$")
axes[1].plot(pl, [b["a10"] for b in B], color=ps.C1, marker="s", markersize=4, linewidth=1.4, label=r"$1\to0$")
axes[1].plot(pl, [b["dis_nt"] for b in B], color=ps.MUTED, marker="^", markersize=4, linewidth=1.4, linestyle="--",
             label="Non-target disagreement")
axes[1].set_xlabel("Plants")
axes[1].set_ylabel("Rate")
axes[1].set_title("(b) Direction and side effects", fontsize=9)
ps.style_axes(axes[1], legend_kwargs=dict(loc="upper left", fontsize=6.5))
out(fig, "fig9_backdoor_tradeoff")

# ============ Fig. 12: radius ============
fig, axes = ps.new_fig(ncols=3, wide=True)
for ax, ds in zip(axes, ("adult", "heart", "credit")):
    Rr = H["radius"][ds]
    rs = sorted(int(r) for r in Rr)
    ys = np.array([Rr[str(r)]["attacked"] for r in rs])
    ax.plot(rs, ys, color=COL[ds], marker="o", markersize=4, linewidth=1.6, label="Attacked")
    ax.axhline(Rr["0"]["clean"], color=ps.MUTED, linestyle="--", linewidth=1, label="No attack")
    ax.axhline(0.5, color=ps.C1, linestyle=":", linewidth=1, label="Random guessing")
    ax.set_xlabel("Attack radius $r$")
    ax.set_ylabel("Test AUC")
    ax.set_xticks(rs)
    ax.set_title(NAMES[ds], fontsize=9)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower left", fontsize=6.5))
out(fig, "fig12_attack_depth_scoping")

# ============ Fig. 13: target selection ============
fig, ax = ps.new_fig(wide=True, height=3.4)
for i, (c, lab, col) in enumerate((("fixed", "Fixed feature", ps.C2), ("variance", "Maximum variance", ps.C1))):
    T = [V["tgt"][f"{d}|{c}"]["loss"] for d in ("adult", "heart", "credit")]
    ys = np.array([t[0] for t in T])
    ax.bar(np.arange(3) + (i - 0.5) * 0.35, ys, 0.35, color=col, label=lab,
           yerr=[ys - [t[1] for t in T], [t[2] for t in T] - ys], capsize=3)
ax.set_xticks(range(3))
ax.set_xticklabels([NAMES[d] for d in ("adult", "heart", "credit")])
ax.set_ylabel("Paired AUC loss (95% CI)")
ps.style_axes(ax, legend_kwargs=dict(loc="upper left", fontsize=7))
out(fig, "fig13_variance_targeting")

# ============ Fig. 14: federation size ============
fig, axes = ps.new_fig(ncols=2, wide=True, height=3.2)
Ks = [5, 10, 20, 40]
L = [lossci(H["federation"][str(k)]) for k in Ks]
eb(axes[0], range(len(Ks)), [x[0] for x in L], [x[1] for x in L], [x[2] for x in L], color=ps.C1, marker="o")
axes[0].set_xticks(range(len(Ks)))
axes[0].set_xticklabels(Ks)
axes[0].set_xlabel("Total HFL clients (one malicious)")
axes[0].set_ylabel("Paired AUC loss")
axes[0].set_title("(a) HFL", fontsize=9)
axes[0].set_ylim(0, 0.014)
L = [V["party"][str(n)]["loss"] for n in (1, 2, 3)]
eb(axes[1], range(3), [x[0] for x in L], [x[1] for x in L], [x[2] for x in L], color=ps.C2, marker="o")
axes[1].set_xticks(range(3))
axes[1].set_xticklabels([1, 2, 3])
axes[1].set_xlabel("Passive parties (one malicious)")
axes[1].set_ylabel("Paired AUC loss")
axes[1].set_title("(b) VFL (20 rounds)", fontsize=9)
for a in axes:
    ps.style_axes(a, legend=False)
out(fig, "fig14_federation_size_scaling")

# ============ Fig. 15: leaf hijacking ============
fig, axes = ps.new_fig(ncols=3, wide=True)
ps_ = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
for ax, ds in zip(axes, ("adult", "heart", "credit")):
    Lf = V["leaf"][ds]
    m = np.array([Lf[str(p)]["auc"] for p in ps_])
    sd = np.array([Lf[str(p)]["sd"] for p in ps_])
    ax.plot(ps_, m, color=ps.C4, linewidth=1.8, marker="^", markersize=4)
    ax.fill_between(ps_, m - sd, m + sd, color=ps.C4, alpha=0.2)
    ax.axhline(m[0], color=ps.MUTED, linestyle="--", linewidth=1, label="No attack")
    ax.axhline(0.5, color=ps.C1, linestyle=":", linewidth=1, label="Random guessing")
    ax.set_ylabel("Test AUC")
    ax.set_title(NAMES[ds], fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ps.style_axes(ax, legend_kwargs=dict(loc="lower left", fontsize=6.5))
fig.supxlabel("Fraction of samples misrouted")
out(fig, "fig15_leaf_misdirection")

# ============ Fig. R1: knowledge, budget, matched baselines ============
order = [(D["variants"]["full_zero"], "Full-know.\nzeroing\n(implemented)", ps.C1),
         (D["variants"]["full_one_cell"], "Full-know.\none cell", ps.C1),
         (D["variants"]["cons_pair"], "Full-know.\nconservation-\npreserving", ps.C1),
         (D["variants"]["limited_x1.5"], "Own report\nonly (x1.5)", ps.C2),
         (X["variants"]["released_x1.5"], "Own report +\nreleased trees\n(x1.5)", ps.C2),
         (D["variants"]["matched_noise"], "Matched-L1\nrandom\nnoise", ps.MUTED),
         (D["variants"]["matched_random_cell"], "Matched-L1\nrandom\ncell", ps.MUTED),
         (X["matched"]["label_flip"], "Matched-L1\nlabel flip", ps.MUTED)]
fig, ax = ps.new_fig(wide=True, height=3.7)
vals = np.array([o[0]["loss"] for o in order])
lo = np.array([o[0]["ci"][0] for o in order])
hi = np.array([o[0]["ci"][1] for o in order])
ax.bar(range(len(order)), vals, yerr=[vals - lo, hi - vals], color=[o[2] for o in order], capsize=3, width=0.65)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([o[1] for o in order], fontsize=6)
ax.set_ylabel("Paired AUC loss (95% CI)")
ps.style_axes(ax, legend=False)
out(fig, "figR1_knowledge_budget")
