from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

PALETTE = {
    "yellow": "#eda100", "red": "#e34948", "grid": "#e1e0d9", "muted": "#898781",
    "ink": "#0b0b0b", "secondary": "#52514e", "surface": "#fcfcfb",
}

fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
naive = [0.8992, 0.8954, 0.8952, 0.8952, 0.8940, 0.8939]
margin_aware = [0.8992, 0.8793, 0.8793, 0.8793, 0.8793, 0.8793]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
fig.patch.set_facecolor(PALETTE["surface"])
ax.set_facecolor(PALETTE["surface"])

ax.plot(fractions, naive, color=PALETTE["yellow"], linewidth=2, marker="o",
        markersize=6, label="Naive noise (generic FL baseline)")
ax.plot(fractions, margin_aware, color=PALETTE["red"], linewidth=2, marker="o",
        markersize=6, label="Margin-aware split-flip (ours, category 1)")
ax.axhline(naive[0], color=PALETTE["muted"], linewidth=1, linestyle="--",
           label="No attack (baseline)")

ax.annotate("saturates at 1 malicious client\n(2, 3 add nothing further)",
            xy=(0.1, 0.8793), xytext=(0.22, 0.865),
            fontsize=8, color=PALETTE["secondary"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1))

ax.set_xlabel("Fraction of malicious clients", color=PALETTE["ink"])
ax.set_ylabel("Test AUC", color=PALETTE["ink"])
ax.set_title("Category-1 histogram/split-integrity attack — HFL, Adult dataset\n"
              "(5 clients, Dirichlet non-IID α=0.5, attack on depth 0-1 of every tree)",
              color=PALETTE["ink"], fontsize=10)
ax.set_ylim(0.855, 0.905)
ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(PALETTE["muted"])
ax.tick_params(colors=PALETTE["secondary"])
ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=PALETTE["secondary"])

fig.tight_layout()
out_path = FIG_DIR / "category1_attack_sweep.png"
fig.savefig(out_path, facecolor=fig.get_facecolor())
print(f"Saved to {out_path}")
