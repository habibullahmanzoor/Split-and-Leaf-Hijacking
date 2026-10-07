"""Shared publication style for every figure in results/figures/main/.

Design choices (see paper.md Working Notes, W8):
  - Pure white surface (#ffffff), not the earlier off-white dashboard tint --
    prints cleanly and matches a plain LaTeX figure environment.
  - A fixed-order categorical palette, each color paired with a distinct
    marker/linestyle so identity never depends on color alone.
  - Titles are omitted or kept to a handful of words; the explanation that
    used to live in a two-line plot title belongs in the LaTeX caption
    instead (each consolidated script prints its suggested caption text).
  - dpi=300, sized to a target print width (single vs. double column) rather
    than an arbitrary figsize.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "main"
MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cached(key, compute_fn):
    """Disk-cache the JSON-serializable result of an expensive experiment
    computation (federated training, seed sweeps) under a name unique to the
    figure/panel/dataset it belongs to. Every paper_figures/*.py script fuses
    "run the experiment" and "plot it" together, so without this, a purely
    cosmetic replot (font size, layout, label wording) re-trains every model
    from scratch. Delete results/figures/_cache/<key>.json (or the whole
    directory) to force recomputation after an attack or dataset code change.
    """
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    result = compute_fn()
    with open(path, "w") as f:
        json.dump(result, f)
    return result

INK = "#1a1a1a"
SECONDARY = "#4d4d4d"
MUTED = "#8f8f8f"
GRID = "#e6e6e6"
SURFACE = "#ffffff"

# Fixed-order categorical palette. Use in this order; never cycle past it --
# a 6th series folds into small multiples instead.
C1 = "#c1443c"  # red-vermillion   -- attack / undefended / primary finding
C2 = "#2f6f9f"  # blue             -- defense / mitigation / comparison A
C3 = "#dba43a"  # amber            -- comparison B
C4 = "#5c4a8a"  # violet           -- comparison C
C5 = "#2f9e6f"  # teal-green       -- reserved: "escapes the bound" / success

CATEGORICAL = [C1, C2, C3, C4, C5]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "-", "-", "-", "-"]

# Print widths in inches, matching a two-column venue (USENIX/IEEE S&P/CCS style).
WIDTH_SINGLE = 3.6
WIDTH_WIDE = 7.4
DPI = 300

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
})


def new_fig(ncols=1, nrows=1, wide=False, height=None):
    # Every figure is placed in the paper at \includegraphics[width=\textwidth]
    # (~6.5in), which uniformly rescales the whole canvas -- including font
    # points -- to that width regardless of how wide the canvas was drawn.
    # Widening the canvas for figures with more panels (as a previous version
    # of this function did, proportionally to ncols) therefore backfires: it
    # shrinks the final printed font size instead of preserving it. Cap the
    # canvas at WIDTH_WIDE for every multi-panel figure so the printed scale
    # factor (textwidth / canvas width) stays the same ~0.88 regardless of
    # panel count.
    if wide or ncols > 1:
        width = WIDTH_WIDE
    else:
        width = WIDTH_SINGLE
    if height is not None:
        h = height
    elif ncols > 2:
        h = 3.1 * nrows
    else:
        h = (2.6 if ncols > 1 else 2.8) * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, h), dpi=DPI)
    return fig, axes


def style_axes(ax, legend=True, legend_kwargs=None):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.8)
    if legend:
        kwargs = dict(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92,
                      fontsize=7.5, labelcolor=SECONDARY)
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def save_fig(fig, name, caption=""):
    fig.tight_layout()
    out_path = MAIN_FIG_DIR / f"{name}.png"
    fig.savefig(out_path, facecolor=SURFACE)
    pdf_path = MAIN_FIG_DIR / f"{name}.pdf"
    fig.savefig(pdf_path, facecolor=SURFACE)
    print(f"\nSaved {out_path.name} (+ .pdf)")
    if caption:
        print(f"Suggested caption:\n  {caption}")
    return out_path
