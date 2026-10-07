"""Formatting helpers and data loading for the generated Results text."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
H = json.load(open(R / "u_hfl.json"))
D = json.load(open(R / "hfl_default.json"))
X = json.load(open(R / "u_extra.json"))
C = json.load(open(R / "constant_check.json"))
FB = json.load(open(R / "fedbins.json"))


def vfl():
    return json.load(open(R / "u_vfl_summary.json"))


def n(x, d=4):
    """Number as it appears in text; negatives get a math minus."""
    s = f"{abs(x):.{d}f}"
    return f"${'-' if x < -0.5 * 10 ** (-d) else ''}{s}$" if x < -0.5 * 10 ** (-d) else s


def ci(lo, hi, d=4):
    return f"{n(lo, d)} to {n(hi, d)}"


def lc(v, d=4):
    """'loss (95% CI lo to hi)' for a dict with loss/ci, or a [mean, lo, hi] triple."""
    if isinstance(v, dict):
        m, lo, hi = v["loss"], v["ci"][0], v["ci"][1]
    else:
        m, lo, hi = v
    return f"{n(m, d)} (95\\% CI {ci(lo, hi, d)})"


def zero_in(lo, hi):
    return lo <= 0 <= hi


def pc(x, d=0):
    return f"{100 * x:.{d}f}\\%"
