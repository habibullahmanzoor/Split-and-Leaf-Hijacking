"""Highlight whole figure/table captions of new or changed floats in paper.tex.

A caption written as  \\caption{\\rev{TEXT}}  only turns TEXT blue; the "Figure 5." label stays black.
This pass rewrites such captions into

    \\revfloat{\\caption{TEXT}\\label{L}}

so that the label and the text are both blue.  The \\label has to sit inside the colour group because
\\refstepcounter sets \\@currentlabel locally.  For tables the label is moved from after the tabular
to right behind the caption.  Algorithms are left alone.  Running it twice changes nothing.
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper" / "paper.tex"
OPEN = "\\caption{\\rev{"


def _end(s, start):
    """Index of the brace closing the one at s[start]."""
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "\\":
            continue
        if c in "{}" and i > 0 and s[i - 1] == "\\":
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def transform(s):
    """returns (new_text, number_of_captions_rewritten)"""
    out, pos, n = [], 0, 0
    while True:
        i = s.find(OPEN, pos)
        if i < 0:
            out.append(s[pos:])
            break
        env = max(s.rfind("\\begin{figure", 0, i), s.rfind("\\begin{table", 0, i),
                  s.rfind("\\begin{algorithm", 0, i))
        env_name = re.match(r"\\begin\{(\w+)", s[env:]).group(1) if env >= 0 else ""
        cap_open = i + len("\\caption")                       # the "{" after \caption
        cap_close = _end(s, cap_open)
        rev_open = i + len("\\caption{\\rev")                 # the "{" after \rev
        rev_close = _end(s, rev_open)
        whole = rev_close == cap_close - 1
        if not whole or env_name not in ("figure", "table"):
            out.append(s[pos:cap_close + 1])
            pos = cap_close + 1
            continue
        inner = s[rev_open + 1:rev_close]
        stop = s.index("\\end{" + env_name, cap_close)
        m = re.search(r"\\label\{[^}]+\}", s[cap_close + 1:stop])
        if not m:
            raise ValueError("float without label near: " + inner[:60])
        label = m.group(0)
        a, b = cap_close + 1 + m.start(), cap_close + 1 + m.end()
        if s[b:b + 1] == "\n":
            b += 1
        before_label = s[cap_close + 1:a]
        out.append(s[pos:i] + "\\revfloat{\\caption{" + inner + "}" + label + "}" + before_label)
        pos = b
        n += 1
    return "".join(out), n


def apply(path=PAPER):
    raw = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    s = raw.replace("\r\n", "\n")
    new, n = transform(s)
    macro = "\\newcommand{\\revfloat}[1]{{\\color{blue}#1}}"
    if macro not in new:
        anchor = "\\newcommand{\\rev}[1]{{\\color{blue}#1}}\n"
        assert anchor in new, "\\rev macro line not found"
        new = new.replace(anchor, anchor + macro + "\n")
    assert "\r" not in new
    open(path, "w", encoding="utf-8", newline="").write(new.replace("\n", nl))
    return n


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER
    print("captions rewritten:", apply(p))
