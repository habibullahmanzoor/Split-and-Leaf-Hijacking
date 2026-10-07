"""Compute section / figure / table numbers exactly as LaTeX will, from the order in paper.tex."""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper" / "paper.tex"

TOKEN = re.compile(
    r"\\(section|subsection|subsubsection)(\*?)\{"
    r"|\\begin\{(figure|table|algorithm)\*?\}"
    r"|\\end\{(figure|table|algorithm)\*?\}"
    r"|\\label\{([^}]+)\}"
    r"|\\caption\{")


def _read(text):
    s = text if text is not None else open(PAPER, encoding="utf-8", newline="").read()
    return s.replace("\r\n", "\n")


def _brace_end(s, start):
    """Index of the brace that closes the one opened at s[start] == '{'."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def compute(text=None):
    """label -> number string, for sections, figures, tables and algorithms."""
    s = _read(text)
    sec = sub = subsub = 0
    fig = tab = alg = 0
    nums = {}
    last_heading = None
    in_float = None
    float_number = None
    for m in TOKEN.finditer(s):
        if m.group(1):
            if m.group(2):
                last_heading = None
            elif m.group(1) == "section":
                sec += 1
                sub = subsub = 0
                last_heading = str(sec)
            elif m.group(1) == "subsection":
                sub += 1
                subsub = 0
                last_heading = f"{sec}.{sub}"
            else:
                subsub += 1
                last_heading = f"{sec}.{sub}.{subsub}"
        elif m.group(3):
            in_float, float_number = m.group(3), None
        elif m.group(4):
            in_float, float_number = None, None
        elif m.group().startswith("\\caption"):
            if in_float == "figure":
                fig += 1
                float_number = fig
            elif in_float == "table":
                tab += 1
                float_number = tab
            elif in_float == "algorithm":
                alg += 1
                float_number = alg
        elif m.group(5):
            lab = m.group(5)
            if in_float and float_number is not None:
                nums[lab] = str(float_number)
            elif not in_float and last_heading and lab.startswith("sec:"):
                nums[lab] = last_heading
    return nums


def headings(text=None):
    """list of (number, title) for every numbered heading, titles with \\rev wrappers stripped."""
    s = _read(text)
    sec = sub = subsub = 0
    out = []
    for m in re.finditer(r"\\(section|subsection|subsubsection)(\*?)\{", s):
        end = _brace_end(s, m.end() - 1)
        title = s[m.end():end]
        title = re.sub(r"\\rev\{(.*)\}", r"\1", title)
        if m.group(2):
            continue
        if m.group(1) == "section":
            sec += 1
            sub = subsub = 0
            out.append((str(sec), title))
        elif m.group(1) == "subsection":
            sub += 1
            subsub = 0
            out.append((f"{sec}.{sub}", title))
        else:
            subsub += 1
            out.append((f"{sec}.{sub}.{subsub}", title))
    return out


def heading_number(title_start, text=None):
    hs = [n for n, t in headings(text) if t.startswith(title_start)]
    assert len(hs) == 1, (title_start, hs)
    return hs[0]


def cite_numbers(text=None):
    """citation key -> number printed by an order-of-appearance numeric style."""
    s = _read(text)
    s = s[: s.index("\\bibliographystyle")] if "\\bibliographystyle" in s else s
    order = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", s):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    return {k: i + 1 for i, k in enumerate(order)}


def caption_text(label, text=None):
    """caption body (with the \\rev wrapper removed) of the float carrying `label`."""
    s = _read(text)
    for m in re.finditer(r"\\caption\{", s):
        end = _brace_end(s, m.end() - 1)
        body = s[m.end():end]
        window = s[end:end + 400]
        lab = re.match(r"\s*\}?\s*\\label\{([^}]+)\}", window)
        before = s[max(0, m.start() - 20):m.start()]
        if lab and lab.group(1) == label:
            body = re.sub(r"^\\rev\{(.*)\}$", r"\1", body, flags=re.S)
            return body
    # caption before a table: label comes later in the same environment
    for m in re.finditer(r"\\caption\{", s):
        end = _brace_end(s, m.end() - 1)
        env_end = re.search(r"\\end\{(figure|table)\*?\}", s[end:])
        seg = s[end:end + env_end.end()] if env_end else ""
        if f"\\label{{{label}}}" in seg:
            body = s[m.end():end]
            return re.sub(r"^\\rev\{(.*)\}$", r"\1", body, flags=re.S)
    raise KeyError(label)


if __name__ == "__main__":
    for k, v in compute().items():
        print(f"{k:26s} {v}")
    print()
    for n, t in headings():
        print(n, t[:70])
    print()
    cn = cite_numbers()
    print("wu2025vflpractice:", cn["wu2025vflpractice"], " total keys:", len(cn))
