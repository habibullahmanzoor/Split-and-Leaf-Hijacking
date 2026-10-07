import re
s = open("paper.tex", encoding="utf-8", newline="").read()
pats = [r"original submission", r"earlier version", r"an earlier", r"withdraw", r"no longer", r"of the revision",
        r"the revision", r"previous version", r"originally", r"we corrected", r"was inaccurate", r"in the original",
        r"Appendix", r"appendix", r"\\appendix", r"app:impl", r"substitute", r"reviewer", r"Reviewer", r"we now report", r"We now report",
        r"We now ", r"we now "]
for p in pats:
    for m in re.finditer(p, s):
        print(f"{p!r:28s}", repr(s[max(0, m.start() - 70):m.end() + 70]))
print("sections:", [l for l in s.split("\n") if l.startswith("\\section") or l.startswith("\\subsection")][-34:-14])
