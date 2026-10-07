import re
import os
c = open("paper.tex", encoding="utf-8").read()
L = set(re.findall(r"\\label\{([^}]+)\}", c))
R = set(re.findall(r"\\ref\{([^}]+)\}", c))
print("dangling refs:", R - L)
print("duplicate labels:", [l for l in L if c.count("\\label{" + l + "}") > 1])
print("brace balance:", c.count("{") - c.count("}"))
for env in ["figure", "table", "equation", "proposition", "corollary", "algorithm", "itemize", "abstract"]:
    b = len(re.findall(r"\\begin\{" + env + r"\*?\}", c))
    e = len(re.findall(r"\\end\{" + env + r"\*?\}", c))
    print(env, b, e, "OK" if b == e else "MISMATCH")
for f in re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", c):
    print("figure file", f, os.path.exists("../figures/" + f))
cites = set()
for m in re.findall(r"\\cite\{([^}]+)\}", c):
    cites.update(k.strip() for k in m.split(","))
bib = open("ref.bib", encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("missing bib keys:", cites - keys)
print("groups >3:", [m for m in re.findall(r"\\cite\{([^}]+)\}", c) if len(m.split(",")) > 3])
print("rev blocks:", c.count("\\rev{"))
