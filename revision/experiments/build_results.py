"""Rebuild the Results section of revision/paper/paper.tex from generated blocks."""
from pathlib import Path
import rb_sections_hfl as SH
import rb_sections_vfl as SV

P = Path(__file__).resolve().parents[1] / "paper" / "paper.tex"
s = open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")

a = s.index("\\section{Results}")
b = s.index("\\section{Discussion and Limitations}")
old = s[a:b]

# keep: results intro paragraph and 6.1 as they are
i1 = old.index("\\subsection{Split flip margin validation}")
i2 = old.index("\\subsection{Split hijacking: attacker count and injection magnitude}")
head = old[:i1]
margin = old[i1:i2]

if "shared protocol of Section" not in head:
  head = head.replace(
    "We present the results from mechanism to impact.",
    "\\rev{Every experiment uses the shared protocol of Section~\\ref{sec:trainconfig}, so the honest baseline of a dataset is the same in every figure.} We present the results from mechanism to impact.")

new = (head + margin
       + SH.sec_saturation(SV.scale_text()) + "\n"
       + SH.sec_realistic() + "\n"
       + SH.sec_radius(SV.party_text()) + "\n"
       + SH.sec_persistence() + "\n"
       + SV.sec_disruption() + "\n"
       + SV.sec_leaf() + "\n"
       + SV.sec_combined() + "\n"
       + SV.sec_cross() + "\n"
       + SV.summary_table() + "\n")
s = s[:a] + new + s[b:]
P.write_text(s.replace("\n", nl), encoding="utf-8", newline="")
import revfloat
print("captions highlighted:", revfloat.apply(P))
print("rebuilt, bytes", len(new))
