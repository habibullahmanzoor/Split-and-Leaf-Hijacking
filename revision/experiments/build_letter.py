"""Write revision/response_to_reviewers.tex from the final result files."""
import json
import re
from pathlib import Path
import numpy as np
from rb_common import H, D, X, FB, n, ci
import numbering as NB

N = NB.compute()                       # label -> number, in the order LaTeX will print them
CN = NB.cite_numbers()
REPO_URL = "https://github.com/habibullahmanzoor/Split-and-Leaf-Hijacking"

# Section numbers typed by hand in the replies below. The build fails if the manuscript is renumbered.
HAND = {"4.1": "sec:parameters", "4.2.1": None, "4.3.1": "sec:splithijack", "5.2": "sec:trainconfig",
        "5.6": "sec:impl", "5.7": "sec:xgbcheck", "6.1": "sec:marginresults", "6.3": "sec:realistic",
        "6.5": "sec:persistenceresults", "6.6": "sec:backdoorresults"}
for num, lab in HAND.items():
    if lab:
        assert N[lab] == num, f"manuscript renumbered: {lab} is {N[lab]}, letter says {num}"
assert NB.heading_number("Properties of the margin") == "4.2.1"


def _names(labs):
    v = [N[x] if x in N else NB.heading_number(x) for x in labs]      # label, or start of a heading title
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and~" + v[-1]


def S(*labs):
    return ("Section~" if len(labs) == 1 else "Sections~") + _names(labs)


def F(*labs):
    return ("Figure~" if len(labs) == 1 else "Figures~") + _names(labs)


def T(*labs):
    return ("Table~" if len(labs) == 1 else "Tables~") + _names(labs)


def FIG(label, pdf, where):
    """A new manuscript figure, reproduced in the letter with its blue caption."""
    cap = NB.caption_text(label)
    return ("\n\\letterfig{" + pdf + "}{" + N[label] + "}{" + N[where] + "}{" + cap + "}\n")


R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[1] / "response_to_reviewers.tex"
V = json.load(open(R / "u_vfl_summary.json"))
E = json.load(open(R / "excess_flip.json"))
XG = json.load(open(R / "xgb_conformance.json"))
OV = json.load(open(R / "overflow_events.json"))
C = json.load(open(R / "constant_check.json"))

z = H["count"]["adult"]["1"]
v = D["variants"]
rad = H["radius"]
A, Hh, Cc = V["scale"]["adult"], V["scale"]["heart"], V["scale"]["credit"]
ex = E["adult"]["plants"]
M = X["matched"]
RL = X["variants"]


def pc(x):
    return f"{100 * x:.0f}\\%"


def L(t, d=4):
    return f"{n(t[0], d)} ({ci(t[1], t[2], d)})"


b = D["bounded"]

# ---- where each addition is in the revised manuscript (numbers computed from paper.tex) ----
FIG_NEW = ["fig:knowledgebudget", "fig:detectors", "fig:bounded", "fig:persistencesweep"]
FIG_REGEN = ["fig:saturation", "fig:attackdepth", "fig:federationscaling", "fig:selfheal", "fig:backdoortradeoff",
             "fig:leafmisdirection", "fig:crossdataset", "fig:variancetargeting"]
FIG_SAME = ["fig:framework", "fig:baseline", "fig:margin"]
TAB_NEW = ["tab:config", "tab:xgbcheck", "tab:radius"]
TAB_CHANGED = ["tab:threatmodel", "tab:summary"]
TAB_SAME = ["tab:parameters", "tab:datasets"]
assert sorted(FIG_NEW + FIG_REGEN + FIG_SAME) == sorted(k for k in N if k.startswith("fig:")), "figure list drifted"
assert sorted(TAB_NEW + TAB_CHANGED + TAB_SAME) == sorted(k for k in N if k.startswith("tab:")), "table list drifted"

ROWS = [
    ("Attackers with limited knowledge: own report only, and own report plus the released trees", "R2.1, R4.1",
     f"{S('sec:threatmodel', 'sec:realistic')}; {T('tab:threatmodel')} (new rows); {F('fig:knowledgebudget')} (new)."),
    (r"Integrity checks (conservation, cell magnitude, $L_1$ norm) and the conservation-preserving attack", "R2.2, R4.3",
     f"{S('sec:realistic')}; {F('fig:detectors')} (new); discussion in {S('Detectability')}."),
    (r"Bounded reports (cap at $\rho$ times the largest honest cell)", "R2.2, R4.2",
     f"Statement in {S('sec:splithijack')}; experiment in {S('sec:realistic')}; {F('fig:bounded')} (new); "
     f"discussion in {S('What the results establish')}."),
    ("Budget-matched baselines: Gaussian noise, random cell, client label flipping, bagging report forgery", "R4.9",
     f"{S('sec:realistic')}; {F('fig:knowledgebudget')} (grey bars); the one-cell construction is in {S('sec:splithijack')}."),
    ("Federated binning with merged quantile sketches", "R4.4",
     f"Paragraph ``Federated binning'' in {S('sec:marginresults')}; statement in {S('Properties of the margin')}; "
     f"limitation in {S('Implementation scope')}."),
    ("Attack radius indexed from depth zero, and the radius sweep", "correction",
     f"Definition in {S('sec:parameters')}; sweep in {S('sec:radiusresults')} with {T('tab:radius')} (new) and "
     f"{F('fig:attackdepth')} (regenerated)."),
    ("Label-free prediction disruption: new name, directional rates, honest-model false-trigger baseline, trigger specificity",
     "R2.3, R4.5",
     f"{S('sec:intro', 'Related work', 'sec:splithijack', 'sec:backdoorresults')}; {F('fig:backdoortradeoff')} (regenerated); "
     f"Algorithm~{N['alg:backdoor']} retitled; discussion in {S('Prediction-disruption interpretation')}."),
    ("All VFL experiments rerun at 20 boosting rounds", "R4.6",
     f"Round count justified in {S('sec:baselinequality')}; results in {S('sec:saturationresults')} ({F('fig:saturation')}b), "
     f"{S('sec:radiusresults')} ({F('fig:federationscaling')}b), {S('sec:leafresults')} ({F('fig:leafmisdirection')}), "
     f"{S('sec:combinedresults')} and {S('sec:crossdataset')} ({F('fig:crossdataset')}b and {F('fig:variancetargeting')}); "
     f"discussion in {S('Vertical undertraining')}."),
    (r"Persistence sweep: five values of $\tau$ under five corruption schedules", "R4.7",
     f"{S('sec:persistenceresults')}; {F('fig:persistencesweep')} (new); {F('fig:selfheal')} (regenerated)."),
    ("Implementation details, honest-training check against XGBoost, code release, status of the platform port", "R4.8",
     f"{S('sec:impl', 'sec:xgbcheck')}; {T('tab:xgbcheck')} (new); release statement and limitations in {S('Implementation scope')}."),
    (r"One shared protocol for every experiment; paired differences with 95\% intervals", "R4.10",
     f"{S('sec:trainconfig')} with {T('tab:config')} (new); {S('sec:statmethod')}; every results figure; "
     f"{T('tab:radius', 'tab:summary')}."),
    ("Reframed contribution, abstract, discussion and conclusion", "R2.1, R4.2",
     f"Abstract; {S('sec:intro', 'sec:discussion', 'sec:conclusion')}; {T('tab:summary')} (regenerated)."),
    ("Paillier overflow events and reruns disclosed", "disclosure", f"{S('Paillier encoding')}."),
    ("Reference added: Wu et al., arXiv:2502.08160", "R2.4",
     f"{S('Gradient boosting and federated histograms')}, reference [{CN['wu2025vflpractice']}]."),
]
WHERE = (r"""\section*{Where to find each addition}
Every addition listed below is marked in blue in the revised manuscript, including the captions of the figures and tables concerned.

\small
\begin{longtable}{>{\raggedright\arraybackslash}p{4.1cm}>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{9.1cm}}
\toprule
\textbf{Addition} & \textbf{Points} & \textbf{Where in the revised manuscript} \\
\midrule
\endhead
""" + "".join(a + " & " + p + " & " + w + " \\\\\n\\addlinespace\n" for a, p, w in ROWS) + r"""\bottomrule
\end{longtable}
\normalsize

\textbf{Figures.} New: """ + F(*FIG_NEW) + r""" (reproduced below). Regenerated under the shared protocol, with updated captions: """ + F(*FIG_REGEN) + r""". Unchanged: """ + F(*FIG_SAME) + r""".
\textbf{Tables.} New: """ + T(*TAB_NEW) + r""". Changed: """ + T(*TAB_CHANGED) + r""" (new rows in Table~""" + N['tab:threatmodel'] + r"""; regenerated numbers in Table~""" + N['tab:summary'] + r"""). Unchanged: """ + T(*TAB_SAME) + r""".

""")

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\newcommand{\point}[1]{\par\medskip\noindent\textbf{\textcolor{blue!50!black}{#1}}\par}
\newcommand{\reply}{\noindent\textbf{Response.}\ }
% a new manuscript figure with its blue caption: file, figure number, section number, caption text
\newcommand{\letterfig}[4]{\par\medskip\begin{center}\begin{minipage}{0.97\textwidth}\centering
\includegraphics[width=\linewidth]{#1}\par\smallskip
\raggedright{\color{blue}\small\textbf{Figure~#2.} #4\ \emph{(New figure, Section~#3 of the revised manuscript.)}}
\end{minipage}\end{center}\par}

\title{Response to Reviewers\\[2pt]\large Split and Leaf Hijacking: Integrity Attacks on Federated Gradient-Boosted Trees}
\date{}
\begin{document}
\maketitle

We thank both reviewers for careful and constructive reports. Every new or changed passage in the revised manuscript is shown in blue, and so is the caption (including its ``Figure~$n$'' label) of every new or regenerated figure and table. Section, figure and table numbers in this letter are those of the revised manuscript. The section ``Where to find each addition'' lists every addition with its location, and the four new figures are reproduced below at the replies they support. We ran new experiments for every point that asked for them. Where an experiment weakened a claim of the original submission, we say so below and changed the claim.

\textbf{Changes that affect the conclusions.}
\begin{itemize}
\item \textbf{Attack radius corrected.} The ``root-only'' default attack of the original submission corrupted the root and its two children. We redefined the radius (Section~4.1), added the radius sweep (""" + T('tab:radius') + r""" and """ + F('fig:attackdepth') + r""") and corrected every statement. The headline loss is unchanged (""" + f"{n(z['loss'])}" + r""" in the matched design, against $0.0087$ before).
\item \textbf{One protocol for every experiment.} Every experiment now uses the same train-test split, subsample, client partition, round count and depth per dataset (Section~5.2, """ + T('tab:config') + r"""). All results figures were regenerated. All effects are paired differences with 95\% intervals over ten seeds (HFL and the Adult VFL experiments) or five seeds (the other VFL experiments).
\item \textbf{All VFL experiments run at convergence.} We reran every VFL experiment at 20 boosting rounds. Several claims did not survive and were withdrawn: the local-variance target heuristic no longer helps, VFL injection-size saturation holds on Adult only, and leaf hijacking has a threshold and inverts predictions instead of degrading gradually.
\item \textbf{The VFL ``backdoor'' is mostly a baseline effect.} The same test-time manipulation flips """ + f"{100 * E['adult']['honest_flip']:.1f}" + r"""\% of the selected predictions of an honest Adult model. The planted trigger adds """ + f"{100 * min(ex[p]['excess'][0] for p in ex):.0f}--{100 * max(ex[p]['excess'][0] for p in ex):.0f}" + r""" points, nothing measurable on Heart Disease, and nothing at all on Credit Default.
\item \textbf{The one-cell injection is weaker than the implemented attack}: """ + f"{n(v['full_one_cell']['loss'])} against {n(v['full_zero']['loss'])}" + r""". The original submission suggested the two were equivalent.
\end{itemize}

""" + WHERE + r"""\section*{Reviewer 2}

\point{R2.1 --- The threat model is permissive, so single-attacker sufficiency is expected.}
\reply We agree and no longer present the result as tree-specific. The abstract, the contributions and the conclusion attribute the tree-specific contribution to the finite, closed-form split flip margin. We also evaluate attackers with less knowledge (Section~6.3). An attacker that sees only its own report costs """ + L([v['limited_x1.5']['loss'], *v['limited_x1.5']['ci']]) + r""" against """ + L([v['full_zero']['loss'], *v['full_zero']['ci']]) + r""" for the full-knowledge attacker, about a quarter. An attacker that also reads the trees released after each round costs """ + L([RL['released_x1.5']['loss'], *RL['released_x1.5']['ci']]) + r""" and wins the target split less often, so we do not claim that released trees help. The full-knowledge result is now framed as a worst case.

\point{R2.2 --- Stealthiness is weak; evaluate realistic integrity checks and bounded reports.}
\reply Section~6.3 adds three integrity checks (cross-feature conservation, cell magnitude and $L_1$ norm) with a measured false-alarm rate on honest clients (""" + pc(v['full_zero']['fpr_any']) + r""" for the last two, none for conservation). We built a conservation-preserving attack that places its compensation on the same feature, on the other side of the split. It keeps target control (""" + pc(v['cons_pair']['hit']) + r""" of attacked nodes, against """ + pc(v['full_one_cell']['hit']) + r""" for the one-cell attack), costs """ + n(v['cons_pair']['loss']) + r""", and is never flagged by the conservation check. The magnitude and norm checks flag """ + pc(v['cons_pair']['tpr_mag']) + r""" and """ + pc(v['cons_pair']['tpr_l1']) + r""" of its reports. Conservation checking therefore does not prevent targeted split hijacking, which reverses a conclusion of the original submission. For bounded reports we cap each cell at $\rho$ times the client's largest honest cell. At $\rho=0.05$ the attack disappears for up to four colluders, and at $\rho=0.1$ to $0.25$ target control rises with the number of colluders (""" + f"{pc(b['rho=0.25|k=1']['hit'])} to {pc(b['rho=0.25|k=3']['hit'])}" + r""" from one to three colluders at $\rho=0.25$), as Corollary~1 predicts. The detector results are in """ + F('fig:detectors') + r""" and the bounded-report results in """ + F('fig:bounded') + r""", both new.
""" + FIG('fig:detectors', 'figR3_detectors.pdf', 'sec:realistic') + FIG('fig:bounded', 'figR2_bounded_reports.pdf', 'sec:realistic') + r"""

\point{R2.3 --- The VFL ``backdoor'' is weak and should be carefully qualified.}
\reply We renamed it \emph{label-free prediction disruption} throughout (Sections~""" + _names(['sec:intro', 'Related work', 'sec:splithijack', 'sec:backdoorresults']).replace('and~', 'and ') + r""", Algorithm~""" + N['alg:backdoor'] + r""", """ + F('fig:backdoortradeoff') + r"""). We report directional rates, clean accuracy, the disagreement on untouched inputs and, as Reviewer 4 asked, a false-trigger rate: the same manipulation applied to an honest model. That baseline is large. On Adult the planted trigger adds """ + f"{100 * min(ex[p]['excess'][0] for p in ex):.0f}--{100 * max(ex[p]['excess'][0] for p in ex):.0f}" + r""" points to a """ + f"{100 * E['adult']['honest_flip']:.1f}" + r"""\% baseline, with no measurable AUC cost. On Heart Disease the excess is not measurable, and on Credit Default the flip rate equals the honest model's for every seed. The mechanism has no fixed source or target class. We withdrew the earlier figure of $12.6\%$ at $12.8\%$ cost, because that experiment was undertrained and did not measure the baseline.

\point{R2.4 --- Missing reference.}
\reply We added ``Vertical Federated Learning in Practice: The Good, the Bad, and the Ugly'' (Wu et al., arXiv:2502.08160), cited in the VFL background.

\section*{Reviewer 4}

\point{R4.1 --- The adaptive attacker knows the honest aggregate; evaluate a limited-knowledge attacker.}
\reply Done (Section~6.3). Two limited-knowledge attackers were built. The first sees only its own histogram and extrapolates the aggregate as $K$ times its own report (""" + L([v['limited_x1.5']['loss'], *v['limited_x1.5']['ci']]) + r""" at multiplier $1.5$, target control """ + pc(v['limited_x1.5']['hit']) + r"""). The second also reads, at the same node, the split chosen by the previous tree, which is the ``previously released global-tree information'' that the reviewer suggested (""" + L([RL['released_x1.5']['loss'], *RL['released_x1.5']['ci']]) + r""", target control """ + pc(RL['released_x1.5']['hit']) + r"""). The gap to the full-knowledge attacker is quantified and is large (""" + F('fig:knowledgebudget') + r""", new).
""" + FIG('fig:knowledgebudget', 'figR1_knowledge_budget.pdf', 'sec:realistic') + r"""

\point{R4.2 --- Single-attacker sufficiency depends on unrestricted reports; evaluate bounded reports.}
\reply The contribution is now attributed to the margin (abstract, introduction, Section~4.3.1, conclusion). The bounded case is evaluated in Section~6.3 (see R2.2).

\point{R4.3 --- The default attack violates conservation; build a conservation-preserving targeted attack or narrow the claim.}
\reply We built the attack (see R2.2). The earlier uniform-spread construction lost control because it spread the compensation over the other features, and the manuscript now says so.

\point{R4.4 --- Invariance assumes centralized bin edges.}
\reply We implemented federated binning with merged local quantile sketches (Section~6.1). On a fixed split with six partitions the runner-up margin is identical with centralized edges ($10{,}639.8$) but takes the values $1{,}888.4$ and $10{,}468.0$ with sketch edges, and for two of six partitions no single-cell injection can flip the runner-up. Invariance therefore holds only approximately, and the abstract and Section~""" + NB.heading_number("Properties of the margin") + r""" say so. Honest accuracy is unchanged ($0.9098$ against $0.9103$) and the attack survives (root-only paired loss $0.0031$ with sketches against $0.0025$ with centralized edges).

\point{R4.5 --- Terminology of the VFL backdoor.}
\reply See R2.3. The directional success rates, false-trigger rate and clean performance are in Section~6.6 (""" + F('fig:backdoortradeoff') + r"""). We did not find a fixed source-to-target mapping and the trigger's effect over the honest baseline is small, so we do not call it a backdoor.

\point{R4.6 --- The VFL experiments use an undertrained model.}
\reply We reran every VFL experiment at 20 rounds, where the Adult baseline has plateaued: the ciphertext-scale sweep, leaf hijacking, the combined attack, prediction disruption, target selection and the federation-size check, on all three datasets. On Adult the scale attack costs """ + L(A['10.0']['loss']) + r""" at scale 10, against $0.0764$ at three rounds. The headline Adult experiments use ten seeds. The undertrained numbers remain, labelled as such. The reruns are in """ + S('sec:saturationresults') + r""" (""" + F('fig:saturation') + r"""b), """ + S('sec:radiusresults') + r""" (""" + F('fig:federationscaling') + r"""b), """ + S('sec:leafresults') + r""" (""" + F('fig:leafmisdirection') + r"""), """ + S('sec:combinedresults') + r""" and """ + S('sec:crossdataset') + r""" (""" + F('fig:crossdataset') + r"""b and """ + F('fig:variancetargeting') + r"""), and Section~""" + N['sec:baselinequality'] + r""" justifies the round count from the honest convergence curves.

\point{R4.7 --- Persistence is under-sampled; test intermediate ratios and schedules.}
\reply Done (Section~6.5). We corrupt $\tau\in\{0.1,0.25,0.5,0.75,1\}$ of the rounds under contiguous-early, contiguous-middle, contiguous-late, evenly spaced and random schedules. Damage grows with $\tau$ ($\le0.0006$ at $0.1$ up to $0.0088$ at $1$) and depends little on the schedule (""" + F('fig:persistencesweep') + r""", new).
""" + FIG('fig:persistencesweep', 'figR4_persistence.pdf', 'sec:persistenceresults') + r"""

\point{R4.8 --- Reproducibility and external validity.}
\reply \textbf{Release.} The code, seeds, scripts and result files are released at \url{""" + REPO_URL + r"""}, and we name the reference system (FedTree v1.0.5). \textbf{Port to an established platform: not done.} FedTree exposes no per-client hook for a forged report without patching its C++ source, and we judged that effort out of scope. As a partial substitute we added the implementation details (Section~5.6) and an honest-training check against XGBoost's histogram method on the same data (Section~5.7, """ + T('tab:xgbcheck') + r""", new). The first-tree root split agrees on every seed for Adult and Credit Default and on """ + pc(XG['heart']['root_feature_agree']) + r""" of seeds for Heart Disease. XGBoost reaches a higher test AUC by """ + f"{XG['adult']['xgb_minus_ours']:.3f}, {XG['heart']['xgb_minus_ours']:.3f} and {XG['credit']['xgb_minus_ours']:.3f}" + r""" on Adult, Heart Disease and Credit Default, and we could not identify the cause. The check supports the honest baseline. It does not show that the attacks work on a production system, and the manuscript says so.

\point{R4.9 --- Baselines and budgets should be matched.}
\reply Section~6.3 adds baselines with the same $L_1$ deviation as the one-cell injection (""" + F('fig:knowledgebudget') + r""", grey bars). Random noise costs """ + L([v['matched_noise']['loss'], *v['matched_noise']['ci']]) + r""" and the same deviation in a random cell costs """ + L([v['matched_random_cell']['loss'], *v['matched_random_cell']['ci']]) + r""", against """ + n(v['full_one_cell']['loss']) + r""" for the split-aware injection. Client-level label flipping, sized so that the client's root histogram in the first tree changes by the same amount, costs """ + L([M['label_flip']['loss'], *M['label_flip']['ci']]) + r""". Report forgery in bagging, sized to the same relative deviation, costs """ + L([M['bagging_forgery']['loss'], *M['bagging_forgery']['ci']]) + r""". None causes measurable damage. The matching is exact at the root of the first tree only, which the text states.

\point{R4.10 --- Statistical consistency across experiments.}
\reply Every experiment now follows the protocol of Section~5.2: per seed, the same subsample, train-test split and client partition for the clean and attacked runs, one round count and depth per dataset, and 95\% $t$-intervals on paired differences. All results figures were regenerated (""" + F(*FIG_REGEN) + r""") and the new ones follow the same protocol (""" + F(*FIG_NEW) + r"""). As a consequence the honest baseline of a dataset is the same in every figure, and the old explanation of differing baselines is gone. Two experiments remain unchanged because they validate theory and do not report attack damage: the margin validation and the honest convergence curves (five seeds).

\section*{Limits and corrections we want to state plainly}
\begin{itemize}
\item The damage at the default radius is small (""" + f"{n(z['loss'])}" + r""" AUC on Adult) and is not measurable on Credit Default. Large effects need full-tree coverage.
\item VFL injection-size saturation holds on Adult only. Heart Disease keeps degrading up to scale 200 ("""+ n(Hh['200.0']['loss'][0]) + r"""), and Credit Default is not measurably affected.
\item At full radius the attacked model is a constant predictor on Adult in """ + f"{C['adult']['n_constant']}" + r""" of 10 seeds, on Credit Default in """ + f"{C['credit']['n_constant']}" + r""" of 10 and on Heart Disease in """ + f"{C['heart']['n_constant']}" + r""". The original manuscript described Adult and Credit Default as constant predictors.
\item """ + f"{len(OV['events']) + 1}" + r""" VFL tasks raised a Paillier overflow error on first execution. Reruns with fresh keys succeeded with identical results. We report the reruns, disclose the events, and cannot explain them.
\item Our honest HFL model is a little weaker than XGBoost's, and heart-disease intervals are wide.
\end{itemize}

\end{document}
"""
assert "\r" not in tex, "stray carriage return in the letter"
LISTSEP = r"(?:, | and~| and )"
_secs = {h[0] for h in NB.headings()}
_figs = {v for k, v in N.items() if k.startswith("fig:")}
_tabs = {v for k, v in N.items() if k.startswith("tab:")}
_algs = {v for k, v in N.items() if k.startswith("alg:")}
_bad = []
for word, valid, num in (("Section", _secs, r"\d+(?:\.\d+)*"), ("Figure", _figs, r"\d+"), ("Table", _tabs, r"\d+"),
                         ("Algorithm", _algs, r"\d+")):
    for m in re.finditer(word + r"s?~(" + num + r")[a-z]?(?:" + LISTSEP + num + r"[a-z]?)*", tex):
        for x in re.findall(num, m.group(0)):
            if x not in valid:
                _bad.append(f"{word} {x}")
assert not _bad, "the letter cites things the manuscript does not have: " + ", ".join(_bad)
for f in re.findall(r"\\letterfig\{([^}]+)\}", tex):
    assert (Path(__file__).resolve().parents[1] / "figures" / f).exists(), f
OUT.write_text(tex, encoding="utf-8")
print("written", len(tex), "| new figures embedded:", tex.count("\\letterfig{"))
