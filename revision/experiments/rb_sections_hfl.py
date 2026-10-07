"""Whole LaTeX subsections of the HFL half of the Results section, generated from JSON."""
import numpy as np
import rb_hfl as B
from rb_common import H, D, n, ci

NAMES = {"adult": "Adult", "heart": "Heart Disease", "credit": "Credit Default"}


def sec_saturation(vfl_scale_text):
    return r"""\subsection{Split hijacking: attacker count and injection magnitude}
\label{sec:saturationresults}

""" + B.block_counts() + r"""

\rev{The VFL panel of Figure~\ref{fig:saturation} varies the injection size of one passive party and not the number of colluders. That is saturation in injection size.} """ + vfl_scale_text + r"""

""" + B.block_aggregation() + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{fig1_saturation.pdf}
\caption{\rev{Saturation under discrete split selection (paired AUC loss, 95\% CI). (a) HFL: number of malicious clients, for the implemented zeroing attack, the one-cell injection and Gaussian noise (Adult, ten seeds). (b) VFL: ciphertext scale of one passive party, at 20 rounds and at 3 rounds, where the model is undertrained (Adult). (c) Bagging with label shuffling and with report forgery. (d) The same label-shuffle attack under boosting and under bagging.}}
\label{fig:saturation}
\end{figure}

\paragraph{Cross-feature conservation.}
The default HFL forgery fails a simple conservation test because a client's bin sums should produce the same total for every feature. \rev{A construction that spreads a compensating amount evenly over the other features preserves every total but loses target control, because another inflated feature then wins the node. A construction that places the compensation on the same feature, on the other side of the split, keeps both the totals and the target (Section~\ref{sec:realistic}). Conservation checking therefore does not by itself prevent targeted split hijacking.}

\FloatBarrier
"""


def sec_realistic():
    t1, t2, t3, t4, t5 = B.block_realistic()
    return r"""\subsection{Realistic attacker conditions}
\label{sec:realistic}

""" + t1 + "\n\n" + t2 + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figR1_knowledge_budget.pdf}
\caption{\rev{Paired AUC loss (ten matched seeds, 95\% CI) for full-knowledge attacks (red), attackers that see only their own report, with and without the released trees (blue), and budget-matched baselines (grey).}}
\label{fig:knowledgebudget}
\end{figure}

""" + t3 + "\n\n" + t4 + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figR3_detectors.pdf}
\caption{\rev{Fraction of malicious reports flagged by three integrity checks. The dashed line is the false-alarm rate of the combined check on honest clients. The conservation-preserving attack passes the conservation check.}}
\label{fig:detectors}
\end{figure}

""" + t5 + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figR2_bounded_reports.pdf}
\caption{\rev{Bounded reports. Each cell of a client's report is capped at $\rho$ times the largest cell of its honest report. (a) Paired AUC loss and (b) target control against the number of colluding clients.}}
\label{fig:bounded}
\end{figure}

\FloatBarrier
"""


def radius_table():
    rows = ""
    for ds in ("adult", "heart", "credit"):
        rr = H["radius"][ds]
        for i, r in enumerate(sorted(int(x) for x in rr)):
            v = rr[str(r)]
            full = " (full)" if r == max(int(x) for x in rr) else ""
            lab = NAMES[ds] if i == 0 else ""
            lo, hi = v["ci"]
            rows += (f"\\rev{{{lab}}} & \\rev{{${r}${full}}} & \\rev{{{v['attacked']:.4f}}} & "
                     f"\\rev{{{n(v['loss'])} ({ci(lo, hi)})}} \\\\\n")
        rows += "\\midrule\n" if ds != "credit" else ""
    return (r"""\begin{table}[htbp]
\centering
\caption{\rev{Attack radius (one malicious client, ten matched seeds). Attacked test AUC and paired AUC loss with a 95\% interval. Clean AUC: Adult $""" + f"{np.mean(H['clean']['adult']):.4f}" + r"""$, Heart Disease $""" + f"{np.mean(H['clean']['heart']):.4f}" + r"""$, Credit Default $""" + f"{np.mean(H['clean']['credit']):.4f}" + r"""$.}}
\begin{tabular}{llcc}
\toprule
\rev{Dataset} & \rev{Radius $r$} & \rev{Attacked AUC} & \rev{Paired loss (95\% CI)} \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\label{tab:radius}
\end{table}
""")


def sec_radius(party_text):
    a, b, c, d = B.block_radius_fed_depth()
    return r"""\subsection{Attack radius, federation size, and tree depth}
\label{sec:radiusresults}

""" + a + "\n\n" + radius_table() + "\n" + b + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{fig12_attack_depth_scoping.pdf}
\caption{\rev{Test AUC with one malicious HFL client as the attack radius grows (ten matched seeds, mean AUC). Full-tree coverage collapses Adult to a constant predictor. Credit Default is not affected below $r=3$, and Heart Disease collapses from $r=2$.}}
\label{fig:attackdepth}
\end{figure}

""" + c + "\n\n" + party_text + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{fig14_federation_size_scaling.pdf}
\caption{\rev{Federation-size checks (paired AUC loss, 95\% CI). (a) Total number of HFL clients with one attacker (ten seeds). (b) Number of VFL passive parties with one attacker, at 20 rounds (five seeds).}}
\label{fig:federationscaling}
\end{figure}

""" + d + r"""

\FloatBarrier
"""


def sec_persistence():
    return r"""\subsection{Persistence across boosting rounds}
\label{sec:persistenceresults}

""" + B.block_persistence() + r"""

The two transient cases recover for different reasons. Early corruption causes an immediate loss, after which several honest trees fit the altered residuals and correct much of the error. Late corruption has little visible effect because the final tree fits a small residual and has limited influence on the converged ensemble. The first case reflects correction, while the second reflects low leverage. Sustained corruption receives neither protection and causes the lasting damage shown in Figure~\ref{fig:saturation}.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig2_selfheal.pdf}
\caption{\rev{Test AUC by boosting round with no attack, with the first round corrupted, and with the last round corrupted (ten matched seeds, mean and one standard deviation).}}
\label{fig:selfheal}
\end{figure}

\rev{\paragraph{Intermediate persistence and schedules.} We corrupt $m$ of the 20 rounds with $\tau\in\{0.1,0.25,0.5,0.75,1\}$ under five schedules: contiguous early, contiguous middle, contiguous late, evenly spaced, and random (Adult, ten matched seeds, default radius, implemented attack; Figure~\ref{fig:persistencesweep}). Damage grows with $\tau$. The paired loss is at most $0.0006$ at $\tau=0.1$, $0.0007$--$0.0012$ at $0.25$, $0.0019$--$0.0034$ at $0.5$, $0.0042$--$0.0055$ at $0.75$, and $0.0088$ at $\tau=1$ for every schedule. At $\tau=0.1$ four of the five schedules are indistinguishable from zero. The schedule matters little. The largest difference is at $\tau=0.5$, where an early contiguous block gives $0.0034$ ($0.0022$--$0.0045$) and a late block gives $0.0019$ ($0.0009$--$0.0029$). At the other values the intervals overlap. Final damage therefore depends mainly on the number of corrupted rounds, and we do not claim an effect of position.}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figR4_persistence.pdf}
\caption{\rev{Paired AUC loss against persistence $\tau$ for five corruption schedules (Adult, ten matched seeds, 95\% CI).}}
\label{fig:persistencesweep}
\end{figure}

\FloatBarrier
"""
