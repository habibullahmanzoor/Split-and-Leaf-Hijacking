"""VFL half of the Results section, generated from u_vfl_summary.json and trigger_specificity.json."""
import json
import numpy as np
from rb_common import R, n, ci, pc, zero_in

V = json.load(open(R / "u_vfl_summary.json"))
NAMES = {"adult": "Adult", "heart": "Heart Disease", "credit": "Credit Default"}
TS = json.load(open(R / "trigger_specificity.json"))


def L(t, d=4):
    m, lo, hi = t
    return f"{n(m, d)} (95\\% CI {ci(lo, hi, d)})"


def scale_text():
    a, h, c = V["scale"]["adult"], V["scale"]["heart"], V["scale"]["credit"]
    s3 = V["scale3"]
    return (
        f"\\rev{{On Adult at 20 boosting rounds (ten seeds, honest AUC {V['honest']['adult']['mean']:.4f}) the paired loss is {L(a['2.0']['loss'])} at scale 2, {n(a['5.0']['loss'][0])} at scale 5, {L(a['10.0']['loss'])} at scale 10 and {n(a['200.0']['loss'][0])} at scale 200. "
        "The loss grows up to scale 5 and is flat after that, which is a threshold in the injection size. "
        f"At 3 rounds, where the model is undertrained (five seeds, honest AUC {V['honest']['adult_3rounds']['mean']:.4f}) the loss is {n(s3['2.0']['loss'][0])} at every scale from 2 to 200, about {s3['2.0']['loss'][0] / a['10.0']['loss'][0]:.0f} times the converged value (the two settings use different seed sets), so a model that has not converged exaggerates the damage. "
        f"The threshold behavior is specific to Adult. On Credit Default no scale causes a measurable loss (largest {n(max(c[k]['loss'][0] for k in c))}, every interval includes zero). "
        f"On Heart Disease the loss does not saturate: it is {n(h['5.0']['loss'][0])} at scale 5, {n(h['10.0']['loss'][0])} at scale 10, {n(h['50.0']['loss'][0])} at scale 50 and {n(h['200.0']['loss'][0])} at scale 200, and only the last two intervals exclude zero. "
        "We did not investigate why Heart Disease keeps degrading with scale. The test set has about 60 rows, so these values are imprecise.}")


def party_text():
    p = V["party"]
    return (
        "\\rev{Figure~\\ref{fig:federationscaling}b divides the passive feature space among one, two or three passive parties and rescales one cell of the first party at scale 10 (Adult, 20 rounds, five seeds). "
        f"The paired loss is {L(p['1']['loss'])} in all three settings, with honest AUC {p['1']['honest']:.4f}, {p['2']['honest']:.4f} and {p['3']['honest']:.4f}. "
        "The attack depends on the feature block of the malicious party and not on how the other features are divided. This experiment does not change the number of malicious passive parties.}")


def sec_disruption():
    E = json.load(open(R / "excess_flip.json"))
    A, Hh, Cc = V["bd"]["adult"], V["bd"]["heart"], V["bd"]["credit"]
    pl = ["1", "3", "5", "8", "13"]
    flips = ", ".join(f"{100 * A[p]['flip']:.1f}\\%" for p in pl)
    dis = [A[p]["dis_nt"] for p in pl]
    maxcost = max(abs(A[p]["cost"][0]) for p in pl)
    allzero = all(zero_in(A[p]["cost"][1], A[p]["cost"][2]) for p in pl)
    ea = E["adult"]["plants"]
    he, ce = E["heart"], E["credit"]
    hb, cb = 100 * E["adult"]["honest_flip"], None

    def pts(t):
        return f"{100 * t[0]:.1f} points (95\\% CI {100 * t[1]:.1f} to {100 * t[2]:.1f})"

    ts = TS
    def mf(ds):
        v = [r["flip"] for r in ts[ds] if r["flip"] is not None]
        return 100 * float(np.mean(v)), 100 * float(np.std(v, ddof=1))
    ma, mh, mc = mf("adult"), mf("heart"), mf("credit")
    heart_all_zero = all(zero_in(he["plants"][p]["excess"][1], he["plants"][p]["excess"][2]) for p in he["plants"])
    return (
        r"""\subsection{\rev{Label-free prediction disruption}}
\label{sec:backdoorresults}

\rev{Figure~\ref{fig:backdoortradeoff} varies the number of trees that contain the same trigger split. """
        f"On Adult at 20 rounds (ten seeds, about {A['1']['n_target']:.0f} target samples per seed) the flip rate on the selected inputs is {flips} for 1, 3, 5, 8 and 13 plants. "
        f"The paired AUC cost lies between {n(min(A[p]['cost'][0] for p in pl))} and {n(max(A[p]['cost'][0] for p in pl))} for every plant count, so it is at most a few thousandths (for 13 plants, {L(A['13']['cost'])}); a negative value means that the manipulated model scored slightly higher. "
        ""
        "These flip rates cannot be read on their own, because the test-time manipulation changes some predictions even when no trigger was planted.}"
        r"""

\rev{\paragraph{Trigger specificity.} We applied the same manipulation to an honest model of the same seed, which has no planted trigger. """
        f"It flips {ma[0]:.1f}\\% ($\\pm{ma[1]:.1f}$) of the selected predictions on Adult, {mh[0]:.1f}\\% ($\\pm{mh[1]:.1f}$) on Heart Disease and {mc[0]:.1f}\\% ($\\pm{mc[1]:.1f}$) on Credit Default. "
        "This false-trigger rate is the baseline. The effect of the planted trigger is the paired excess over it. "
        f"On Adult the excess is {pts(ea['1']['excess'])} with one plant, {pts(ea['3']['excess'])} with three, {pts(ea['5']['excess'])} with five, {pts(ea['8']['excess'])} with eight and {pts(ea['13']['excess'])} with thirteen. "
        "The planted trigger therefore adds a small but measurable effect on Adult, which grows with the number of plants. Most of the flip rate in the previous paragraph is not caused by the trigger. "
        f"On Heart Disease the excess is {'not distinguishable from zero for any plant count' if heart_all_zero else 'not zero for every plant count'} (for 8 plants, {pts(he['plants']['8']['excess'])}). "
        f"On Credit Default the flip rate is identical to the honest model's for every seed, so the planted trigger has no effect there at all. "
        "A planted trigger that does nothing on one dataset and little on another is weak evidence for a backdoor.}"
        r"""

\rev{\paragraph{Direction and side effects.} """
        f"On Adult the raw flip rates are asymmetric: among target samples predicted as class 1 before the trigger, {100 * min(A[p]['a10'] for p in pl):.0f}--{100 * max(A[p]['a10'] for p in pl):.0f}\\% move to class 0, and among those predicted as class 0, {100 * min(A[p]['a01'] for p in pl):.0f}--{100 * max(A[p]['a01'] for p in pl):.0f}\\% move to class 1. "
        f"The honest baseline is asymmetric in the same way ({100 * np.mean([r['a10'] for r in ts['adult']]):.1f}\\% to class 0 and {100 * np.mean([r['a01'] for r in ts['adult']]):.1f}\\% to class 1), so the asymmetry is mostly a property of the manipulation and the model, and the excess for 13 plants is {pts(ea['13']['excess_1to0'])} toward class 0 and {pts(ea['13']['excess_0to1'])} toward class 1. "
        f"On Heart Disease the raw asymmetry points the other way ({100 * Hh['8']['a01']:.0f}\\% to class 1 and {100 * Hh['8']['a10']:.0f}\\% to class 0 at 8 plants), again close to the honest baseline ({100 * np.mean([r['a01'] for r in ts['heart']]):.0f}\\% and {100 * np.mean([r['a10'] for r in ts['heart']]):.0f}\\%). "
        f"On untouched inputs the honest and manipulated models disagree on {100 * min(dis):.1f}--{100 * max(dis):.1f}\\% of non-target test samples on Adult, and clean accuracy changes by at most {max(abs(A[p]['acc_b'] - A[p]['acc_h']) for p in pl):.4f}. "
        "Training is deterministic given the data, so this disagreement is a cost of planting the trigger and not seed noise. "
        "We conclude that the mechanism has no fixed source class or target class, and that its effect beyond the baseline is small. We therefore call it prediction disruption, and we do not claim a backdoor.}"
        r"""



\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig9_backdoor_tradeoff.pdf}
\caption{\rev{Label-free prediction disruption on Adult at 20 rounds (ten seeds). (a) Flip rate against paired AUC cost (95\% CI), labelled by the number of plants. (b) Flip rate in each direction and disagreement between the honest and manipulated models on non-target inputs. The honest model's flip rate under the same manipulation is the baseline.}}
\label{fig:backdoortradeoff}
\end{figure}

\paragraph{Comparison with label inference.}
In a five-seed experiment, a standard instance-clustering method that receives true labels for one fifth of the training set reaches $81.3\pm3.1\%$ label-inference accuracy, with individual results from about $78\%$ to $85\%$. Confidence filtering improves accuracy but reduces coverage. A minimum agreement of $70\%$ gives $84.9\%$ accuracy at $88.9\%$ coverage. A minimum agreement of $85\%$ gives $88.7\%$ accuracy at $74.8\%$ coverage. Unanimous voting gives $92.3\pm3.1\%$ accuracy at only $39\%$ coverage.

Label-inference accuracy and prediction-disruption flip rate measure different outcomes. Inference accuracy is measured over candidate labels, while flip rate is measured on selected triggered inputs. The comparison supports only one advantage: the label-free mechanism cannot inherit label-inference errors because it performs no inference. It does not make the mechanism reliable, and its effect beyond the false-trigger baseline is small.

\FloatBarrier
""")


def sec_leaf():
    a, h, c = V["leaf"]["adult"], V["leaf"]["heart"], V["leaf"]["credit"]
    pp = ["0.1", "0.25", "0.5", "0.75", "1.0"]

    def row(d):
        return ", ".join(f"{d[p]['auc']:.4f}" for p in pp)

    return (
        r"""\subsection{Leaf hijacking response curve}
\label{sec:leafresults}

\rev{Figure~\ref{fig:leafmisdirection} varies the misrouting probability $p$ at 20 rounds (Adult ten seeds, the other datasets five seeds). The test AUC at $p=0.1$, $0.25$, $0.5$, $0.75$ and $1$ is """
        f"{row(a)} on Adult (honest {a['0.0']['auc']:.4f}), {row(h)} on Heart Disease (honest {h['0.0']['auc']:.4f}) and {row(c)} on Credit Default (honest {c['0.0']['auc']:.4f}). "
        f"The curve is continuous, as Proposition~\\ref{{prop:leafcontinuity}} suggests for the expected leaf weight, but it is not gradual in the sense of rising linearly with $p$. "
        f"Misrouting up to $p=0.25$ costs little (Adult {L(a['0.25']['loss'])}, with the loss at $p=0.1$ indistinguishable from zero). The AUC then falls steeply between $p=0.25$ and $p=0.75$, and it is below chance at $p=0.75$ and $p=1$ on every dataset. "
        "An AUC below $0.5$ means that the model ranks samples in the wrong order, so the attack inverts predictions and does not merely add noise. "
        f"Full misrouting is not always the worst case: on Adult the AUC at $p=1$ ({a['1.0']['auc']:.4f}) is slightly higher than at $p=0.75$ ({a['0.75']['auc']:.4f}), and the difference is within the spread between seeds. "
        "The Heart Disease and Credit Default intervals are wide because the test sets are small. "
        "This is a different response shape from split hijacking, which saturates once the target split wins. Leaf hijacking has a threshold in $p$ below which little happens, and a steep region above it.}"
        r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig15_leaf_misdirection.pdf}
\caption{\rev{Test AUC as the leaf-misrouting probability increases from zero to one, at 20 rounds (mean and one standard deviation; Adult ten seeds, others five).}}
\label{fig:leafmisdirection}
\end{figure}

\FloatBarrier
""")


def sec_combined():
    sy = V["comb"]["synergy"]
    cells = V["comb"]["cells"]
    lines = []
    for k in ("1.5|0.15", "1.5|0.3", "3.0|0.15", "3.0|0.3"):
        he, mis = k.split("|")
        lines.append(f"$({he},{float(mis):.2f})$: {L(sy[k]['syn'])}")
    top = sy["3.0|0.3"]
    return (
        r"""\subsection{Combined split and leaf attacks}
\label{sec:combinedresults}

\rev{We combine a uniform rescaling of the passive party's histogram with a misrouting probability, on Adult at 20 rounds over five matched seeds. The honest AUC is """
        f"{cells['1.0|0.0']['auc']:.4f}. The largest tested combination (rescaling $3.0$, misrouting $0.30$) reaches {cells['3.0|0.3']['auc']:.4f}, a paired loss of {n(top['comb_loss'])}, against {n(top['split_loss'])} for the rescaling alone and {n(top['leaf_loss'])} for the misrouting alone. "
        "The effects therefore accumulate. The paired interaction statistic and its 95\\% interval for each combination are "
        + "; ".join(lines) + ". "
        "Every interval includes zero. With five seeds and wide intervals the supported conclusion is that we do not detect synergy. This is not evidence that the attacks are exactly additive, and an equivalence test with a defined tolerance and more seeds would be needed to support that claim.}"
        "\n")


def sec_cross(he_table_note=""):
    t = V["tgt"]
    s = V["scale"]
    hon = V["honest"]
    return (
        r"""\subsection{Cross-dataset transfer and blind target selection}
\label{sec:crossdataset}

\rev{Figure~\ref{fig:crossdataset} applies the default attacks to all three datasets under one protocol. In HFL (panel a), the aimed attack at the default radius costs """
        f"{n(0.0088)} on Adult, and on Heart Disease and Credit Default the effect is smaller than the spread between seeds (Section~\\ref{{sec:saturationresults}}). "
        f"In VFL (panel b, 20 rounds) the loss at scale 10 is {L(s['adult']['10.0']['loss'])} on Adult, {L(s['heart']['10.0']['loss'])} on Heart Disease and {L(s['credit']['10.0']['loss'])} on Credit Default. "
        "Heart Disease keeps degrading at larger scales (Section~\\ref{sec:saturationresults}), and Credit Default is not measurably affected at any scale. "
        "Credit Default looks resistant at the default settings for two different reasons. In HFL the default radius is too small, and a larger radius collapses the model (Section~\\ref{sec:radiusresults}). In VFL the blind attacker uses a fixed feature, which may be a poor target.}"
        r"""

\rev{We test a label-free VFL heuristic that selects the passive feature with the largest variance in its locally binned values. It uses no labels, gradients, or information from another party. Figure~\ref{fig:variancetargeting} compares it with the fixed first feature at scale 10 and 20 rounds. """
        f"On Adult (ten seeds) the paired loss is {L(t['adult|variance']['loss'])} with the heuristic and {L(t['adult|fixed']['loss'])} with the fixed feature, so the heuristic does not help and may be worse. "
        f"On Heart Disease (five seeds) the losses are {L(t['heart|variance']['loss'])} and {L(t['heart|fixed']['loss'])}. On Credit Default they are {L(t['credit|variance']['loss'])} and {L(t['credit|fixed']['loss'])}. All Heart Disease and Credit Default intervals include zero. "
        "The heuristic is therefore not a reliable target-selection rule on converged models, and finding a blind rule that works remains open.}"
        r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig7_cross_dataset.pdf}
\caption{\rev{Cross-dataset evaluation. (a) HFL: paired AUC loss against the number of malicious clients at the default radius (ten seeds). (b) VFL: paired AUC loss against the ciphertext scale at 20 rounds (Adult ten seeds, others five).}}
\label{fig:crossdataset}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{fig13_variance_targeting.pdf}
\caption{\rev{Fixed-feature and maximum-variance target selection at scale 10 and 20 rounds (paired AUC loss, 95\% CI; Adult ten seeds, others five). The heuristic does not reliably improve on the fixed feature.}}
\label{fig:variancetargeting}
\end{figure}

\FloatBarrier
""")


def summary_table():
    return r"""\subsection{Summary of empirical findings}

Table~\ref{tab:summary} states what each experiment supports and where its scope ends.

\begin{table}[htbp]
\centering
\caption{\rev{Summary of the findings and the scope of each claim. Effects are paired AUC losses over matched seeds.}}
\begin{tabular}{p{4.0cm}p{9.8cm}}
\toprule
\rev{Finding} & \rev{Scope} \\
\midrule
\rev{Attacker-count saturation (HFL)} & \rev{Exact for the aimed attacks on Adult. Nearly flat on Heart Disease with wide intervals. Not measurable on Credit Default at the default radius.} \\
\rev{Injection-size saturation (VFL)} & \rev{Flat from scale 5 on Adult. Not saturating on Heart Disease. No measurable effect on Credit Default.} \\
\rev{Coverage} & \rev{Small effect up to $r=2$ on Adult, collapse at $r\geq3$. Credit Default collapses only at $r\geq3$.} \\
\rev{Transient corruption} & \rev{One corrupted round has no measurable loss on Adult and Credit Default. Damage tracks the number of corrupted rounds, not their position.} \\
\rev{Limited knowledge} & \rev{About a quarter of the full-knowledge damage. Released trees do not help.} \\
\rev{Conservation-preserving attack} & \rev{Keeps target control and passes the conservation check. Magnitude and norm checks flag it at about three times the honest false-alarm rate.} \\
\rev{Bounded reports} & \rev{A cap at $0.05$ of the largest honest cell removes the attack. Caps of $0.1$ to $0.25$ restore dependence on colluder count.} \\
\rev{Leaf hijacking} & \rev{Continuous in the misrouting probability, with a threshold near $0.25$ and inverted predictions at high $p$.} \\
\rev{Combined attack} & \rev{No detectable synergy with five seeds.} \\
\rev{Label-free prediction disruption} & \rev{On Adult the planted trigger adds $2$--$6$ points to the flip rate that the same manipulation causes on an honest model, with no measurable AUC cost. It adds nothing measurable on Heart Disease or Credit Default. No fixed source or target class.} \\
\rev{Blind target selection} & \rev{The local-variance heuristic does not help on converged models.} \\
\bottomrule
\end{tabular}
\label{tab:summary}
\end{table}

\FloatBarrier
"""
