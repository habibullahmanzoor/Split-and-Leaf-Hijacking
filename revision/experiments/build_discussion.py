"""Rebuild Discussion, Conclusion (and abstract) of revision/paper/paper.tex from the result files."""
import json
import re
from pathlib import Path
import numpy as np
from rb_common import H, D, X, FB, n, ci

R = Path(__file__).resolve().parents[1] / "results"
P = Path(__file__).resolve().parents[1] / "paper" / "paper.tex"
REPO_URL = "https://github.com/habibullahmanzoor/Split-and-Leaf-Hijacking"
V = json.load(open(R / "u_vfl_summary.json"))
OV = json.load(open(R / "overflow_events.json"))
XG = json.load(open(R / "xgb_conformance.json"))

zero = H["count"]["adult"]["1"]
v = D["variants"]
cp = v["cons_pair"]
lk = v["limited_x1.5"]
rel = X["variants"]["released_x1.5"]
rad = H["radius"]
A, Hh, Cc = V["scale"]["adult"], V["scale"]["heart"], V["scale"]["credit"]
bd = V["bd"]["adult"]
leaf = V["leaf"]["adult"]
flips = [bd[p]["flip"] for p in ("1", "3", "5", "8", "13")]
EX = json.load(open(R / "excess_flip.json"))
exa = [EX["adult"]["plants"][p]["excess"][0] for p in ("1", "3", "5", "8", "13")]
base_a = EX["adult"]["honest_flip"]


def pc(x, d=0):
    return f"{100 * x:.{d}f}\\%"


discussion = rf"""\section{{Discussion and Limitations}}
\label{{sec:discussion}}

\subsection{{What the results establish}}

\rev{{The main result is a difference in how damage scales, and it comes with conditions. Under unrestricted HFL reports, additional attackers after the first do not expand the set of reachable aggregate histograms, and on Adult the aimed attacks cost exactly the same with one, two and three attackers ({n(zero['loss'])}, 95\% CI {ci(*zero['ci'])}). Node coverage matters far more than attacker count: the same single attacker costs {n(rad['adult']['1']['loss'])} at the default radius and collapses the model to an AUC of {rad['adult']['4']['attacked']:.4f} when it covers the whole tree. Boosting dynamics create a second distinction. A single corrupted round leaves no measurable loss, while corruption in every round accumulates, and damage tracks the number of corrupted rounds and not their position. A security evaluation based only on the Byzantine fraction can therefore miss the parameters that control damage in this setting.}}

\rev{{The conditions matter as much as the result. The damage at the default radius is small ({n(zero['loss'])} AUC on Adult), and it is not measurable on Credit Default. An attacker that sees only its own report causes about a quarter of the full-knowledge damage ({n(lk['loss'])} against {n(v['full_zero']['loss'])}), and giving it the trees released after each round does not help ({n(rel['loss'])}). Random corruption with the same deviation, and label flipping or report forgery sized to the same deviation, cause no measurable damage, so the effect comes from aiming at the split decision. The large effects need full-tree coverage together with unrestricted reports. We state these limits because they decide how much weight the result deserves in practice.}}

The Byzantine fraction is still important when reports are bounded. Corollary~\ref{{cor:bounded}} shows that a verifiable per-client limit below the split margin forces collusion\rev{{, and Section~\ref{{sec:realistic}} confirms it: with a cap of $0.1$ to $0.25$ times the largest honest cell, target control rises with the number of colluders, and a cap of $0.05$ removes the attack in the tested range}}. This suggests defenses such as range proofs, per-record sensitivity bounds, or proofs that each histogram was computed from committed samples. Robust aggregation rules that estimate a consensus update do not directly solve the problem because honest HFL requires an exact sum of disjoint contributions, not a consensus estimate.

\subsection{{Detectability and structural checks}}

The evaluated attacks favor clear causal behavior over stealth. The HFL attack injects a large multiple of the honest aggregate, and the VFL sweep uses scales as large as 200. Simple sanity checks may detect these values. For example, the active VFL party can compare a passive feature's reported total with the node total computed from the labels. In HFL, each client's bin sums should produce the same total across all features because every local sample contributes once to each feature histogram.

\rev{{Section~\ref{{sec:realistic}} shows that a conservation-preserving attack that places its compensation on the same feature keeps target control ({pc(cp['hit'])} of attacked nodes) and passes the conservation check, so this check alone is not sufficient. Magnitude and norm checks flag {pc(cp['tpr_mag'])} and {pc(cp['tpr_l1'])} of its reports at a {pc(cp['fpr_any'])} false-alarm rate on honest clients, which is weak evidence under non-IID data.}} A broader defense would require parties to prove that their reports match a committed computation without revealing raw data \cite{{xu2020verifynet}}. Monitoring reports across rounds may also help because persistence can be visible over time even when one report is plausible under non-IID sampling. Formal detection limits and full defense evaluation remain future work.

These checks all assume the server or active party can inspect a per-client report. A secure aggregation protocol that sums client contributions without revealing them individually \cite{{bonawitz2017secureagg}} would remove that option entirely, trading detectability for confidentiality. Our threat model assumes the server sees each client's report and does not need to resolve this tension, but a deployment that adds secure aggregation for privacy would also need a different, aggregate-only integrity check.

\subsection{{Implementation scope and limitations}}

The experiments use our own implementations of the two protocols. \rev{{We did not run the attacks on a named production system. FedTree v1.0.5 builds and trains in our environment, but it exposes no per-client hook for a malicious report without patching its C++ source. We compared our honest HFL training with XGBoost's histogram method (Section~\ref{{sec:xgbcheck}}). The first tree's root split agrees on every seed for Adult and Credit Default and on {pc(XG['heart']['root_feature_agree'])} of seeds for Heart Disease, but XGBoost reaches a higher test AUC by {XG['adult']['xgb_minus_ours']:.3f} on Adult, {XG['heart']['xgb_minus_ours']:.3f} on Heart Disease and {XG['credit']['xgb_minus_ours']:.3f} on Credit Default, and we did not identify the cause. Our honest model is therefore a little weaker than a tuned production model, and the damage measured here need not equal the damage on a stronger one.}} Deployed systems may include clipping, authentication, secure aggregation \cite{{bonawitz2017secureagg}}, quantile sketches, auditing, or other checks that change which attacks are possible. \rev{{We release the code, the exact seeds and the experiment scripts at \url{{{REPO_URL}}}.}}

\rev{{The main HFL simulator builds shared bin edges with centralized access to all feature values. We tested merged local quantile sketches (Section~\ref{{sec:marginresults}}). They leave honest accuracy and the attack effect essentially unchanged, but the root margin then depends on the partition, so Proposition~\ref{{prop:heterogeneity}} holds only for fixed edges.}}

\rev{{The 512-bit Paillier keys are intentionally too small for deployment. They support the required homomorphic operations, but they are not a security recommendation. Across our VFL experiments, {len(OV['events']) + 1} runs raised a Paillier overflow error on their first execution. We reran the affected tasks with fresh keys, three times each where we could identify them. Every rerun succeeded and gave identical results, so we report the rerun values, and we cannot explain the errors on first execution. The overflow is therefore not caused by the attack, but it is a reason not to treat the no-wraparound check of Section~\ref{{sec:setup}} as a proof for every key.}}

\rev{{Statistical power is limited. HFL results use ten matched seeds and VFL results use five, except for the headline Adult VFL experiments, which use ten. Heart Disease has a test set of about 60 rows, and many of its intervals include zero. We report a result as not measurable when its interval includes zero, and we do not claim that it is absent.}}

\subsection{{Coverage and tree depth}}

Most HFL comparisons attack \rev{{only the root and its two children}} of a depth-four tree. This design keeps different conditions comparable without collapsing every model to a floor value, but it understates the largest possible effect. The radius sweep shows that one attacker can cause complete failure on Adult when it can corrupt the full tree.

The separate radius and depth sweeps in Section~\ref{{sec:radiusresults}} do not prove that damage follows a general function of $r/D$. A matched design would need to test the same ratios across several depths. \rev{{Under the default radius, the client forges the histograms of the root and its two children and reports honestly below them.}} The altered \rev{{splits change}} the partition, but the attacker does not directly forge later leaf weights. Full-radius experiments repeat the split attack at more nodes and therefore represent a stronger attacker.

\subsection{{Vertical undertraining and target selection}}

\rev{{On Adult the ciphertext-scale attack costs {n(A['10.0']['loss'][0])} AUC at scale 10 (95\% CI {ci(*A['10.0']['loss'][1:])}), flat from scale 5. A model trained for only three rounds exaggerates the damage to {n(V['scale3']['10.0']['loss'][0])}. The threshold behavior does not transfer to the other datasets: Credit Default shows no measurable loss at any scale, and Heart Disease keeps degrading up to scale 200 ({n(Hh['200.0']['loss'][0])}).}}

\rev{{Blind VFL target selection remains open. The local-variance rule does not help on converged models (Section~\ref{{sec:crossdataset}}). Stronger label-free rules based on local data structure may improve targeting. Alternatively, feature commitments and range proofs may limit the attacker's choices.}}

\subsection{{\rev{{Prediction-disruption interpretation}}}}

\rev{{The VFL mechanism changes predictions in both directions, and most of its apparent effect is not caused by the planted trigger. On Adult at convergence the manipulation flips {100 * min(flips):.0f}--{100 * max(flips):.0f}\% of the selected predictions, but the same manipulation flips {100 * base_a:.1f}\% on an honest model, so the trigger adds {100 * min(exa):.0f}--{100 * max(exa):.0f} points, with no measurable AUC cost. On Heart Disease the excess is not distinguishable from zero, and on Credit Default it is exactly zero (Section~\ref{{sec:backdoorresults}}). The mechanism should not be treated as a backdoor: it has no fixed source class or target class, and its effect beyond the false-trigger baseline is small and dataset dependent. We report directional rates, the clean-input disagreement and the false-trigger baseline, and they support calling it prediction disruption.}}

\subsection{{\rev{{Dual use and responsible disclosure}}}}

\rev{{This paper describes attacks, so we state how we limited their misuse. All attacks run against our own simulators, not against a deployed system or real participants, and all data are public benchmark datasets. The strongest effects need either unrestricted histogram reports or full-tree coverage, and Section~\ref{{sec:realistic}} shows that a verifiable per-client bound removes most of the damage. We therefore present the bound, the cross-feature conservation check and the cell-magnitude check as the first mitigations for any system that sums client histograms. The released code contains the attacks, because reproducing the results requires it. We have not contacted the maintainers of a named federated tree system, because we did not test any.}}

\section{{Conclusion}}
\label{{sec:conclusion}}

\rev{{Federated gradient-boosted trees make repeated discrete decisions from aggregated histograms. Under unrestricted HFL reports, one malicious participant can reproduce any aggregate change that several colluders could create, so additional attackers do not expand the reachable aggregate histograms. The tree-specific result is that success is controlled by a finite, closed-form split flip margin, and a verifiable per-client limit below that margin restores dependence on the number of colluders. We tested the claim under limited attacker knowledge, bounded reports, integrity checks and budget-matched baselines. Over ten matched seeds, one full-knowledge attacker among five lowers Adult test AUC by {n(zero['loss'])} (95\% CI {ci(*zero['ci'])}) at the default radius. An attacker that sees only its own report causes about a quarter of this damage, a conservation-preserving attack keeps target control and passes the conservation check, and a cap of $0.05$ times the largest honest cell removes the attack. Coverage and persistence matter more than attacker count. Covering the whole tree collapses Adult to an AUC of {rad['adult']['4']['attacked']:.4f} and Credit Default to {rad['credit']['4']['attacked']:.4f}, one corrupted round leaves no measurable loss, and damage tracks the number of corrupted rounds. In VFL at convergence, ciphertext rescaling costs {n(A['10.0']['loss'][0])} AUC on Adult and has no measurable effect on Credit Default. Leaf hijacking is continuous in the misrouting probability but has a threshold near $0.25$ and inverts predictions at high probability (Adult AUC {leaf['0.75']['auc']:.2f} at $p=0.75$). A planted label-free trigger adds {100 * min(exa):.0f}--{100 * max(exa):.0f} points to the {100 * base_a:.1f}\% flip rate that the same manipulation already causes on an honest Adult model, adds nothing measurable on Heart Disease or Credit Default, costs no measurable AUC, and has no fixed target class. Overall, security analysis should follow the aggregation and decision rules of the learning protocol. For federated trees, useful guarantees must consider split margins, verifiable report bounds, node coverage, and attack duration. The Byzantine fraction alone does not describe this attack surface.}}

"""

abstract = (
    "\\rev{Hospitals and banks often hold tabular data that they cannot share. Federated learning (FL) lets them train a shared model without moving raw data, and gradient-boosted trees suit this setting. "
    "Federated tree training creates an integrity risk, less studied than privacy. At each node, the protocol combines gradient histograms and picks a split by a discrete argmax. "
    "For horizontal FL (HFL), we derive a closed-form split flip margin, invariant to repartitioning for fixed bin edges and approximately so under federated binning. "
    "We introduce split hijacking, which forges or rescales a histogram contribution, and leaf hijacking, which corrupts sample routing. "
    "Under unrestricted reports one attacker suffices; per-client bounds restore dependence on colluders. "
    f"Over ten matched seeds, one full-knowledge HFL attacker lowers AUC by ${zero['loss']:.4f}$ (95\\% CI ${zero['ci'][0]:.4f}$--${zero['ci'][1]:.4f}$). "
    "An attacker seeing only its own report reaches about a quarter of this, and a conservation-preserving variant keeps target control while passing the cross-feature check. "
    f"Full-tree coverage collapses Adult to AUC ${rad['adult']['4']['attacked']:.2f}$. "
    f"In vertical FL, ciphertext rescaling costs ${A['10.0']['loss'][0]:.3f}$ AUC at convergence, leaf hijacking inverts predictions above a misrouting threshold, "
    f"and a planted label-free trigger adds only ${100 * min(exa):.0f}$--${100 * max(exa):.0f}$ points to the ${100 * base_a:.1f}\\%$ flip rate that the manipulation causes on an honest model.}}")

s = open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")
a = s.index("\\section{Discussion and Limitations}")
b = s.index("\\section*{Acknowledgements}")
s = s[:a] + discussion + s[b:]
a0 = s.index("\\begin{abstract}") + len("\\begin{abstract}")
a1 = s.index("\\end{abstract}")
s = s[:a0] + "\n" + abstract + "\n" + s[a1:]
open(P, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
txt = re.sub(r"\\rev\{|\\%|\$|\\", "", abstract).replace("}", " ").replace("--", " ")
print("abstract words ~", len(txt.split()))
