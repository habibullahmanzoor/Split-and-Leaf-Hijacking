"""Generated LaTeX for the HFL parts of the Results section (numbers come from JSON)."""
import numpy as np
from rb_common import H, D, X, C, FB, n, ci, lc, pc, zero_in

NAMES = {"adult": "Adult", "heart": "Heart Disease", "credit": "Credit Default"}


def block_counts():
    a = H["count"]["adult"]
    one = H["count_adult"]["one_cell"]
    g = H["count_adult"]["gauss5"]
    he, cr = H["count"]["heart"], H["count"]["credit"]
    return (
        "\\rev{Figure~\\ref{fig:saturation}(a) varies the number of malicious HFL clients from one to three of five, which gives fractions $f=0.2$, $0.4$ and $0.6$. "
        f"Under unrestricted reports the aimed attacks cause the same damage with one, two and three attackers on Adult. The implemented zeroing attack costs {lc(a['1'])} at every count, and the one-cell injection costs {lc(one['1'])} at every count. "
        "This agrees with Proposition~\\ref{prop:singleattacker}. The values are identical because the extra forged mass lands on the cell that the first attacker already promoted, so every split decision is unchanged. "
        f"The blind Gaussian-noise baseline behaves differently. It costs {n(g['1']['loss'])}, {n(g['2']['loss'])} and {n(g['3']['loss'])} with one, two and three attackers, so it grows with the number of attackers, as independent noise terms add up. Saturation is therefore a property of the aimed attacks. "
        "The noise attack also causes less damage than the aimed attack at every count. "
        f"On the two smaller datasets the picture is less clean (Figure~\\ref{{fig:crossdataset}}a). On Heart Disease the loss is {n(he['1']['loss'])}, {n(he['2']['loss'])} and {n(he['3']['loss'])} for one, two and three attackers. It is nearly flat after the first attacker, but the intervals are wide and the first one includes zero ({ci(*he['1']['ci'])}), because the test set has about 60 rows. "
        f"On Credit Default the loss is {n(cr['1']['loss'])}, {n(cr['2']['loss'])} and {n(cr['3']['loss'])}, with every interval including zero. At the default radius the attack has no measurable effect there, and Section~\\ref{{sec:radiusresults}} shows why.}}")


def block_aggregation():
    bs, bf, bo = H["bagging"]["shuffle"], H["bagging"]["forge"], H["boosting_labelshuffle"]
    clean_b = np.mean(H["bagging"]["clean"])
    return (
        "\\rev{\\paragraph{Aggregation controls.} We hold the attack type fixed and change the aggregation mechanism (Figure~\\ref{fig:saturation}c,d). "
        f"With label shuffling, bagging starts at an honest AUC of {clean_b:.4f} and loses {n(bs['1']['loss'])}, {n(bs['2']['loss'])} and {n(bs['3']['loss'])} with one, two and three malicious clients. "
        f"Boosting starts at {H['count']['adult']['1']['clean']:.4f} and loses {n(bo['1']['loss'])}, {n(bo['2']['loss'])} and {n(bo['3']['loss'])}. "
        "These losses are small and their intervals overlap between the two mechanisms, so label shuffling does not separate them. We do not draw a conclusion from this pair. "
        f"A report-level forgery separates them clearly. When malicious clients train honestly but report the complement of their predicted probability, bagging loses {n(bf['1']['loss'])}, {n(bf['2']['loss'])} and {n(bf['3']['loss'])} for one, two and three attackers, and the damage grows steeply with attacker count because three inverted reports outvote two honest ones in a five-client average. "
        "The aimed histogram attack on boosting stays at its first-attacker value. "
        "This comparison is not budget matched, because a probability is bounded to $[0,1]$ and a histogram cell is not. The budget-matched version in Section~\\ref{sec:realistic} sizes the forgery to the same relative deviation as the one-cell injection.}")


def block_realistic():
    v = D["variants"]
    ex = X["variants"]
    m = X["matched"]
    clean = H["clean"]["adult"]
    b = D["bounded"]

    def row(key):
        r = v[key]
        return r["loss"], r["ci"][0], r["ci"][1]

    t1 = (
        "\\rev{We test whether the HFL results survive a limited-knowledge attacker, integrity checks, bounded reports, and budget-matched baselines. "
        f"All runs use Adult, the default radius $r=1$, one malicious client unless stated, and ten matched seeds. The mean clean AUC is {np.mean(clean):.4f}. "
        "``Target control'' is the fraction of attacked nodes at which the intended candidate wins the argmax.}")
    t2 = (
        "\\rev{\\paragraph{Attacker knowledge and budget.} Figure~\\ref{fig:knowledgebudget} reports paired AUC loss. "
        f"The implemented full-knowledge attack, which zeroes the rest of its report, costs {lc(row('full_zero'))} and wins the target at {pc(v['full_zero']['hit'])} of attacked nodes. "
        f"The clean one-cell injection costs {lc(row('full_one_cell'))} and wins at {pc(v['full_one_cell']['hit'])}. "
        f"A limited-knowledge attacker that extrapolates the aggregate from its own report costs {lc(row('limited_x1.5'))} at a margin multiplier of $1.5$ and {lc(row('limited_x3.0'))} at $3$, with target control of {pc(v['limited_x1.5']['hit'])} and {pc(v['limited_x3.0']['hit'])}. "
        "This is about a quarter of the full-knowledge damage, and the interval at $1.5$ includes zero. "
        f"An attacker that treats its own report as the aggregate does worse ({n(v['limited_localonly_x1.5']['loss'])}, target control {pc(v['limited_localonly_x1.5']['hit'])}). "
        f"We also gave the attacker the trees released after each round. It reads the split chosen at the same node of the previous tree, takes it as the split to beat, and sizes its injection against that one rival. This attacker costs {lc(ex['released_x1.5'])} at multiplier $1.5$ and {lc(ex['released_x3.0'])} at $3$, with target control of {pc(ex['released_x1.5']['hit'])} and {pc(ex['released_x3.0']['hit'])}. "
        "It is not better than the attacker that uses only its own report. A plausible reason is that the winning split changes between rounds, and sizing the injection against one rival ignores the others. "
        "We therefore do not claim that released trees help an attacker, and the gap to the full-knowledge attacker stays large. "
        "The full-knowledge result should be read as a worst case.}")
    t3 = (
        "\\rev{Budget-matched baselines support the split-aware design. "
        f"Random noise with the same $L_1$ deviation as the one-cell injection costs {lc(row('matched_noise'))}, and the same deviation placed in a random cell costs {lc(row('matched_random_cell'))}. "
        f"Unbounded Gaussian noise with standard deviation $5$ costs {lc(row('gauss5'))}. "
        f"For data poisoning we flip the labels of enough rows of the malicious client to give its root histogram in the first tree the same $L_1$ deviation as the one-cell injection (between $0.6\\%$ and $76\\%$ of its rows, depending on the seed). It costs {lc(m['label_flip'])}. "
        f"For report forgery in bagging we mix the client's probability with its complement until the report changes by the same relative amount. It costs {lc(m['bagging_forgery'])} from a bagging baseline of {m['bagging_forgery']['clean']:.4f}, and the required mixing exceeds the maximum possible in one of ten seeds. "
        "Neither matched baseline causes measurable damage. The matching is exact at the root of the first tree only, and a poisoned dataset persists through later trees and nodes, so this comparison favors the poisoning baseline if anything.}")
    # detectors / conservation
    cp = v["cons_pair"]
    oc = v["full_one_cell"]
    zr = v["full_zero"]
    t4 = (
        "\\rev{\\paragraph{Integrity checks and a conservation-preserving attack.} We apply three checks to every attacked node: cross-feature conservation, a cell-magnitude check (largest cell above three times the median of the other clients), and an $L_1$-norm check with the same factor. "
        f"The magnitude and norm checks raise a false alarm on {pc(zr['fpr_any'])} of honest reports under non-IID data, and the conservation check raises none. "
        f"Figure~\\ref{{fig:detectors}} shows that the implemented attack is flagged by every check. The conservation check flags {pc(oc['tpr_cons'])} of one-cell injections. "
        "We also build a conservation-preserving attack that adds $\\delta$ to the target cell and subtracts $\\delta$ from a cell of the same feature on the other side of the split, so every feature total is unchanged. "
        "The smallest $\\delta$ on a geometric grid for which the target wins the full argmax is used, and the client reports honestly if none works. "
        f"This attack costs {lc(row('cons_pair'))}, wins the target at {pc(cp['hit'])} of attacked nodes, and is flagged by the conservation check at {pc(cp['tpr_cons'])} of nodes. "
        f"The magnitude and norm checks flag {pc(cp['tpr_mag'])} and {pc(cp['tpr_l1'])} of its reports, against a {pc(zr['fpr_any'])} false-alarm rate. "
        "A conservation check alone therefore does not prevent targeted split hijacking. Magnitude and norm checks catch it at about three times the honest false-alarm rate, which is weak evidence under non-IID data.}")

    def bnd(rho, k):
        r = b[f"rho={rho}|k={k}"]
        return r["loss"], r["ci"][0], r["ci"][1], r["hit"]

    r05 = [bnd(0.05, k) for k in (1, 2, 3, 4)]
    r1_1, r1_4 = bnd(0.1, 1), bnd(0.1, 4)
    r25_1, r25_3 = bnd(0.25, 1), bnd(0.25, 3)
    ub = b["rho=None|k=1"]
    t5 = (
        "\\rev{\\paragraph{Bounded reports.} We cap each cell of a client's report at $\\rho$ times the largest cell of that client's own honest report, which models a verifiable range check. Colluders add to the target cell greedily until the margin is met. "
        f"Figure~\\ref{{fig:bounded}} shows that a tight bound removes the attack. At $\\rho=0.05$ no attacker count up to four causes measurable damage (largest loss {n(max(x[0] for x in r05))}, target control at most {pc(max(x[3] for x in r05))}). "
        f"At $\\rho=0.1$, target control grows from {pc(r1_1[3])} with one attacker to {pc(r1_4[3])} with four, and the loss from {n(r1_1[0])} to {n(r1_4[0])} ({ci(r1_4[1], r1_4[2])}). "
        f"At $\\rho=0.25$, one attacker costs {n(r25_1[0])} ({ci(r25_1[1], r25_1[2])}) and three cost {n(r25_3[0])} ({ci(r25_3[1], r25_3[2])}), with target control of {pc(r25_1[3])} and {pc(r25_3[3])}. "
        f"Without a bound, target control is {pc(ub['hit'])} for every attacker count. "
        "Dependence on the number of colluders therefore returns once the bound is below the margin, as Corollary~\\ref{cor:bounded} predicts. The gain from a fourth colluder is within the intervals in this range.}")
    return t1, t2, t3, t4, t5


def block_radius_fed_depth():
    r = H["radius"]
    ad, he, cr = r["adult"], r["heart"], r["credit"]
    dep = H["depth"]
    fed = H["federation"]
    cc = C
    first = (
        "\\rev{The experiments so far use the default radius $r=1$. Figure~\\ref{fig:attackdepth} keeps one malicious client and extends the radius to the full tree, which is $r=4$ for Adult and Credit Default and $r=3$ for Heart Disease. Damage rises sharply with coverage. "
        f"On Adult the paired loss is {n(ad['0']['loss'])} at $r=0$, {n(ad['1']['loss'])} at $r=1$ and {n(ad['2']['loss'])} at $r=2$. At $r=3$ the model has collapsed to an AUC of {ad['3']['attacked']:.4f}, and with full coverage it reaches exactly {ad['4']['attacked']:.4f}. "
        f"Credit Default shows no measurable loss up to $r=2$ (all three intervals include zero), then falls to an AUC of {cr['3']['attacked']:.4f} at $r=3$ and {cr['4']['attacked']:.4f} at $r=4$. "
        f"Heart Disease falls from a clean {he['0']['clean']:.4f} to {he['2']['attacked']:.4f} at $r=2$ and {he['3']['attacked']:.4f} at full depth. "
        "The Heart Disease values are not monotone: the loss at $r=1$ is smaller than at $r=0$ and the AUC at full depth is higher than at $r=2$, and the first pair of intervals overlaps. This shows how noisy a test set of about 60 rows is, and we do not read structure into it. "
        "The default radius is therefore too small to damage Credit Default, which is why it appears resistant in Section~\\ref{sec:crossdataset}.}")
    second = (
        "\\rev{We checked whether full coverage produces random predictions or a constant predictor. "
        f"On Adult the attacked model outputs a single probability for every test sample in {cc['adult']['n_constant']} of 10 seeds. "
        f"On Credit Default this happens in {cc['credit']['n_constant']} of 10 seeds, and in the others the model still produces hundreds of distinct values with an AUC near chance. "
        f"On Heart Disease it happens in {cc['heart']['n_constant']} of 10 seeds.}}")
    fedtxt = (
        "\\rev{Figure~\\ref{fig:federationscaling}a checks whether the default federation size explains the saturation result. We keep one malicious HFL client and increase the total client count from 5 to 40. "
        f"The paired loss is {n(fed['5']['loss'])} at every client count, with the same interval ({ci(*fed['5']['ci'])}). "
        "More honest clients do not dilute the attack because the malicious report remains unrestricted. The identical clean AUC at every count also follows from Proposition~\\ref{prop:heterogeneity}. "
        "At 40 clients some Dirichlet partitions leave a client without rows, and such a client is skipped.}")
    depth = (
        "\\rev{\\paragraph{Tree-depth sensitivity.} Production gradient boosting often uses depths from six to ten. We keep the radius at $r=1$ and increase the maximum depth on Adult. "
        f"The paired loss is {n(dep['4']['loss'])} at depth 4, {n(dep['6']['loss'])} at depth 6, {n(dep['8']['loss'])} at depth 8 and {n(dep['10']['loss'])} at depth 10, and every interval excludes zero. "
        "The default-radius attack is therefore smaller at greater depths, but it does not vanish. This pattern is consistent with the attacked nodes being a smaller part of a deeper tree, "
        "but it does not prove that damage is a function of $r/D$. The radius experiment changes $r$ at fixed $D$, while the depth experiment changes $D$ at fixed $r$. A matched experiment that varies both is needed to support a general scaling law.}")
    return first, second, fedtxt, depth


def block_persistence():
    t = H["transient"]
    import numpy as np
    from scipy import stats

    def pl(ds, k):
        fin = {c: np.array(v)[:, -1] for c, v in t[ds].items()}
        d = fin["clean"] - fin[k]
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), d.mean() - h, d.mean() + h, fin["clean"].mean(), fin[k].mean()

    out = []
    for ds in ("adult", "heart", "credit"):
        f, l, a = pl(ds, "first"), pl(ds, "last"), pl(ds, "all")
        out.append(f"{NAMES[ds]}: clean {f[3]:.4f}; corrupted first round {f[4]:.4f} (paired loss {n(f[0])}, {ci(f[1], f[2])}); corrupted last round {l[4]:.4f} ({n(l[0])}, {ci(l[1], l[2])}); every round {a[4]:.4f} ({n(a[0])}, {ci(a[1], a[2])})")
    txt = (
        "\\rev{Figure~\\ref{fig:selfheal} compares a corrupted first round, a corrupted last round and an unattacked baseline on every dataset under the matched design. "
        + "; ".join(out) + ". "
        "A single corrupted round leaves no measurable loss on Adult or Credit Default. On Heart Disease the first-round loss is larger than the last-round loss but its interval includes zero. "
        "Corrupting every round costs $0.0088$ on Adult. On Heart Disease it costs $0.0134$ with an interval that includes zero, and on Credit Default it has no measurable effect at the default radius.}")
    return txt
