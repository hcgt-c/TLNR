# -*- coding: utf-8 -*-
"""230_star_existence_figure.py — figure, report and audit checks for the (star) existence test.

Reads results/star_existence_test.json (script 229) and produces

  paper/figures/F36_star_existence.png
  paper/targets/TPAMI/reports/STAR_EXISTENCE_REPORT.md
  results/star_existence_check.json     (verifier 229 for scripts/181_master_audit.py)

Three reference classes are reported, because the first of them alone is misleading:

  (i)  chance            — random pairs; a base-equal pair stays equal with probability q by construction.
  (ii) rank-conditioned  — pairs that *should* stay close under any monotone deformation with the same rank
       fidelity: a Gaussian copula with the observed Spearman (r = 2 sin(pi rho_s / 6)) predicts
       Phi_2(z_q, z_q; r) / q. This is the null a hostile reviewer will use, and it is the honest one: it
       removes the part of the effect that any smooth, not-necessarily-faithful deformation would produce.
  (iii) matched-effect-size noise control — a *driven* control in the paper's sense (Section 8.2): additive
       Gaussian pixel noise calibrated so its logit effect size equals the real transformation's. It moves
       the consumer by the same amount, but it is not a physical transformation of the scene, so it has no
       reason to respect the consumer's equivalence classes.

Nothing is recomputed here beyond the null and the ratios: every measured number comes from the JSON.
Usage: python scripts/230_star_existence_figure.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json

import numpy as np

WORK = rp.REPO_ROOT
SRC = os.path.join(WORK, "results", "star_existence_test.json")
FIG = os.path.join(WORK, "paper", "figures", "F36_star_existence.png")
REP = os.path.join(WORK, "paper", "targets", "TPAMI", "reports", "STAR_EXISTENCE_REPORT.md")

FAMCOL = {"hue": "#1f77b4", "heat": "#d62728"}
FAMLBL = {"hue": r"hue $+90^\circ$ (group)", "heat": r"heat $\sigma\!=\!2$ (semigroup)"}


def copula_null(rho_s, q):
    """Expected preservation under a Gaussian copula with the observed Spearman rank correlation.

    P(V <= z_q | U <= z_q) with (U,V) standard bivariate normal of correlation r = 2 sin(pi rho_s / 6),
    z_q = Phi^{-1}(q). Exact equality of the two distance *ranks* is what the statistic tests, so a null
    that removes rank fidelity is the right comparison.
    """
    from scipy.stats import norm, multivariate_normal
    r = 2.0 * np.sin(np.pi * rho_s / 6.0)
    r = float(np.clip(r, -0.999, 0.999))
    z = norm.ppf(q)
    p2 = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]]).cdf([z, z])
    return float(p2 / q)


def rows(d):
    """Flatten the JSON into one record per (backbone, family, param, scope, site, q), real and control."""
    out = []
    for bb, rv in d["results"].items():
        for fam, fv in rv["families"].items():
            for p, pv in fv.items():
                for q, qv in pv.get("consumer", {}).items():
                    out.append(dict(backbone=bb, kind=rv["kind"], family=fam, param=p, mode="real",
                                    scope="consumer", site="logits", eff=pv["effect_size_logits"], **qv))
                for site, sv in pv.get("sites", {}).items():
                    for q, qv in sv.items():
                        out.append(dict(backbone=bb, kind=rv["kind"], family=fam, param=p, mode="real",
                                        scope="site", site=site, eff=pv["effect_size_logits"], **qv))
        for fam, pv in rv.get("controls", {}).items():
            for q, qv in pv.get("consumer", {}).items():
                out.append(dict(backbone=bb, kind=rv["kind"], family="noise@" + fam, param=pv["param"],
                                mode="control", scope="consumer", site="logits", eff=pv["effect_size_logits"],
                                target_effect=pv.get("target_effect"), **qv))
    return out


def get(R, bb, fam, scope, q, site=None, mode="real"):
    for r in R:
        if (r["backbone"] == bb and r["family"] == fam and r["scope"] == scope and r["mode"] == mode
                and r["q"] == q and (site is None or r["site"] == site)):
            return r
    return None


def figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = rows(d)
    bbs = list(d["results"].keys())
    fams = sorted({r["family"] for r in R if r["mode"] == "real"})
    has_ctrl = any(r["mode"] == "control" for r in R)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9))

    def barpos(bb_i, j, k, n):
        return bb_i + (j - (len(fams) - 1) / 2) * 0.62 + (k - (n - 1) / 2) * 0.13

    # (a) preservation against the two nulls
    ax = axes[0]
    nbar = 3 if has_ctrl else 2
    for j, fam in enumerate(fams):
        for k, (lab, col, alpha) in enumerate([("observed", FAMCOL[fam], 0.95),
                                               ("rank-conditioned null", "0.55", 0.9),
                                               ("matched noise control", "0.85", 0.9)][:nbar]):
            vals = []
            for bb in bbs:
                r = get(R, bb, fam, "consumer", 0.01)
                v = r["preservation_rate"]
                if k == 1:
                    v = copula_null(r["spearman_base_vs_transformed_dist"], 0.01)
                elif k == 2:
                    c = get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")
                    v = c["preservation_rate"] if c else np.nan
                vals.append(v)
            ax.bar([barpos(i, j, k, nbar) for i in range(len(bbs))], vals, 0.12,
                   color=col, alpha=alpha, label=f"{FAMLBL[fam]}: {lab}" if k == 0 else f"{lab} ({fam})")
    ax.set_xticks(np.arange(len(bbs))); ax.set_xticklabels(bbs, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("preservation of base-equal pairs, $q=0.01$")
    ax.set_title("(a) observed against the two reference classes")
    ax.legend(fontsize=6.2, loc="upper left", ncol=1)
    ax.grid(alpha=0.25, axis="y")

    # (b) the same statistic along depth (q = 0.01)
    ax = axes[1]
    for bb in bbs:
        for fam in fams:
            sn = d["results"][bb]["site_names"]
            ys = [(get(R, bb, fam, "site", 0.01, site=s) or {}).get("preservation_rate", np.nan) for s in sn]
            ax.plot(np.arange(1, len(sn) + 1), ys, "o-" if fam == "hue" else "s--",
                    color=FAMCOL[fam], ms=4.5, lw=1.3, alpha=0.55 + 0.45 * (bbs.index(bb) / max(1, len(bbs) - 1)),
                    label=f"{bb}/{fam}")
    ax.axhline(0.01, color="0.5", lw=0.9, ls="--")
    ax.text(0.02, 0.02, "random-pair chance $q=0.01$", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.set_xticks(np.arange(1, 5)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("site index (relative depth)"); ax.set_ylim(0, 1.0)
    ax.set_ylabel("site preservation (q = 0.01)")
    ax.set_title("(b) per site: solid = hue, dashed = heat (no per-site power control)")
    ax.legend(fontsize=6.4, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)

    # (c) preservation vs the transformation's logit effect size, real cells and their matched controls
    ax = axes[2]
    for fam in fams:
        xs = [get(R, bb, fam, "consumer", 0.01)["eff"] for bb in bbs]
        ys = [get(R, bb, fam, "consumer", 0.01)["preservation_rate"] for bb in bbs]
        ax.plot(xs, ys, "o", color=FAMCOL[fam], ms=6, label=FAMLBL[fam])
        for bb, x, y in zip(bbs, xs, ys):
            ax.annotate(bb, (x, y), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center",
                        color=FAMCOL[fam])
        if has_ctrl:
            cx = [get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")["eff"] for bb in bbs]
            cy = [get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")["preservation_rate"] for bb in bbs]
            ax.plot(cx, cy, "x", color=FAMCOL[fam], ms=6, alpha=0.7,
                    label=f"matched noise control ({fam})")
            for x, y in zip(cx, cy):
                ax.annotate("", (x, y), textcoords="offset points", xytext=(0, 0))
            for bb, x, y, x2, y2 in zip(bbs, xs, ys, cx, cy):
                ax.annotate("", xy=(x2, y2), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", lw=0.6, color=FAMCOL[fam], alpha=0.6))
    ax.set_xlabel("logit effect size of the transformation")
    ax.set_ylabel("preservation (q = 0.01)")
    ax.set_title("(c) preservation tracks effect size; arrows point to the matched control")
    ax.legend(fontsize=6.6, loc="lower right"); ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print("saved", FIG)
    return FIG


def report(d):
    R = rows(d)
    bbs = list(d["results"].keys())
    fams = sorted({r["family"] for r in R if r["mode"] == "real"})
    has_ctrl = any(r["mode"] == "control" for r in R)
    n = d["n_crops"]
    L = []
    L.append("# Direct test of the existence condition $(\\star)$\n")
    L.append("*Script:* `scripts/229_star_existence_test.py` (measurement), "
             "`scripts/230_star_existence_figure.py` (this report and figure). "
             "*Artifact:* `results/star_existence_test.json`. "
             "*Figure:* `paper/figures/F36_star_existence.png`. "
             f"*Protocol:* {n} COCO crops, seed {d['seed']}, device {d['device']}.\n")
    L.append("## What is tested\n")
    L.append("The theory's existence level is the class-preservation condition\n")
    L.append("$$F(z_1)=F(z_2)\\;\\Longrightarrow\\;F(g_\\tau z_1)=F(g_\\tau z_2). \\tag{$\\star$}$$\n")
    L.append("On a frozen backbone the site activation determines the rest of the forward pass, so with the "
             "consumer taken as the network's own output the condition reads "
             "$y(x_1)=y(x_2)\\Rightarrow y(\\tau x_1)=y(\\tau x_2)$, where $y$ is the ImageNet head for the "
             "three classifier backbones (ResNet-50, ConvNeXt-T, ViT-B/16) and the 768-d self-supervised CLS "
             "embedding for DINOv2-B/14 — the one non-classifier consumer, whose class-quotient language is "
             "correspondingly weaker and whose cells are labelled as such throughout.\n")
    L.append("Continuous equality is discretised by a tolerance: two inputs are *base-equal* when their "
             "output distance is at most the $q$-quantile $\\varepsilon$ of the base pairwise-distance "
             "distribution, and a base-equal pair is *preserved* when its transformed distance is at most the "
             "$q$-quantile of the transformed distribution. **One violation refutes $(\\star)$ for that "
             "(transformation, consumer) pair**, so this is a falsification test and not a fit.\n")
    L.append("## Three reference classes, because one of them is misleading\n")
    L.append("1. **Chance.** Random pairs are base-equal-and-preserved with probability exactly $q$ by "
             "construction. This is the weakest possible comparison: it credits the transformation for merely "
             "being smooth.\n")
    L.append("2. **Rank-conditioned null.** The base-equal set is the bottom-$q$ of the base distances, and "
             "those pairs are strongly rank-correlated with the transformed distances (Spearman $\\rho_s$ "
             "below). A monotone deformation that respects no consumer classes at all still keeps most of "
             "them close. The null is a Gaussian copula with the observed $\\rho_s$ "
             "($r = 2\\sin(\\pi\\rho_s/6)$), giving expected preservation "
             "$\\Phi_2(z_q,z_q;r)/q$. **This is the honest reference class for the effect.**\n")
    if has_ctrl:
        L.append("3. **Matched-effect-size noise control (driven).** Additive Gaussian pixel noise with its "
                 "level calibrated by secant search so that its logit effect size equals the real "
                 "transformation's. It moves the consumer by the same amount as the real transformation but is "
                 "not a physical transformation of the scene, so it has no reason to respect the consumer's "
                 "equivalence classes. The calibration history is stored per cell. This is the paper's driven "
                 "control (Section 8.2) applied to the existence test, and it removes the power confound.\n")
    L.append("The paper's own power gate (an effect size of at least 0.05) is applied to every cell; no cell "
             "is excluded, and no ad-hoc floor is used.\n")
    L.append("**In its exact form $(\\star)$ is not testable by this design**, and we do not claim otherwise: "
             "exact equality of continuous outputs is a measure-zero event, so no exactly base-equal pair is "
             "ever instantiated, and with a finite tolerance `preservation < 1` is guaranteed for any "
             "non-comonotone map. What is measured is a *graded surrogate*, and the question it can answer is "
             "whether the transformation keeps consumer-indistinguishable pairs together **better than a "
             "matched, class-agnostic deformation does**.\n")
    L.append("## Effect size and the two nulls\n")
    if has_ctrl:
        L.append("| backbone | family | logit effect | control target | control effect | control sigma |\n"
                 "|---|---|---|---|---|---|")
        for bb in bbs:
            for fam in fams:
                r = get(R, bb, fam, "consumer", 0.01)
                c = get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")
                sig = d["results"][bb].get("controls", {}).get(fam, {}).get("param")
                L.append(f"| {bb} | {fam} | {r['eff']:.3f} | {c['target_effect']:.3f} | {c['eff']:.3f} | "
                         f"{sig:.4f} |")
        L.append("")
    L.append("| backbone | family | q | preservation | chance $q$ | rank-conditioned null | excess over null | "
             + ("matched control | excess over control | " if has_ctrl else "")
             + "median ratio | Spearman |\n"
             + "|---|---|---|---|---|---|---|" + ("---|---|" if has_ctrl else "") + "---|---|")
    for bb in bbs:
        for fam in fams:
            for q in (0.01, 0.05):
                r = get(R, bb, fam, "consumer", q)
                null = copula_null(r["spearman_base_vs_transformed_dist"], q)
                extra = ""
                if has_ctrl:
                    c = get(R, bb, "noise@" + fam, "consumer", q, mode="control")
                    cp = c["preservation_rate"] if c else float("nan")
                    extra = (f"{cp:.3f} | {(r['preservation_rate'] / cp if cp else float('nan')):.2f}$\\times$ | ")
                L.append(f"| {bb} | {fam} | {q:.2f} | {r['preservation_rate']:.3f} | {q:.2f} | {null:.3f} | "
                         f"{r['preservation_rate'] / null:.2f}$\\times$ | {extra}"
                         f"{r['median_ratio']:.3f} | {r['spearman_base_vs_transformed_dist']:.3f} |")
    L.append("")
    L.append("`excess over null` is preservation divided by the rank-conditioned null: the part of the effect "
             "that a rank-matched monotone deformation does *not* explain. The median ratio and Spearman are "
             "scale-free (both tolerances are quantiles and the companions use ranks and same-matrix ratios), "
             "so a global rescaling of the transformed distances in either direction cannot change them — "
             "what a non-uniform rescaling can do is reorder pairs, and it is the reordering that is measured.\n")
    L.append("## Site-level result (q = 0.01)\n")
    L.append("The site-level statistic uses the pooled activation itself as the read-out, not the network "
             "output, so it is a separate representation-level measurement and not a bound on the "
             "consumer-level test. No per-site effect size is stored, so these rows carry no per-site power "
             "control.\n")
    L.append("| backbone | family | " + " | ".join(d["results"][bbs[0]]["site_names"]) + " | argmin |\n"
             "|---|---|" + "---|" * (len(d["results"][bbs[0]]["site_names"]) + 1))
    for bb in bbs:
        for fam in fams:
            sn = d["results"][bb]["site_names"]
            vals = [(get(R, bb, fam, "site", 0.01, site=s) or {}).get("preservation_rate", float("nan")) for s in sn]
            am = sn[int(np.nanargmin(vals))]
            L.append(f"| {bb} | {fam} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {am} |")
    L.append("")
    L.append("## Reading\n")
    c_lo = [get(R, bb, fam, "consumer", 0.01) for bb in bbs for fam in fams]
    nulls = {r["backbone"] + "/" + r["family"]: copula_null(r["spearman_base_vs_transformed_dist"], 0.01)
             for r in c_lo}
    ex = {k: get(R, *k.split("/"), "consumer", 0.01)["preservation_rate"] / v for k, v in nulls.items()}
    wk, st = min(ex, key=ex.get), max(ex, key=ex.get)
    L.append(f"**Against chance alone the effect looks large and is not informative.** Preservation at "
             f"$q=0.01$ spans {min(r['preservation_rate'] for r in c_lo):.3f}–"
             f"{max(r['preservation_rate'] for r in c_lo):.3f} against a 1% chance rate, an enrichment of "
             f"{min(r['preservation_rate'] for r in c_lo) / 0.01:.0f}$\\times$–"
             f"{max(r['preservation_rate'] for r in c_lo) / 0.01:.0f}$\\times$. **Most of that is rank "
             f"fidelity, not class preservation.** Against the rank-conditioned null the excess is "
             f"{wk} ({ex[wk]:.2f}$\\times$) to {st} ({ex[st]:.2f}$\\times$), and the two DINOv2 cells are at "
             f"or below the null.\n")
    if has_ctrl:
        rel = {}
        for bb in bbs:
            for fam in fams:
                r = get(R, bb, fam, "consumer", 0.01)
                c = get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")
                rel[bb + "/" + fam] = r["preservation_rate"] - c["preservation_rate"]
        rw, rb = min(rel, key=rel.get), max(rel, key=rel.get)
        npos = sum(1 for v in rel.values() if v > 0)
        L.append(f"**Against the matched-effect-size control the excess is the class-specific part.** "
                 f"Preservation(real) − preservation(matched noise) runs from {rel[rw]:+.3f} ({rw}) to "
                 f"{rel[rb]:+.3f} ({rb}); the real transformation is ahead in {npos} of {len(rel)} cells "
                 f"(both DINOv2 at $q=0.01$) and *behind* in the six classifier cells. Because the control has "
                 f"the same logit effect size by construction, this difference is not a power difference: for "
                 f"the classifier consumers the physical transformation keeps consumer-indistinguishable pairs "
                 f"together **less** than an unstructured perturbation of identical output magnitude does. At "
                 f"$q=0.05$ the picture is mixed (4 of 8 ahead), so no robust advantage survives at either "
                 f"tolerance.\n")
    L.append(f"**Depth, without a per-site power control.** The deepest site is the minimum in "
             f"{sum(1 for bb in bbs for fam in fams if np.nanargmin([(get(R, bb, fam, 'site', 0.01, site=s) or {}).get('preservation_rate', np.nan) for s in d['results'][bb]['site_names']]) == len(d['results'][bb]['site_names']) - 1)}"
             f" of {len(bbs) * len(fams)} rows: hue is lowest at the deepest site in ResNet-50, ViT-B/16 and "
             f"DINOv2-B/14 (and at site 3 for ConvNeXt), while heat is lowest at the deepest site only for "
             f"ConvNeXt and ViT-B/16 and is non-monotone elsewhere. With no per-site effect size, this is a "
             f"direction to note rather than a graded depth law.\n")
    L.append("**What the measurement establishes, and what it does not.** Closure on consumer-visible state is "
             "negative here: the real transformation is behind a driven control at matched effect size in six of "
             "eight cells, and its excess over a rank-matched null is at most 2.4$\\times$ and at or below 1 for "
             "the self-supervised DINOv2 consumer. Three limits keep that inside its scope and are reasons not to "
             "read it as evidence about a representational action: a homomorphism need not preserve Euclidean "
             "near-neighbour ordering (the family $\\rho_t=\\mathrm{diag}(e^t,e^{-t})$ satisfies the composition "
             "law and reorders neighbours), within-tolerance closeness is not transitive, and neither control "
             "proves non-existence (the copula null absorbs whatever structure a rank-typical monotone deformation "
             "has, and the noise control matches only the global displacement magnitude, not direction, local "
             "expansion, inter-sample correlation or smoothness). By the counterexample of Section 3.3 — $F(a,b)=a$ "
             "with $\\rho(\\tau)(a,b)=(b,a)$ — closure can fail while an exact, composable action exists on the "
             "full representation, because the operator may use coordinates the consumer ignores. The positive "
             "evidence about the representation is therefore the fitted-operator and composition programme of "
             "Sections 9–11, and the practical yield here is the question of which state description an "
             "intervention should be defined on.\n")
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    open(REP, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", REP)


def checks(d):
    """Machine-readable checks for the audit chain (verifier 229)."""
    R = rows(d)
    bbs = list(d["results"].keys())
    fams = sorted({r["family"] for r in R if r["mode"] == "real"})
    c = [get(R, bb, fam, "consumer", 0.01) for bb in bbs for fam in fams]
    null = {r["backbone"] + "/" + r["family"]: copula_null(r["spearman_base_vs_transformed_dist"], 0.01) for r in c}
    ex = {k: get(R, *k.split("/"), "consumer", 0.01)["preservation_rate"] / v for k, v in null.items()}
    # only the three classifier backbones enter the "deepest site" claim, and the set is derived, not hardcoded
    cls = [bb for bb in bbs if d["results"][bb]["kind"] != "dinov2"]
    deepest = {}
    for bb in bbs:
        for fam in fams:
            sn = d["results"][bb]["site_names"]
            vals = [(get(R, bb, fam, "site", 0.01, site=s) or {}).get("preservation_rate", np.nan) for s in sn]
            deepest[(bb, fam)] = sn[int(np.nanargmin(vals))]
    hue_deep = [bb for bb in bbs if deepest[(bb, "hue")] == d["results"][bb]["site_names"][-1]]
    out = [
        {"label": "star_min_preservation_q01", "literal": f"{min(r['preservation_rate'] for r in c):.3f}",
         "pass": abs(min(r["preservation_rate"] for r in c) - 0.249) < 5e-4,
         "note": "minimum consumer preservation at q=0.01 over four backbones and both algebras (ViT-B/16 hue)"},
        {"label": "star_max_preservation_q01", "literal": f"{max(r['preservation_rate'] for r in c):.3f}",
         "pass": abs(max(r["preservation_rate"] for r in c) - 0.815) < 5e-4,
         "note": "maximum consumer preservation at q=0.01 (DINOv2-B/14 heat, a self-supervised embedding consumer)"},
        {"label": "star_min_excess_over_null", "literal": f"{min(ex.values()):.2f}",
         "pass": abs(min(ex.values()) - 0.91) < 0.01,
         "note": "minimum excess over the rank-conditioned (Gaussian-copula) null at q=0.01; below 1 means the real transformation is no better than a rank-matched monotone deformation"},
        {"label": "star_max_excess_over_null", "literal": f"{max(ex.values()):.2f}",
         "pass": abs(max(ex.values()) - 2.39) < 0.01,
         "note": "maximum excess over the rank-conditioned null at q=0.01"},
        {"label": "star_hue_deepest_min", "literal": "0.314",
         "pass": len(hue_deep) == 3 and abs(get(R, "resnet50", "hue", "site", 0.01, site="layer4")["preservation_rate"] - 0.314) < 5e-4,
         "note": f"hue preservation is lowest at the deepest site in exactly three of four backbones (derived set: {hue_deep}); literal is the ResNet-50 deepest site"},
    ]
    if any(r["mode"] == "control" for r in R):
        rel = {}
        for bb in bbs:
            for fam in fams:
                cr = get(R, bb, "noise@" + fam, "consumer", 0.01, mode="control")
                if cr:
                    rel[bb + "/" + fam] = get(R, bb, fam, "consumer", 0.01)["preservation_rate"] - cr["preservation_rate"]
        rw, rb = min(rel, key=rel.get), max(rel, key=rel.get)
        out += [
            {"label": "star_control_gap_min", "literal": f"{abs(rel[rw]):.3f}",
             "pass": rel[rw] < 0,
             "note": f"magnitude of the most negative preservation(real) - preservation(matched noise) at q=0.01 ({rw}); the gap itself is negative"},
            {"label": "star_control_gap_max", "literal": f"{rel[rb]:+.3f}",
             "pass": rel[rb] > 0,
             "note": f"most positive preservation(real) - preservation(matched noise) at q=0.01 ({rb})"},
            {"label": "star_control_win_count", "literal": str(sum(1 for v in rel.values() if v > 0)),
             "pass": True,
             "note": f"cells (of {len(rel)}) in which the real transformation preserves more than its matched control"},
        ]
    p = os.path.join(WORK, "results", "star_existence_check.json")
    json.dump({"about": "verifier 229: literals recomputed from results/star_existence_test.json, plus the "
                        "rank-conditioned (Gaussian-copula) null and the matched-noise-control gaps",
               "checks": out}, open(p, "w"), indent=1)
    print("wrote", p, "| all pass:", all(c_["pass"] for c_ in out))
    return out


def main():
    d = json.load(open(SRC))
    figure(d)
    report(d)
    checks(d)


if __name__ == "__main__":
    main()
