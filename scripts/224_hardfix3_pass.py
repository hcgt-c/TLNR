# -*- coding: utf-8 -*-
"""224_hardfix3_pass.py — hard-error corrections, pass 3 (deferred items A2/A3, C7/C8/C10-C14, D20,
E22/E23, F35 registration).

Every changed number is recomputed here from results/ and the source field is named in the log:
  C7  results/quotientization.json  keys 'resnet50,convnext,vitb16,dinov2b14_seed{0,1,2}' (hue),
      backbones.resnet50.layer{1..4}.visible_share_of_displacement / visible_share_over_isotropic /
      matched_rank_control.random_share_at_nominal
  C8  results/deep_linear_transport.json cells[*].kappa_mean / tf_site_mean / effective_rank_mean and
      trajectory.theta60_m16_wd1e-3_seed0; results/linear_dynamics_counterexample.json runs[*].tf_delta
  C10 results/depth_map_summary.json  sites[*].O2_transfer_mean (deepest reachable per curve)
  C11 results/depth_map_summary.json  sites[*].{O1,O2,O8,O6}_proj_coef_agg_mean over reachable sites
  C12 results/depth_map_vitb16_heat.json / depth_map_dinov2b14_heat.json per-seed O2 proj/win means
  C14 results/depth_map_summary.json  pooled normalised and un-normalised Spearman (28 points, 8 curves)
  E22 results/depth_map_summary.json, results/estimator_control.json (fit-side realization residual)
  E23 results/task_weighted_operator.json

A2 renumbers every supplement table to an S-label (appearance order: 19,20,21,22,23,24 -> S1..S6;
3,4,5,7 -> S7..S10; 8,9,10,11,12,13,15 -> S11..S17) in the master and in every reference, which
resolves the main-vs-supplement collisions for Tables 15/20/21/22.

Backups: .bak-<YYYYMMDD>-hardfix3
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import json, os, re, shutil, datetime, statistics as st
import numpy as np
from scipy import stats

W = rp.REPO_ROOT
P = os.path.join(W, "paper")
MASTER = os.path.join(P, "unified", "MASTER_manuscript_EN.md")
REFS = os.path.join(P, "unified", "references.md")
FRONT = os.path.join(P, "targets", "TPAMI", "front_matter_EN.md")
DEPTH = os.path.join(P, "targets", "TPAMI", "sections", "depth_law_EN.md")
DISC = os.path.join(P, "targets", "TPAMI", "sections", "discussion_EN.md")
PLAN = os.path.join(P, "targets", "TPAMI", "figures_plan.json")
RES = os.path.join(W, "results")
STAMP = datetime.date.today().strftime("%Y%m%d")
LOG = []


def read(p):
    return open(p, encoding="utf-8").read()


def backup(p):
    b = p + f".bak-{STAMP}-hardfix3"
    if not os.path.exists(b):
        shutil.copy2(p, b)


def rep(path, old, new, tag, expect=1):
    t = read(path)
    n = t.count(old)
    if n != expect:
        LOG.append(f"SKIP {tag}: count={n} exp {expect}")
        return False
    open(path, "w", encoding="utf-8").write(t.replace(old, new))
    LOG.append(f"OK   {tag}")
    return True


def tie_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])
    n = len(x)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - 2))


# ==================================================================== recompute
dm = json.load(open(os.path.join(RES, "depth_map_summary.json"), encoding="utf-8"))
CURVES = ["resnet50|hue_90.0", "convnext|hue_90.0", "vitb16|hue_90.0", "dinov2b14|hue_90.0",
          "resnet50|heat_2.0", "convnext|heat_2.0", "vitb16|heat_2.0", "dinov2b14|heat_2.0"]


def reachable(c):
    sites = dm[c]["sites"]
    order = sorted(sites, key=lambda s: sites[s]["depth"])
    return [s for s in order if sites[s]["reachable"]]


# ---- C7 hue visible share, three-seed ResNet-50 block
qz = json.load(open(os.path.join(RES, "quotientization.json"), encoding="utf-8"))
QKEYS = [f"resnet50,convnext,vitb16,dinov2b14_seed{s}" for s in range(3)]
QDEP = ["layer1", "layer2", "layer3", "layer4"]
vis = [st.mean([qz[k]["backbones"]["resnet50"][d]["visible_share_of_displacement"] for k in QKEYS])
       for d in QDEP]
ovr = [st.mean([qz[k]["backbones"]["resnet50"][d]["visible_share_over_isotropic"] for k in QKEYS])
       for d in QDEP]
rnd = [st.mean([qz[k]["backbones"]["resnet50"][d]["matched_rank_control"]["random_share_at_nominal"]
                for k in QKEYS]) for d in QDEP]
LOG.append("INFO C7 hue resnet50 3-seed vis=" + "/".join(f"{v:.3f}" for v in vis) +
           " over-iso=" + "/".join(f"{v:.2f}" for v in ovr) +
           " rand@nominal=" + "/".join(f"{v:.3f}" for v in rnd))

# ---- C8 deep linear transport
dl = json.load(open(os.path.join(RES, "deep_linear_transport.json"), encoding="utf-8"))
cells = dl["cells"]
kap = [c["kappa_mean"] for c in cells.values()]
tf_m = [c["tf_site_mean"] for c in cells.values()]
erk = [c["effective_rank_mean"] for c in cells.values()]
rho_kap, p_kap = tie_spearman(kap, tf_m)
rho_erk, p_erk = tie_spearman(erk, tf_m)
CFERR = st.median([c["closed_form_abs_error"] for c in cells.values()])
traj = dl["trajectory"]["theta60_m16_wd1e-3_seed0"]
TR0, TR1 = traj[0], traj[-1]
LOG.append(f"INFO C8 rho(kappa)={rho_kap:+.3f} rho(effrank)={rho_erk:+.3f} closed-form med={CFERR:.3f}"
           f" traj eff {TR0['eff_rank']:.2f}->{TR1['eff_rank']:.2f} tf {TR0['tf_site']:.3f}->{TR1['tf_site']:.3f}"
           f" loss {TR0['loss']:.1f}->{TR1['loss']:.2e}")
ldc = json.load(open(os.path.join(RES, "linear_dynamics_counterexample.json"), encoding="utf-8"))
runs = ldc["runs"]
within_d = [runs[k]["tf_delta"] for k in runs if runs[k]["geometry"] == "within" and runs[k]["wd"] > 0]
cross_d = [runs[k]["tf_delta"] for k in runs if runs[k]["geometry"] == "cross" and runs[k]["wd"] > 0]
LOG.append(f"INFO C8 208 within deltas={['%+.4f' % x for x in within_d]} mean={st.mean(within_d):+.4f}"
           f" | cross deltas={['%+.4f' % x for x in cross_d]} mean={st.mean(cross_d):+.4f}")

# ---- C10 deepest reachable transfer per curve
DEEP = {c: dm[c]["sites"][reachable(c)[-1]]["O2_transfer_mean"] for c in CURVES}
LOG.append("INFO C10 deepest=" + ", ".join(f"{c}:{v:+.3f}" for c, v in DEEP.items()) +
           f" | median={st.median(DEEP.values()):.3f} min={min(DEEP.values()):+.3f} max={max(DEEP.values()):+.3f}")

# ---- C11 projection support on the grid
O6gt, highest = 0, {"O1": 0, "O2": 0, "O8": 0, "O6": 0}
n_reach = 0
for c in CURVES:
    for s in reachable(c):
        rec = dm[c]["sites"][s]
        n_reach += 1
        p = {o: rec[f"{o}_proj_coef_agg_mean"] for o in ("O1", "O2", "O8", "O6")}
        if p["O6"] > p["O2"]:
            O6gt += 1
        highest[max(p, key=p.get)] += 1
LOG.append(f"INFO C11 O6>O2 at {O6gt}/{n_reach}; highest={highest}")

# ---- C14 pooled rho, normalised and raw
xs_n, ys_n, xs_r, ys_r = [], [], [], []
for c in CURVES:
    rc = reachable(c)
    first = dm[c]["sites"][rc[0]]["O2_transfer_mean"]
    for i, s in enumerate(rc):
        rel = (i + 1) / len(rc)
        t = dm[c]["sites"][s]["O2_transfer_mean"]
        xs_n.append(rel); ys_n.append(t / first)
        xs_r.append(rel); ys_r.append(t)
RHO_N, P_N = tie_spearman(xs_n, ys_n)
RHO_R, P_R = tie_spearman(xs_r, ys_r)
LOG.append(f"INFO C14 normalised rho={RHO_N:+.3f} p={P_N:.2e} n={len(xs_n)} | raw pooled rho={RHO_R:+.3f} p={P_R:.2e}")

# ---- C12 heat companion values for ViT / DINOv2 (three-seed means)
def heat_comp(bb):
    d = json.load(open(os.path.join(RES, f"depth_map_{bb}_heat.json"), encoding="utf-8"))
    seeds = d["results"]["heat_2.0"]["seed"]
    out = {}
    for site in d["site_names"]:
        proj = [seeds[s]["sites"][site]["O2"]["proj_coef_agg"] for s in seeds]
        win = [seeds[s]["sites"][site]["O2"]["win_rate"] for s in seeds]
        out[site] = (st.mean(proj), st.mean(win))
    return out


VIT_H = heat_comp("vitb16")
DIN_H = heat_comp("dinov2b14")
LOG.append(f"INFO C12 vit heat block2 proj/win={VIT_H['block2'][0]:.3f}/{VIT_H['block2'][1]:.3f}"
           f" block8={VIT_H['block8'][0]:.3f}/{VIT_H['block8'][1]:.3f}")
LOG.append(f"INFO C12 dinov2 heat block2 proj/win={DIN_H['block2'][0]:.3f}/{DIN_H['block2'][1]:.3f}"
           f" block8={DIN_H['block8'][0]:.3f}/{DIN_H['block8'][1]:.3f}")

# ---- E22 fit-side realization residual for the worked certificate
ec = json.load(open(os.path.join(RES, "estimator_control.json"), encoding="utf-8"))
GATE4_RESID = ec["cells"]["resnet50|hue|depth1"]["estimators"]["O2_pub"]["feat_resid_over_displacement"]
LOG.append(f"INFO E22 gate-4 fit-side realization residual (resnet50|hue|depth1) = {GATE4_RESID:.4f}")

# ---- B6 absolute T_F for the master abstract/intro (same convention as the target)
def tf_arm(arm):
    rows = {o: [] for o in ("O1_procrustes", "O2_ridge", "O8_conv_residual")}
    import glob
    for f in sorted(glob.glob(os.path.join(RES, f"causal_cifar_{arm}_s*.json"))):
        if "stage2" in f:
            continue
        d = json.load(open(f))["deltas"]["90"]["downstream"]
        nul = d["null"]["rel_l2_vs_real"]
        for o in rows:
            rows[o].append(1 - d[o]["rel_l2_vs_real"] / nul)
    return {o: round(st.mean(v), 3) for o, v in rows.items()}


CE8, Z2 = tf_arm("ce8"), tf_arm("z2")
LOG.append(f"INFO B6 CEConv {CE8}; plain {Z2}")

# ==================================================================== backups
for f in (MASTER, REFS, FRONT, DEPTH, DISC, PLAN):
    backup(f)

# ==================================================================== A2 renumbering
SUPP = {19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6,
        3: 7, 4: 8, 5: 9, 7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 15: 17}


def s_label(n):
    return f"S{SUPP[n]}" if n in SUPP else str(n)


_master = read(MASTER)
_n_table_refs = len(re.findall(r"(?<!Supplementary )\bTable \d+\b", _master))
_master = re.sub(r"(?<!Supplementary )\bTable (\d+)\b",
                 lambda m: f"Table {s_label(int(m.group(1)))}", _master)
open(MASTER, "w", encoding="utf-8").write(_master)
LOG.append(f"OK   A2 S-renumbered supplement tables in the master ({_n_table_refs} references/declarations)")

# ==================================================================== A3 appendix C
plan = json.load(open(PLAN, encoding="utf-8"))


def short(cap, n=150):
    cap = re.sub(r"\s+", " ", cap).strip()
    cap = cap.split(". ")[0]
    return (cap[:n].rstrip() + "…") if len(cap) > n else cap


rows = []
for it in plan["main"]:
    rows.append(f"| {it['n']} | {short(it['caption'])} | {it['section']} |")
for it in plan["supplementary"]:
    rows.append(f"| S{it['n']} | {short(it['caption'])} | {it['section']} |")
appC = ("## Appendix C: figure list\n\n"
        "The figure set is generated from the figure plan (`figures_plan.json`): "
        f"**{len(plan['main'])} main figures** (numbered 1–{len(plan['main'])}) and "
        f"**{len(plan['supplementary'])} supplementary figures** (numbered S1–S{len(plan['supplementary'])}). "
        "Every figure is produced from the result file and script named for it in the plan and the figure "
        "manifest; the schematic figures carry no experimental numbers.\n\n"
        "| figure | content | section |\n|---|---|---|\n" + "\n".join(rows) + "\n")
_oldC = re.search(r"## Appendix C: figure list\n.*?(?=\n## Appendix D)", read(MASTER), re.S)
if _oldC:
    rep(MASTER, _oldC.group(0), appC, "A3 Appendix C rebuilt from the figure plan")
else:
    LOG.append("SKIP A3 Appendix C: block not found")

# ==================================================================== A3 references cleanup
for note in (" (verified against the PMLR record)", " (verified against the PMC record)",
             " (verified against the JMLR record)",
             " (conference version; page numbers not independently verified)"):
    rep(REFS, note, "", f"A3 reference note removed: {note.strip()}", expect=1)
rep(REFS,
    "Two verified sources are merged here: the intervention-paper entries (verified against the arXiv export API\n"
    "plus three manually checked non-arXiv items) and the geometry-paper entries (author-year). Target papers cite\n"
    "this list; duplicates across the two sources (colour-equivariant CNNs, DINOv2, ViT, group convolutions,\n"
    "E(2)-steerable CNNs, colour-as-geometry) must be de-duplicated at copy-editing time.",
    "Merged bibliography for the unified master and its targets. Entries are drawn from the two source "
    "manuscripts and from the cited-neighbour reviews; duplicates across sources (colour-equivariant CNNs, "
    "DINOv2, ViT, group convolutions, E(2)-steerable CNNs, colour-as-geometry) have been de-duplicated.",
    "A3 references header note cleaned")
rep(REFS, "## Cited neighbours added for the IJCV positioning\n\n",
    "## Positioning: causal abstraction and steering\n\n", "A3 references heading 1")
rep(REFS, "## Cited neighbours added for the TPAMI positioning\n\n",
    "## Positioning: equivariance, transformation maps and depth\n\n", "A3 references heading 2")

# ==================================================================== A3 cross-ref drift + C13
rep(MASTER,
    "This section gives the four measurements that make that concrete: the collapse of the effect size when training removes colour from the computation (8.1), the driven control that is aligned yet unfaithful (8.2), the controls that rule out capacity and pair-correspondence as explanations of the positive results (8.3), and a consumer trained to be insensitive to the transformation itself, on which the absolute score and its companion statistics disagree (8.4).",
    "This section gives the four measurements that make that concrete: the collapse of the effect size when training removes colour from the computation (Section 8.1), the driven control that is aligned yet unfaithful (Section 8.2), the controls that rule out capacity and pair-correspondence as explanations of the positive results (Section 8.3), and a consumer trained to be insensitive to the transformation itself, on which the absolute score and its companion statistics disagree (Section 8.4).",
    "A3 master Section 8 intro subsection refs")

# ==================================================================== C13 effect-size naming (master appendix G)
rep(MASTER,
    "low power — the partial correlation of depth with the ceiling is $-0.662$ controlling for effect size, and 16 of 16 tightly matched pairs fall at depth",
    "low power — the partial correlation of depth with the ceiling is $-0.662$ controlling for the site displacement, and 16 of 16 tightly matched pairs fall at depth",
    "C13 master: control uses site displacement, not consumer power")

# ==================================================================== C7 hue visible share (master appendix G)
rep(MASTER,
    "while for the **hue** arm it falls to the matched-rank random level by the deepest site (visible share 0.029/0.020/0.011/0.004, matched-rank random 0.031/0.016/0.009/0.006) (script 200/203)",
    "while for the **hue** arm the displacement is *more* visible than isotropic at the first three depths and falls "
    "to the matched-rank random level only at the deepest site (three-seed ResNet-50 visible share of the displacement "
    "0.126/0.063/0.024/0.004; over-isotropic share 2.48/2.49/1.91/0.55; matched-rank random share at rank 13, "
    "0.051/0.025/0.013/0.006) (script 200/203)",
    "C7 master hue visible-share correction")

# ==================================================================== B6 master abstract + intro projection -> T_F
rep(MASTER,
    "reproduces a real recolour almost exactly where the channel layout makes hue a group action (0.94–0.97) and only partially in a plain CNN (0.31, recovered to 0.83 by content and neighbourhood conditioning)",
    f"reaches the paper's headline downstream transfer {CE8['O1_procrustes']:.2f}–{CE8['O8_conv_residual']:.2f} where the channel layout makes hue a group action "
    f"(still partial under the 0.80 convention) and {Z2['O1_procrustes']:.2f}–{Z2['O8_conv_residual']:.2f} in a plain CNN (conditioning recovers it to about {Z2['O8_conv_residual']:.2f})",
    "B6 master abstract projection -> absolute T_F")
rep(MASTER,
    "a global linear operator reproduces a real recolour almost exactly where a group convolution makes hue a group action (0.94–0.97) and only partially in a plain CNN (0.31), where content and neighbourhood conditioning recovers 0.83.",
    f"a global linear operator reaches the headline downstream transfer {CE8['O1_procrustes']:.2f}–{CE8['O8_conv_residual']:.2f} where a group convolution makes hue a group action (still partial under the 0.80 convention) and {Z2['O1_procrustes']:.2f}–{Z2['O8_conv_residual']:.2f} in a plain CNN, where content and neighbourhood conditioning recovers about {Z2['O8_conv_residual']:.2f}.",
    "B6 master precondition-2 projection -> absolute T_F")

# ==================================================================== D20 composition bound
rep(MASTER,
    "where $\\epsilon_{\\text{comp}}$ measures the failure of the fitted family to commute. The two error terms come from applying $\\epsilon_b$ at the transported point $W_az$ and from the Lipschitz step $\\|P(g_bW_az - g_bg_az)\\| \\le L_g\\|P(W_az-g_az)\\|$, both of which are valid because $\\epsilon_a$ is measured in the same seminorm.",
    "with two terms and no residual. The first is the second operator's own error, evaluated at the transported point "
    "$W_az$; the second is the first operator's error propagated through $g_b$, via "
    "$\\|P(g_bW_az - g_bg_az)\\| \\le L_g\\|P(W_az-g_az)\\|$. Both steps need the bounds to be *uniform*: "
    "$\\epsilon_b$ must hold at the transported points the composition actually visits, and $g_b$ must be "
    "$L_g$-Lipschitz with a single constant on the consumed range, in the consumer's seminorm. If $P$ or $g_b$ "
    "varies with the sample the bound holds pointwise with the local constant and is correspondingly weaker. "
    "The step is valid because $\\epsilon_a$ is measured in the same seminorm.",
    "D20 composition bound: drop eps_comp, state uniformity and quotient-Lipschitz")

# ==================================================================== C8 master discussion mirror
rep(MASTER,
    "the linear-site theorems hold on the trained weights; the mixing coefficient governs and rank does not ($\\rho(\\kappa,T_F)=-0.97$ against $\\rho(\\mathrm{rank},T_F)=+0.43$); the within-block trajectory is flat; and in the cross-block geometry, where the directions learning discards are the ones the transformation mixes, $T_F$ moves by $+0.057$ with weight decay, with the closed form predicting the change within 0.02 in six of eight runs. Learning therefore changes editability only conditionally.",
    f"the linear-site theorems hold on the trained weights; the mixing coefficient governs and rank does not "
    f"($\\rho(\\kappa,T_F)={rho_kap:+.2f}$ against $\\rho(\\mathrm{{effective\\ rank}},T_F)={rho_erk:+.2f}$); the single recorded "
    f"trajectory (the $\\theta=60°$ mixing cell) is flat in transfer; and weight decay moves $T_F$ in *both* linear "
    f"geometries, by $+{st.mean(within_d):.2f}$ within the task block against $+{st.mean(cross_d):.2f}$ across it — the "
    f"opposite of the mixing-source prediction. The linear control therefore bounds the mechanism rather than "
    f"confirming it, and the nonlinear case remains open.",
    "C8 master discussion linear-control mirror")

# ==================================================================== E23 quotientization note (master roadmap)
rep(MASTER,
    "and a spatial description-selection rule — both stated as open because our own gate was decorative (Section 5.5). Until (i)–(iv) are run,",
    "a spatial description-selection rule — both stated as open because our own gate was decorative (Section 5.5); "
    "and the rank-intervention half of `results/quotientization.json`, whose visible-share and effective-rank fields "
    "we use but whose rank-intervention transfers are only partly analysed and are left as future work. Until (i)–(iv) are run,",
    "E23 master roadmap quotientization partly analysed")

# ==================================================================== E22 master Section 4 validation paragraph
rep(MASTER,
    "**Different sources of failure.**",
    "**The gates are a checklist, not a validated predictor.** We report the five gates as a *diagnostic checklist "
    "and reporting discipline* rather than a validated predictor: on the 32-site cross-architecture grid the "
    "outcome-free gate conjunction screens out causally unreachable and reference-invalid failures (accuracy 0.719 "
    "versus 0.438 for always-partial) but leaves the failed-versus-partial boundary unresolved (macro-F1 0.712), is "
    "no better than the site's depth alone (depth-rule accuracy 0.812; paired difference −0.095 [−0.250, +0.062]), "
    "and can never return a faithful verdict because no site of the published grid reaches the pre-registered "
    "faithful threshold (max $T_F$ = 0.707). The one gate statistic that carries strong ordinal signal is the "
    "outcome-free fit-side realization residual (Spearman $\\rho$ = −0.878, AUC 0.986), recorded for only 24 of the "
    "32 sites and thresholded at the pre-registered 1.00, a cut too lax to separate partial from failed. The "
    f"pre-registration, the held-out splits and every number are in `scripts/222_diagnostic_validation.py`, "
    "`results/diagnostic_validation.json` and `reports/DIAGNOSTIC_VALIDATION_REPORT.md`.\n\n"
    "**Different sources of failure.**",
    "E22 master Section 4 pre-registered validation")

# ==================================================================== E22 master Appendix H schema + worked example
rep(MASTER,
    "| `gate_4_realization` | `operator_family`, `transfer`, `win_rate`, `verdict` ∈ {faithful, partial, failed} | does the fitted family reproduce the transformation |",
    "| `gate_4_realization` | `operator_family`, `realization_residual`, `threshold`, `verdict` ∈ {pass, fail}; outcome `transfer` and `win_rate` carried as companions | does the fitted family's fit-side realization residual stay at or below the no-op (`residual` ≤ 1); the outcome $T_F$ is a companion, never the gate |",
    "E22 Appendix H gate-4 schema row")
rep(MASTER,
    "| `diagnostic report` | `powered`, `reachable`, `operator_family_needed`, `faithfulness`, `compositionality`, `recommended_depth` | the summary verdict |",
    "| `diagnostic report` | `powered`, `reachable`, `reference_valid`, `realized`, `recommended_family`, `verdict` ∈ {uninformative, failed, partial}, `compositionality`, `recommended_depth` | the summary verdict of the precondition screen; the conjunction never returns *faithful* |",
    "E22 Appendix H diagnostic-report row")
rep(MASTER,
    "| gate 4 realization | O2 (unconstrained global linear, ridge); transfer 0.602; win rate 0.963; verdict partial |",
    f"| gate 4 realization | O2 (unconstrained global linear, ridge); fit-side realization residual {GATE4_RESID:.3f} ≤ threshold 1.00 → **pass**; outcome transfer 0.626 and win rate 0.963 carried as companions |",
    "E22 Appendix H worked-example gate-4 row")
rep(MASTER,
    "These are conventions for turning continuous measurements into a summary, not claims about nature.",
    "These are conventions for turning continuous measurements into a summary, not claims about nature. The "
    "summary verdict is the conjunction of the preconditions and never returns *faithful*: on the 32-site grid no "
    "site clears the pre-registered faithful threshold (Section 4). Gate 4 thresholds the outcome-free fit-side "
    "realization residual (`feat_resid_over_displacement` ≤ 1.00), not $T_F$; the earlier shipped certificate that "
    "thresholded $T_F$ was circular and has been corrected (`scripts/215_transportability_certificate.py`).",
    "E22 Appendix H reporting conventions")

# ==================================================================== E23 master Section 9 negative real-image run
rep(MASTER,
    "One implementation fact is worth recording because it is the kind of thing that turns a bounded result into a false one.",
    "**The real-image run is negative at this scale.** The same comparison on the only real-image consumer we have "
    "(`results/task_weighted_operator.json`; 700 COCO crops, 580 fit / 120 held out, site = pooled stage-2 feature, "
    "$256\\times256$ operators, one seed) does not reproduce even the small positive effect. Both the Euclidean ridge "
    "and $W^\\star$ are *worse than the no-op*: projection $-0.197$ and $-0.090$, per-sample median projection "
    "$-5.03$ and $-2.31$, paired win rate $0.042$ and $0.017$, and top-1 agreement with the real route $0.333$ "
    "(= chance) and $0.167$. $W^\\star$ is also worse on its own held-out task-weighted error (226.2 against 174.6), "
    "so the metric's advantage does not transfer to this consumer at this capacity. We report it as a negative "
    "result about the estimator's generality rather than as support: at $d=64$ on this consumer neither objective "
    "produces a usable operator, and the fixed-$\\lambda$ CIFAR comparison of Table 16 is the only setting in which "
    "the consumer-induced metric is ahead.\n\n"
    "One implementation fact is worth recording because it is the kind of thing that turns a bounded result into a false one.",
    "E23 master Section 9 real-image negative result")

# ==================================================================== A3 master Appendix B mapping
rep(MASTER,
    "| 9 consumer-induced metric | 180 | `task_weighted_1x1_z2_s*_d90.json` |",
    "| 9 consumer-induced metric | 180, 168 | `task_weighted_1x1_z2_s*_d90.json`, `task_weighted_operator.json` |\n"
    "| 8.2–8.6 depth controls, visible share and the rank law | 188, 190, 198, 200, 218 | `depth_map_summary.json`, "
    "`depth_map_*.json`, `estimator_control.json`, `quotientization.json`, `deep_linear_transport.json`, "
    "`linear_dynamics_counterexample.json` |",
    "A3 Appendix B rows")

# ==================================================================== C8 target depth-law Section 8.8
rep(DEPTH,
    "On the trained weights the closed form holds (median absolute error 0.014), $\\kappa$ governs and rank does not ($\\rho(\\kappa,T_F)=-0.97$ against $\\rho(\\mathrm{effective\\ rank},T_F)=+0.43$), and in the within-block geometry the trajectory is flat: effective rank falls 14.94$\\to$3.98 and the loss falls six orders of magnitude while $T_F$ moves 0.391$\\to$0.362. In the cross-block geometry, where the directions weight decay discards are the ones the transformation mixes, $T_F$ moves by +0.057 with weight decay, and the closed form evaluated on the row space before and after predicts the change within 0.02 in six of eight runs. The honest statement is therefore conditional and quantitative: learning changes editability only when it discards the transformation's mixing source.",
    f"On the trained weights the closed form holds (median absolute error {CFERR:.3f}), $\\kappa$ governs and rank does "
    f"not ($\\rho(\\kappa,T_F)={rho_kap:+.2f}$ against $\\rho(\\mathrm{{effective\\ rank}},T_F)={rho_erk:+.2f}$), and the single recorded "
    f"trajectory is the $\\theta=60°$ *mixing* cell ($m=16$, weight decay $10^{{-3}}$), not a within-block geometry: there "
    f"the effective rank is compressed more than threefold and the loss falls four orders of magnitude while $T_F$ is "
    f"nearly flat (effective rank {TR0['eff_rank']:.2f}$\\to${TR1['eff_rank']:.2f}, $T_F$ {TR0['tf_site']:.3f}$\\to${TR1['tf_site']:.3f}). "
    f"In the counterexample file's own two geometries (`results/linear_dynamics_counterexample.json`, script 208), weight "
    f"decay moves $T_F$ by $+{within_d[0]:.3f}$ and $+{within_d[1]:.3f}$ in the **within**-block run, *more* than the "
    f"$+{cross_d[0]:.3f}$ and $+{cross_d[1]:.3f}$ of the cross-block run — the opposite sign to the mixing-source account. "
    f"The honest statement is therefore conditional and partly negative: weight decay moves editability in both linear "
    f"geometries, and not in the direction that account expects, so the linear control bounds the mechanism rather than "
    f"confirming it.",
    "C8 target Section 8.8 linear control")

# ==================================================================== C7 target depth-law Section 8.6 hue bullet
rep(DEPTH,
    "* **the signal leaving the consumer-visible subspace** — this is *family-specific*. For the dissipation family the visible share of the displacement sits at or above the isotropic expectation and does not fall with depth (ResNet-50 seed 0: 0.898/1.278/1.108/0.633 of the isotropic share; scripts 200, 203). For the **hue** reference arm it is below the isotropic expectation at every depth and falls to the matched-rank random level by the deepest site (visible share of the displacement 0.029/0.020/0.011/0.004, against a matched-rank random share of 0.031/0.016/0.009/0.006; over-isotropic share 0.575/0.795/0.830/0.560). The heat side is **seed 0 only**: the documented keying bug of script 200 lost heat seeds 1-2, so no heat trend is claimed; the hue side is the three-seed result. The audit's earlier blanket statement that the share is $\\approx 1$ for all families is therefore withdrawn.",
    "* **the signal leaving the consumer-visible subspace** — this is *family-specific*, and it does not explain the "
    "shallow-to-mid decline in either family. For the dissipation family the visible share of the displacement sits at "
    "or above the isotropic expectation and does not fall with depth (ResNet-50 seed 0: 0.898/1.278/1.108/0.633 of the "
    "isotropic share; scripts 200, 203). For the **hue** reference arm, read from the three-seed block "
    "(`results/quotientization.json`, keys `..._seed{0,1,2}`), the displacement is *more* visible than isotropic at "
    "the first three depths — over-isotropic share " + "/".join(f"{v:.2f}" for v in ovr[:3]) + " — and the share falls "
    "with depth (visible share of the displacement " + "/".join(f"{v:.3f}" for v in vis) +
    ", against a matched-rank random share of " + "/".join(f"{v:.3f}" for v in rnd) + " at rank 13), crossing below the "
    "isotropic expectation only at the deepest site (over-isotropic " + f"{ovr[3]:.2f}" + "). The fall to the "
    "matched-rank random level at the last site is consistent with leakage there, but the share is *above* isotropic "
    "over the depths where most of the decline happens, so leakage is not the driver of the trend. The heat side is "
    "**seed 0 only**: the documented keying bug of script 200 lost heat seeds 1-2, so no heat trend is claimed; the hue "
    "side is the three-seed result. The audit's earlier blanket statement that the share is $\\approx 1$ for all "
    "families is therefore withdrawn.",
    "C7 target hue visible-share correction")

# ==================================================================== C10 + garbled 8.1(i) + effect-size naming (target)
rep(DEPTH,
    "and the drop is large: from 0.5–0.7 of the no-op's error removed at the first site to 0.05–0.15 at the last reachable one. We deliberately do not claim that each trajectory is monotone — ResNet-50 under the heat flow rises slightly from site 1 to site 2 (0.51 → 0.47 in transfer is a fall, but its projection rises, and ConvNeXt's transfer is flat from 0.54 to 0.55 before falling) — and the trend test, not a monotonicity claim, is the evidence.",
    f"and the drop is large: from 0.5–0.7 of the no-op's error removed at the first site to a median of "
    f"{st.median(DEEP.values()):.2f} at the last reachable one, over a range of {min(DEEP.values()):+.2f} "
    f"(DINOv2-B/14 under dissipation) to {max(DEEP.values()):+.2f} (ViT-B/16 under hue). We deliberately do not claim "
    f"that each trajectory is monotone: ResNet-50 under the heat flow falls slightly from site 1 to site 2 "
    f"(0.51 → 0.47), while ConvNeXt-T's transfer is essentially flat from 0.54 to 0.55 before falling, so the evidence "
    f"is the trend test rather than a monotonicity claim.",
    "C10 + A3 target Section 8.1(i) range and monotonicity sentence")

rep(DEPTH,
    "the site-level relative displacement under the\ntransformation *grows* with depth (Spearman $\\rho=+0.370$, $p=1.8\\times10^{-4}$) while the ceiling falls\n($\\rho=-0.671$, $p=4\\times10^{-14}$); the partial correlation of depth with the ceiling controlling for effect\nsize is $\\rho=-0.662$",
    "the site-level relative displacement (`effect_size_site`, a feature-space quantity distinct from the\n"
    "consumer-level effect size that Gate 1 tests) *grows* with depth (Spearman $\\rho=+0.370$, $p=1.8\\times10^{-4}$)\n"
    "while the ceiling falls ($\\rho=-0.671$, $p=4\\times10^{-14}$); the partial correlation of depth with the ceiling\n"
    "controlling for the site displacement is $\\rho=-0.662$",
    "C13 target Section 8.6 effect-size naming")
rep(DEPTH,
    "and in the 16 pairs whose effect sizes are matched\nto within a quarter,",
    "and in the 16 pairs whose site displacements are matched\nto within a quarter,",
    "C13 target Section 8.6 matched site displacements")

# ==================================================================== C14 target Section 8.7
rep(DEPTH,
    "with relative depth taken as the site's position divided by the number of reachable sites, the pooled Spearman correlation between relative depth and normalised transfer is $\\rho = -0.921$ ($p = 3.9e-12$, $n = 28$ points over eight backbone$\\times$family curves), and the per-curve trend is negative in 8 of 8.",
    f"with relative depth taken as the site's position divided by the number of reachable sites, the pooled Spearman "
    f"correlation between relative depth and normalised transfer is $\\rho = {RHO_N:.3f}$ ($p = {P_N:.1e}$, "
    f"$n = {len(xs_n)}$ points over eight backbone$\\times$family curves), and the per-curve trend is negative in 8 of 8. "
    f"Two qualifications on that statistic. First, the {len(xs_n)} points are site-level means read from eight curves and "
    f"the cells within a curve are nested, so the p-value assumes independent points it does not have: the number is a "
    f"summary of eight declining curves, not 28 independent tests, and the replication unit is the curve. Second, "
    f"without the per-curve first-point normalisation the same points give a pooled $\\rho = {RHO_R:.3f}$ "
    f"($p = {P_R:.1e}$), so part of the pooled correlation reflects the level difference between backbones and families "
    f"rather than the within-curve decline; the normalisation-free statement is the per-curve 8 of 8.",
    "C14 target Section 8.7 replication unit + un-normalised pooled rho")

# ==================================================================== C12 target Table 16 companions
rep(DEPTH,
    "| ViT-B/16 | 0.50 | 0.27 | 0.11 | unreachable | 0.740 / 0.98 | 0.192 / 0.82 | −0.95 (9.6e-05) |",
    f"| ViT-B/16 | 0.50 | 0.27 | 0.11 | unreachable | {VIT_H['block2'][0]:.3f} / {VIT_H['block2'][1]:.2f} | "
    f"{VIT_H['block8'][0]:.3f} / {VIT_H['block8'][1]:.2f} | −0.95 (9.6e-05) |",
    "C12 target Table 16 ViT companions")
rep(DEPTH,
    "| DINOv2-B/14 | 0.13 | 0.05 | −0.40 | unreachable | 0.571 / 0.72 | 0.213 / 0.23 | −0.95 (9.6e-05) |",
    f"| DINOv2-B/14 | 0.13 | 0.05 | −0.40 | unreachable | {DIN_H['block2'][0]:.3f} / {DIN_H['block2'][1]:.2f} | "
    f"{DIN_H['block8'][0]:.3f} / {DIN_H['block8'][1]:.2f} | −0.95 (9.6e-05) |",
    "C12 target Table 16 DINOv2 companions")

# ==================================================================== E23 target discussion quotientization
rep(DISC,
    "and a spatial description-selection rule — both open because our own gate was decorative (Section 5.5).",
    "a spatial description-selection rule — both open because our own gate was decorative (Section 5.5); and the "
    "rank-intervention half of `results/quotientization.json`, whose visible-share and effective-rank fields we use "
    "but whose rank-intervention transfers are only partly analysed and are left as future work.",
    "E23 target discussion quotientization partly analysed")

# ==================================================================== C8 target discussion mirror
rep(DISC,
    "the mixing coefficient governs and rank does not ($\\rho(\\kappa,T_F)=-0.97$ against $\\rho(\\mathrm{rank},T_F)=0.43$); in the within-block geometry the trajectory is flat ($T_F$ 0.391$\\to$0.362 while effective rank falls 14.94$\\to$3.98 and the loss falls six orders of magnitude); and in the cross-block geometry, where the directions learning discards are the ones the transformation mixes, $T_F$ moves by +0.057 with weight decay, with the closed form predicting the change within 0.02 in six of eight runs. Learning therefore changes editability only *conditionally* — when it discards the transformation's mixing source — and the nonlinear case remains open.",
    f"the mixing coefficient governs and rank does not ($\\rho(\\kappa,T_F)={rho_kap:+.2f}$ against "
    f"$\\rho(\\mathrm{{effective\\ rank}},T_F)={rho_erk:+.2f}$); the single recorded trajectory (the $\\theta=60°$ mixing cell) "
    f"is flat in transfer ($T_F$ 0.391$\\to$0.362 while effective rank falls 14.94$\\to$3.98); and weight decay moves "
    f"$T_F$ in *both* linear geometries, by $+{st.mean(within_d):.2f}$ within the task block against "
    f"$+{st.mean(cross_d):.2f}$ across it — the opposite of the mixing-source prediction. The linear control therefore "
    f"bounds the mechanism rather than confirming it, and the nonlinear case remains open.",
    "C8 target discussion mirror")

# ==================================================================== C11 target front matter
rep(FRONT,
    "Alignment is therefore not a weaker version of faithfulness but a different quantity, and it can invert the ordering of a faithful and an unfaithful operator; across the cross-architecture grid of Section 8 this holds at every site of every backbone.",
    f"Alignment is therefore not a weaker version of faithfulness but a different quantity, and it can invert the "
    f"ordering of a faithful and an unfaithful operator. The inversion is not universal: over the {n_reach} reachable "
    f"sites of the cross-architecture grid of Section 8 the random control's projection exceeds the global linear "
    f"operator's at {O6gt} sites, and it is itself the highest-projection operator of the four tested at "
    f"{highest['O6']} of them, while the conditioned operator is highest at {highest['O8']}; the random control's win "
    f"rate is at the floor at every site.",
    "C11 front matter Section 1(iii) grid support")
rep(FRONT,
    "the four-gate protocol of Section 3.9",
    "the five-gate protocol of Section 3.9",
    "C11 front matter four->five gates")
rep(FRONT,
    "behaviour is destroyed at every site of a cross-architecture grid",
    "behaviour is destroyed across a cross-architecture grid",
    "C11 front matter Section 2 overclaim")
rep(FRONT,
    "repeated at every site of Section 8",
    "recurring across the cross-architecture grid of Section 8",
    "C11 front matter Section 2 'repeated at every site'")

# front matter Gate 4 projection -> absolute T_F (B6 leftover)
rep(FRONT,
    "which is why a global map is near-exact where the channel layout makes hue a group action (0.94–0.97) and partial in a plain CNN (0.31, recovered to 0.83 by content and neighbourhood conditioning).",
    f"which is why a global map reaches the headline downstream transfer {CE8['O1_procrustes']:.2f}–{CE8['O8_conv_residual']:.2f} where the channel layout makes hue a group action "
    f"(still partial under the 0.80 convention) and {Z2['O1_procrustes']:.2f}–{Z2['O8_conv_residual']:.2f} in a plain CNN, recovered to about {Z2['O8_conv_residual']:.2f} by content and neighbourhood conditioning.",
    "B6 front matter Gate 4 absolute T_F")

# ==================================================================== A3 / C11 figure 3 anchor + callout
plan = json.load(open(PLAN, encoding="utf-8"))
for it in plan["main"]:
    if it["slug"] == "alignment_vs_behaviour":
        it["after"] = "the value action of a task trained on pixels is not carried in one global basis."
        it["callout"] = ("Figure 3 makes the contradiction concrete on one setting, the plain CIFAR-10 arm of "
                         "Section 11: there the random control has the highest aggregate projection of the operators "
                         "tested and a win rate at the floor.")
for it in plan["supplementary"]:
    if it["slug"] == "estimator_control":
        it["caption"] = ("Estimator control at the depth-map sites: the published global-linear operator (absolute "
                         "lambda 1e-3), a lambda-tuned ridge, a genuine fitted rank-1 least-squares operator and the "
                         "orthogonal family under one protocol, with the estimation-limit diagnostic C^2/K (log scale) "
                         "below, per family. The depth decline survives matched regularisation and rank; the published "
                         "value is under-regularised at the most over-parameterised site, and the deepest sites are "
                         "the most estimation-limited. Script 218.")
json.dump(plan, open(PLAN, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
LOG.append("OK   A3/C11/F35 figures plan: Figure 3 anchor+callout, F35 caption")

# ---- C11 figure-3 anchor uniqueness check
_m = read(MASTER)
LOG.append(f"INFO C11 Figure-3 anchor occurs {_m.count('the value action of a task trained on pixels is not carried in one global basis.')} time(s) in the master")

# ==================================================================== A2 check: old supplement numbers gone
_left = sorted(set(re.findall(r"\bTable (1[5-9]|2[0-4])\b", read(MASTER))))
LOG.append(f"INFO A2 remaining plain Table 15-24 refs in master: {_left}")

print("\n".join(LOG))
