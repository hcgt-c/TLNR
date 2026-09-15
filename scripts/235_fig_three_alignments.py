# -*- coding: utf-8 -*-
"""235_fig_three_alignments.py — one figure, three levels of alignment, none of them faithfulness.

Upgrades F34 (single CIFAR setting) to the cross-architecture grid.  Three panels, three levels:

  A  OPERATOR level  (results/depth_map_summary.json)
     4 backbones x 2 transformation families (hue_90.0, heat_2.0) x 4 depths x 4 operator
     families = 112 points on 28 causally reachable cells.  Within an operator family the
     aggregate projection coefficient tracks the downstream transfer T_F; ACROSS operator
     types the ordering inverts -- the random orthogonal control (O6) carries a high
     projection (it beats the global linear operator at 9/28 sites and is the single
     highest-projection operator at 8/28) while its T_F is negative at 28/28 sites and its
     paired win rate is at the noise floor (<= 0.05) at 27/28.

  B  SUBSPACE level  (results/quotientization.json)
     Visible share of the transformation displacement against a matched-rank (rank 13)
     random subspace, per depth, ResNet-50, hue arm, three seeds (the only 3-seed arm).
     The displacement is *more* visible than the matched-rank random subspace at the first
     three depths (over-isotropic 2.48/2.49/1.91) and crosses below only at the deepest site
     (0.55).  The heat arm is seed-0 ONLY -- script 200's documented JSON keying bug lost
     heat seeds 1-2 -- so it is drawn as a marked, exploratory single-seed curve and no heat
     trend is claimed.

  C  STATISTIC level  (results/estimator_control.json)
     Feature-space residual against downstream T_F over the 32 (backbone x family x depth)
     cells and six estimators.  The audited Spearman correlations reproduce exactly
     (-0.846 published / -0.831 tuned ridge; -0.512 / -0.541 partialling out depth;
     ridge-tuned by family heat -0.592, hue -0.183).

     !! SIGN FINDING, reported rather than hidden !!
     `feat_resid_over_displacement` is an ERROR ratio, ||pred - Y|| / ||Y - X|| (script 218,
     line 123), so larger = WORSE feature-space fit.  rho < 0 therefore says that a BETTER
     feature-space fit goes with a BETTER downstream transfer -- the opposite of the
     manuscript's sentence "fitting the feature displacement more accurately goes with
     transporting it less faithfully" (paper/full/TPAMI_comprehensive_EN.md:486).  The same
     file family interprets the statistic the other way in
     paper/targets/TPAMI/reports/DIAGNOSTIC_VALIDATION_REPORT.md:134 (residual is an ordinal
     predictor of T_F, high residual -> failure).  Every audited NUMBER is reproduced here;
     the claimed direction is not, and the panel says so on its face.

Every plotted value comes from a JSON under results/; source file + key are recorded in
results/fig_three_alignments.json.

Usage
  OMP_NUM_THREADS=4 nice -n 15 ~/miniconda3/envs/yolo11/bin/python scripts/235_fig_three_alignments.py
Outputs
  paper/figures/F36_three_alignments.png
  results/fig_three_alignments.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

WORK = rp.REPO_ROOT
SRC_DEPTH = os.path.join(WORK, "results", "depth_map_summary.json")
SRC_QUOT = os.path.join(WORK, "results", "quotientization.json")
SRC_EST = os.path.join(WORK, "results", "estimator_control.json")
OUT_PNG = os.path.join(WORK, "paper", "figures", "F36_three_alignments.png")
OUT_JSON = os.path.join(WORK, "results", "fig_three_alignments.json")

# ------------------------------------------------------------------ Panel A constants
FAMILIES_A = ["hue_90.0", "heat_2.0"]          # both 3-seed families -> 28 reachable cells
OPS_A = [("O1", "O1 orthogonal", "#4c72b0", "o"),
         ("O2", "O2 linear", "#dd8452", "s"),
         ("O8", "O8 content+neighbourhood", "#55a868", "^"),
         ("O6", "O6 random orthogonal control", "#c44e52", "X")]

# ------------------------------------------------------------------ helpers
def rank_residual_partial_rho(x, y, z):
    """Audited partial correlation: Spearman between rank-residualised x and y (control z).

    Reproduces the manuscript's -0.512 / -0.541 / -0.592 / -0.183 exactly.
    """
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    return float(stats.spearmanr(resid(rx, rz), resid(ry, rz))[0])


def series_from_quotient(rec, backbone):
    """Ordered (visible share, isotropic expectation, over-isotropic, matched-rank random,
    effective rank, nominal rank) for the four depth sites of one backbone."""
    out = []
    for site in rec["backbones"][backbone]:
        s = rec["backbones"][backbone][site]
        mrc = s["matched_rank_control"]
        out.append(dict(site=site,
                        visible=float(s["visible_share_of_displacement"]),
                        isotropic=float(s["visible_share_isotropic_expectation"]),
                        over_isotropic=float(s["visible_share_over_isotropic"]),
                        random_share=float(mrc["random_share_at_nominal"]),
                        effective_rank=float(s["readout_effective_rank"]),
                        nominal_rank=int(mrc["nominal_rank"]),
                        visible_dim=int(s["consumer_visible_dim_of_probe"])))
    return out


# ------------------------------------------------------------------ Panel A
def panel_a(ax, rec_out):
    d = json.load(open(SRC_DEPTH, encoding="utf-8"))
    rows = []
    for key, rec in d.items():
        backbone, fam_full = key.split("|")
        if fam_full not in FAMILIES_A:
            continue
        family = fam_full.split("_")[0]
        for site, s in rec["sites"].items():
            if not s["reachable"]:
                continue
            for op, _lab, _c, _m in OPS_A:
                rows.append(dict(backbone=backbone, family=family, family_key=fam_full,
                                 site=site, depth=int(s["depth"]), op=op,
                                 projection=float(s["%s_proj_coef_agg_mean" % op]),
                                 transfer=float(s["%s_transfer_mean" % op]),
                                 win_rate=float(s["%s_win_rate_mean" % op])))
    n_cells = len({(r["backbone"], r["family"], r["depth"]) for r in rows})

    # within-family Spearman
    rho = {}
    for op, lab, _c, _m in OPS_A:
        x = [r["projection"] for r in rows if r["op"] == op]
        y = [r["transfer"] for r in rows if r["op"] == op]
        rho[op] = float(stats.spearmanr(x, y)[0])
    rho_hue = {op: float(stats.spearmanr([r["projection"] for r in rows if r["op"] == op and r["family"] == "hue"],
                                         [r["transfer"] for r in rows if r["op"] == op and r["family"] == "hue"])[0])
               for op, _l, _c, _m in OPS_A}
    rho_heat = {op: float(stats.spearmanr([r["projection"] for r in rows if r["op"] == op and r["family"] == "heat"],
                                          [r["transfer"] for r in rows if r["op"] == op and r["family"] == "heat"])[0])
                for op, _l, _c, _m in OPS_A}

    # across-type statistics
    aligned = [r for r in rows if r["op"] in ("O1", "O2", "O8")]
    rho_aligned = float(stats.spearmanr([r["projection"] for r in aligned],
                                        [r["transfer"] for r in aligned])[0])
    rho_all = float(stats.spearmanr([r["projection"] for r in rows],
                                    [r["transfer"] for r in rows])[0])
    mean_proj = {op: float(np.mean([r["projection"] for r in rows if r["op"] == op]))
                 for op, _l, _c, _m in OPS_A}
    mean_tf = {op: float(np.mean([r["transfer"] for r in rows if r["op"] == op]))
               for op, _l, _c, _m in OPS_A}
    rho_across_type = float(stats.spearmanr([mean_proj[o] for o, _l, _c, _m in OPS_A],
                                            [mean_tf[o] for o, _l, _c, _m in OPS_A])[0])

    # site-level counts
    cells = {}
    for r in rows:
        cells.setdefault((r["backbone"], r["family"], r["depth"]), {})[r["op"]] = r
    counts = {
        "n_reachable_cells": n_cells,
        "O6_proj_gt_O2": int(sum(1 for c in cells.values() if c["O6"]["projection"] > c["O2"]["projection"])),
        "O6_proj_gt_O1": int(sum(1 for c in cells.values() if c["O6"]["projection"] > c["O1"]["projection"])),
        "O6_proj_gt_O8": int(sum(1 for c in cells.values() if c["O6"]["projection"] > c["O8"]["projection"])),
        "O6_highest_projection": int(sum(1 for c in cells.values()
                                         if c["O6"]["projection"] == max(c[o]["projection"] for o, _l, _c, _m in OPS_A))),
        "O8_highest_projection": int(sum(1 for c in cells.values()
                                         if c["O8"]["projection"] == max(c[o]["projection"] for o, _l, _c, _m in OPS_A))),
        "O2_highest_projection": int(sum(1 for c in cells.values()
                                         if c["O2"]["projection"] == max(c[o]["projection"] for o, _l, _c, _m in OPS_A))),
        "O1_highest_projection": int(sum(1 for c in cells.values()
                                         if c["O1"]["projection"] == max(c[o]["projection"] for o, _l, _c, _m in OPS_A))),
        "O6_win_rate_le_0p05": int(sum(1 for c in cells.values() if c["O6"]["win_rate"] <= 0.05)),
        "O6_transfer_negative": int(sum(1 for c in cells.values() if c["O6"]["transfer"] < 0)),
    }

    # ---- draw.  O6's transfer tail reaches -4.6, so the y axis is symlog
    # (linthresh 0.3): every one of the 112 points is drawn, the dense 0..0.7 band
    # stays linear, and the negative O6 tail is compressed rather than cut.
    short_lab = {"O1": "O1 orthogonal", "O2": "O2 linear",
                 "O8": "O8 content+neigh.", "O6": "O6 random control"}
    for op, lab, colour, marker in OPS_A:
        pts = [r for r in rows if r["op"] == op]
        ax.scatter([r["projection"] for r in pts], [r["transfer"] for r in pts],
                   s=27 if op != "O6" else 40, marker=marker, color=colour,
                   label=short_lab[op], alpha=0.78, linewidths=0.5,
                   edgecolors="white" if op != "O6" else "#7a2020", zorder=3 if op != "O6" else 4)
    # pooled O1/O2/O8 trend (the within-family relation the manuscript claims)
    xs = np.array([r["projection"] for r in aligned])
    ys = np.array([r["transfer"] for r in aligned])
    coef = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(grid, np.polyval(coef, grid), color="0.25", lw=1.4, ls="--", zorder=2,
            label="OLS pooled O1/O2/O8")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_yscale("symlog", linthresh=0.3)
    ax.set_ylim(-5.2, 0.9)
    ax.set_yticks([-4, -2, -1, -0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["-4", "-2", "-1", "-0.5", "-0.2", "0", "0.2", "0.4", "0.6", "0.8"])

    ax.set_xlabel("aggregate projection coefficient")
    ax.set_ylabel("downstream transfer $T_F$\n(symlog, linear within $\\pm$0.3)")
    ax.set_title("A. Operator level: projection tracks transfer within a family,\n"
                 "but the random control tops projection with $T_F<0$", fontsize=9.5)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=5.8, loc="lower right", framealpha=0.92, handletextpad=0.3,
              borderpad=0.35, labelspacing=0.28)
    box = ("within-family $\\rho$(proj, $T_F$), n=%d\n"
           "  O1 +%.2f   O2 +%.2f   O8 +%.2f\n"
           "pooled O1/O2/O8 +%.2f\n"
           "pooled all four +%.2f    O6 %.2f\n"
           "O6 > O2 proj %d/%d; O6 top %d/%d\n"
           "O8 top %d/%d; O6 win$\\leq$0.05 %d/%d\n"
           "O6 mean proj %.2f > O1 %.2f, but\n"
           "  mean $T_F$ %+.2f (negative %d/%d)"
           ) % (n_cells,
                rho["O1"], rho["O2"], rho["O8"],
                rho_aligned, rho_all, rho["O6"],
                counts["O6_proj_gt_O2"], n_cells, counts["O6_highest_projection"], n_cells,
                counts["O8_highest_projection"], n_cells,
                counts["O6_win_rate_le_0p05"], n_cells,
                mean_proj["O6"], mean_proj["O1"], mean_tf["O6"], counts["O6_transfer_negative"], n_cells)
    ax.text(0.018, 0.975, box, transform=ax.transAxes, fontsize=5.4, va="top", ha="left",
            family="monospace", bbox=dict(fc="white", ec="0.75", alpha=0.94, lw=0.6))
    rec_out["panel_A_operator_level_extra"] = {
        "y_axis": "symlog with linthresh=0.3; all %d points drawn, no clipping" % len(rows),
        "O6_transfer_quantiles": [round(float(v), 4) for v in
                                  np.quantile([r["transfer"] for r in rows if r["op"] == "O6"],
                                              [0.0, 0.25, 0.5, 0.75, 1.0])]}

    rec_out["panel_A_operator_level"] = {
        "source_file": "results/depth_map_summary.json",
        "source_keys": ["<backbone>|<family_key>.sites.<site>.reachable",
                        "<backbone>|<family_key>.sites.<site>.<op>_proj_coef_agg_mean",
                        "<backbone>|<family_key>.sites.<site>.<op>_transfer_mean",
                        "<backbone>|<family_key>.sites.<site>.<op>_win_rate_mean"],
        "family_keys_used": FAMILIES_A,
        "families_excluded": {"heat_1.0": "1 seed only; excluding it keeps the grid at n_seeds=3 and the 28-site count"},
        "operators": {op: lab for op, lab, _c, _m in OPS_A},
        "n_points": len(rows),
        "within_family_spearman_proj_vs_TF": rho,
        "within_family_spearman_by_family": {"hue": rho_hue, "heat": rho_heat},
        "across_type": {"pooled_O1_O2_O8_rho": rho_aligned,
                        "pooled_all_four_rho": rho_all,
                        "O6_alone_rho": rho["O6"],
                        "mean_projection_per_type": mean_proj,
                        "mean_transfer_per_type": mean_tf,
                        "rho_of_type_mean_proj_vs_type_mean_TF": rho_across_type},
        "site_counts": counts,
        "points": rows,
    }
    return n_cells


# ------------------------------------------------------------------ Panel B
def panel_b(ax, rec_out):
    q = json.load(open(SRC_QUOT, encoding="utf-8"))
    hue_keys = [k for k in q if q[k]["family"] == "hue" and q[k].get("readout") is None]
    hue_keys = sorted(hue_keys, key=lambda k: q[k]["seed"])
    heat_key = [k for k in q if q[k]["family"] == "heat"
                and q[k].get("readout") == "hf" and "resnet50," in k][0]

    BB = "resnet50"
    hue_seeds = [series_from_quotient(q[k], BB) for k in hue_keys]
    num_fields = [f for f in hue_seeds[0][0] if f != "site"]
    hue_mean = [{**{"site": hue_seeds[0][i]["site"]},
                 **{f: float(np.mean([s[i][f] for s in hue_seeds])) for f in num_fields}}
                for i in range(4)]
    heat0 = series_from_quotient(q[heat_key], BB)

    depths = np.arange(1, 5)
    vis = np.array([s["visible"] for s in hue_mean])
    rnd = np.array([s["random_share"] for s in hue_mean])
    over = np.array([s["over_isotropic"] for s in hue_mean])
    heat_vis = np.array([s["visible"] for s in heat0])
    heat_over = np.array([s["over_isotropic"] for s in heat0])

    for i, sd in enumerate(hue_seeds):
        ax.plot(depths, [s["visible"] for s in sd], marker="o", ms=3.0, lw=0.9,
                color="#4c72b0", alpha=0.45, zorder=3,
                label="hue seeds 0/1/2 (ResNet-50)" if i == 0 else None)
    ax.plot(depths, vis, marker="o", ms=6, lw=2.4, color="#1f3f66", zorder=5,
            label="hue seed mean (n=3)")
    ax.plot(depths, rnd, marker="D", ms=4.5, lw=1.5, ls="--", color="#7f7f7f", zorder=4,
            label="matched-rank random subspace, rank %d" % hue_mean[0]["nominal_rank"])
    ax.plot(depths, heat_vis, marker="s", ms=4.5, lw=1.6, ls=":", color="#dd8452", zorder=4,
            label="heat, seed 0 only (exploratory)")

    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("depth site (1 = shallowest, 4 = deepest)")
    ax.set_ylabel("visible share of the transformation displacement\n(log scale)")
    ax.set_title("B. Subspace level: the displacement is more visible than a\n"
                 "matched-rank random subspace until the deepest site", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=5.8, loc="upper right", framealpha=0.93, handletextpad=0.35,
              borderpad=0.35, labelspacing=0.28)
    box = ("hue, ResNet-50, 3-seed mean\n"
           " visible   %.3f %.3f %.3f %.4f\n"
           " random    %.3f %.3f %.3f %.4f\n"
           " over-iso  %.2f  %.2f  %.2f  %.2f\n"
           "above the random subspace at depths 1-3;\n"
           "crosses below only at depth 4\n"
           "heat = seed 0 ONLY (script 200 keying bug\n"
           "lost seeds 1-2) -- no heat trend claimed"
           ) % (vis[0], vis[1], vis[2], vis[3],
                rnd[0], rnd[1], rnd[2], rnd[3],
                over[0], over[1], over[2], over[3])
    ax.text(0.018, 0.02, box, transform=ax.transAxes, fontsize=5.3, va="bottom", ha="left",
            family="monospace", bbox=dict(fc="white", ec="0.75", alpha=0.94, lw=0.6))

    rec_out["panel_B_subspace_level"] = {
        "source_file": "results/quotientization.json",
        "source_keys": {
            "hue_seeds": {k: "backbones.resnet50.<layer{1..4}>.{visible_share_of_displacement,"
                             "visible_share_isotropic_expectation,visible_share_over_isotropic,"
                             "matched_rank_control.random_share_at_nominal,readout_effective_rank}" for k in hue_keys},
            "heat_seed0": heat_key + ".backbones.resnet50.<layer{1..4}> (readout=hf)",
        },
        "backbone": BB,
        "depth_sites": [s["site"] for s in hue_mean],
        "hue_per_seed": {("seed%d" % q[k]["seed"]): [{f: s[f] for f in
                         ("site", "visible", "isotropic", "over_isotropic", "random_share",
                          "effective_rank", "nominal_rank", "visible_dim")} for s in sd]
                         for k, sd in zip(hue_keys, hue_seeds)},
        "hue_seed_mean": [{f: s[f] for f in ("site", "visible", "isotropic", "over_isotropic",
                                             "random_share", "effective_rank", "nominal_rank")} for s in hue_mean],
        "heat_seed0_only": [{f: s[f] for f in ("site", "visible", "isotropic", "over_isotropic",
                                               "random_share", "effective_rank", "nominal_rank")} for s in heat0],
        "heat_caveat": "documented script-200 JSON keying bug lost heat seeds 1-2; no heat trend is claimed",
        "manuscript_audited": {"visible_share": [0.126, 0.063, 0.024, 0.004],
                               "over_isotropic": [2.48, 2.49, 1.91, 0.55],
                               "matched_rank_random": [0.051, 0.025, 0.013, 0.006]},
    }
    return over


# ------------------------------------------------------------------ Panel C
def panel_c(ax, rec_out):
    cells = json.load(open(SRC_EST, encoding="utf-8"))["cells"]
    EST = [("O2_pub", "O2 published (linear)", "#4c72b0", "o"),
           ("ridge_tun", "ridge tuned (validation)", "#dd8452", "s"),
           ("O1", "O1 orthogonal", "#c44e52", "P"),
           ("rank1_ls", "rank-1 least squares", "#55a868", "^"),
           ("rank1_mean", "rank-1 mean-aligned", "#8172b3", "v"),
           ("rank1_rnd", "rank-1 random dir.", "#937860", "D")]

    rec = {}
    for name, lab, colour, marker in EST:
        x = np.array([c["estimators"][name]["feat_resid_over_displacement"] for c in cells.values()])
        y = np.array([c["estimators"][name]["transfer"] for c in cells.values()])
        z = np.array([c["depth"] for c in cells.values()])
        fam = np.array([c["family"] for c in cells.values()])
        rec[name] = {"label": lab,
                     "n": int(len(x)),
                     "rho_raw": float(stats.spearmanr(x, y)[0]),
                     "rho_partial_depth": rank_residual_partial_rho(x, y, z),
                     "by_family_partial_depth": {
                         f: rank_residual_partial_rho(x[fam == f], y[fam == f], z[fam == f])
                         for f in ("heat", "hue")}}
        ax.scatter(x, y, s=26, marker=marker, color=colour, alpha=0.72, linewidths=0.5,
                   edgecolors="white", label=lab, zorder=3)

    # pooled trend over all six estimators (the "fit better -> transport ..." relation)
    X = np.array([c["estimators"][n]["feat_resid_over_displacement"] for c in cells.values() for n, _l, _c, _m in EST])
    Y = np.array([c["estimators"][n]["transfer"] for c in cells.values() for n, _l, _c, _m in EST])
    Z = np.array([c["depth"] for c in cells.values() for n, _l, _c, _m in EST])
    pooled_rho = float(stats.spearmanr(X, Y)[0])
    pooled_partial = rank_residual_partial_rho(X, Y, Z)
    coef = np.polyfit(X, Y, 1)
    grid = np.linspace(X.min(), X.max(), 50)
    ax.plot(grid, np.polyval(coef, grid), color="0.2", lw=1.5, ls="--", zorder=2,
            label="OLS, all six estimators pooled (n=%d)" % len(X))
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")

    ax.set_xlabel("feature-space residual $\\|{\\rm pred}-Y\\|\\,/\\,\\|Y-X\\|$   "
                  "(larger = worse feature-space fit)")
    ax.set_ylabel("downstream transfer $T_F$")
    ax.set_title("C. Statistic level: the feature residual anti-correlates with $T_F$\n"
                 "(audited $\\rho$ = -0.846 raw, -0.512 depth-partial)", fontsize=9.5)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=5.6, loc="upper right", framealpha=0.93, handletextpad=0.35,
              borderpad=0.35, labelspacing=0.28)
    box = ("audited numbers (32 cells, Spearman)\n"
           " O2 published  rho %+.3f  partial %+.3f\n"
           " ridge tuned   rho %+.3f  partial %+.3f\n"
           " ridge by family, partial:\n"
           "   heat %+.3f   hue %+.3f\n"
           " rank-1 LS %+.3f   O1 %+.3f\n"
           " all six pooled (n=%d) rho %+.3f\n"
           "SIGN CHECK: x is an ERROR ratio\n"
           " (script 218 line 123), so rho<0 means\n"
           " a BETTER feature fit gives BETTER T_F.\n"
           " The manuscript's 'fitting better\n"
           " transports it worse' is the OPPOSITE\n"
           " sign and is NOT supported by this data."
           ) % (rec["O2_pub"]["rho_raw"], rec["O2_pub"]["rho_partial_depth"],
                rec["ridge_tun"]["rho_raw"], rec["ridge_tun"]["rho_partial_depth"],
                rec["ridge_tun"]["by_family_partial_depth"]["heat"],
                rec["ridge_tun"]["by_family_partial_depth"]["hue"],
                rec["rank1_ls"]["rho_raw"], rec["O1"]["rho_raw"],
                len(X), pooled_rho)
    ax.text(0.018, 0.02, box, transform=ax.transAxes, fontsize=5.2, va="bottom", ha="left",
            family="monospace", bbox=dict(fc="#fff6f6", ec="#c44e52", alpha=0.95, lw=0.9))

    rec_out["panel_C_statistic_level"] = {
        "source_file": "results/estimator_control.json",
        "source_keys": "cells.<backbone>|<family>|depth<n>.estimators.<name>.{feat_resid_over_displacement,transfer}",
        "field_definition": "scripts/218_estimator_control.py line 123: "
                            "feat_resid_over_displacement = ||pred - Y|| / ||Y - X||  (an ERROR ratio; "
                            "larger = worse feature fit; verified equal to feat_err_to_real / feat_displacement_norm)",
        "partial_method": "Spearman correlation of rank-residualised x and y, controlling rank(depth); "
                          "all 32 cells for the headline, all 16 cells per family for the by-family split",
        "per_estimator": rec,
        "pooled_six_estimators": {"n": int(len(X)), "rho_raw": pooled_rho,
                                  "rho_partial_depth": pooled_partial},
        "manuscript_audited": {"O2_pub_rho": -0.846, "ridge_tun_rho": -0.831,
                               "O2_pub_partial_depth": -0.512, "ridge_tun_partial_depth": -0.541,
                               "ridge_tun_partial_heat": -0.592, "ridge_tun_partial_hue": -0.183},
        "sign_discrepancy": "ALL audited numbers reproduce. The claimed direction does not: because the "
                            "plotted x is an error ratio, rho<0 means better feature-space fit -> better "
                            "downstream transfer. The manuscript's 'fitting the feature displacement more "
                            "accurately goes with transporting it less faithfully' "
                            "(paper/full/TPAMI_comprehensive_EN.md:486) is the opposite sign. "
                            "The same statistic is read with this sign in "
                            "paper/targets/TPAMI/reports/DIAGNOSTIC_VALIDATION_REPORT.md:134.",
    }


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
    rec_out = {"figure": "paper/figures/F36_three_alignments.png",
               "title": "Three kinds of alignment, and none of them is faithfulness",
               "sources": {"panel_A": "results/depth_map_summary.json",
                           "panel_B": "results/quotientization.json",
                           "panel_C": "results/estimator_control.json"},
               "style_reference": "scripts/217_fig_random_contradiction.py"}
    panel_a(axes[0], rec_out)
    panel_b(axes[1], rec_out)
    panel_c(axes[2], rec_out)
    fig.suptitle("Three kinds of alignment, and none of them is faithfulness", fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    # ---- explicit audited-vs-reproduced check (RULES: report, never silently adjust) ----
    A = rec_out["panel_A_operator_level"]
    B = rec_out["panel_B_subspace_level"]
    C = rec_out["panel_C_statistic_level"]
    counts = A["site_counts"]
    over = [x["over_isotropic"] for x in B["hue_seed_mean"]]
    vis = [x["visible"] for x in B["hue_seed_mean"]]
    rnd = [x["random_share"] for x in B["hue_seed_mean"]]
    heat_over = [x["over_isotropic"] for x in B["heat_seed0_only"]]
    rec_out["reproduction_check"] = {
        "panel_A": {
            "audited": {"within_family_rho_approx": 0.92,
                        "O6_proj_gt_O2": 9, "O6_highest_projection": 8,
                        "O8_highest_projection": 20, "O6_win_rate_le_0p05": 27,
                        "n_reachable_cells": 28},
            "reproduced": {"within_family_rho": A["within_family_spearman_proj_vs_TF"],
                           "O6_proj_gt_O2": counts["O6_proj_gt_O2"],
                           "O6_highest_projection": counts["O6_highest_projection"],
                           "O8_highest_projection": counts["O8_highest_projection"],
                           "O6_win_rate_le_0p05": counts["O6_win_rate_le_0p05"],
                           "n_reachable_cells": counts["n_reachable_cells"],
                           "across_type_pooled_O1_O2_O8_rho": A["across_type"]["pooled_O1_O2_O8_rho"],
                           "across_type_pooled_all_four_rho": A["across_type"]["pooled_all_four_rho"]},
            "verdict": "MATCH on every count (9/8/20/27 of 28). The '+0.92' summary is the "
                       "within-family value of the global-linear family (O2 hue +0.934, heat +0.916, "
                       "mean 0.925); recomputed per family directly it is O1 +0.823, O2 +0.944, "
                       "O8 +0.941 (n=28 each), so the '+0.92' holds for O2/O8 and understates O1 on heat.",
        },
        "panel_B": {
            "audited": {"over_isotropic_first_three": [2.48, 2.49, 1.91],
                        "over_isotropic_deepest": 0.55,
                        "visible_share": [0.126, 0.063, 0.024, 0.004],
                        "matched_rank_random": [0.051, 0.025, 0.013, 0.006]},
            "reproduced": {"over_isotropic": [round(v, 4) for v in over],
                           "visible_share": [round(v, 5) for v in vis],
                           "matched_rank_random": [round(v, 4) for v in rnd],
                           "heat_seed0_over_isotropic": [round(v, 4) for v in heat_over]},
            "verdict": "MATCH except the deepest visible share: the audited 0.004 is 0.00347 in the "
                       "file (0.126/0.063/0.024/0.0035), i.e. the manuscript rounds 0.0035 up to 0.004. "
                       "over-isotropic 2.4816/2.4871/1.9087/0.5462 -> 2.48/2.49/1.91/0.55 exactly; "
                       "matched-rank random 0.0512/0.0254/0.0126/0.0063 -> 0.051/0.025/0.013/0.006 exactly. "
                       "heat seed-0 over-isotropic 0.8976/1.2777/1.1080/0.6328 -> 0.898/1.278/1.108/0.633 exactly.",
        },
        "panel_C": {
            "audited": {"O2_pub_rho": -0.846, "ridge_tun_rho": -0.831,
                        "O2_pub_partial_depth": -0.512, "ridge_tun_partial_depth": -0.541,
                        "ridge_tun_partial_heat": -0.592, "ridge_tun_partial_hue": -0.183,
                        "rank1_ls_rho": -0.570, "O1_rho": -0.720},
            "reproduced": {
                "O2_pub_rho": round(C["per_estimator"]["O2_pub"]["rho_raw"], 4),
                "ridge_tun_rho": round(C["per_estimator"]["ridge_tun"]["rho_raw"], 4),
                "O2_pub_partial_depth": round(C["per_estimator"]["O2_pub"]["rho_partial_depth"], 4),
                "ridge_tun_partial_depth": round(C["per_estimator"]["ridge_tun"]["rho_partial_depth"], 4),
                "ridge_tun_partial_heat": round(C["per_estimator"]["ridge_tun"]["by_family_partial_depth"]["heat"], 4),
                "ridge_tun_partial_hue": round(C["per_estimator"]["ridge_tun"]["by_family_partial_depth"]["hue"], 4),
                "rank1_ls_rho": round(C["per_estimator"]["rank1_ls"]["rho_raw"], 4),
                "O1_rho": round(C["per_estimator"]["O1"]["rho_raw"], 4)},
            "verdict": "MATCH on all eight numbers (to 3 dp). ONE NON-NUMERIC DISCREPANCY: the "
                       "direction. As stored, the x axis is an ERROR ratio, so rho<0 means a better "
                       "feature-space fit gives a BETTER T_F; the manuscript's stated direction "
                       "('fitting better transports it worse') is the opposite sign and is not "
                       "supported by these data.",
        },
    }

    json.dump(rec_out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT_PNG)
    print("wrote", OUT_JSON)
    print("A: n=%d reachable cells, %d points" % (counts["n_reachable_cells"], A["n_points"]))
    print("A: within-family rho(proj,T_F) ",
          {k: round(v, 3) for k, v in A["within_family_spearman_proj_vs_TF"].items()})
    print("A: across-type pooled O1/O2/O8 rho=%.4f, pooled all four rho=%.4f, O6 alone rho=%.4f"
          % (A["across_type"]["pooled_O1_O2_O8_rho"], A["across_type"]["pooled_all_four_rho"],
             A["across_type"]["O6_alone_rho"]))
    print("A: counts  O6>O2 %d/28  O6 top %d/28  O8 top %d/28  O6 win<=0.05 %d/28  O6 T_F<0 %d/28"
          % (counts["O6_proj_gt_O2"], counts["O6_highest_projection"],
             counts["O8_highest_projection"], counts["O6_win_rate_le_0p05"],
             counts["O6_transfer_negative"]))
    print("B: hue visible %s | random %s | over-iso %s"
          % ([round(v, 4) for v in vis], [round(v, 4) for v in rnd], [round(v, 3) for v in over]))
    print("B: heat seed0 only over-iso %s (no heat trend claimed)" % [round(v, 3) for v in heat_over])
    print("C: O2_pub rho=%.4f partial=%.4f | ridge_tun rho=%.4f partial=%.4f | by family %s"
          % (C["per_estimator"]["O2_pub"]["rho_raw"], C["per_estimator"]["O2_pub"]["rho_partial_depth"],
             C["per_estimator"]["ridge_tun"]["rho_raw"], C["per_estimator"]["ridge_tun"]["rho_partial_depth"],
             {k: round(v, 4) for k, v in C["per_estimator"]["ridge_tun"]["by_family_partial_depth"].items()}))
    print("C: DISCREPANCY -- all audited numbers reproduce, but the x field is an ERROR ratio, "
          "so rho<0 = better fit -> better transport; the manuscript's stated direction is opposite.")


if __name__ == "__main__":
    main()
