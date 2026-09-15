# -*- coding: utf-8 -*-
"""234_fig_correspondence.py — the paper's strongest POSITIVE result: "the correspondence holds,
and it is cheap".

2x2 panels, all values recomputed from stored results under results/ (no new runs):

  A  the fixed closed-form operator rho(D) = diag(1, 1, R(D), R(2D)) on the learned harmonic colour
     code, evaluated ZERO-SHOT on unseen shapes: per-(s,v)-cell hue error over the 64-cell grid.
     Source: results/code3d_sv_lowext.json  ->  per_cell[*].median_err / .mean_err, summary.*
     The Delta = 40 deg equivariance residual (2.4 deg) is NOT stored in that file (key absent);
     it lives in results/code3d_v3.json -> rho_d40_deg = 2.36.  Used as an annotation only.

  B  band-limited linear map on frozen features vs copying, per (test shape, harmonic order K).
     Source: results/hue_prediction_demo.json -> dense5[0..3]  (relMSE_op, copy, gain, cos).

  C  cost of the region-level operation (shift the code field in a region) against a pixel
     recolour + re-forward.  Source: results/region_rho_demo.json -> region_rho_cost_s,
     pixel_route_cost_s.

  D  architecture contrast — the channel layout sets the ceiling.  Downstream transfer
     T_F = 1 - d(F(Wz), F(gz)) / d(F(z), F(gz)) at Delta = 90 deg, stage 1, hue.
     Source: results/causal_cifar_summary.json -> [ce8_stage1|z2_stage1]["90"].ops[*].rel_l2_vs_real
     (reference null in ops.null.rel_l2_vs_real); per-seed sd from results/causal_cifar_{ce8,z2}_s{0,1,2}.json.

Outputs: paper/figures/F35_correspondence_and_cost.png (dpi=200), results/fig_correspondence.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

WORK = rp.REPO_ROOT
SRC_A = os.path.join(WORK, "results", "code3d_sv_lowext.json")
SRC_A40 = os.path.join(WORK, "results", "code3d_v3.json")          # Delta=40 residual provenance
SRC_B = os.path.join(WORK, "results", "hue_prediction_demo.json")
SRC_C = os.path.join(WORK, "results", "region_rho_demo.json")
SRC_D = os.path.join(WORK, "results", "causal_cifar_summary.json")
SRC_D_SEEDS = {"ce8_stage1": [os.path.join(WORK, "results", "causal_cifar_ce8_s%d.json" % i) for i in range(3)],
               "z2_stage1": [os.path.join(WORK, "results", "causal_cifar_z2_s%d.json" % i) for i in range(3)]}
OUT_PNG = os.path.join(WORK, "paper", "figures", "F35_correspondence_and_cost.png")
OUT_JSON = os.path.join(WORK, "results", "fig_correspondence.json")

DELTA_D = "90"                       # hue shift at which the audited architecture numbers were taken
CHECK = []                           # (name, got, audited, tol, ok)


def check(name, got, audited, tol):
    ok = abs(float(got) - float(audited)) <= tol
    CHECK.append({"name": name, "recomputed": round(float(got), 4), "audited": audited,
                  "tol": tol, "ok": bool(ok)})
    print("  [%s] %-46s recomputed=%.4f audited=%s (tol %s)" %
          ("PASS" if ok else "FAIL", name, got, audited, tol))
    return ok


def panel_a(ax):
    d = json.load(open(SRC_A, encoding="utf-8"))
    gs, gv, pc = d["grid_s"], d["grid_v"], d["per_cell"]
    alias = d.get("per_cell_alias", {})
    kf = lambda s, v: alias.get("s%g_v%g" % (s, v), "s%g_v%g" % (s, v))
    med = np.zeros((len(gv), len(gs)))
    avg = np.zeros_like(med)
    for i, v in enumerate(gv):
        for j, s in enumerate(gs):
            c = pc[kf(s, v)]
            med[i, j] = c["median_err"]
            avg[i, j] = c["mean_err"]
    all_med, all_avg = med.ravel(), avg.ravel()

    med_of_cells = float(np.median(all_med))              # headline "median per-cell error"
    n_med30 = int((all_med < 30).sum())
    n_avg30 = int((all_avg < 30).sum())
    n_avg30_summary = int(d["summary"]["n_cells_err_le_thr"])
    corner_mean = float(d["summary"]["err_deg_at_s1_v1"])
    corner_med = float(pc[kf(1.0, 1.0)]["median_err"])

    check("A median per-cell hue error (deg)", med_of_cells, 3.4, 0.1)
    check("A cells <=30 deg, per-cell mean criterion", n_avg30, 59, 0)
    check("A cells <=30 deg, summary.n_cells_err_le_thr", n_avg30_summary, 59, 0)
    check("A cells <=30 deg, per-cell median criterion", n_med30, 61, 0)
    check("A saturated corner (s=v=1) mean error (deg)", corner_mean, 2.1, 0.1)

    # Delta = 40 deg equivariance residual: audited 2.4 deg, claimed provenance code3d_sv_lowext.json
    has_a40 = False
    try:
        d40 = json.load(open(SRC_A, encoding="utf-8"))
        has_a40 = "rho_d40_deg" in d40
    except Exception:
        pass
    a40, a40_src = None, None
    if has_a40:
        a40, a40_src = float(d40["rho_d40_deg"]), "results/code3d_sv_lowext.json:rho_d40_deg"
    else:
        d3 = json.load(open(SRC_A40, encoding="utf-8"))
        if "rho_d40_deg" in d3:
            a40, a40_src = float(d3["rho_d40_deg"]), "results/code3d_v3.json:rho_d40_deg"
    if a40 is not None:
        check("A Delta=40 deg residual (deg)", a40, 2.4, 0.05)

    norm = Normalize(vmin=0.0, vmax=30.0)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_over("#4a0c3b")
    im = ax.pcolormesh(np.arange(len(gs) + 1), np.arange(len(gv) + 1), med,
                       cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
    for i in range(len(gv)):
        for j in range(len(gs)):
            v = med[i, j]
            ax.text(j + 0.5, i + 0.5, ("%.1f" % v) if v < 30 else ("%.0f" % v),
                    ha="center", va="center", fontsize=5.4,
                    color="white" if v > 20 else "black")
    ax.set_xticks(np.arange(len(gs)) + 0.5); ax.set_xticklabels(["%g" % s for s in gs], fontsize=7)
    ax.set_yticks(np.arange(len(gv)) + 0.5); ax.set_yticklabels(["%g" % v for v in gv], fontsize=7)
    ax.set_xlabel("saturation  $s$", fontsize=8.5)
    ax.set_ylabel("value  $v$", fontsize=8.5)
    cb = ax.figure.colorbar(im, ax=ax, extend="max", pad=0.02)
    cb.set_label("per-cell hue error, median (deg)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("A  Zero-shot closed form $\\rho(\\Delta)$ on the learned code\n"
                 "median %.1f$^\\circ$  ·  %d/64 cells $\\leq$30$^\\circ$  ·  saturated corner %.1f$^\\circ$%s\n"
                 "%d/64 by per-cell mean; %d/64 by median; dark cells exceed 30$^\\circ$"
                 % (med_of_cells, n_avg30, corner_mean,
                    ("  ·  $\\Delta$=40$^\\circ$ residual %.1f$^\\circ$" % a40
                     if a40 is not None else "  ·  $\\Delta$=40$^\\circ$: not stored"),
                    n_avg30, n_med30),
                 fontsize=7.6)
    return {"median_per_cell_median_err_deg": med_of_cells,
            "median_per_cell_mean_err_deg": float(np.median(all_avg)),
            "n_cells_le_30_mean": n_avg30, "n_cells_le_30_median": n_med30,
            "n_cells_total": int(all_med.size),
            "corner_s1_v1_mean_err_deg": corner_mean, "corner_s1_v1_median_err_deg": corner_med,
            "delta40_residual_deg": a40, "delta40_source": a40_src,
            "delta40_present_in_code3d_sv_lowext": bool(has_a40),
            "grid_s": gs, "grid_v": gv, "median_err_grid_v_by_s": med.tolist(),
            "threshold_deg": 30.0,
            "source": "results/code3d_sv_lowext.json: per_cell[*].median_err/.mean_err, summary.*"}


def panel_b(ax):
    d = json.load(open(SRC_B, encoding="utf-8"))
    rows = d["dense5"]
    shapes = ["pentagon", "hexagon"]                 # scripts/64 test shapes si=(4,5) in this order
    labels, op, cp, gain, cos = [], [], [], [], []
    for i, r in enumerate(rows):
        labels.append("%s\n$K$=%d\ncos %.3f" % (shapes[i // 2] if len(rows) == 4 else "setting %d" % i,
                                                r["K"], r["cos"]))
        op.append(100.0 * r["relMSE_op"]); cp.append(100.0 * r["copy"])
        gain.append(r["gain"]); cos.append(r["cos"])
    op, cp = np.array(op), np.array(cp)
    check("B min relative error of predicted map (%)", op.min(), 0.17, 0.005)
    check("B max relative error of predicted map (%)", op.max(), 0.40, 0.005)
    check("B min gain over copy (x)", min(gain), 6.7, 0.05)
    check("B max gain over copy (x)", max(gain), 16.6, 0.05)
    check("B min cosine", min(cos), 0.955, 0.001)
    check("B max cosine", max(cos), 0.976, 0.001)

    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, op, w, color="#2b6cb0", label="band-limited predicted map")
    ax.bar(x + w / 2, cp, w, color="#b0b0b0", label="copy anchor (baseline)")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 12)
    for xi, (a, b, g) in enumerate(zip(op, cp, gain)):
        ax.text(xi - w / 2, a * 1.18, "%.2f%%" % a, ha="center", fontsize=6.8)
        ax.text(xi + w / 2, b * 1.18, "%.2f%%" % b, ha="center", fontsize=6.8)
        ax.text(xi, 6.2, "%.1f$\\times$" % g, ha="center", fontsize=7.4, color="#1a365d",
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("relative feature error (%)", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("B  Band-limited maps on frozen features beat copying\n"
                 "%.2f–%.2f%% relative error, %.1f–%.1f$\\times$ gain, cos %.3f–%.3f"
                 % (op.min(), op.max(), min(gain), max(gain), min(cos), max(cos)), fontsize=7.6)
    return {"settings": labels, "relMSE_op_pct": op.tolist(), "copy_pct": cp.tolist(),
            "gain_over_copy": gain, "cosine": cos,
            "source": "results/hue_prediction_demo.json:dense5[0..3] (relMSE_op, copy, gain, cos)"}


def panel_c(ax):
    d = json.load(open(SRC_C, encoding="utf-8"))
    rho_us = d["region_rho_cost_s"] * 1e6
    px_ms = d["pixel_route_cost_s"] * 1e3
    ratio = d["pixel_route_cost_s"] / d["region_rho_cost_s"]
    check("C region-level field shift (us)", rho_us, 20.9, 0.05)
    check("C pixel recolour + re-forward (ms)", px_ms, 38.1, 0.05)
    check("C ratio (x)", ratio, 1825, 1.0)
    vals = [rho_us, px_ms * 1e3]                                  # microseconds on a common axis
    names = ["region-level\nfield shift\n$\\rho_r(\\Delta)$", "pixel recolour\n+ re-forward"]
    ax.bar([0, 1], vals, 0.55, color=["#2b6cb0", "#b0b0b0"])
    ax.set_yscale("log"); ax.set_ylim(5, 2e5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("wall-clock cost ($\\mu$s, log scale)", fontsize=8.5)
    ax.text(0, vals[0] * 1.35, "%.1f $\\mu$s" % rho_us, ha="center", fontsize=8.5, fontweight="bold")
    ax.text(1, vals[1] * 1.35, "%.1f ms  (%.0f $\\mu$s)" % (px_ms, vals[1]), ha="center", fontsize=8.5,
            fontweight="bold")
    ax.annotate("", xy=(1, vals[1] * 0.7), xytext=(0, vals[0] * 0.7),
                arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1.1))
    ax.text(0.5, np.sqrt(vals[0] * vals[1]) * 0.75, "%.0f$\\times$" % ratio, ha="center", fontsize=10,
            color="#c44e52", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("C  A region-level shift is %.0f$\\times$ cheaper than a pixel re-forward\n"
                 "%.1f $\\mu$s against %.1f ms, both regions in one pass" % (ratio, rho_us, px_ms),
                 fontsize=7.6)
    return {"region_rho_cost_us": rho_us, "pixel_route_cost_ms": px_ms, "ratio": ratio,
            "source": "results/region_rho_demo.json:region_rho_cost_s, pixel_route_cost_s"}


def panel_d(ax):
    d = json.load(open(SRC_D, encoding="utf-8"))
    ops = [("O1_procrustes", "O1 global\northogonal"), ("O2_ridge", "O2 global\nlinear"),
           ("O8_conv_residual", "O8 content +\nneighbourhood")]
    arms = [("ce8_stage1", "CEConv (group-structured)", "#2b6cb0"),
            ("z2_stage1", "plain CNN", "#c44e52")]
    audited = {"ce8_stage1": [0.75, 0.78, 0.83], "z2_stage1": [0.22, 0.31, 0.54]}
    out = {"delta_deg": float(DELTA_D), "arms": {}}
    means, sds = {}, {}
    for arm, lab, _ in arms:
        dn = d[arm][DELTA_D]["ops"]
        ref = dn["null"]["rel_l2_vs_real"]["mean"]
        m, s = [], []
        for op, _lab in ops:
            m.append(1.0 - dn[op]["rel_l2_vs_real"]["mean"] / ref)
            per_seed = []
            for f in SRC_D_SEEDS[arm]:
                dd = json.load(open(f, encoding="utf-8"))["deltas"][DELTA_D]["downstream"]
                per_seed.append(1.0 - dd[op]["rel_l2_vs_real"] / dd["null"]["rel_l2_vs_real"])
            s.append(float(np.std(per_seed)))
        means[arm], sds[arm] = np.array(m), np.array(s)
        out["arms"][lab] = {"T_F": m, "T_F_sd_seed": s,
                            "operator_order": [o for o, _ in ops],
                            "audited": audited[arm]}
        for (op, opl), got, aud in zip(ops, m, audited[arm]):
            check("D %s T_F %s" % (lab, op), got, aud, 0.01)
    x = np.arange(len(ops)); w = 0.36
    for k, (arm, lab, col) in enumerate(arms):
        ax.bar(x + (k - 0.5) * w, means[arm], w, yerr=sds[arm], capsize=3, color=col, label=lab)
        for xi, mv in zip(x, means[arm]):
            ax.text(xi + (k - 0.5) * w, mv + 0.03, "%.2f" % mv, ha="center", fontsize=7)
    ax.axhline(0.80, color="#555555", ls="--", lw=0.9)
    ax.text(2.45, 0.815, "0.80 reporting\nconvention", fontsize=6.4, color="#555555", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ops], fontsize=7.2)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("downstream transfer  $T_F$", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("D  The channel layout sets the ceiling ($\\Delta$=%s$^\\circ$, stage 1, hue)\n"
                 "group conv 0.75/0.78/0.83 vs 0.22/0.31/0.54 (plain CNN)" % DELTA_D, fontsize=7.6)
    out["source"] = ("results/causal_cifar_summary.json:[ce8_stage1|z2_stage1]['%s'].ops[*].rel_l2_vs_real "
                     "(null ref ops.null.rel_l2_vs_real); sd from results/causal_cifar_{ce8,z2}_s{0,1,2}.json:"
                     "deltas['%s'].downstream" % (DELTA_D, DELTA_D))
    return out


def main():
    print("234_fig_correspondence.py — audited-value check")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.8))
    rec = {"figure": "F35_correspondence_and_cost.png", "claim": "the correspondence holds, and it is cheap",
           "generated_by": "scripts/234_fig_correspondence.py"}
    rec["panel_A_closed_form"] = panel_a(axes[0, 0])
    rec["panel_B_band_limited_vs_copy"] = panel_b(axes[0, 1])
    rec["panel_C_cost"] = panel_c(axes[1, 0])
    rec["panel_D_architecture"] = panel_d(axes[1, 1])
    fig.suptitle("The correspondence holds, and it is cheap", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    rec["checks"] = CHECK
    rec["all_checks_pass"] = all(c["ok"] for c in CHECK)
    json.dump(rec, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("\nwrote", OUT_PNG, "(%.1f KB)" % (os.path.getsize(OUT_PNG) / 1024.0))
    print("wrote", OUT_JSON)
    n_bad = sum(1 for c in CHECK if not c["ok"])
    print("checks: %d/%d pass" % (len(CHECK) - n_bad, len(CHECK)))
    for c in CHECK:
        if not c["ok"]:
            print("  NOT REPRODUCED:", c)
    assert n_bad == 0, "%d audited value(s) did not reproduce" % n_bad


if __name__ == "__main__":
    main()
