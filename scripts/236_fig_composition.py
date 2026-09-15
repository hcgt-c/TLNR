# -*- coding: utf-8 -*-
"""236_fig_composition.py — one step does not imply composition: paths, agreement, and the detector boundary.

Reproducible main figure with three panels, all values read from existing JSON under results/ (nothing is
recomputed, no GPU is touched):

  (A) composition along ALTERNATIVE PATHS at a fixed total transformation. results/multipath_composition.json
      (script 231) holds the total transformation fixed and varies its decomposition (two/four/eight steps)
      beside the single-step substitute and the direct fit (which sees the target and is therefore an upper
      reference, not a competitor). Two families (hue = rotation, heat = dissipative semigroup) x two operator
      families (O2 global linear, O8 content+neighbourhood).
  (B) path AGREEMENT is not path FIDELITY: the disagreement between equivalent decompositions of the SAME total
      against the disagreement with a DIFFERENT total (the single-step substitute). Same source.
  (C) the detector boundary. results/detector_o8_multiseed.json (script 158) gives the box-level agreement of
      each routed operator with the real recolour (a real recolour destroys it, fitted transports restore it,
      and the random orthogonal control sits at zero despite a non-trivial projection coefficient) at delta=90.
      results/causal_intervention_hue.json gives the never-fitted shift: composing the operators fitted at 30
      and 60 and evaluating at 90, against a direct fit at 90 (AP50).

Outputs: paper/figures/F37_composition_and_boundary.png (dpi=200) and results/fig_composition.json.
Usage: python scripts/236_fig_composition.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = rp.REPO_ROOT
MP_SRC = os.path.join(WORK, "results", "multipath_composition.json")
DET_SRC = os.path.join(WORK, "results", "detector_o8_multiseed.json")
CI_SRC = os.path.join(WORK, "results", "causal_intervention_hue.json")
OUT_PNG = os.path.join(WORK, "paper", "figures", "F37_composition_and_boundary.png")
OUT_JSON = os.path.join(WORK, "results", "fig_composition.json")

FAMCOL = {"heat": "#d62728", "hue": "#1f77b4"}
OPALPHA = {"O2": 0.45, "O8": 0.98}
CHAINS = ["direct_fit", "two_step", "four_step", "eight_step", "single"]
XLAB = ["direct fit\n(sees target)", "2-step", "4-step", "8-step", "single step\n(sub-total)"]
FAMS = ["hue", "heat"]
OPS = ["O2", "O8"]


def load_multipath():
    recs = json.load(open(MP_SRC, encoding="utf-8"))["records"]
    by = {}
    for _, r in recs.items():
        by.setdefault(r["family"], []).append(r)
    return by


def chain_of(key):
    for p in CHAINS:
        if key.startswith(p):
            return p
    return None


def panel_a_data(by):
    """chain length -> {fam|op: [values]} ; values are T_F over paths x seeds."""
    out = {}
    for fam in FAMS:
        for op in OPS:
            for ch in CHAINS:
                vals = []
                for r in by.get(fam, []):
                    for k, v in r["paths"].items():
                        if k.endswith("|" + op) and chain_of(k) == ch:
                            vals.append(float(v["transfer"]))
                if vals:
                    out[f"{fam}|{op}|{ch}"] = vals
    return out


def panel_b_data(by):
    out = {}
    for fam in FAMS:
        for op in OPS:
            same, diff = [], []
            for r in by.get(fam, []):
                for k, v in r["path_consistency"].items():
                    k1, k2 = k.split(" vs ")
                    if k1.split("|")[1] != op:
                        continue
                    if k1.split("|")[0].startswith("single") or k2.split("|")[0].startswith("single"):
                        diff.append(float(v))
                    else:
                        same.append(float(v))
            out[f"{fam}|{op}"] = {"same_total": same, "different_total": diff}
    return out


def panel_c_data():
    det = json.load(open(DET_SRC, encoding="utf-8"))
    routes = det["deltas"]["90"]["routes"]
    RLAB = {"null": "null\n(no transport)", "O1_procrustes": "O1\northogonal",
            "O2_ridge": "O2\nlinear", "O8_conv_residual": "O8\ncontent+neigh.",
            "O6_random": "O6\nrandom orth."}
    route_out = {}
    for r, lab in RLAB.items():
        d = routes[r]
        route_out[r] = {"label": lab, "f1_mean": float(d["f1_mean"]), "f1_sd": float(d["f1_sd"]),
                        "ap50_mean": float(d["ap50_mean"]), "ap50_sd": float(d["ap50_sd"]),
                        "proj_coef_mean": float(d.get("proj_coef_mean", 0.0))}
    ci = json.load(open(CI_SRC, encoding="utf-8"))["generalisation_to_unseen_delta"]
    unseen = {
        "compose_W60W30": {"label": "chain\n$W_{60}W_{30}$", "ap50": float(ci["composition_of_fitted_30_and_60"]["box_agreement_AP50"])},
        "generator_exp90": {"label": "generator\nexp(90$G$)", "ap50": float(ci["generator_exp_90_from_log_W30"]["box_agreement_AP50"])},
        "direct_fit_W90": {"label": "direct fit\n$W_{90}$", "ap50": float(ci["per_delta_fitted_W90_reference"]["box_agreement_AP50"])},
    }
    return route_out, unseen


def figure():
    by = load_multipath()
    A = panel_a_data(by)
    B = panel_b_data(by)
    Crow, Cunseen = panel_c_data()

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(12.0, 4.6), gridspec_kw=dict(width_ratios=[1.05, 0.95, 1.45]))

    # ---------------------------------------------------------------- Panel A
    ncat = len(CHAINS)
    x = np.arange(ncat)
    series = [(fam, op) for fam in FAMS for op in OPS]      # hue O2, hue O8, heat O2, heat O8
    w = 0.20
    off = {s: (i - 1.5) * w for i, s in enumerate(series)}
    for fam, op in series:
        means, sds, xs = [], [], []
        for ci, ch in enumerate(CHAINS):
            vals = A.get(f"{fam}|{op}|{ch}")
            means.append(float(np.mean(vals)) if vals else np.nan)
            sds.append(float(np.std(vals, ddof=1)) if vals and len(vals) > 1 else 0.0)
            xs.append(ci + off[(fam, op)])
            if vals:
                axA.scatter([ci + off[(fam, op)]] * len(vals), vals, s=4.5, color="0.15", zorder=5, alpha=0.6)
        axA.bar(xs, means, w, color=FAMCOL[fam], alpha=OPALPHA[op],
                edgecolor="white", linewidth=0.4, label=f"{fam} / {op}",
                yerr=sds, error_kw=dict(ecolor="0.25", lw=0.7, capsize=1.6))
    # hatch the direct-fit group: it sees the target and is an upper reference, not a competitor
    for b in axA.patches:
        if abs(b.get_x() + b.get_width() / 2.0) < 0.5:
            b.set_hatch("//")
    axA.set_xticks(x)
    axA.set_xticklabels(XLAB, fontsize=7.2)
    axA.set_ylabel("downstream $T_F$ vs the real composed transformation")
    axA.set_title("A  one fixed total, many decompositions\n(hue 90$^\\circ$; heat $t^*=1$; hatched = direct fit sees the target)",
                  fontsize=9.0)
    axA.legend(fontsize=6.4, ncol=2, loc="upper left", framealpha=0.9)
    axA.grid(alpha=0.25, axis="y")
    axA.set_ylim(0, 1.02)
    # heat single-step saturation caveat
    hs = A.get("heat|O8|single", [0.0])
    h2 = A.get("heat|O2|single", [0.0])
    axA.annotate("heat single step is high\n(%.3f / %.3f): saturation,\nnot composition" % (float(np.mean(h2)), float(np.mean(hs))),
                 xy=(4 + off[("heat", "O8")], float(np.mean(hs))), xytext=(2.55, 0.86),
                 fontsize=6.0, color="0.25", ha="left",
                 arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7))
    axA.text(0.99, 0.02, "hue has no 8-step path", transform=axA.transAxes, fontsize=5.8,
             color="0.4", ha="right")

    # ---------------------------------------------------------------- Panel B
    keys = [f"{fam}|{op}" for fam in FAMS for op in OPS]
    xb = np.arange(len(keys))
    same_m = [float(np.mean(B[k]["same_total"])) for k in keys]
    diff_m = [float(np.mean(B[k]["different_total"])) for k in keys]
    for i, k in enumerate(keys):
        fam, op = k.split("|")
        axB.bar(i - 0.19, same_m[i], 0.36, color=FAMCOL[fam], alpha=OPALPHA[op],
                edgecolor="white", linewidth=0.4)
        axB.bar(i + 0.19, diff_m[i], 0.36, color="0.55", alpha=0.9, edgecolor="white", linewidth=0.4)
        axB.annotate(f"{diff_m[i] / same_m[i]:.1f}$\\times$", (i, max(same_m[i], diff_m[i]) + 0.045),
                     ha="center", fontsize=7.4, color="0.15", fontweight="bold")
    axB.set_xticks(xb)
    axB.set_xticklabels([k.replace("|", "\n") for k in keys], fontsize=7.4)
    axB.set_ylabel("mean relative logit difference between paths")
    axB.set_title("B  agreement between equivalent paths $\\neq$ fidelity\n(a different total is 2.2-7.1$\\times$ more distinct)",
                  fontsize=9.0)
    hs_ = float(np.mean(B["hue|O2"]["same_total"] + B["hue|O8"]["same_total"]))
    hd_ = float(np.mean(B["hue|O2"]["different_total"] + B["hue|O8"]["different_total"]))
    axB.text(0.30, 0.56, "hue pooled: %.3f (same) vs %.3f (different) = %.1f$\\times$" % (hs_, hd_, hd_ / hs_),
             transform=axB.transAxes, fontsize=6.3, color="0.2", ha="center")
    axB.legend([plt.Rectangle((0, 0), 1, 1, color="0.55", alpha=0.9),
                plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=0.45)],
               ["different total (single step)", "same total (bars shaded by operator)"],
               fontsize=6.0, loc="upper left", framealpha=0.9)
    axB.grid(alpha=0.25, axis="y")
    axB.set_ylim(0, 1.02)

    # ---------------------------------------------------------------- Panel C
    order = ["null", "O1_procrustes", "O2_ridge", "O8_conv_residual", "O6_random"]
    xc = np.arange(len(order))
    f1 = [Crow[r]["f1_mean"] for r in order]
    f1sd = [Crow[r]["f1_sd"] for r in order]
    proj = [Crow[r]["proj_coef_mean"] for r in order]
    axC.bar(xc, f1, 0.62, yerr=f1sd, capsize=2.0,
            color=["#8c8c8c", "#4c72b0", "#4c72b0", "#2ca02c", "#c44e52"],
            alpha=0.9, edgecolor="white", linewidth=0.4)
    axC.scatter(xc, proj, marker="D", s=26, color="0.1", zorder=6)
    for xi, (a, p) in enumerate(zip(f1, proj)):
        axC.annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=6.0)
    axC.text(0.0, 1.02, "\u25c6 = projection coefficient", fontsize=6.0, color="0.1")
    axC.annotate("random orthogonal: box 0.000,\nprojection coef %.2f" % proj[4],
                 xy=(4, 0.02), xytext=(4.0, 0.97), fontsize=6.0, color="#8b1a1a", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#8b1a1a", lw=0.8))
    axC.axvline(5.15, color="0.7", lw=0.9, ls="--")
    # unseen-shift group: a different metric (AP50) from the same held-out recolour reference
    xu = np.arange(len(Cunseen)) + 5.9
    uorder = ["compose_W60W30", "generator_exp90", "direct_fit_W90"]
    uvals = [Cunseen[k]["ap50"] for k in uorder]
    axC.bar(xu, uvals, 0.62, color=["#dd8452", "#937860", "#4c72b0"], alpha=0.9,
            edgecolor="white", linewidth=0.4)
    for xi, v in zip(xu, uvals):
        axC.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=6.0)
    axC.set_xticks(list(xc) + list(xu))
    axC.set_xticklabels([Crow[r]["label"] + "\nF1@.5" for r in order] +
                        [Cunseen[k]["label"] + "\nAP50" for k in uorder], fontsize=6.3)
    axC.set_ylabel("box-level agreement with the real recolour\n(F1@IoU0.5 | AP50)")
    axC.set_title("C  the detector boundary at 90$^\\circ$\n(left: routed operators; right: a shift never fitted)",
                  fontsize=9.0)
    axC.text(2.0, 1.10, "same shift, fitted transports", ha="center", fontsize=6.6, color="0.25")
    axC.text(7.0, 1.10, "composition at an unseen shift", ha="center", fontsize=6.6, color="0.25")
    axC.grid(alpha=0.25, axis="y")
    axC.set_ylim(0, 1.16)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.14, wspace=0.32)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)

    # ---------------------------------------------------------------- record
    import platform
    rec = {
        "script": "scripts/236_fig_composition.py",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sources": {
            "panelA": MP_SRC.replace(WORK + "/", ""),
            "panelB": MP_SRC.replace(WORK + "/", ""),
            "panelC_detector": DET_SRC.replace(WORK + "/", ""),
            "panelC_unseen": CI_SRC.replace(WORK + "/", ""),
        },
        "panelA": {"description": "T_F by decomposition at a fixed total transformation (mean over paths and seeds)",
                   "categories": CHAINS, "series": {}},
        "panelB": {"description": "path-consistency: same total vs different total"},
        "panelC": {"description": "detector box agreement at delta=90 and AP50 at the never-fitted 90 shift"},
        "discrepancies": [
            "Panel A: the audited hue chain range 0.237-0.249 (O2) / 0.472-0.488 (O8) reproduces for the three "
            "TWO-step paths (O2 0.2373/0.2492/0.2453; O8 0.4724/0.4883/0.4839). The hue FOUR-step path is 0.2533 "
            "(O2) and 0.4514 (O8), i.e. just outside the stated range. No data were adjusted; the 2-step bar is "
            "the mean of the three 2-step variants and the 4-step bar is plotted separately.",
            "Panel C: results/detector_o8_multiseed.json contains no never-fitted-shift AP50. The audited 0.37 vs "
            "0.81 pair comes from results/causal_intervention_hue.json "
            "generalisation_to_unseen_delta: generator_exp_90_from_log_W30 = 0.3654, "
            "per_delta_fitted_W90_reference = 0.8052 (and composition_of_fitted_30_and_60 = 0.5110).",
            "Panel C: 'box-level agreement 0.008 to 0.965' is F1@IoU0.5 at delta=90 (null_unintervened vs "
            "O8_conv_residual) in detector_o8_multiseed.json; the AP50 pair at the same delta is 0.0030 vs 0.9686.",
        ],
    }
    for fam in FAMS:
        for op in OPS:
            for ch in CHAINS:
                key = f"{fam}|{op}|{ch}"
                if key not in A:
                    continue
                v = A[key]
                rec["panelA"]["series"][key] = {
                    "values": v, "mean": float(np.mean(v)),
                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
                    "source_key": "records[*].paths[<path>|%s].transfer" % op}
    for k in keys:
        fam, op = k.split("|")
        s, d = B[k]["same_total"], B[k]["different_total"]
        rec["panelB"][k] = {
            "same_total_values": s, "same_total_mean": float(np.mean(s)),
            "different_total_values": d, "different_total_mean": float(np.mean(d)),
            "ratio": float(np.mean(d) / np.mean(s)),
            "source_key": "records[*].path_consistency[<equivalent> vs <single>]"}
    rec["panelB"]["hue_pooled"] = {
        "same_total_mean": hs_, "different_total_mean": hd_, "ratio": hd_ / hs_}
    rec["panelC"]["detector_delta90"] = Crow
    rec["panelC"]["unseen_shift_90"] = Cunseen
    rec["panelC"]["detector_source_key"] = "deltas.90.routes.<route>.{f1_mean,f1_sd,ap50_mean,ap50_sd,proj_coef_mean}"
    rec["panelC"]["unseen_source_key"] = "generalisation_to_unseen_delta.<variant>.box_agreement_AP50"
    json.dump(rec, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT_JSON)
    return rec


if __name__ == "__main__":
    figure()
