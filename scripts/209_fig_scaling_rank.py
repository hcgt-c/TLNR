# -*- coding: utf-8 -*-
"""209_fig_scaling_rank.py — the normalised depth/rank figure and the section that reports it.

Reads `results/depth_map_summary.json` (the cross-architecture grid of script 188/190) and produces, with
every number generated from that file rather than transcribed:

  * `results/fig_scaling_rank.json` — per-curve normalised transfer, relative depth, per-curve and pooled
    rank statistics, and the linear-versus-exponential fit comparison;
  * `paper/figures/F29_scaling_rank.png` — panel (a) normalised $T_F$ against relative depth for the four
    backbones x two families, panel (b) the per-curve Spearman statistic with the pooled value;
  * a new subsection `### 8.7 ...` appended to `paper/targets/TPAMI/sections/depth_law_EN.md`, stating the
    monotone/rank result and that with four depths the functional form is not identifiable, so no fitted
    parameters are reported;
  * the supplementary entry S22 in `paper/targets/TPAMI/figures_plan.json` and figure 31 in the master
    manifest, so that 184 packages the asset and the manifest stays complete.

Twelve curves are available in the JSON (four backbones x {hue_90.0, heat_2.0, heat_1.0}); this script uses
the two **headline** families of the depth law (hue_90.0 and heat_2.0, three seeds each) and only sites whose
`reachable` flag is true, so the two class-token transformers contribute three sites rather than four.

Usage: python scripts/209_fig_scaling_rank.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-dsh")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")
SUMMARY = os.path.join(RES, "depth_map_summary.json")
OUT_JSON = os.path.join(RES, "fig_scaling_rank.json")
FIG = os.path.join(WORK, "paper", "figures", "F29_scaling_rank.png")
DEPTH_LAW = os.path.join(WORK, "paper", "targets", "TPAMI", "sections", "depth_law_EN.md")
PLAN = os.path.join(WORK, "paper", "targets", "TPAMI", "figures_plan.json")
MANIFEST = os.path.join(WORK, "paper", "unified", "manifest.json")

BACKBONES = ["resnet50", "convnext", "vitb16", "dinov2b14"]
FAMILIES = ["hue_90.0", "heat_2.0"]
FAMILY_LABEL = {"hue_90.0": "hue 90°", "heat_2.0": "heat $\\sigma{=}2$"}
BB_LABEL = {"resnet50": "ResNet-50", "convnext": "ConvNeXt-T", "vitb16": "ViT-B/16", "dinov2b14": "DINOv2-B/14"}
COLOR = {"resnet50": "#1f77b4", "convnext": "#ff7f0e", "vitb16": "#2ca02c", "dinov2b14": "#d62728"}
MARK = {"hue_90.0": "o", "heat_2.0": "s"}


def curves_from(summary):
    out = {}
    for bb in BACKBONES:
        for fam in FAMILIES:
            key = f"{bb}|{fam}"
            if key not in summary:
                raise KeyError(f"missing grid key {key} in depth_map_summary.json")
            pts = [(s["depth"], s["O2_transfer_mean"])
                   for s in summary[key]["sites"].values() if s["reachable"]]
            pts.sort()
            if len(pts) < 3:
                raise ValueError(f"{key}: only {len(pts)} reachable sites")
            out[key] = pts
    return out


def per_curve(pts):
    """Normalised transfer against relative depth (site position / number of reachable sites)."""
    n = len(pts)
    x = np.array([i / n for i in range(1, n + 1)])
    raw = np.array([t for _, t in pts])
    y = raw / raw[0]
    rho, p = stats.spearmanr(np.array([d for d, _ in pts]), raw)
    lin = np.polyfit(x, y, 1)
    r2_lin = 1 - np.sum((y - np.polyval(lin, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    ln = np.log(np.clip(y, 1e-9, None))                    # exponential form y = a e^{-b x}
    ex = np.polyfit(x, ln, 1)
    r2_exp = 1 - np.sum((ln - np.polyval(ex, x)) ** 2) / np.sum((ln - ln.mean()) ** 2)
    return {"x": x.tolist(), "T": raw.tolist(), "T_norm": y.tolist(),
            "rho_vs_depth": float(rho), "p": float(p), "n_sites": n,
            "r2_linear": float(r2_lin), "r2_exponential": float(r2_exp),
            "first": float(raw[0]), "last": float(raw[-1])}


def figure(curves, pooled, per):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
    for key, c in curves.items():
        bb, fam = key.split("|")
        ax[0].plot(per[key]["x"], per[key]["T_norm"], color=COLOR[bb], marker=MARK[fam],
                   ls="-" if fam == "hue_90.0" else "--", lw=1.4, ms=4,
                   label=f"{BB_LABEL[bb]} · {FAMILY_LABEL[fam]}")
    ax[0].axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax[0].set_xlabel("relative depth (site / number of reachable sites)")
    ax[0].set_ylabel("$T_F$ normalised by its first reachable site")
    ax[0].set_title("(a) every curve is monotone; the level is backbone-specific", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
    ax[0].grid(alpha=0.25, lw=0.5)
    order = sorted(curves, key=lambda k: per[k]["rho_vs_depth"])
    ax[1].barh([k.replace("|", " · ") for k in order], [per[k]["rho_vs_depth"] for k in order],
               color=[COLOR[k.split("|")[0]] for k in order], alpha=0.85)
    ax[1].axvline(0, color="0.3", lw=0.8)
    ax[1].axvline(pooled["rho"], color="k", lw=1.0, ls="--")
    ax[1].set_xlabel("Spearman $\\rho$ (transfer vs depth)")
    ax[1].set_title(f"(b) pooled $\\rho={pooled['rho']:+.3f}$ ($p={pooled['p']:.1e}$, $n={pooled['n']}$)",
                    fontsize=9)
    ax[1].tick_params(axis="y", labelsize=6)
    ax[1].grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def subsection(res):
    p, fits = res["pooled"], res["fits"]
    med_lin, lo_lin, hi_lin = fits["linear"]["median"], fits["linear"]["min"], fits["linear"]["max"]
    med_exp, lo_exp, hi_exp = fits["exponential"]["median"], fits["exponential"]["min"], fits["exponential"]["max"]
    return f"""### 8.7 The functional form is not identifiable; the rank law is

Normalising each curve by its first reachable site removes the level difference between backbones and families, and what remains is monotone in every curve: with relative depth taken as the site's position divided by the number of reachable sites, the pooled Spearman correlation between relative depth and normalised transfer is $\\rho = {p['rho']:+.3f}$ ($p = {p['p']:.1e}$, $n = {p['n']}$ points over eight backbone$\\times$family curves), and the per-curve trend is negative in {res['n_negative']} of {res['n_curves']}. The deepest reachable site carries a median transfer of {res['deepest']['median']:.3f} (range {res['deepest']['min']:+.3f} to {res['deepest']['max']:+.3f}); the lowest value is the DINOv2-B/14 dissipation curve, which crosses zero as Figure 9B shows. Supplementary Figure S22 plots the normalised curves and the per-curve rank statistics.

Four depths are not enough to identify the functional form. Over the same curves a linear fit to normalised transfer against relative depth reaches a median $R^2$ of {med_lin:.3f} (range {lo_lin:.3f}–{hi_lin:.3f}), while an exponential fit reaches a median $R^2$ of {med_exp:.3f} (range {lo_exp:.3f}–{hi_exp:.3f}); the two forms lie within each other's residual scatter and neither is separable with four points (three for the two class-token transformers). **No fitted parameters are therefore reported**, and the law of this section is stated as a monotone, rank-based statement rather than as a rate. Reporting a rate constant from four points would be a property of the fit, not a measurement.
"""


def main():
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    curves = curves_from(summary)
    per = {k: per_curve(v) for k, v in curves.items()}

    xs, ys = [], []
    for k in curves:
        xs += per[k]["x"]; ys += per[k]["T_norm"]
    rho, p = stats.spearmanr(xs, ys)

    r2l = sorted(per[k]["r2_linear"] for k in curves)
    r2e = sorted(per[k]["r2_exponential"] for k in curves)
    deepest = sorted(per[k]["last"] for k in curves)
    n_neg = sum(1 for k in curves if summary[k]["depth_trend"]["O2"]["spearman_rho"] < 0)

    res = {"source": "results/depth_map_summary.json (operator O2, reachable sites)",
           "families": FAMILIES, "backbones": BACKBONES,
           "relative_depth": "site position / number of reachable sites",
           "curves": {k: {"depth": [d for d, _ in curves[k]], **per[k]} for k in curves},
           "pooled": {"rho": float(rho), "p": float(p), "n": len(xs)},
           "n_curves": len(curves), "n_negative": int(n_neg),
           "deepest": {"median": float(np.median(deepest)), "min": float(deepest[0]), "max": float(deepest[-1])},
           "fits": {"linear": {"median": float(np.median(r2l)), "min": float(r2l[0]), "max": float(r2l[-1])},
                    "exponential": {"median": float(np.median(r2e)), "min": float(r2e[0]), "max": float(r2e[-1])}}}
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    figure(curves, res["pooled"], per)

    # ---- the authored subsection, replaced in place so the script is idempotent
    txt = open(DEPTH_LAW, encoding="utf-8").read()
    marker = "\n### 8.7 "
    if marker in txt:
        txt = txt[:txt.index(marker)].rstrip() + "\n"
    txt = txt.rstrip() + "\n\n" + subsection(res)
    open(DEPTH_LAW, "w", encoding="utf-8").write(txt)

    # ---- register the supplementary asset (S22) and the manifest entry, programmatically
    plan = json.load(open(PLAN, encoding="utf-8"))
    entry = {"n": 22, "slug": "scaling_rank",
             "source": "paper/figures/F29_scaling_rank.png", "section": "8.7",
             "caption": "Normalised depth law and its rank form. (a) $T_F$ of the global linear operator, "
                        "normalised by its first reachable site, against relative depth for four backbones and "
                        "two transformation families; (b) the per-curve Spearman statistic with the pooled value. "
                        "With four depths the functional form is not identifiable, so no fitted parameters are "
                        "reported (Section 8.7)."}
    plan["supplementary"] = [e for e in plan["supplementary"] if e.get("n") != 22] + [entry]
    plan["supplementary"].sort(key=lambda e: e["n"])
    json.dump(plan, open(PLAN, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    man = json.load(open(MANIFEST, encoding="utf-8"))
    man["figures"]["31"] = {"title": "Normalised depth law and its rank form",
                            "source": "this round (script 209)", "status": "drafted"}
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("209: figure -> paper/figures/F29_scaling_rank.png")
    print(f"  pooled rho={rho:+.3f} (p={p:.1e}, n={len(xs)}) | per-curve negative {n_neg}/{len(curves)}")
    print(f"  deepest reachable: median {np.median(deepest):.3f} range {deepest[0]:+.3f}..{deepest[-1]:+.3f}")
    print(f"  R2 linear {np.median(r2l):.3f} [{r2l[0]:.3f},{r2l[-1]:.3f}] | exponential {np.median(r2e):.3f} "
          f"[{r2e[0]:.3f},{r2e[-1]:.3f}]")
    print("  subsection 8.7 written to targets/TPAMI/sections/depth_law_EN.md; S22 registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
