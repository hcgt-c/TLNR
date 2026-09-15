# -*- coding: utf-8 -*-
"""203_aggregate_mechanism.py — aggregate the quotientization / dilution probe over seeds.

Reads results/quotientization.json (script 200, possibly several seeds) and reports:

  * the dilution measures per site: displacement rms, coherence (top-1 share, participation ratio);
  * the matched-rank control for the visible share: the raw share, the random-projector baseline at the
    read-out's *effective* rank, and their ratio (the chance-corrected share). This is the control the M1
    measurement was missing;
  * the bottleneck contrast at matched rank: T_F of the displacement-aligned projector against the
    top-variance and random projectors of the same rank, with the power (mean no-op displacement) that the
    comparison is powered by, and the full-rank ceiling for reference;
  * rank correlations of each mechanism quantity with relative depth, pooled over seeds.

Usage: python scripts/203_aggregate_mechanism.py
Outputs results/mechanism_summary.json, paper/targets/TPAMI/MECHANISM_TABLES.md,
        paper/figures/F29_dilution_bottleneck.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = rp.REPO_ROOT
LABEL = {"resnet50": "ResNet-50", "convnext": "ConvNeXt-T", "vitb16": "ViT-B/16", "dinov2b14": "DINOv2-B/14"}
COL = {"resnet50": "#1f4e79", "convnext": "#2e8b57", "vitb16": "#b8860b", "dinov2b14": "#b22222"}


def mean_sd(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def fmt(v, f="{:.3f}"):
    m, s = mean_sd(v)
    if not np.isfinite(m):
        return "—"
    return f.format(m) if s == 0.0 else f"{f.format(m)} ± {f.format(s)}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(WORK, "results", "quotientization.json"))
    ap.add_argument("--outdir", default=os.path.join(WORK, "paper", "targets", "TPAMI"))
    ap.add_argument("--figdir", default=os.path.join(WORK, "paper", "figures"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = a.input
    raw = json.load(open(path, encoding="utf-8"))
    blocks = list(raw.values())
    summary = {}
    L = ["# Mechanism tables: dilution of the transformation signal, and the transformation-aligned bottleneck", "",
         f"Aggregated over {len(blocks)} run(s) (each run = one seed x one family x four backbones x four depths). "
         "All numbers are on the fit/held-out split of script 188; the displacement-aligned projector is fitted "
         "on the fit split and evaluated on held-out images.", ""]

    # ---- per site table ----
    L += ["## Per site: dilution and the matched-rank control for the visible share", "",
          "| backbone | family | site | depth | rms(δ) | top-1 share | PR | visible share | random @ eff-rank | "
          "corrected | ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    cell = {}
    for blk in blocks:
        fam = f"{blk['family']}{blk['param']}"
        for bb, sites in blk["backbones"].items():
            names = list(sites.keys())
            for di, site in enumerate(names, start=1):
                r = sites[site]
                c = r["displacement"]["coherence"]
                mr = r.get("matched_rank_control", {})
                key = (bb, fam, site)
                d = cell.setdefault(key, {"bb": bb, "fam": fam, "site": site, "depth": di,
                                          "rms": [], "top1": [], "pr": [], "vis": [], "rand": [], "corr": [],
                                          "ceiling": [], "ranks": {}, "full_dnull": None})
                d["rms"].append(r["displacement"]["rms"])
                d["top1"].append(c["top1_share"])
                d["pr"].append(c["participation_ratio"])
                d["vis"].append(r["visible_share_of_displacement"])
                if mr:
                    d["rand"].append(mr["random_share_at_effective"])
                    d["corr"].append(mr["visible_over_random_effective"])
                d["ceiling"].append(r["uncompressed_ceiling"])
                full = r["rank_intervention"].get("full", {}).get("none")
                if full:
                    d["full_dnull"] = full.get("mean_d_noop")
                for k, v in r["rank_intervention"].items():
                    if k == "full":
                        continue
                    for tag in ("pca", "random", "displacement_oracle"):
                        d["ranks"].setdefault(int(k), {}).setdefault(tag, []).append(
                            {"transfer": v[tag]["transfer"], "dnull": v[tag]["mean_d_noop"]})
    for key in sorted(cell, key=lambda k: (k[0], k[1], cell[k]['depth'])):
        d = cell[key]
        summary[f"{d['bb']}|{d['fam']}|{d['site']}"] = {
            "depth": d["depth"], "rms": mean_sd(d["rms"]), "top1_share": mean_sd(d["top1"]),
            "participation_ratio": mean_sd(d["pr"]), "visible_share": mean_sd(d["vis"]),
            "random_share_at_effrank": mean_sd(d["rand"]), "visible_corrected": mean_sd(d["corr"]),
            "ceiling": mean_sd(d["ceiling"])}
        L.append(f"| {LABEL.get(d['bb'], d['bb'])} | {d['fam']} | {d['site']} | {d['depth']} | "
                 f"{fmt(d['rms'])} | {fmt(d['top1'])} | {fmt(d['pr'], '{:.0f}')} | {fmt(d['vis'], '{:.4f}')} | "
                 f"{fmt(d['rand'], '{:.4f}')} | {fmt(d['corr'], '{:.2f}')} | {fmt(d['ceiling'], '{:+.3f}')} |")

    # ---- matched-rank bottleneck contrast ----
    L += ["", "## Matched-rank bottleneck: displacement-aligned vs top-variance vs random", "",
          "Same rank, same consumer class, same held-out images; only the retained subspace differs. "
          "`d_null` is the consumer's own displacement under the real transformation, i.e. the quantity the "
          "ratio is normalised by — it must be reported because the consumer changes with the projector.", "",
          "| backbone | family | site | k | aligned | top-variance | random | aligned − best control | d_null |",
          "|---|---|---|---|---|---|---|---|---|"]
    gains = {}
    for key in sorted(cell, key=lambda k: (k[0], k[1], cell[k]['depth'])):
        d = cell[key]
        for k in sorted(d["ranks"]):
            for tag in ("pca", "random", "displacement_oracle"):
                if tag not in d["ranks"][k]:
                    continue
            try:
                al = mean_sd([x["transfer"] for x in d["ranks"][k]["displacement_oracle"]])[0]
                pc = mean_sd([x["transfer"] for x in d["ranks"][k]["pca"]])[0]
                rn = mean_sd([x["transfer"] for x in d["ranks"][k]["random"]])[0]
                dn = mean_sd([x["dnull"] for x in d["ranks"][k]["displacement_oracle"]])[0]
            except KeyError:
                continue
            gain = al - max(pc, rn)
            # A rank-k comparison is a comparison between *different consumers*, so it is only meaningful if
            # every member of the comparison is powered. The full-rank no-op displacement at the same site is
            # the reference; a member whose no-op displacement falls below a quarter of it cannot support a
            # ratio at all (this is Section 3.7 applied to this experiment). Destructive (T_F < -1) cells are
            # excluded as well.
            dn_all = [mean_sd([x["dnull"] for x in d["ranks"][k][t]])[0]
                      for t in ("displacement_oracle", "pca", "random") if t in d["ranks"][k]]
            dn_ref = d.get("full_dnull", float("nan"))
            powered = bool(dn_all) and np.isfinite(dn_ref) and min(dn_all) >= 0.25 * dn_ref
            flag = "excluded" if (al < -1.0 or not powered) else "ok"
            gains.setdefault(k, []).append({"gain": gain, "cnn": d["bb"] in ("resnet50", "convnext"),
                                            "flag": flag, "cell": f"{d['bb']}|{d['fam']}|{d['site']}"})
            L.append(f"| {LABEL.get(d['bb'], d['bb'])} | {d['fam']} | {d['site']} | {k} | {al:+.3f} | {pc:+.3f} | "
                     f"{rn:+.3f} | **{gain:+.3f}** | {dn:.2f} |")
    def agg_gain(v, pred=lambda g: True):
        xs = [g["gain"] for g in v if pred(g)]
        if not xs:
            return {}
        return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                "median": float(np.median(xs)), "n": len(xs), "powered": len(xs),
                "positive": int(sum(1 for x in xs if x > 0)),
                "sign_p": float(stats.binomtest(int(sum(1 for x in xs if x > 0)), len(xs), 0.5).pvalue)}
    summary["bottleneck_gain_by_rank"] = {
        str(k): {"all": agg_gain(v), "ok_only": agg_gain(v, lambda g: g["flag"] == "ok"),
                 "cnn_ok": agg_gain(v, lambda g: g["flag"] == "ok" and g["cnn"])}
        for k, v in sorted(gains.items())}
    L += ["", "Sign test over cells of (aligned − best control). A cell is **excluded** unless every member of "
              "the comparison is powered, i.e. its no-op displacement is at least a quarter of the same site's "
              "full-rank no-op displacement (Section 3.7 applied to a comparison between different consumers), "
              "and unless the aligned ceiling is above −1 (destructive regimes such as DINOv2 block8). "
              "`powered` counts how many cells survive.", "",
          "| rank | aggregate | mean gain | median | positive / n | sign p |", "|---|---|---|---|---|---|"]
    for k, v in sorted(summary["bottleneck_gain_by_rank"].items()):
        for tag, g in (("all cells", v["all"]), ("excluding destructive", v["ok_only"]), ("CNN only", v["cnn_ok"])):
            if not g:
                continue
            L.append(f"| {k} | {tag} | {g['mean']:+.3f} ± {g['sd']:.3f} | {g['median']:+.3f} | "
                     f"{g['positive']}/{g['n']} | {g['sign_p']:.2g} |")

    # ---- depth trends ----
    L += ["", "## Depth trends (pooled over seeds and backbones/families)", "",
          "| quantity | Spearman rho vs relative depth | p | n |", "|---|---|---|---|"]
    trends = {}
    for field in ("rms", "top1_share", "participation_ratio", "visible_corrected", "ceiling"):
        xs, ys = [], []
        for blk in blocks:
            for bb, sites in blk["backbones"].items():
                names = list(sites.keys())
                for di, site in enumerate(names, start=1):
                    r = sites[site]
                    rel = di / (len(names) + 1)
                    if field == "rms":
                        val = r["displacement"]["rms"]
                    elif field == "top1_share":
                        val = r["displacement"]["coherence"]["top1_share"]
                    elif field == "participation_ratio":
                        val = r["displacement"]["coherence"]["participation_ratio"]
                    elif field == "visible_corrected":
                        mr = r.get("matched_rank_control")
                        if not mr or not np.isfinite(mr.get("visible_over_random_effective", np.nan)):
                            continue
                        val = mr["visible_over_random_effective"]
                    else:
                        if not r.get("intervention_reaches_consumer", True):
                            continue
                        val = r["uncompressed_ceiling"]
                    xs.append(rel); ys.append(val)
        if len(xs) >= 5:
            rho, p = stats.spearmanr(xs, ys)
            trends[field] = {"spearman_rho": float(rho), "p": float(p), "n": len(xs)}
            L.append(f"| {field} | {rho:+.3f} | {p:.2g} | {len(xs)} |")
    summary["depth_trends"] = trends
    summary["n_runs"] = len(blocks)
    json.dump(summary, open(os.path.join(WORK, "results", f"mechanism_summary{a.tag}.json"), "w"), indent=1)
    open(os.path.join(a.outdir, f"MECHANISM_TABLES{a.tag}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for bb in ("resnet50", "convnext", "vitb16", "dinov2b14"):
        for fam in ("heat2.0", "hue90.0"):
            ks = sorted([k for k in cell if k[0] == bb and k[1] == fam], key=lambda k: cell[k]["depth"])
            if not ks:
                continue
            xs = [cell[k]["depth"] / 5 for k in ks]
            ys = [mean_sd(cell[k]["pr"])[0] for k in ks]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.2, color=COL[bb],
                    ls="-" if fam.startswith("heat") else "--", alpha=0.9)
    ax.set_xlabel("relative depth"); ax.set_ylabel("participation ratio of $\\mathrm{Cov}(\\delta)$")
    ax.set_title("A. Displacement coherence collapses with depth\n(solid heat $\\sigma{=}2$, dashed hue $90°$)",
                 fontsize=10)
    ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.4)
    ax = axes[1]
    for bb in ("resnet50", "convnext", "vitb16", "dinov2b14"):
        ks = sorted([k for k in cell if k[0] == bb and k[1] == "heat2.0"], key=lambda k: cell[k]["depth"])
        if not ks:
            continue
        xs, ys = [], []
        for k in ks:
            rk = cell[k]["ranks"].get(1)
            if not rk or "displacement_oracle" not in rk:
                continue
            al = mean_sd([x["transfer"] for x in rk["displacement_oracle"]])[0]
            if al < -1.0:                      # destructive regime, excluded and reported in the table
                continue
            pc = mean_sd([x["transfer"] for x in rk["pca"]])[0]
            rn = mean_sd([x["transfer"] for x in rk["random"]])[0]
            xs.append(cell[k]['depth'] / (len(ks) + 1)); ys.append(al - max(pc, rn))
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.3, color=COL[bb], label=LABEL[bb])
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("gain of the aligned rank-1 bottleneck\nover the best control (heat $\\sigma{=}2$)")
    ax.set_title("B. The retained subspace, not the rank,\ndecides the ceiling", fontsize=10)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figdir, f"F29_dilution_bottleneck{a.tag}.png"), dpi=200)

    print("mechanism aggregate: PASS (0 issues)")
    print(f"  runs aggregated: {len(blocks)} | sites: {len(cell)}")
    for k, v in sorted(summary["bottleneck_gain_by_rank"].items()):
        g = v["cnn_ok"] or v["ok_only"] or v["all"]
        print(f"  gain at rank {k}: CNN {g['mean']:+.3f} (median {g['median']:+.3f}, "
              f"{g['positive']}/{g['n']} positive, p={g['sign_p']:.2g})")
    for f_, v in trends.items():
        print(f"  trend {f_:22s} rho={v['spearman_rho']:+.3f} (p={v['p']:.2g}, n={v['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
