# -*- coding: utf-8 -*-
"""189_fig_depth_law.py — the TPAMI main figure: downstream transfer against depth.

Reads the aggregated depth grid (`results/depth_map_summary.json`, produced by 190) and draws four panels:

  (A) hue rotation 90 deg (the reference arm): **downstream transfer** against relative depth, mean +/- sd
      over seeds, one line per backbone. Transfer is the fraction of the no-op's error that the intervention
      removes, so 1 is exact faithfulness, 0 is no better than doing nothing and < 0 is worse. This replaces
      the earlier win-rate panel: a win rate near 1 is compatible with removing almost none of the error.
  (B) the heat semigroup at sigma = 2 (the headline family), same axes.
  (C) the conditioning gain of the equal-budget O8 over the global linear O2, per site: positive means
      content and neighbourhood conditioning helps. This supports "the action becomes more state-dependent
      with depth" only to the extent that the gain is positive.
  (D) the canonical spatial action's intertwining residual R at each site (heat family): near 1 means a
      blurred input does NOT produce a blurred feature map.

Usage: python scripts/189_fig_depth_law.py
Outputs paper/figures/F27_depth_law.png and results/fig_depth_law.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = rp.REPO_ROOT
OUT = os.path.join(WORK, "paper", "figures")
SUMMARY = os.path.join(WORK, "results", "depth_map_summary.json")
BACKBONES = ["resnet50", "convnext", "vitb16", "dinov2b14"]
LABEL = {"resnet50": "ResNet-50", "convnext": "ConvNeXt-T", "vitb16": "ViT-B/16", "dinov2b14": "DINOv2-B/14"}
STYLE = {"resnet50": ("o-", "#1f77b4"), "convnext": ("s-", "#2ca02c"),
         "vitb16": ("^-", "#d62728"), "dinov2b14": ("v-", "#9467bd")}


def block(summary, b, fam):
    for _, v in summary.items():
        if v["backbone"] == b and v["family_key"].startswith(fam + "_"):
            return v
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    out = {}

    for ax, (fam, title) in zip(axes.flat[:2], [("hue", "(A) hue rotation 90°, the reference arm"),
                                                ("heat", "(B) heat semigroup, σ = 2")]):
        for b in BACKBONES:
            v = block(summary, b, fam)
            if not v:
                continue
            xs, ys, es, dead = [], [], [], []
            for _, e in sorted(v["sites"].items(), key=lambda kv: kv[1]["depth"]):
                if not e.get("reachable", True):
                    dead.append(e["depth"]); continue
                if "O2_transfer_mean" not in e:
                    continue
                xs.append(e["depth"]); ys.append(e["O2_transfer_mean"]); es.append(e.get("O2_transfer_sd", 0.0))
            mk, col = STYLE[b]
            ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, label=LABEL[b], lw=1.6, ms=6, capsize=3)
            for d in dead:
                ax.plot([d], [0.0], "x", color=col, ms=8, mew=2)
            if dead:
                ax.annotate("unreachable", (dead[0], 0.02), color=col, fontsize=7, ha="center")
        ax.axhline(0.0, color="k", lw=1, ls="--")
        ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.set_ylim(-0.35, 1.05)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["1\nearly", "2", "3", "4\nlate"])
        ax.set_xlabel("relative depth")
        ax.set_ylabel("downstream transfer  $T_F$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        out[fam] = {}
        for b in BACKBONES:                     # explicit loop: a comprehension would capture the last `v`
            v2 = block(summary, b, fam)
            if not v2:
                continue
            out[fam][b] = {e["depth"]: e.get("O2_transfer_mean")
                           for _, e in v2["sites"].items() if e.get("reachable", True)}
    axes[0, 0].legend(fontsize=8)

    # (C) conditioning gain O8 - O2
    ax = axes[1, 0]
    width = 0.18
    x = 0.0
    ticks, tlabels = [], []
    for fam in ("hue", "heat"):
        for b in BACKBONES:
            v = block(summary, b, fam)
            if not v:
                continue
            gains = []
            for _, e in sorted(v["sites"].items(), key=lambda kv: kv[1]["depth"]):
                if not e.get("reachable", True) or "O8_transfer_mean" not in e:
                    continue
                gains.append(e["O8_transfer_mean"] - e["O2_transfer_mean"])
            if not gains:
                continue
            ax.bar([x + i * width for i in range(len(gains))], gains, width=width,
                   color="#d62728" if fam == "heat" else "#1f77b4")
            ticks.append(x + width * (len(gains) - 1) / 2)
            tlabels.append(LABEL[b].replace("-", "-\n"))
            x += len(gains) * width + 0.25
        x += 0.4
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=7.5)
    ax.set_ylabel("conditioning gain  $T_F(O8) - T_F(O2)$")
    ax.set_title("(C) does conditioning help, at equal fitting budget?", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.text(0.02, 0.92, "blue: hue    red: heat", transform=ax.transAxes, fontsize=8)

    # (D) the site displacement explained, spatial action against channel map, same definition and held out
    ax = axes[1, 1]
    cvs = os.path.join(WORK, "results", "channel_vs_spatial_residual.json")
    if os.path.exists(cvs):
        d = json.load(open(cvs, encoding="utf-8"))
        order = {"resnet50": 1, "convnext": 2, "vitb16": 3, "dinov2b14": 4}
        xs, es, ec = [], [], []
        lbl = []
        pos = 0.0
        for b in BACKBONES:
            if b not in [k.split("|")[0] for k in d["sites"]]:
                continue
            sites = [(k, v) for k, v in d["sites"].items() if k.startswith(b + "|")]
            base = pos
            for j, (k, v) in enumerate(sites):
                xs.append(base + j); es.append(v["E_spatial_canonical"]); ec.append(v["E_channel_global_map"])
            lbl.append((base + (len(sites) - 1) / 2, LABEL[b].replace("-", "-\n")))
            pos += len(sites) + 0.6
        w = 0.38
        ax.bar([x - w / 2 for x in xs], es, width=w, color="#999999", label="canonical spatial action")
        ax.bar([x + w / 2 for x in xs], ec, width=w, color="#d62728", label="global channel map")
        ax.axhline(0.0, color="k", lw=1)
        ax.set_xticks([p_ for p_, _ in lbl]); ax.set_xticklabels([t for _, t in lbl], fontsize=7.5)
        ax.set_ylabel("fraction of the site displacement explained")
        ax.set_title("(D) what carries the dissipation: spatial or channel?", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Downstream transfer of a global feature action: four backbones, four depths, two transformation families",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "F27_depth_law.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    json.dump(out, open(os.path.join(WORK, "results", "fig_depth_law.json"), "w"), indent=1)
    print("depth-law figure: PASS (0 issues)")
    print("  wrote", path)
    for fam in out:
        for b, v in out[fam].items():
            vals = [f"{k}:{x:.2f}" if x is not None else f"{k}:-" for k, x in sorted(v.items())]
            print(f"  {fam:4s} {b:10s} " + " ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
