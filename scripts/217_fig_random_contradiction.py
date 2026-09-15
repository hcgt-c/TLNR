# -*- coding: utf-8 -*-
"""217_fig_random_contradiction.py — the alignment-vs-behaviour contradiction, from existing results only.

One setting (plain CIFAR-10 arm, stage 1, hue shift 60 deg, three seeds) plotted twice: the aggregate projection
coefficient of each operator family, and the paired win rate against the no-op. The random orthogonal control has
the *highest* projection coefficient of the families and a win rate at the noise floor.

Source: results/causal_cifar_summary.json (no new runs).
Outputs: paper/figures/F34_random_contradiction.png, results/fig_random_contradiction.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = rp.REPO_ROOT
SRC = os.path.join(WORK, "results", "causal_cifar_summary.json")
OUT_PNG = os.path.join(WORK, "paper", "figures", "F34_random_contradiction.png")
OUT_JSON = os.path.join(WORK, "results", "fig_random_contradiction.json")

ARM, DELTA = "z2_stage1", "60"
OPS = [("O1_procrustes", "O1 orthogonal"), ("O2_ridge", "O2 linear"),
       ("O5_mlp", "O5 content"), ("O8_conv_residual", "O8 content+neighbourhood"),
       ("O6_random", "O6 random")]


def main():
    d = json.load(open(SRC, encoding="utf-8"))[ARM][DELTA]
    labels, proj, proj_sd, win, win_sd = [], [], [], [], []
    rec = {"arm": ARM, "delta_deg": float(DELTA), "operators": {}}
    for key, lab in OPS:
        o = d["ops"][key]
        p, ps = o["proj_coef_agg"]["mean"], o["proj_coef_agg"]["std"]
        w, ws = o["vs_null_win_rate"]["mean"], o["vs_null_win_rate"]["std"]
        labels.append(lab); proj.append(p); proj_sd.append(ps); win.append(w); win_sd.append(ws)
        rec["operators"][lab] = {"projection_mean": p, "projection_sd": ps,
                                 "win_rate_mean": w, "win_rate_sd": ws}
    x = np.arange(len(labels)); colors = ["#4c72b0"] * 4 + ["#c44e52"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].bar(x, proj, yerr=proj_sd, color=colors, capsize=3)
    axes[0].set_ylabel("aggregate projection coefficient")
    axes[0].set_title("Alignment: the random control is not last", fontsize=10)
    axes[1].bar(x, win, yerr=win_sd, color=colors, capsize=3)
    axes[1].set_ylabel("paired win rate vs no-op")
    axes[1].set_title("Behaviour: the random control is at the floor", fontsize=10)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=7.5)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Plain CNN, stage 1, hue shift 60 deg, three seeds: alignment is not faithfulness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=200)
    rec["source"] = "results/causal_cifar_summary.json"
    json.dump(rec, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT_PNG); print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
