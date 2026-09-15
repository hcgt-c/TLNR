# -*- coding: utf-8 -*-
"""137_causal_summary.py — aggregate results/causal_cifar_*.json into paper tables.

Outputs results/causal_cifar_summary.json and results/causal_cifar_summary.md:
per (arm, delta, operator): power of real recolour, downstream equivalence to the real recolour
(rel L2), direction alignment (cos), aggregate projection coefficient, win rate vs null
(fraction of held-out images where the intervention is closer to the real recolour than no-op),
and feature-space fidelity. Mean +/- std across seeds; arms with fewer than 3 seeds are flagged.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, glob, statistics as st

WORK = rp.REPO_ROOT
OPS = ["null", "O1_procrustes", "O2_ridge", "O4_blockrot", "O5_mlp", "O7_lowrank_mlp", "O8_conv_residual", "O6_random"]
ARMS = ["z2", "z2hue", "ce8", "lcer8", "ocode"]


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def main():
    files = sorted(glob.glob(os.path.join(WORK, "results", "causal_cifar_*.json")))
    data = {}
    for fn in files:
        base = os.path.basename(fn)
        if base.startswith("causal_cifar_summary"):
            continue
        d = json.load(open(fn))
        data.setdefault((d["arm"], d.get("stage", 1)), {})[d["seed"]] = d
    summ = {}
    for akey in sorted(data):
        arm, stage = akey
        seeds = sorted(data[akey])
        per_delta = {}
        for delta in ["30", "60", "90"]:
            have = [data[akey][s]["deltas"][delta] for s in seeds if delta in data[akey][s]["deltas"]]
            if not have:
                continue
            row = {"n_seeds": len(have), "power": {}, "ops": {}}
            for k in ["rel_l2_y_real_vs_null", "top1_flip_rate_real_vs_null", "acc_null", "acc_real"]:
                m, s, n = ms([h["power"][k] for h in have])
                row["power"][k] = {"mean": m, "std": s, "n": n}
            for op in OPS:
                got = [h["downstream"][op] for h in have if op in h["downstream"]]
                fid = [h["feature_fidelity"].get(op) for h in have]
                if not got:
                    continue
                rec = {}
                for k in ["rel_l2_vs_real", "rel_l2_vs_null", "kl_real_int", "top1_agree_real",
                          "acc_int", "align_cos", "proj_coef_agg", "vs_null_win_rate", "vs_null_mean_gain"]:
                    m, s, n = ms([g.get(k) for g in got])
                    rec[k] = {"mean": m, "std": s, "n": n}
                m, s, n = ms([f["rel_l2"] for f in fid if f])
                rec["feature_rel_l2"] = {"mean": m, "std": s, "n": n}
                m, s, n = ms([f["angle_deg"] for f in fid if f])
                rec["feature_angle_deg"] = {"mean": m, "std": s, "n": n}
                row["ops"][op] = rec
            per_delta[delta] = row
        summ[f"{arm}_stage{stage}"] = per_delta
    json.dump(summ, open(os.path.join(WORK, "results", "causal_cifar_summary.json"), "w"), indent=1)

    # markdown
    L = ["# Causal intervention on colour-sensitive CIFAR arms — summary", "",
         "Metric definitions: `rel_l2_vs_real` = distance of intervened output to the real-recolour output;",
         "`align_cos` = cosine between (intervened - null) and (real - null); `proj` = aggregate projection",
         "coefficient (1 = direction and magnitude reproduced); `win` = fraction of held-out images where the",
         "intervention is closer to the real recolour than no-op; `power` = ||real - null||.", ""]
    for arm, per_delta in summ.items():
        L.append(f"## {arm}")
        for delta, row in per_delta.items():
            p = row["power"]
            L.append(f"### delta={delta}° (seeds={row['n_seeds']})")
            L.append(f"power: rel_l2 {p['rel_l2_y_real_vs_null']['mean']:.4f}±{p['rel_l2_y_real_vs_null']['std']:.4f}, "
                     f"top1 flip {p['top1_flip_rate_real_vs_null']['mean']:.3f}, "
                     f"acc null/real {p['acc_null']['mean']:.3f}/{p['acc_real']['mean']:.3f}")
            L.append("")
            L.append("| op | rel_l2_vs_real | align_cos | proj | win | feat rel_l2 | feat angle |")
            L.append("|---|---|---|---|---|---|---|")
            for op, r in row["ops"].items():
                L.append(f"| {op} | {r['rel_l2_vs_real']['mean']:.3f}±{r['rel_l2_vs_real']['std']:.3f} | "
                         f"{r['align_cos']['mean']:.2f} | {r['proj_coef_agg']['mean']:.2f} | "
                         f"{r['vs_null_win_rate']['mean']:.2f} | {r['feature_rel_l2']['mean']:.3f} | "
                         f"{r['feature_angle_deg']['mean']:.1f}° |")
            L.append("")
    md = "\n".join(L)
    open(os.path.join(WORK, "results", "causal_cifar_summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
