# -*- coding: utf-8 -*-
"""158_detector_o8_summary.py — aggregate the multi-seed detector-domain intervention runs.

Reads results/detector_o8_seed{0,1,2}.json (script 156) and reports, per delta and route, the
box-level F1@IoU0.5 / AP50 and the tensor-level projection coefficient as mean +- sd across seeds,
plus the paired per-seed gain of O8 over O1 / O2 and the random-orthogonal null.

Outputs results/detector_o8_multiseed.json and .md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import numpy as np

WORK = rp.REPO_ROOT
SEEDS = [0, 1, 2]
ROUTES = ["null", "O1_procrustes", "O2_ridge", "O8_conv_residual", "O6_random"]


def main():
    runs = {}
    for s in SEEDS:
        fn = os.path.join(WORK, "results", f"detector_o8_seed{s}.json")
        runs[s] = json.load(open(fn))
    deltas = sorted(runs[SEEDS[0]]["deltas"].keys(), key=float)
    out = {"seeds": SEEDS, "n_train": runs[0]["n_train"], "n_held": runs[0]["n_held"],
           "weights": runs[0]["weights"], "deltas": {}}
    for d in deltas:
        rec = {"routes": {}, "paired": {}}
        for r in ROUTES:
            f1 = [runs[s]["deltas"][d]["routes"][r]["box"]["f1_iou50"] for s in SEEDS]
            ap = [runs[s]["deltas"][d]["routes"][r]["box"]["ap50_11pt"] for s in SEEDS]
            prec = [runs[s]["deltas"][d]["routes"][r]["box"]["precision"] for s in SEEDS]
            rec_ = runs[0]["deltas"][d]["routes"][r]
            rec["routes"][r] = {
                "f1_mean": float(np.mean(f1)), "f1_sd": float(np.std(f1, ddof=0)),
                "f1_per_seed": [float(v) for v in f1],
                "ap50_mean": float(np.mean(ap)), "ap50_sd": float(np.std(ap, ddof=0)),
                "ap50_per_seed": [float(v) for v in ap],
                "precision_mean": float(np.mean(prec)),
                "tensor_rel_l2_vs_real_mean": float(np.mean(
                    [runs[s]["deltas"][d]["routes"][r]["tensor_rel_l2_vs_real"] for s in SEEDS])),
                "proj_coef_mean": float(np.mean([runs[s]["deltas"][d]["routes"][r].get("proj_coef_agg", 0.0)
                                                 for s in SEEDS])) if "proj_coef_agg" in rec_ else 0.0,
            }
        for a, b in [("O8_conv_residual", "O1_procrustes"), ("O8_conv_residual", "O2_ridge")]:
            g = [runs[s]["deltas"][d]["routes"][a]["box"]["f1_iou50"] -
                 runs[s]["deltas"][d]["routes"][b]["box"]["f1_iou50"] for s in SEEDS]
            rec["paired"][f"{a}_minus_{b}"] = {"f1_gain_per_seed": [float(v) for v in g],
                                               "f1_gain_mean": float(np.mean(g)),
                                               "wins": int(sum(v > 0 for v in g)), "n": len(SEEDS)}
        for a in ["O1_procrustes", "O2_ridge", "O8_conv_residual"]:
            g = [runs[s]["deltas"][d]["routes"][a]["box"]["f1_iou50"] -
                 runs[s]["deltas"][d]["routes"]["null"]["box"]["f1_iou50"] for s in SEEDS]
            rec["paired"][f"{a}_minus_null"] = {"f1_gain_mean": float(np.mean(g)), "wins": int(sum(v > 0 for v in g))}
        out["deltas"][d] = rec

    jf = os.path.join(WORK, "results", "detector_o8_multiseed.json")
    json.dump(out, open(jf, "w"), indent=1)

    L = ["# Detector-domain operator intervention — multi-seed summary", "",
         f"YOLO11n on the hue-class synthetic detector (scripts 156/158); seeds {SEEDS}; "
         f"n_train={out['n_train']} scenes, n_held={out['n_held']}; weights `{out['weights']}`.", "",
         "| delta | route | F1@0.5 (mean±sd) | AP50 (mean±sd) | tensor relL2 | proj coef |",
         "|---|---|---|---|---|---|"]
    for d in deltas:
        for r in ROUTES:
            v = out["deltas"][d]["routes"][r]
            L.append(f"| {d}° | {r} | {v['f1_mean']:.3f}±{v['f1_sd']:.3f} | "
                     f"{v['ap50_mean']:.3f}±{v['ap50_sd']:.3f} | {v['tensor_rel_l2_vs_real_mean']:.3f} | "
                     f"{v['proj_coef_mean']:.2f} |")
    L += ["", "Paired gains (F1@0.5, per-seed wins out of %d):" % len(SEEDS), ""]
    for d in deltas:
        for k, v in out["deltas"][d]["paired"].items():
            L.append(f"- delta {d}°: {k} = {v['f1_gain_mean']:+.3f} "
                     f"({v['wins']}/{v.get('n', len(SEEDS))} seeds positive)")
    mf = os.path.join(WORK, "results", "detector_o8_multiseed.md")
    open(mf, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("saved", jf, "and", mf)
    for d in deltas:
        r = out["deltas"][d]["routes"]
        print(f"delta {d}: null F1 {r['null']['f1_mean']:.3f} O1 {r['O1_procrustes']['f1_mean']:.3f} "
              f"O2 {r['O2_ridge']['f1_mean']:.3f} O8 {r['O8_conv_residual']['f1_mean']:.3f} "
              f"O6 {r['O6_random']['f1_mean']:.3f}")


if __name__ == "__main__":
    main()
