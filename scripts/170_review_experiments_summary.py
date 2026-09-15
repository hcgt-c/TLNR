# -*- coding: utf-8 -*-
"""170_review_experiments_summary.py — aggregate the four experiments run in response to the reviews.

Reads
  results/capacity_controls_z2_s{0,1,2}_d90.json         (script 167 — capacity / shuffled-pair controls)
  results/causal_unseen_z2_s{0,1,2}_stage1.json          (script 146 at n_train=1000 — composition at scale)
  results/real_task_consumer.json                        (script 168 — real-image value-task consumer)
  results/boundary_largescale.json                       (script 169 — pre-registered boundary validation)
and writes results/review_experiments_summary.{json,md}, one section per experiment, each labelled with
the reviewer objection it answers. Missing inputs are reported as "not yet run" rather than skipped.

Usage: python scripts/170_review_experiments_summary.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, glob
import numpy as np

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(a, s):
    return f"{a:.3f}±{s:.3f}"


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def main():
    out = {}

    # ---- 1. capacity / shuffled-pair controls
    files = sorted(glob.glob(os.path.join(RES, "capacity_controls_z2_s*_d90.json")))
    if files:
        runs = [json.load(open(f)) for f in files]
        names = [n for n in runs[0]["controls"].keys() if n != "O8_unfit"]   # identical to the frozen row by construction
        tbl = {}
        for nm in names:
            tbl[nm] = {
                "proj": mean_sd([r["controls"][nm]["proj_coef_agg"] for r in runs]),
                "cos": mean_sd([r["controls"][nm]["align_cos"] for r in runs]),
                "win": mean_sd([r["controls"][nm]["win_rate"] for r in runs]),
                "feat_relL2": mean_sd([r["controls"][nm]["feature_rel_l2"] for r in runs]),
                "top1": mean_sd([r["controls"][nm]["top1_agree_real"] for r in runs]),
            }
        out["capacity_controls"] = {"n_seeds": len(runs), "files": [os.path.basename(f) for f in files], "rows": tbl,
                                    "answers": "Is O8's advantage capacity, or the correct content/neighbourhood pairing?"}

    # ---- 2. composition / unseen shift at scale
    files = sorted(glob.glob(os.path.join(RES, "causal_unseen_z2_s*_stage1.json")))
    files = [f for f in files if json.load(open(f)).get("n_train", 0) >= 1000]
    if files:
        runs = [json.load(open(f)) for f in files]
        keys = [k for k in ("O1_unseen_compose", "O1_ref_fit90", "O8_unseen_compose", "O8_ref_fit90", "O6_random")
                if k in runs[0]["variants"]]
        tbl = {k: {"proj": mean_sd([r["variants"][k]["proj_coef_agg"] for r in runs]),
                   "cos": mean_sd([r["variants"][k]["align_cos"] for r in runs]),
                   "win": mean_sd([r["variants"][k]["vs_null_win_rate"] for r in runs])} for k in keys}
        out["composition_scale"] = {"n_seeds": len(runs), "n_train": runs[0].get("n_train"),
                                    "files": [os.path.basename(f) for f in files], "rows": tbl,
                                    "answers": "Does the composed operator still reach the directly fitted one at n_train = 1000?"}

    # ---- 3. real-image task consumer
    d = load(os.path.join(RES, "real_task_consumer.json"))
    if d:
        out["real_task_consumer"] = {
            "n_regions": d["n_regions"], "n_test": d["n_test"], "task_acc": d["task_acc_three_levels"],
            "power": d["power"],
            "rows": {k: {"proj": v["proj_coef_agg"], "cos": v["align_cos"], "win": v["win_rate"],
                         "top1": v["top1_agree_real"], "feat_relL2": v["feature_rel_l2"]}
                     for k, v in d["ops"].items()},
            "answers": "Does the intervention reproduce a real-image task consumer's response to a real transform?"}

    # ---- 4. large-scale boundary validation
    d = load(os.path.join(RES, "boundary_largescale.json"))
    if d:
        out["boundary_largescale"] = {
            "n_regions": d["n_regions"], "n_images": d["n_images"], "m_star": d["m_star"],
            "test_half": d["test_half"], "test_half_raw": d["test_half_raw"],
            "auc_align": d["auc_test_align"], "auc_align_ci95": d["auc_test_align_ci95"],
            "auc_raw": d["auc_test_raw"], "auc_gtbox": d["auc_test_gtbox"],
            "gap_ci95": d.get("gap_ci95_corrected"), "gap_p_le0": d.get("gap_p_le0"),
            "calibration_deciles": d["calibration"]["deciles_test"],
            "answers": "Does the pre-registered m* = 0.70 rule transfer out of sample and out of the aligned regime?"}

    json.dump(out, open(os.path.join(RES, "review_experiments_summary.json"), "w"), indent=1)

    L = ["# Review-response experiments — summary", ""]
    if "capacity_controls" in out:
        c = out["capacity_controls"]
        L += [f"## 1. Capacity / shuffled-pair controls (script 167, {c['n_seeds']} seeds)", "",
              f"*answers:* {c['answers']}", "",
              "| control | projection | cosine | win rate | feature rel-L2 | top-1 agreement |", "|---|---|---|---|---|---|"]
        for nm, v in c["rows"].items():
            L.append(f"| {nm} | {fmt(*v['proj'])} | {fmt(*v['cos'])} | {fmt(*v['win'])} | "
                     f"{fmt(*v['feat_relL2'])} | {fmt(*v['top1'])} |")
        L.append("")
    if "composition_scale" in out:
        c = out["composition_scale"]
        L += [f"## 2. Composition / unseen shift at scale (script 146, n_train = {c['n_train']}, {c['n_seeds']} seeds)", "",
              f"*answers:* {c['answers']}", "", "| variant | projection | cosine | win rate |", "|---|---|---|---|"]
        for nm, v in c["rows"].items():
            L.append(f"| {nm} | {fmt(*v['proj'])} | {fmt(*v['cos'])} | {fmt(*v['win'])} |")
        L.append("")
    if "real_task_consumer" in out:
        c = out["real_task_consumer"]
        L += ["## 3. Real-image task consumer (script 168)", "",
              f"*answers:* {c['answers']}", "",
              f"Trained task accuracy (three value levels, held-out regions): **{c['task_acc']:.3f}**; "
              f"transform effect size {c['power']['effect_rel_l2']:.3f}, flip rate {c['power']['flip_rate']:.3f}.", "",
              "| operator | projection | cosine | win rate | top-1 agreement | feature rel-L2 |", "|---|---|---|---|---|---|"]
        for nm, v in c["rows"].items():
            L.append(f"| {nm} | {v['proj']:.2f} | {v['cos']:.2f} | {v['win']:.2f} | {v['top1']:.2f} | {v['feat_relL2']:.3f} |")
        L.append("")
    if "boundary_largescale" in out:
        c = out["boundary_largescale"]
        L += ["## 4. Pre-registered boundary validation (script 169)", "",
              f"*answers:* {c['answers']}", "",
              f"{c['n_regions']} regions from {c['n_images']} images; m* = {c['m_star']} fixed in advance; "
              f"test half: success above m* {c['test_half']['success_hi']:.3f} vs below {c['test_half']['success_lo']:.3f} "
              f"(gap {c['test_half']['gap']:+.3f}); ROC AUC aligned {c['auc_align']:.3f} "
              f"(95% CI {c['auc_align_ci95'][0]:.3f}-{c['auc_align_ci95'][1]:.3f}), "
              f"ground-truth-box route {c['auc_gtbox']:.3f}, raw route {c['auc_raw']:.3f}.", ""]
    if len(L) == 2:
        L.append("_No experiment results found yet._")
    open(os.path.join(RES, "review_experiments_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("saved results/review_experiments_summary.{json,md}")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
