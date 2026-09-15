# -*- coding: utf-8 -*-
"""161_multiattr_summary.py — aggregate the multi-attribute intervention runs for Paper B.

Reads results/multiattr_intervention_vitb16_b4_s{0,1,2}.json (script 157, synthetic controlled family)
and results/realmultiattr_intervention_<backbone>_b4_s0.json (script 159, real COCO instance crops) and
writes results/multiattr_summary.{json,md} with, per attribute and operator, the mean and sd across
seeds of: input power, transfer fraction, projection coefficient, alignment cosine and win rate.

Usage: python scripts/161_multiattr_summary.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, glob
import numpy as np

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")
ATTRS = ["hue", "saturation", "value", "quantity"]
OPS = ["O1_procrustes", "O5_mlp", "O8_conv_residual", "O6_random_orthogonal"]
FIELDS = ["power_block_rel_l2", "probe_err_null", "probe_err_real", "attr_transfer",
          "proj_coef_agg", "align_cos", "win_rate", "repr_rel_l2_vs_real", "final_rel_l2_vs_real"]


def collect(paths):
    runs = [json.load(open(p)) for p in paths]
    out = {"n_runs": len(runs), "files": [os.path.basename(p) for p in paths], "attributes": {}}
    for attr in ATTRS:
        if attr not in runs[0]["attributes"]:
            continue
        rec = {"power_mean": float(np.mean([r["attributes"][attr]["power_block_rel_l2"] for r in runs])),
               "power_sd": float(np.std([r["attributes"][attr]["power_block_rel_l2"] for r in runs])),
               "probe_err_null_mean": float(np.mean([r["attributes"][attr]["probe_err_null"] for r in runs])),
               "probe_err_real_mean": float(np.mean([r["attributes"][attr]["probe_err_real"] for r in runs])),
               "ops": {}}
        for op in OPS:
            if op not in runs[0]["attributes"][attr]["ops"]:
                continue
            vals = {}
            for f in FIELDS:
                v = [r["attributes"][attr]["ops"][op].get(f) for r in runs]
                v = [x for x in v if x is not None]
                if not v:
                    continue
                vals[f + "_mean"] = float(np.mean(v))
                vals[f + "_sd"] = float(np.std(v))
                vals[f + "_min"] = float(min(v))
                vals[f + "_max"] = float(max(v))
            rec["ops"][op] = vals
        out["attributes"][attr] = rec
    return out


def main():
    synth = sorted(glob.glob(os.path.join(RES, "multiattr_intervention_vitb16_b4_s*.json")))
    real = sorted(glob.glob(os.path.join(RES, "realmultiattr_intervention_*_b4_s*.json")))
    assert synth, "no synthetic multi-attribute runs found"
    out = {"synthetic_vitb16": collect(synth), "real": {}}
    by_backbone = {}
    for p in real:
        bb = os.path.basename(p).replace("realmultiattr_intervention_", "").split("_b4_s")[0]
        by_backbone.setdefault(bb, []).append(p)
    for name, paths in by_backbone.items():
        out["real"][name] = collect(sorted(paths))

    jf = os.path.join(RES, "multiattr_summary.json")
    json.dump(out, open(jf, "w"), indent=1)

    def fmt(a, s):
        return f"{a:.3f}±{s:.3f}"

    L = ["# Multi-attribute intervention — aggregated results", "",
         f"Synthetic controlled family: ViT-B/16 block 4, seeds {out['synthetic_vitb16']['n_runs']} "
         f"(`{'`, `'.join(out['synthetic_vitb16']['files'])}`).", "",
         "| attribute | power | operator | transfer | projection | cosine | win rate |", "|---|---|---|---|---|---|---|"]
    for attr, rec in out["synthetic_vitb16"]["attributes"].items():
        for op, v in rec["ops"].items():
            L.append(f"| {attr} | {fmt(rec['power_mean'], rec['power_sd'])} | {op} | "
                     f"{fmt(v['attr_transfer_mean'], v['attr_transfer_sd'])} | "
                     f"{fmt(v['proj_coef_agg_mean'], v['proj_coef_agg_sd'])} | "
                     f"{fmt(v['align_cos_mean'], v['align_cos_sd'])} | "
                     f"{fmt(v['win_rate_mean'], v['win_rate_sd'])} |")
    for name, blk in out["real"].items():
        L += ["", f"## Real domain — {name} (COCO instance crops, {blk['n_runs']} seed(s))", "",
              "| attribute | power | err null | err real | operator | transfer | projection | cosine | win rate |",
              "|---|---|---|---|---|---|---|---|---|"]
        for attr, rec in blk["attributes"].items():
            for op, v in rec["ops"].items():
                L.append(f"| {attr} | {fmt(rec['power_mean'], rec['power_sd'])} | "
                         f"{rec['probe_err_null_mean']:.3f} | {rec['probe_err_real_mean']:.3f} | {op} | "
                         f"{fmt(v['attr_transfer_mean'], v['attr_transfer_sd'])} | "
                         f"{fmt(v['proj_coef_agg_mean'], v['proj_coef_agg_sd'])} | "
                         f"{fmt(v['align_cos_mean'], v['align_cos_sd'])} | "
                         f"{fmt(v['win_rate_mean'], v['win_rate_sd'])} |")
    mf = os.path.join(RES, "multiattr_summary.md")
    open(mf, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("saved", jf, "and", mf)
    for attr, rec in out["synthetic_vitb16"]["attributes"].items():
        line = f"{attr:11s} power {rec['power_mean']:.3f} | " + " ".join(
            f"{op.split('_')[0]} {v['attr_transfer_mean']:.2f}/{v['proj_coef_agg_mean']:.2f}" for op, v in rec["ops"].items())
        print(line)


if __name__ == "__main__":
    main()
