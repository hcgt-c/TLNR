# -*- coding: utf-8 -*-
"""148_verify_paperB_numbers.py — self-audit: every headline number in the Paper B draft vs its source.

For each claim it (i) looks for the literal in paperB_manuscript_v0.md, (ii) recomputes the value from
results/causal_cifar_summary.json, results/causal_intervention_hue.json,
results/causal_unseen_z2_s0_stage1.json or the detector results.csv, and (iii) reports PASS/FAIL with
the rounding tolerance. Writes results/paperB_numbers_check.{json,md}.

Usage: python scripts/148_verify_paperB_numbers.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, re, json, csv
import numpy as np

WORK = rp.REPO_ROOT
DRAFT = os.path.join(WORK, "paper", "paperB_manuscript_v0.md")
SUM = json.load(open(os.path.join(WORK, "results", "causal_cifar_summary.json")))
DET = json.load(open(os.path.join(WORK, "results", "causal_intervention_hue.json")))
UNS = json.load(open(os.path.join(WORK, "results", "causal_unseen_z2_s0_stage1.json")))
PAIR = [json.load(open(os.path.join(WORK, "results", f"paired_stats_z2_s{s}_stage1.json"))) for s in (0, 1, 2)]
VIT = [json.load(open(os.path.join(WORK, "results", f"vit_causal_vitb16_b4_s{s}.json"))) for s in (0, 1, 2)]
VITD = json.load(open(os.path.join(WORK, "results", "vit_causal_dinov2b14_b4_s0.json")))
MSEED = json.load(open(os.path.join(WORK, "results", "detector_o8_multiseed.json")))
MSUM = json.load(open(os.path.join(WORK, "results", "multiattr_summary.json")))
MATTR = json.load(open(os.path.join(WORK, "results", "multiattr_intervention_vitb16_b4_s0.json")))


def ms(delta, route, field="f1_mean"):
    return MSEED["deltas"][delta]["routes"][route][field]


def msgain(delta, key):
    return MSEED["deltas"][delta]["paired"][key]["f1_gain_mean"]


def ma(attr, field, op=None):
    r = MATTR["attributes"][attr]
    if op is None:
        return r[field]
    return r["ops"][op][field]


def ms_(attr, op, field):
    return MSUM["synthetic_vitb16"]["attributes"][attr]["ops"][op][field + "_mean"]


def msp(attr):
    return MSUM["synthetic_vitb16"]["attributes"][attr]["power_mean"]


def real(backbone, attr, op, field):
    return MSUM["real"][backbone]["attributes"][attr]["ops"][op][field + "_mean"]


def realp(backbone, attr):
    return MSUM["real"][backbone]["attributes"][attr]["power_mean"]


DINOV2 = [json.load(open(os.path.join(WORK, "results", f"vit_causal_dinov2b14_b4_s{s}.json"))) for s in (0, 1, 2)]
CAP = [json.load(open(os.path.join(WORK, "results", f"capacity_controls_z2_s{s}_d90.json"))) for s in (0, 1, 2)]
UNS3 = [json.load(open(os.path.join(WORK, "results", f"causal_unseen_z2_s{s}_stage1.json"))) for s in (0, 1, 2)]
RTASK = json.load(open(os.path.join(WORK, "results", "real_task_consumer.json")))
PSP = json.load(open(os.path.join(WORK, "results", "per_sample_projection_z2_s0_d90.json")))
HREG = json.load(open(os.path.join(WORK, "results", "hue_readout_regimes.json")))
TW = [json.load(open(os.path.join(WORK, "results", f"task_weighted_1x1_z2_s{s}_d90.json"))) for s in (0, 1, 2)]


def tw(op, field):
    """Mean over the three seeds of script 180 (Table 9)."""
    return float(np.mean([d["operators"][op][field] for d in TW]))


def tw_train(op, field):
    return float(np.mean([d["train_diagnostic"][op][field] for d in TW]))


def uns3(name, field):
    return float(np.mean([r["variants"][name][field] for r in UNS3]))


def cap(name, field):
    return float(np.mean([r["controls"][name][field] for r in CAP]))


def op_mean(arm, delta, op, field="proj_coef_agg"):
    return SUM[arm][delta]["ops"][op][field]["mean"]


def power(arm, delta, field="rel_l2_y_real_vs_null"):
    return SUM[arm][delta]["power"][field]["mean"]


def det_ap50(delta, route):
    de = DET["downstream_equivalence"][delta]
    if route == "null":
        return de["null_unintervened"]["box_agreement_AP50"]
    if route in de and isinstance(de[route], dict):
        return de[route]["box_agreement_AP50"]
    return de["variants"][route]["box_agreement_AP50"]


def unseen(name, field):
    return UNS["variants"][name][field]


def map50_from_csv():
    """Read the saved post-training validation (results/yolo_hue_val.json) if present."""
    p = os.path.join(WORK, "results", "yolo_hue_val.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return float(d["mAP50"]), float(d["mAP50_95"])
    p = os.path.join(WORK, "runs/detect/runs_det_hue/hue_cls8/results.csv")
    rows = [r for r in csv.DictReader(open(p))]
    last = rows[-1]
    return float(last["metrics/mAP50(B)"]), float(last["metrics/mAP50-95(B)"])


def main():
    txt = open(DRAFT, encoding="utf-8").read().replace("\u2212", "-")  # U+2212 in prose/tables
    m50, m5095 = map50_from_csv()
    checks = [
        # (label, expected, tolerance, must_appear_in_draft)
        ("power plain stage1 d90", power("z2_stage1", "90"), 0.005, "0.389"),
        ("O1 plain stage1 d90", op_mean("z2_stage1", "90", "O1_procrustes"), 0.005, "0.31"),
        ("O2 plain stage1 d90", op_mean("z2_stage1", "90", "O2_ridge"), 0.005, "0.61"),
        ("O5 plain stage1 d90", op_mean("z2_stage1", "90", "O5_mlp"), 0.005, "0.73"),
        ("O8 plain stage1 d90", op_mean("z2_stage1", "90", "O8_conv_residual"), 0.005, "0.83"),
        ("O1 plain stage2 d90", op_mean("z2_stage2", "90", "O1_procrustes"), 0.005, "0.15"),
        ("O2 plain stage2 d90", op_mean("z2_stage2", "90", "O2_ridge"), 0.005, "0.45"),
        ("O5 plain stage2 d90", op_mean("z2_stage2", "90", "O5_mlp"), 0.005, "0.55"),
        ("O8 plain stage2 d90", op_mean("z2_stage2", "90", "O8_conv_residual"), 0.005, "0.69"),
        ("O1 ce8 stage1 d90", op_mean("ce8_stage1", "90", "O1_procrustes"), 0.005, "0.94"),
        ("O8 ce8 stage1 d90", op_mean("ce8_stage1", "90", "O8_conv_residual"), 0.005, "0.97"),
        ("O1 ce8 stage2 d90", op_mean("ce8_stage2", "90", "O1_procrustes"), 0.005, "0.93"),
        ("O1 ocode d90", op_mean("ocode_stage1", "90", "O1_procrustes"), 0.005, "0.30"),
        ("O1 hue-aug d90", op_mean("z2hue_stage1", "90", "O1_procrustes"), 0.005, "0.049"),
        ("O1 lcer8 d90", op_mean("lcer8_stage1", "90", "O1_procrustes"), 0.005, "0.045"),
        ("power hue-aug d90", power("z2hue_stage1", "90"), 0.005, "0.141"),
        ("power lcer8 d90", power("lcer8_stage1", "90"), 0.005, "0.013"),
        ("det multi-seed null d30", ms("30", "null"), 0.002, "0.218"),
        ("det multi-seed null d60", ms("60", "null"), 0.002, "0.016"),
        ("det multi-seed null d90", ms("90", "null"), 0.002, "0.008"),
        ("det multi-seed O1 d30", ms("30", "O1_procrustes"), 0.002, "0.559"),
        ("det multi-seed O1 d60", ms("60", "O1_procrustes"), 0.002, "0.674"),
        ("det multi-seed O1 d90", ms("90", "O1_procrustes"), 0.002, "0.778"),
        ("det multi-seed O2 d30", ms("30", "O2_ridge"), 0.002, "0.696"),
        ("det multi-seed O2 d60", ms("60", "O2_ridge"), 0.002, "0.775"),
        ("det multi-seed O2 d90", ms("90", "O2_ridge"), 0.002, "0.905"),
        ("det multi-seed O8 d30", ms("30", "O8_conv_residual"), 0.002, "0.883"),
        ("det multi-seed O8 d60", ms("60", "O8_conv_residual"), 0.002, "0.845"),
        ("det multi-seed O8 d90", ms("90", "O8_conv_residual"), 0.002, "0.965"),
        ("det multi-seed O8 AP50 d90", ms("90", "O8_conv_residual", "ap50_mean"), 0.002, "0.969"),
        ("det multi-seed O6 d90", ms("90", "O6_random"), 0.002, "0.000"),
        ("det multi-seed gain O8-null d30", msgain("30", "O8_conv_residual_minus_null"), 0.002, "0.666"),
        ("det multi-seed gain O8-null d60", msgain("60", "O8_conv_residual_minus_null"), 0.002, "0.828"),
        ("det multi-seed gain O8-null d90", msgain("90", "O8_conv_residual_minus_null"), 0.002, "0.957"),
        ("det multi-seed gain O8-O1 d30", msgain("30", "O8_conv_residual_minus_O1_procrustes"), 0.002, "0.324"),
        ("det multi-seed gain O8-O1 d60", msgain("60", "O8_conv_residual_minus_O1_procrustes"), 0.002, "0.171"),
        ("det multi-seed gain O8-O1 d90", msgain("90", "O8_conv_residual_minus_O1_procrustes"), 0.002, "0.187"),
        ("unseen O8 composed proj (scaled)", uns3("O8_unseen_compose", "proj_coef_agg"), 0.005, "0.810"),
        ("unseen O8 ref proj (scaled)", uns3("O8_ref_fit90", "proj_coef_agg"), 0.005, "0.808"),
        ("unseen O8 composed win (scaled)", uns3("O8_unseen_compose", "vs_null_win_rate"), 0.005, "0.822"),
        ("unseen O1 composed proj (scaled)", uns3("O1_unseen_compose", "proj_coef_agg"), 0.005, "0.349"),
        ("unseen O6 proj (scaled)", uns3("O6_random", "proj_coef_agg"), 0.005, "0.819"),
        ("unseen O6 win (scaled)", uns3("O6_random", "vs_null_win_rate"), 0.005, "0.005"),
        ("vitb16 effect", VIT[0]["power"]["rel_l2_real_vs_null"], 0.005, "0.79"),
        ("vitb16 flip", VIT[0]["power"]["top1_flip_rate"], 0.005, "0.51"),
        ("vitb16 O5 proj", VIT[0]["ops"]["O5_mlp"]["proj_coef_agg"], 0.005, "0.885"),
        ("vitb16 O8 proj", VIT[0]["ops"]["O8_conv_residual"]["proj_coef_agg"], 0.005, "0.85"),
        ("vitb16 acc real", VIT[0]["power"]["acc_real"], 0.005, "0.37"),
        ("dinov2 effect", VITD["power"]["rel_l2_real_vs_null"], 0.005, "0.13"),
        ("dinov2 O5 proj", VITD["ops"]["O5_mlp"]["proj_coef_agg"], 0.005, "0.46"),
        ("paired O8 win min", min(p["ops"]["O8_conv_residual"]["win_rate"] for p in PAIR), 0.005, "0.84"),
        ("paired O8 win max", max(p["ops"]["O8_conv_residual"]["win_rate"] for p in PAIR), 0.005, "0.88"),
        ("paired O8 gain min", min(p["ops"]["O8_conv_residual"]["mean_gain"] for p in PAIR), 0.005, "0.248"),
        ("paired O8 gain max", max(p["ops"]["O8_conv_residual"]["mean_gain"] for p in PAIR), 0.005, "0.279"),
        ("paired O8 d max", max(p["ops"]["O8_conv_residual"]["cohen_d_paired"] for p in PAIR), 0.005, "0.85"),
        ("paired O8 over O1 win min", min(p["vs_O1"]["O8_conv_residual"]["win_rate_over_O1"] for p in PAIR), 0.005, "0.77"),
        ("dinov2 effect (3 seeds)", DINOV2[0]["power"]["rel_l2_real_vs_null"], 0.005, "0.13"),
        ("dinov2 O1 proj", DINOV2[0]["ops"]["O1_procrustes"]["proj_coef_agg"], 0.005, "0.37"),
        ("dinov2 O5 proj", DINOV2[0]["ops"]["O5_mlp"]["proj_coef_agg"], 0.005, "0.46"),
        ("dinov2 O8 proj mean", float(np.mean([d["ops"]["O8_conv_residual"]["proj_coef_agg"] for d in DINOV2])), 0.005, "0.30"),
        ("multiattr power hue", msp("hue"), 0.002, "0.288±0.021"),
        ("multiattr power saturation", msp("saturation"), 0.002, "0.213±0.016"),
        ("multiattr power value", msp("value"), 0.002, "0.510±0.036"),
        ("multiattr power quantity", msp("quantity"), 0.002, "0.366±0.002"),
        ("multiattr hue O1 transfer", ms_("hue", "O1_procrustes", "attr_transfer"), 0.002, "0.867±0.048"),
        ("multiattr hue O1 proj", ms_("hue", "O1_procrustes", "proj_coef_agg"), 0.002, "0.843±0.054"),
        ("multiattr hue O5 transfer", ms_("hue", "O5_mlp", "attr_transfer"), 0.002, "0.989±0.016"),
        ("multiattr hue O5 proj", ms_("hue", "O5_mlp", "proj_coef_agg"), 0.002, "0.939±0.018"),
        ("multiattr sat O1 transfer", ms_("saturation", "O1_procrustes", "attr_transfer"), 0.002, "0.946±0.034"),
        ("multiattr sat O8 transfer", ms_("saturation", "O8_conv_residual", "attr_transfer"), 0.002, "0.971±0.007"),
        ("multiattr sat O8 proj", ms_("saturation", "O8_conv_residual", "proj_coef_agg"), 0.002, "0.997±0.017"),
        ("multiattr value O1 transfer", ms_("value", "O1_procrustes", "attr_transfer"), 0.002, "0.947±0.013"),
        ("multiattr value O1 proj", ms_("value", "O1_procrustes", "proj_coef_agg"), 0.002, "0.928±0.039"),
        ("multiattr qty O1 transfer", ms_("quantity", "O1_procrustes", "attr_transfer"), 0.002, "0.606±0.022"),
        ("multiattr qty O1 proj", ms_("quantity", "O1_procrustes", "proj_coef_agg"), 0.002, "0.456±0.012"),
        ("multiattr qty O5 transfer", ms_("quantity", "O5_mlp", "attr_transfer"), 0.002, "0.645±0.055"),
        ("multiattr qty O5 proj", ms_("quantity", "O5_mlp", "proj_coef_agg"), 0.002, "0.501±0.029"),
        ("multiattr qty O6 proj", ms_("quantity", "O6_random_orthogonal", "proj_coef_agg"), 0.005, "4.788±1.523"),
        ("multiattr qty O6 transfer", ms_("quantity", "O6_random_orthogonal", "attr_transfer"), 0.005, "-3.630±2.103"),
        ("real vitb16 hue power", realp("vitb16", "hue"), 0.002, "0.689±0.020"),
        ("real vitb16 sat power", realp("vitb16", "saturation"), 0.002, "0.351±0.014"),
        ("real vitb16 value power", realp("vitb16", "value"), 0.002, "0.483±0.005"),
        ("real vitb16 hue O1 transfer", real("vitb16", "hue", "O1_procrustes", "attr_transfer"), 0.002, "0.655±0.073"),
        ("real vitb16 hue O8 transfer", real("vitb16", "hue", "O8_conv_residual", "attr_transfer"), 0.002, "0.563±0.009"),
        ("real vitb16 hue O6 transfer", real("vitb16", "hue", "O6_random_orthogonal", "attr_transfer"), 0.005, "0.222±0.201"),
        ("real vitb16 hue O6 win", real("vitb16", "hue", "O6_random_orthogonal", "win_rate"), 0.002, "0.539"),
        ("real vitb16 sat O5 transfer", real("vitb16", "saturation", "O5_mlp", "attr_transfer"), 0.002, "0.755±0.041"),
        ("real vitb16 sat O6 transfer", real("vitb16", "saturation", "O6_random_orthogonal", "attr_transfer"), 0.005, "-1.220±0.609"),
        ("real vitb16 value O5 transfer", real("vitb16", "value", "O5_mlp", "attr_transfer"), 0.002, "0.846±0.081"),
        ("real vitb16 value O6 transfer", real("vitb16", "value", "O6_random_orthogonal", "attr_transfer"), 0.005, "-3.159±0.837"),
        ("real vitb16 value O6 win", real("vitb16", "value", "O6_random_orthogonal", "win_rate"), 0.002, "0.037"),
        ("real dinov2 hue O1 transfer", real("dinov2b14", "hue", "O1_procrustes", "attr_transfer"), 0.002, "0.484"),
        ("real dinov2 sat O8 transfer", real("dinov2b14", "saturation", "O8_conv_residual", "attr_transfer"), 0.002, "0.992"),
        ("real dinov2 value O1 transfer", real("dinov2b14", "value", "O1_procrustes", "attr_transfer"), 0.002, "0.953"),
        ("real dinov2 value O6 transfer", real("dinov2b14", "value", "O6_random_orthogonal", "attr_transfer"), 0.005, "-10.104"),
        ("capacity O8 correct proj", cap("O8_correct", "proj_coef_agg"), 0.002, "0.812"),
        ("capacity O8 correct win", cap("O8_correct", "win_rate"), 0.002, "0.850"),
        ("capacity O8 shuffled proj", cap("O8_shuffled", "proj_coef_agg"), 0.002, "1.385"),
        ("capacity O8 shuffled cos", cap("O8_shuffled", "align_cos"), 0.002, "0.519"),
        ("capacity O8 shuffled win", cap("O8_shuffled", "win_rate"), 0.002, "0.100"),
        ("capacity ridge-only proj", cap("O8_frozen_residual", "proj_coef_agg"), 0.002, "0.628"),
        ("capacity ridge-only win", cap("O8_frozen_residual", "win_rate"), 0.002, "0.717"),
        ("real task accuracy", RTASK["task_acc_three_levels"], 0.002, "0.796"),
        ("real task effect", RTASK["power"]["effect_rel_l2"], 0.002, "0.956"),
        ("real task O1 proj", RTASK["ops"]["O1_procrustes"]["proj_coef_agg"], 0.005, "0.07"),
        ("real task O2 proj", RTASK["ops"]["O2_ridge"]["proj_coef_agg"], 0.005, "0.79"),
        ("real task O5 proj", RTASK["ops"]["O5_mlp"]["proj_coef_agg"], 0.005, "0.86"),
        ("real task O8 proj", RTASK["ops"]["O8_conv_residual"]["proj_coef_agg"], 0.005, "0.84"),
        ("real task O6 proj", RTASK["ops"]["O6_random"]["proj_coef_agg"], 0.005, "0.75"),
        ("real task O6 top1", RTASK["ops"]["O6_random"]["top1_agree_real"], 0.005, "0.40"),
        ("per-sample O8 agg", PSP["operators"]["O8_conv_residual"]["proj_agg_sum"], 0.002, "0.812"),
        ("per-sample O8 median", PSP["operators"]["O8_conv_residual"]["proj_per_sample_median"], 0.002, "0.824"),
        ("per-sample O8 share<=0", PSP["operators"]["O8_conv_residual"]["proj_share_le_0"], 0.002, "0.01"),
        ("per-sample random agg", PSP["operators"]["O6_random"]["proj_agg_sum"], 0.002, "0.827"),
        ("per-sample random share<=0", PSP["operators"]["O6_random"]["proj_share_le_0"], 0.002, "0.30"),
        ("per-sample random cosine", PSP["operators"]["O6_random"]["cos_median"], 0.002, "0.28"),
        ("O7 lowrank+mlp", PSP["operators"]["O7_lowrank_mlp"]["proj_agg_sum"], 0.002, "0.716"),
        ("O9 lowrank+conv", PSP["operators"]["O9_lowrank_conv"]["proj_agg_sum"], 0.002, "0.785"),
        ("O9 shuffled", PSP["operators"]["O9_shuffled"]["proj_agg_sum"], 0.002, "1.257"),
        ("O10 wide capacity", PSP["operators"]["O10_wide_capacity"]["proj_agg_sum"], 0.002, "0.815"),
        ("hue regime raw err_real", HREG["regimes"]["raw"]["probe_err_real_deg"], 0.05, "12.0"),
        ("hue regime raw err_null", HREG["regimes"]["raw"]["probe_err_null_deg"], 0.05, "82.9"),
        ("hue regime raw transfer", HREG["regimes"]["raw"]["transfer"], 0.005, "1.02"),
        ("hue regime raw proj", HREG["regimes"]["raw"]["projection"], 0.005, "1.01"),
        ("hue regime raw win", HREG["regimes"]["raw"]["win_rate"], 0.005, "0.94"),
        ("hue regime aligned transfer", HREG["regimes"]["aligned"]["transfer"], 0.005, "1.06"),
        ("hue regime aligned win", HREG["regimes"]["aligned"]["win_rate"], 0.005, "0.91"),
        ("hue regime quadratic transfer", HREG["regimes"]["raw_quadratic"]["transfer"], 0.005, "1.02"),
        ("hue regime quadratic win", HREG["regimes"]["raw_quadratic"]["win_rate"], 0.005, "0.95"),
        # Table 9 (Section 6.4): consumer-induced metric vs Euclidean ridge, 1x1 family, three seeds
        ("task-weighted O2 proj", tw("O2_euclidean_ridge_1x1", "proj_coef_agg"), 0.002, "0.627 ± 0.031"),
        ("task-weighted O2 median", tw("O2_euclidean_ridge_1x1", "proj_per_sample_median"), 0.002, "0.659 ± 0.030"),
        ("task-weighted O2 cos", tw("O2_euclidean_ridge_1x1", "align_cos"), 0.002, "0.678 ± 0.016"),
        ("task-weighted O2 win", tw("O2_euclidean_ridge_1x1", "win_rate"), 0.002, "0.722 ± 0.006"),
        ("task-weighted O2 top1", tw("O2_euclidean_ridge_1x1", "top1_agree_real"), 0.002, "0.823 ± 0.041"),
        ("task-weighted O2 weighted err held", tw("O2_euclidean_ridge_1x1", "task_weighted_error_held"), 0.002, "2.064 ± 0.043"),
        ("task-weighted O2 feat rel l2", tw("O2_euclidean_ridge_1x1", "feature_rel_l2"), 0.002, "0.392 ± 0.011"),
        ("task-weighted W* proj", tw("W_task_weighted_1x1", "proj_coef_agg"), 0.002, "0.642 ± 0.017"),
        ("task-weighted W* median", tw("W_task_weighted_1x1", "proj_per_sample_median"), 0.002, "0.675 ± 0.024"),
        ("task-weighted W* cos", tw("W_task_weighted_1x1", "align_cos"), 0.002, "0.692 ± 0.018"),
        ("task-weighted W* win", tw("W_task_weighted_1x1", "win_rate"), 0.002, "0.743 ± 0.025"),
        ("task-weighted W* top1", tw("W_task_weighted_1x1", "top1_agree_real"), 0.002, "0.827 ± 0.038"),
        ("task-weighted W* weighted err held", tw("W_task_weighted_1x1", "task_weighted_error_held"), 0.002, "2.033 ± 0.038"),
        ("task-weighted W* feat rel l2", tw("W_task_weighted_1x1", "feature_rel_l2"), 0.002, "0.402 ± 0.012"),
        ("task-weighted W* weighted err train", tw_train("W_task_weighted_1x1", "task_weighted_error_train"), 0.002, "2.125 ± 0.021"),
        ("task-weighted O2 weighted err train", tw_train("O2_euclidean_ridge_1x1", "task_weighted_error_train"), 0.002, "2.163 ± 0.022"),
        ("task-weighted W* acc int", tw("W_task_weighted_1x1", "acc_int"), 0.002, "0.845 ± 0.026"),
        ("task-weighted O2 acc int", tw("O2_euclidean_ridge_1x1", "acc_int"), 0.002, "0.840 ± 0.010"),
        ("detector mAP50", m50, 0.001, "0.987"),
        ("detector mAP50-95", m5095, 0.001, "0.909"),
    ]
    out = {"draft": os.path.basename(DRAFT), "checks": []}
    n_pass = 0
    for label, val, tol, lit in checks:
        in_draft = lit in txt
        head = re.match(r"[+-]?\d+(?:\.\d+)?", lit.replace("\u2212", "-"))
        lit_num = float(head.group(0)) if head else None
        nd = len(head.group(0).split(".")[1]) if head and "." in head.group(0) else 0
        ok = in_draft and lit_num is not None and abs(round(float(val), nd) - lit_num) <= tol
        n_pass += int(ok)
        out["checks"].append({"label": label, "json": round(float(val), 4), "literal": lit,
                              "in_draft": in_draft, "pass": ok})
    out["n_pass"] = n_pass
    out["n_total"] = len(checks)
    out["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"
    json.dump(out, open(os.path.join(WORK, "results", "paperB_numbers_check.json"), "w"), indent=1)
    L = [f"# Paper B number audit: {n_pass}/{len(checks)} PASS", "",
         "| check | value from JSON | literal in draft | in draft | pass |", "|---|---|---|---|---|"]
    for c in out["checks"]:
        L.append(f"| {c['label']} | {c['json']} | {c['literal']} | {c['in_draft']} | {'OK' if c['pass'] else 'FAIL'} |")
    md = "\n".join(L)
    open(os.path.join(WORK, "results", "paperB_numbers_check.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
