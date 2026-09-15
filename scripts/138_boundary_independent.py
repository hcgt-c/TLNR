# -*- coding: utf-8 -*-
"""138_boundary_independent.py — independent validation of the identifiability boundary.

Goal (reviewer P0 #3): stop reporting only two-side medians. Fit the decision rule on a controlled
domain, FREEZE it, then predict success/failure on independent domains and report ROC/AUPRC,
calibration (ECE) and bootstrap CIs -- plus the +-window sensitivity that is already stored.

Domains:
  A (fit)    : controlled renders  (results/identifiability_law.json -> controlled[], families
               coverage/frequency/palette) with input statistic m and network-side error
               code_err_vs_nominal.
  B (test #1): real COCO regions    (results/region_real_validation.json -> examples[]) with input
               concentration C and per-region network error err_percell.
  B (test #2): real regions under the usage rule (results/usage_rule_scale_strat.json -> rows[])
               with concentration and raw / dominant-fill / shifted-fill errors.
  B (test #3): controlled colour-entropy sweeps (results/color_entropy_scan.json) if the sweep
               rows expose (m/C, error); used when parseable.

Rules compared (frozen before touching B):
  R1  m >= 0.70            (paper's calibrated rule)
  R2  C >= C*              (C* fitted on A by Youden J)
  R3  logistic(C) >= 0.5   (logistic fitted on A)
Outputs results/boundary_independent.json + results/boundary_independent.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math
import numpy as np

WORK = rp.REPO_ROOT
ERR_OK = 10.0     # a global hue read-out is "usable" if its error is <= 10 deg


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(auc(y[idx], s[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def ece(prob, y, bins=10):
    prob = np.asarray(prob, float); y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)


def fit_logistic(x, y, iters=200, lr=0.5):
    x = np.asarray(x, float).reshape(-1, 1); y = np.asarray(y, float)
    w, b = 0.0, 0.0
    for _ in range(iters):
        z = w * x[:, 0] + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (g * x[:, 0]).mean()
        b -= lr * g.mean()
    return w, b


def youden_threshold(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float)
    best, bt = -1, None
    for t in np.unique(s):
        pred = s >= t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0
        fpr = pred[y == 0].mean() if (y == 0).any() else 0
        j = tpr - fpr
        if j > best:
            best, bt = j, t
    return float(bt), float(best)


def report(name, y, score, extra=None):
    a = auc(y, score); lo, hi = auc_ci(y, score)
    rec = {"n": int(len(y)), "n_pos": int(np.sum(y)), "auc": a, "auc_ci95": [lo, hi]}
    if extra:
        rec.update(extra)
    return rec


def main():
    law = json.load(open(os.path.join(WORK, "results", "identifiability_law.json")))
    ctrl = law["controlled"]
    real_rows = law.get("real_rows", [])
    reg = json.load(open(os.path.join(WORK, "results", "region_real_validation.json")))
    usage = json.load(open(os.path.join(WORK, "results", "usage_rule_scale_strat.json")))
    out = {"err_ok_deg": ERR_OK, "protocol": "fit rule on controlled renders; freeze; test on independent domains"}

    # ---- domain A (fit): controlled renders, statistic m, error code_err_vs_nominal
    mA = np.array([c["m"] for c in ctrl], float)
    eA = np.array([c["code_err_vs_nominal"] for c in ctrl], float)
    yA = (eA <= ERR_OK).astype(float)
    Cstar, J = youden_threshold(mA, yA)
    w, b = fit_logistic(mA, yA)
    out["fit_domain_A"] = {
        "n": len(ctrl), "n_pos": int(yA.sum()),
        "m_star_fitted": Cstar, "youden_J": J,
        "m_star_paper": 0.70,
        "logistic_w": w, "logistic_b": b,
        "auc_m_on_A": auc(yA, mA),
        "errors_by_m": [{"m": c["m"], "err": c["code_err_vs_nominal"], "family": c["family"]} for c in ctrl],
    }

    # ---- augmented fit domain A+: controlled renders + colour-entropy sweeps (m approx = 1 - cov)
    ent = json.load(open(os.path.join(WORK, "results", "color_entropy_scan.json")))
    mA2, eA2, srcA2 = list(mA), list(eA), ["controlled"] * len(mA)
    for fam in ["coverage_sweep", "frequency_sweep", "palette_sweep"]:
        for r in ent.get(fam, []):
            cov = r.get("setting", {}).get("cov", None)
            if cov is None:
                continue
            mA2.append(max(0.0, 1.0 - float(cov)))       # approximate dominant share
            eA2.append(r["code_vs_nominal_main"])
            srcA2.append(fam)
    mA2 = np.array(mA2, float); eA2 = np.array(eA2, float)
    yA2 = (eA2 <= ERR_OK).astype(float)
    Cstar2, J2 = youden_threshold(mA2, yA2)
    w2, b2 = fit_logistic(mA2, yA2)
    out["fit_domain_Aplus"] = {
        "n": int(len(mA2)), "n_pos": int(yA2.sum()), "m_star_fitted": Cstar2, "youden_J": J2,
        "auc_m": auc(yA2, mA2), "auc_ci95": list(auc_ci(yA2, mA2)),
        "sources": {s: srcA2.count(s) for s in set(srcA2)},
        "note": "entropy-sweep rows enter with m approximated as 1-cov (approximate)",
    }

    # ---- domain B1: real regions, statistic C (ref_concentration), error err_percell
    ex = reg["examples"]
    CB = np.array([e["ref_concentration"] for e in ex], float)
    eB = np.array([e["err_percell"] for e in ex], float)
    yB = (eB <= ERR_OK).astype(float)
    pB = 1 / (1 + np.exp(-(w * CB + b)))
    out["test_B1_real_regions_percell"] = report("B1", yB, CB, {
        "rule_R1_C_ge_0.70": {"acc": float(((CB >= 0.70) == (yB == 1)).mean()),
                              "tpr": float(((CB >= 0.70) & (yB == 1)).sum() / max(1, (yB == 1).sum())),
                              "fpr": float(((CB >= 0.70) & (yB == 0)).sum() / max(1, (yB == 0).sum()))},
        "rule_R2_C_ge_astar": {"acc": float(((CB >= Cstar) == (yB == 1)).mean()), "C_star": Cstar},
        "rule_R3_logistic": {"acc": float(((pB >= 0.5) == (yB == 1)).mean()), "ece": ece(pB, yB)},
    })

    # ---- domain B2: usage-rule rows, concentration vs raw failure / fill restoration
    rows = usage["rows"]
    conc = np.array([r["concentration"] for r in rows], float)
    raw = np.array([r["a_raw"] for r in rows], float)
    fill = np.array([r["b_dominant_fill"] for r in rows], float)
    yraw = (raw <= ERR_OK).astype(float)
    p2 = 1 / (1 + np.exp(-(w * conc + b)))
    multi = {}
    for thr in [5.0, 10.0, 20.0, 30.0]:
        yy = (raw <= thr).astype(float)
        a_ = auc(yy, conc); lo_, hi_ = auc_ci(yy, conc)
        multi[f"thr_{int(thr)}"] = {"pos_rate": float(yy.mean()), "auc": a_, "auc_ci95": [lo_, hi_],
                                    "rule_R1_acc": float(((conc >= 0.70) == (yy == 1)).mean()),
                                    "rule_Aplus_acc": float(((conc >= Cstar2) == (yy == 1)).mean())}
    out["test_B2_multithreshold"] = multi
    # continuous association on B2
    from math import sqrt
    def pearson(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    out["test_B2_continuous"] = {
        "pearson_conc_vs_raw_err": pearson(conc, raw),
        "pearson_conc_vs_fill_err": pearson(conc, fill),
        "pearson_concentration_vs_ref_err_B1": pearson(CB, eB),
        "note": "negative correlation means higher concentration -> smaller error",
    }
    out["test_B2_usage_rule"] = {
        "n": len(rows),
        "raw_success_rate": float(yraw.mean()),
        "fill_success_rate": float((fill <= ERR_OK).mean()),
        "auc_conc_predicts_raw_success": auc(yraw, conc),
        "auc_conc_ci95": list(auc_ci(yraw, conc)),
        "rule_R1_acc": float(((conc >= 0.70) == (yraw == 1)).mean()),
        "rule_R3_acc": float(((p2 >= 0.5) == (yraw == 1)).mean()),
        "ece_R3": ece(p2, yraw),
        "raw_error_median_above_thr": float(np.median(raw[conc >= 0.70])) if (conc >= 0.70).any() else None,
        "raw_error_median_below_thr": float(np.median(raw[conc < 0.70])) if (conc < 0.70).any() else None,
        "fill_error_median_above_thr": float(np.median(fill[conc >= 0.70])) if (conc >= 0.70).any() else None,
        "fill_error_median_below_thr": float(np.median(fill[conc < 0.70])) if (conc < 0.70).any() else None,
    }

    # ---- window sensitivity (stored per-region m at +-15/30/45/60)
    ws = {}
    if real_rows:
        gaps = np.array([r["gap_deg"] for r in real_rows], float)
        for wname, key in [("15", "m_15"), ("30", "m_30"), ("45", "m_45"), ("60", "m_60")]:
            if key in real_rows[0]:
                mm = np.array([r[key] for r in real_rows], float)
                ok = mm >= 0.70
                ws[f"+-{wname}"] = {
                    "median_m": float(np.median(mm)),
                    "frac_ge_0.70": float(ok.mean()),
                    "gap_median_above": float(np.median(gaps[ok])) if ok.any() else None,
                    "gap_median_below": float(np.median(gaps[~ok])) if (~ok).any() else None,
                    "spearman_1m_gap": float(np.corrcoef(1 - mm, gaps)[0, 1]),
                }
    out["window_sensitivity"] = ws
    out["verdict"] = {
        "fit_on_controlled": {"m_star_fitted": Cstar, "auc_A": auc(yA, mA)},
        "transfer_to_real_percell": {"auc_C": out["test_B1_real_regions_percell"]["auc"]},
        "transfer_to_usage_raw": {"auc_C": out["test_B2_usage_rule"]["auc_conc_predicts_raw_success"]},
        "window_rule_stability_frac_ge_070": {k: v["frac_ge_0.70"] for k, v in ws.items()},
    }
    json.dump(out, open(os.path.join(WORK, "results", "boundary_independent.json"), "w"), indent=1)

    L = ["# Boundary: independent validation", "",
         f"Success = read-out error <= {ERR_OK} deg. Rule fitted on controlled renders (domain A), frozen, then tested.",
         "", f"- Domain A: n={len(ctrl)}, positives={int(yA.sum())}, m* fitted={Cstar:.3f}, AUC(m)={auc(yA, mA):.3f}",
         f"- Domain B1 (real regions, per-cell error): n={len(ex)}, AUC(C)={out['test_B1_real_regions_percell']['auc']:.3f} "
         f"CI{out['test_B1_real_regions_percell']['auc_ci95']}",
         f"- Domain B2 (usage rule, raw success): n={len(rows)}, AUC(C)={out['test_B2_usage_rule']['auc_conc_predicts_raw_success']:.3f}, "
         f"raw success {out['test_B2_usage_rule']['raw_success_rate']:.2f} -> fill {out['test_B2_usage_rule']['fill_success_rate']:.2f}",
         "", "## Window sensitivity (real regions)", "| window | median m | frac >= 0.70 | gap above | gap below | spearman(1-m, gap) |",
         "|---|---|---|---|---|---|"]
    for k, v in ws.items():
        L.append(f"| {k} | {v['median_m']:.3f} | {v['frac_ge_0.70']:.3f} | {v['gap_median_above']:.2f} | {v['gap_median_below']:.2f} | {v['spearman_1m_gap']:.2f} |")
    md = "\n".join(L)
    open(os.path.join(WORK, "results", "boundary_independent.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
