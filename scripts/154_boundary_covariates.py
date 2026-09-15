# -*- coding: utf-8 -*-
"""154_boundary_covariates.py — do mask/texture covariates add predictive power to the boundary?

Targets (real regions under the usage rule, results/usage_rule_scale_strat.json, n=120):
  (T1) raw read-out success   : a_raw <= 10/20/30 deg
  (T2) restored read-out      : d_shifted_fill_grabcut <= 10 deg  (GrabCut mask front end)
Features, increasingly rich:
  F1 concentration            (the paper's input statistic)
  F2 concentration + mask_frac_gc            (mask quality)
  F3 F2 + category one-hot                   (texture/appearance proxy)
Reported with 5-fold stratified CV: AUC for classification, R^2 for the continuous target, plus
bootstrap 95% CI of the CV AUC and the incremental gain of F2/F3 over F1. Also a regression of the
continuous error on the same features (OLS, R^2) to avoid threshold arbitrariness.
Outputs results/boundary_covariates.json + .md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

WORK = rp.REPO_ROOT
D = json.load(open(os.path.join(WORK, "results", "usage_rule_scale_strat.json")))
ROWS = D["rows"]


def boot_auc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(np.asarray(y)[idx])) < 2:
            continue
        vals.append(roc_auc_score(np.asarray(y)[idx], np.asarray(p)[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def design(feats, cats):
    X = np.column_stack([np.asarray(f, float) for f in feats])
    if cats:
        uniq = sorted(set(cats))
        for u in uniq[:-1]:
            col = np.array([1.0 if c == u else 0.0 for c in cats])
            X = np.column_stack([X, col])
    return X.astype(float)


def cv_auc(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    prob = cross_val_predict(clf, StandardScaler().fit_transform(X), y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob)), prob


def cv_r2(X, y, seed=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(LinearRegression(), StandardScaler().fit_transform(X), y, cv=kf)
    return float(r2_score(y, pred)), pred


def main():
    conc = np.array([r["concentration"] for r in ROWS], float)
    mask = np.array([r["mask_frac_gc"] for r in ROWS], float)
    cats = [r["category"] for r in ROWS]
    a_raw = np.array([r["a_raw"] for r in ROWS], float)
    d_gc = np.array([r["d_shifted_fill_grabcut"] for r in ROWS], float)

    feature_sets = {
        "F1_concentration": ([conc], False),
        "F2_+mask": ([conc, mask], False),
        "F3_+category": ([conc, mask], True),
    }
    out = {"n": len(ROWS), "targets": {}, "note": "5-fold CV; AUC for success classification, R^2 for continuous error"}

    for tname, target in [("raw_success", a_raw), ("restored_success", d_gc)]:
        rec = {"classification": {}, "regression": {}}
        for thr in (10.0, 20.0, 30.0):
            y = (target <= thr).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                rec["classification"][f"thr_{int(thr)}"] = {"pos_rate": float(y.mean()), "auc": None}
                continue
            entry = {"pos_rate": float(y.mean())}
            for fname, (feats, use_cats) in feature_sets.items():
                X = design(feats, cats if use_cats else None)
                auc, prob = cv_auc(X, y)
                lo, hi = boot_auc_ci(y, prob)
                entry[fname] = {"auc": auc, "auc_ci95": [lo, hi]}
            entry["gain_F2_minus_F1"] = entry["F2_+mask"]["auc"] - entry["F1_concentration"]["auc"]
            entry["gain_F3_minus_F1"] = entry["F3_+category"]["auc"] - entry["F1_concentration"]["auc"]
            rec["classification"][f"thr_{int(thr)}"] = entry
        for fname, (feats, use_cats) in feature_sets.items():
            X = design(feats, cats if use_cats else None)
            r2, _ = cv_r2(X, target)
            rec["regression"][fname] = {"cv_r2": r2}
        rec["regression"]["gain_F2_minus_F1"] = rec["regression"]["F2_+mask"]["cv_r2"] - rec["regression"]["F1_concentration"]["cv_r2"]
        rec["regression"]["gain_F3_minus_F1"] = rec["regression"]["F3_+category"]["cv_r2"] - rec["regression"]["F1_concentration"]["cv_r2"]
        out["targets"][tname] = rec

    json.dump(out, open(os.path.join(WORK, "results", "boundary_covariates.json"), "w"), indent=1)
    L = ["# Boundary covariates: does mask/texture information add predictive power?", "",
         f"n = {out['n']} real COCO regions under the usage rule. Success = error <= threshold.",
         "", "## Raw read-out success (a_raw)"]
    for thr, e in out["targets"]["raw_success"]["classification"].items():
        if "F1_concentration" not in e:
            L.append(f"- {thr}: degenerate (pos_rate {e['pos_rate']:.2f})"); continue
        L.append(f"- {thr}: AUC F1 {e['F1_concentration']['auc']:.3f} -> F2 {e['F2_+mask']['auc']:.3f} "
                 f"(gain {e['gain_F2_minus_F1']:+.3f}) -> F3 {e['F3_+category']['auc']:.3f} "
                 f"(gain {e['gain_F3_minus_F1']:+.3f})")
    L.append("")
    L.append("## Restored read-out success (GrabCut mask, d_shifted_fill_grabcut)")
    for thr, e in out["targets"]["restored_success"]["classification"].items():
        if "F1_concentration" not in e:
            L.append(f"- {thr}: degenerate (pos_rate {e['pos_rate']:.2f})"); continue
        L.append(f"- {thr}: AUC F1 {e['F1_concentration']['auc']:.3f} -> F2 {e['F2_+mask']['auc']:.3f} "
                 f"(gain {e['gain_F2_minus_F1']:+.3f}) -> F3 {e['F3_+category']['auc']:.3f} "
                 f"(gain {e['gain_F3_minus_F1']:+.3f})")
    L.append("")
    L.append("## Continuous targets (5-fold CV R^2)")
    for tname in out["targets"]:
        r = out["targets"][tname]["regression"]
        L.append(f"- {tname}: F1 {r['F1_concentration']['cv_r2']:.3f} -> F2 {r['F2_+mask']['cv_r2']:.3f} "
                 f"(gain {r['gain_F2_minus_F1']:+.3f}) -> F3 {r['F3_+category']['cv_r2']:.3f} "
                 f"(gain {r['gain_F3_minus_F1']:+.3f})")
    md = "\n".join(L)
    open(os.path.join(WORK, "results", "boundary_covariates.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
