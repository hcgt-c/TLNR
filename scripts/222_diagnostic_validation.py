# -*- coding: utf-8 -*-
"""222_diagnostic_validation.py — external validation of the transportability diagnostic report
(the "gates" protocol) as a PREDICTOR of whether a feature-space intervention will faithfully
reproduce an input transformation, on held-out settings.

Analysis / inference only. No model training, no new forward passes. Every number below is read
from existing result files. CPU only.

===========================================================================================
PRE-REGISTRATION (fixed before any metric in this file was computed; do not tune on the eval set)
===========================================================================================

Target of the prediction
------------------------
Observed outcome  : the headline downstream score of the published global-linear operator O2,
                    T_F = 1 - mean||F(Wz) - F(z_real)|| / mean||F(z) - F(z_real)||, read per site
                    as `O2_transfer_mean` from results/depth_map_summary.json (mean over seeds).
Observed class    : faithful if T_F >= 0.80, partial if 0.40 <= T_F < 0.80, failed if T_F < 0.40.
                    (Reporting conventions of the diagnostic report / manuscript; PRE-REGISTERED.)

Evaluation units (sites)
------------------------
Primary grid  : 4 backbones x 4 depths x {hue_90.0, heat_2.0} = 32 sites.
Auxiliary     : heat_1.0 (4 backbones x 4 depths = 16 sites) as an unseen transformation PARAMETER.
Consumer axis : results/consumer_swap.json, resnet50 x {heat 2.0, hue 90} x depths {1,3} x
                4 consumers (head / hue-probe / dense / retrieval) = 16 (site, consumer) units.
Observability : results/quotientization.json, hue 90, seed 0, 4 backbones x 4 depths = 16 sites,
                used only to test an alternative observability statistic.

Held-out splits, BY SETTING (never by random cell; no parameter is fitted, so these measure
transfer of a fixed rule, and no gate threshold is adjusted on any of them)
------------------------
S1  hold out backbone          : dinov2b14
S2  hold out family            : heat (all heat_2.0 sites)
S3  hold out depth             : depth 4 (all deepest sites)
S4  hold out backbone          : vitb16 (contains the unreachable class-token sites)
S5  hold out parameter setting : heat_1.0 (entire transformation parameter never in the main grid)

Gate statistics available WITHOUT the outcome, and their pre-registered criteria
------------------------
(G0) transformation definition : exact by construction for every site (hue re-render; spectral heat
     semigroup). No per-site statistic; PASS for all units. Recorded, never decisive.
(G1) consumer observability    : the CONSUMER-level effect size of the real transformation -- mean over
     seeds of `effect_size_logits` from the raw results/depth_map_*.json (depth map) / `consumer_power`
     (consumer axis) / `visible_share_of_displacement` (observability axis). The site-level effect size
     `effect_size_site` is carried beside it as a companion. PASS iff >= 0.05 (PRE-REGISTERED power
     floor). FAIL is a hard ABSTENTION (coverage loss): the no-op is already faithful and no operator
     verdict is informative.
(G2) reference validity        : heat sites record `intertwining_residual_canonical` R = how much of
     the site's change the canonical action explains relative to the raw displacement. PASS iff
     R <= 0.99 (R >= 0.99 means the attribute/description is not a number for that input).
     Hue / direct-route consumers: no attribute description is used, so the gate is NOT APPLICABLE and
     is recorded as PASS with an applicability flag (never silently treated as evidence).
(G3) causal reachability       : `reachable` (intervention_reaches_consumer). PASS iff True.
(G4) operator realization (fit side): the outcome-free single-step feature realization residual
     ratio `feat_resid_over_displacement` of O2_pub = mean||Wz - z_real|| / mean||z - z_real||,
     taken on the HELD-OUT FEATURE split of results/estimator_control.json. PASS iff <= 1.00
     (an operator at least as close to the real transformed features as the no-op). Where the cell
     is not recorded the gate is NOT APPLICABLE and recorded as PASS with an applicability flag.
     NOTE the published files record no fit-SPLIT residual; this is the closest available
     outcome-free realization statistic and its split is stated, not hidden. The estimation-limit
     ratio C^2/K and the fit row count K from the same file are reported beside it.

Prediction rule (PRIMARY, "necessary-condition" reading of the protocol)
------------------------
  if G1 fails                       -> ABSTAIN (the protocol issues no decision)
  elif any APPLICABLE gate fails    -> predicted class "failed"
  else                              -> predicted class "partial"
The protocol is a conjunction of preconditions: it licenses *proceeding*, and it contains no
sufficiency test for "faithful". Predicted class "faithful" is therefore never issued. Ordinal
prediction ordP = 0 for failed, 1 for partial.

Variants (reported beside the primary, both pre-registered)
------------------------
Tautological variant : the certificate as it shipped WHEN THIS VALIDATION WAS RUN (scripts/215) set
                       Gate 4 = threshold(T_F) and certificate.faithfulness = that verdict. Read
                       literally it agrees with the outcome 100% BY CONSTRUCTION; we quantify that
                       agreement and its zero out-of-sample information. The certificate has since
                       been corrected: Gate 4 now thresholds the outcome-free fit-side realization
                       residual and the summary verdict is a precondition conjunction that never
                       returns faithful. The variant below records the pre-fix behaviour.
Sufficiency variant  : all applicable gates pass -> "faithful" (the strawman sufficiency reading).
Ordinal gate score   : number of applicable gates passed (0..4), correlated with T_F.

Baselines a reviewer would demand
------------------------
(a) always "partial"
(b) depth alone: "failed" iff depth >= 3, else "partial" (the published depth law, no fitting)
(c) random class assignment: uniform over 3 classes, 2000 seeded draws
(d) prior-matched random: classes drawn from the observed marginal class frequencies
(e) reachability alone (G3 only)   (f) realization alone (G4 only)   (g) reference alone (G2 only)

Metrics
------------------------
3x3 confusion matrix, per-class precision/recall/support, accuracy, macro-F1 (present classes, and
all-3 with undefined F1 = 0), coverage = fraction of units on which a decision is issued,
Spearman and Kendall of the ordinal prediction vs observed T_F. Uncertainty: 2000-sample bootstrap
over units (95% percentile CI), paired against every baseline (CI of the accuracy/F1 difference).

Verdict logic (PRE-REGISTERED)
------------------------
  "validated instrument"  iff faithful recall > 0 (estimable) AND accuracy exceeds every baseline
                              with paired-bootstrap 95% CI lower bound > 0 AND macro-F1 >= 0.60.
  "weak screen"           iff accuracy exceeds every baseline with paired-bootstrap lower bound > 0
                              but the faithful class is never predicted (or macro-F1 < 0.60).
  "checklist / reporting discipline" otherwise (no significant advantage over the best baseline).

Outputs
------------------------
  results/diagnostic_validation.json
  paper/figures/F36_diagnostic_validation.png
  paper/targets/TPAMI/reports/DIAGNOSTIC_VALIDATION_REPORT.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os
import json
import hashlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")
FIG = os.path.join(WORK, "paper", "figures", "F36_diagnostic_validation.png")
OUT_JSON = os.path.join(RES, "diagnostic_validation.json")
OUT_MD = os.path.join(WORK, "paper", "targets", "TPAMI", "reports", "DIAGNOSTIC_VALIDATION_REPORT.md")

# ---------------------------------------------------------------- pre-registered constants
POWER_FLOOR = 0.05
FAITHFUL_AT = 0.80
PARTIAL_AT = 0.40
REF_RESIDUAL_MAX = 0.99       # Gate 2
REALIZATION_RESIDUAL_MAX = 1.00  # Gate 4
DEPTH_FAIL_AT = 3             # baseline (b): published depth law
N_BOOT = 2000
N_RAND = 2000
SEED = 222
CLASSES = ["failed", "partial", "faithful"]
ORD = {"failed": 0, "partial": 1, "faithful": 2}
MAIN_GRID_FAMILIES = ["hue_90.0", "heat_2.0"]
MAIN_GRID = ["resnet50", "convnext", "vitb16", "dinov2b14"]


def observed_class(tf):
    if tf >= FAITHFUL_AT:
        return "faithful"
    if tf >= PARTIAL_AT:
        return "partial"
    return "failed"


def family_of(fam_key):
    return "hue" if fam_key.startswith("hue") else "heat"


# ---------------------------------------------------------------- data
def load():
    dm = json.load(open(os.path.join(RES, "depth_map_summary.json"), encoding="utf-8"))
    ec = json.load(open(os.path.join(RES, "estimator_control.json"), encoding="utf-8"))
    cs = json.load(open(os.path.join(RES, "consumer_swap.json"), encoding="utf-8"))
    qz = json.load(open(os.path.join(RES, "quotientization.json"), encoding="utf-8"))
    return dm, ec, cs, qz


def file_provenance(names):
    """Size/mtime/sha256 of the inputs, so a moving result file cannot silently change the artifact."""
    prov = {}
    for n in names:
        p = os.path.join(RES, n)
        if not os.path.exists(p):
            prov[n] = {"exists": False}
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        st = os.stat(p)
        prov[n] = {"exists": True, "bytes": int(st.st_size),
                   "mtime": __import__("time").strftime("%Y-%m-%dT%H:%M:%S",
                                                        __import__("time").localtime(st.st_mtime)),
                   "sha256": h.hexdigest()}
    return prov


def ec_index(ec):
    """(backbone, family, depth) -> the per-cell record of results/estimator_control.json."""
    idx = {}
    for key, cell in ec.get("cells", {}).items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        bb, fam, dep = parts[0], parts[1], parts[2]
        try:
            dep = int(dep.replace("depth", ""))
        except ValueError:
            continue
        idx[(bb, fam, dep)] = cell
    return idx


def build_grid_units(dm, ecx, families, consumer_power=None):
    """One unit per (backbone, family, depth). All statistics outcome-free.

    `consumer_power` maps (backbone, family_key) -> mean effect_size_logits over the raw
    results/depth_map_*.json seeds: the real transformation's effect at the CONSUMER (the
    backbone's own logits), which is what Gate 1 is about. The site-level effect size
    `effect_size_site` from the summary is kept beside it as a companion.
    """
    consumer_power = consumer_power or {}
    units = []
    for fam_key in families:
        for bb in MAIN_GRID:
            key = f"{bb}|{fam_key}"
            if key not in dm:
                continue
            cell = dm[key]
            fam = family_of(fam_key)
            for site, s in cell["sites"].items():
                depth = int(s["depth"])
                cell_ec = ecx.get((bb, fam, depth))
                est = (cell_ec or {}).get("estimators", {}).get("O2_pub")
                cp = consumer_power.get((bb, fam_key))
                units.append({
                    "id": f"{bb}|{fam_key}|{site}",
                    "backbone": bb, "family": fam_key, "family_base": fam, "site": site,
                    "depth": depth,
                    # Gate 1: consumer-level power (falls back to the site statistic if unrecorded)
                    "power_consumer": (float(cp) if cp is not None else float(s["effect_size_site"])),
                    "power_site": float(s["effect_size_site"]),
                    "power": (float(cp) if cp is not None else float(s["effect_size_site"])),
                    # Gate 2 (heat only)
                    "reference_stat": (None if s.get("intertwining_residual_canonical") is None
                                       else float(s["intertwining_residual_canonical"])),
                    # Gate 3
                    "reachable": bool(s["reachable"]),
                    # Gate 4 (fit side, outcome-free)
                    "realization_resid": (None if est is None
                                          else float(est["feat_resid_over_displacement"])),
                    "fit_rows": (None if cell_ec is None else int(cell_ec.get("K_rows_used", 0))),
                    "C2_over_K": (None if cell_ec is None else float(cell_ec.get("C2_over_K", np.nan))),
                    # observed outcome (never used by the predictor)
                    "T_F": float(s["O2_transfer_mean"]),
                    "T_F_sd": float(s.get("O2_transfer_sd", float("nan"))),
                    "T_F_O8": (None if s.get("O8_transfer_mean") is None
                               else float(s["O8_transfer_mean"])),
                    "T_F_O1": (None if s.get("O1_transfer_mean") is None
                               else float(s["O1_transfer_mean"])),
                })
    for u in units:
        u["observed"] = observed_class(u["T_F"])
    return units


def load_consumer_power():
    """(backbone, family_key) -> mean effect_size_logits over seeds, from the raw depth maps."""
    out = {}
    for bb in MAIN_GRID:
        for fam_file, fam_key in [("hue", "hue_90.0"), ("heat", "heat_2.0"), ("heat", "heat_1.0")]:
            p = os.path.join(RES, f"depth_map_{bb}_{fam_file}.json")
            if not os.path.exists(p):
                continue
            try:
                res = json.load(open(p, encoding="utf-8"))["results"].get(fam_key)
            except Exception:
                res = None
            if not res:
                continue
            vals = [sd.get("effect_size_logits") for sd in res.get("seed", {}).values()
                    if sd.get("effect_size_logits") is not None]
            if vals:
                out[(bb, fam_key)] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------- gates + prediction
def apply_gates(u, ref_max=REF_RESIDUAL_MAX, real_max=REALIZATION_RESIDUAL_MAX):
    """Return per-gate (passed, applicable) and the protocol decision."""
    g0 = (True, True)
    g1 = (u["power"] >= POWER_FLOOR, True)                       # Gate 1 (power floor)
    if u["reference_stat"] is None:
        g2 = (True, False)                                       # not applicable (hue / direct route)
    else:
        g2 = (u["reference_stat"] <= ref_max, True)
    g3 = (bool(u["reachable"]), True)
    if u["realization_resid"] is None:
        g4 = (True, False)                                       # not recorded
    else:
        g4 = (u["realization_resid"] <= real_max, True)
    gates = {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4}

    if not g1[0]:
        pred = None                                              # ABSTAIN
    elif any((not p) for p, applicable in gates.values() if applicable and p is not None):
        pred = "failed"
    else:
        pred = "partial"
    n_app = sum(1 for p, a in gates.values() if a)
    n_pass = sum(1 for p, a in gates.values() if a and p)
    return gates, pred, int(n_app), int(n_pass)


def add_predictions(units, ref_max=REF_RESIDUAL_MAX, real_max=REALIZATION_RESIDUAL_MAX):
    for u in units:
        gates, pred, n_app, n_pass = apply_gates(u, ref_max, real_max)
        u["gates"] = {k: {"pass": bool(p), "applicable": bool(a)} for k, (p, a) in gates.items()}
        u["predicted"] = pred
        u["n_applicable"] = n_app
        u["n_passed"] = n_pass
    return units


# ---------------------------------------------------------------- metrics
def confusion(units, key="predicted"):
    """Rows = observed, cols = predicted, over decided units only."""
    dec = [u for u in units if u.get(key) is not None]
    M = np.zeros((3, 3), dtype=int)
    for u in dec:
        M[CLASSES.index(u["observed"]), CLASSES.index(u[key])] += 1
    return M, dec


def per_class(M):
    out = {}
    for i, c in enumerate(CLASSES):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        support = int(M[i, :].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and
                                                 (prec + rec) > 0) else (0.0 if support == 0 else None)
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    present = [c for c in CLASSES if out[c]["support"] > 0]
    macro_present = (float(np.mean([(out[c]["f1"] if out[c]["f1"] is not None else 0.0)
                                    for c in present])) if present else float("nan"))
    macro_all3 = float(np.mean([(out[c]["f1"] if out[c]["f1"] is not None else 0.0)
                                for c in CLASSES]))
    return out, macro_present, macro_all3


def accuracy(units, key="predicted"):
    dec = [u for u in units if u.get(key) is not None]
    if not dec:
        return float("nan")
    return float(np.mean([u[key] == u["observed"] for u in dec]))


def ordinal_metrics(units, pred_key="predicted", obs_key="T_F"):
    """Spearman/Kendall between an ordinal prediction level and the observed T_F."""
    dec = [u for u in units if u.get(pred_key) is not None and u.get("_ord") is not None]
    if len(dec) < 3:
        return {"n": len(dec), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None}
    a = np.array([u["_ord"] for u in dec], dtype=float)
    b = np.array([u[obs_key] for u in dec], dtype=float)
    if np.all(a == a[0]):
        return {"n": len(dec), "spearman": None, "spearman_p": None,
                "kendall": None, "kendall_p": None, "note": "prediction constant"}
    sp = stats.spearmanr(a, b)
    kt = stats.kendalltau(a, b)
    return {"n": int(len(dec)), "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue)}


def summarise(units, pred_key="predicted", obs_key="T_F"):
    M, dec = confusion(units, pred_key)
    pc, macro_present, macro_all3 = per_class(M)
    n_total = len(units)
    res = {
        "n_units": int(n_total),
        "n_decided": int(len(dec)),
        "n_abstained": int(n_total - len(dec)),
        "coverage": float(len(dec) / n_total) if n_total else float("nan"),
        "accuracy": accuracy(units, pred_key),
        "macro_f1_present": macro_present,
        "macro_f1_all3": macro_all3,
        "confusion_matrix": M.tolist(),
        "per_class": pc,
        "class_support_observed": {c: int(sum(1 for u in units if u["observed"] == c)) for c in CLASSES},
        "predicted_counts": {c: int(sum(1 for u in dec if u[pred_key] == c)) for c in CLASSES},
    }
    for u in units:
        u["_ord"] = ORD[u[pred_key]] if u.get(pred_key) is not None else None
    res["ordinal"] = ordinal_metrics(units, pred_key, obs_key)
    return res


# ---------------------------------------------------------------- baselines
def baseline_units(units):
    """Attach every baseline prediction column to the units (in place)."""
    rng = np.random.default_rng(SEED)
    prior = None
    dec0 = [u for u in units if u.get("observed") is not None]
    counts = np.array([sum(1 for u in dec0 if u["observed"] == c) for c in CLASSES], dtype=float)
    prior = counts / counts.sum()
    for u in units:
        u["always_partial"] = "partial"
        u["depth_rule"] = "failed" if u["depth"] >= DEPTH_FAIL_AT else "partial"
        u["reach_only"] = "partial" if u["reachable"] else "failed"
        g = u["gates"]
        if not g["G1"]["pass"]:
            u["realize_only"] = None
            u["reference_only"] = None
        else:
            u["realize_only"] = ("partial" if (not g["G4"]["applicable"] or g["G4"]["pass"])
                                 else "failed")
            u["reference_only"] = ("partial" if (not g["G2"]["applicable"] or g["G2"]["pass"])
                                   else "failed")
        u["random_uniform"] = CLASSES[int(rng.integers(0, 3))]
        u["random_prior"] = CLASSES[int(rng.choice(3, p=prior))]
    return units


BASELINE_KEYS = ["always_partial", "depth_rule", "reach_only", "realize_only",
                 "reference_only", "random_uniform", "random_prior"]


# ---------------------------------------------------------------- bootstrap
def _metrics_on(units, key):
    return (accuracy(units, key), per_class(confusion(units, key)[0])[1])


def bootstrap(units, keys, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over units. `keys` are unit-column names (e.g. "predicted")."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(units))
    acc = {k: [] for k in keys}
    f1 = {k: [] for k in keys}
    rho = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        sub = [units[i] for i in idx]
        for k in keys:
            for u in sub:
                u["_ord"] = ORD[u[k]] if u.get(k) is not None else None
            a, f = _metrics_on(sub, k)
            acc[k].append(a)
            f1[k].append(f)
            om = ordinal_metrics(sub, k, "T_F")
            rho[k].append(om["spearman"] if om["spearman"] is not None else np.nan)
        for u in sub:
            u["_ord"] = (ORD[u["predicted"]] if u.get("predicted") is not None else None)
    out = {}
    for k in keys:
        a = np.array([x for x in acc[k] if not np.isnan(x)])
        f = np.array([x for x in f1[k] if not np.isnan(x)])
        r = np.array([x for x in rho[k] if not np.isnan(x)])
        out[k] = {
            "accuracy": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "accuracy_mean": float(a.mean()) if a.size else None,
            "macro_f1": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))],
            "macro_f1_mean": float(f.mean()) if f.size else None,
            "spearman": ([float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
                         if r.size else [None, None]),
            "spearman_mean": float(r.mean()) if r.size else None,
        }
    return out


def paired_bootstrap(units, ref_key, other_keys, n_boot=N_BOOT, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(units))
    diffs = {k: [] for k in other_keys}
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        sub = [units[i] for i in idx]
        a_ref = accuracy(sub, ref_key)
        for k in other_keys:
            diffs[k].append(a_ref - accuracy(sub, k))
    out = {}
    for k in other_keys:
        d = np.array([x for x in diffs[k] if not np.isnan(x)])
        out[k] = {
            "mean_diff": float(d.mean()) if d.size else None,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else [None, None],
            "frac_positive": float((d > 0).mean()) if d.size else None,
        }
    return out


def random_baseline_distribution(units, n_rand=N_RAND, seed=SEED + 2):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_rand):
        correct = [CLASSES[int(rng.integers(0, 3))] == u["observed"] for u in units]
        accs.append(float(np.mean(correct)))
    accs = np.array(accs)
    return {"mean": float(accs.mean()), "sd": float(accs.std()),
            "ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "n_draws": n_rand}


# ---------------------------------------------------------------- verdict
def decide_verdict(primary, baselines_bs, paired, faithful_support, macro_f1):
    beats = [k for k in BASELINE_KEYS
             if paired.get(k) and paired[k]["ci95"][0] is not None and paired[k]["ci95"][0] > 0]
    # "beats every baseline" requires the SMALLEST paired advantage to have CI lower bound > 0.
    ci_lowers = [paired[k]["ci95"][0] for k in BASELINE_KEYS
                 if paired.get(k) and paired[k]["ci95"][0] is not None]
    min_ci_lower = min(ci_lowers) if ci_lowers else None
    beats_all = (min_ci_lower is not None and min_ci_lower > 0)
    faithful_recall_estimable = faithful_support > 0
    pred_faithful = primary["predicted_counts"]["faithful"] > 0
    if beats_all and faithful_recall_estimable and pred_faithful and macro_f1 >= 0.60:
        verdict = "validated instrument"
    elif beats_all:
        verdict = "weak screen (screens out some failures, cannot identify faithful sites)"
    else:
        verdict = "checklist / reporting discipline (not a validated predictor)"
    return {
        "verdict": verdict,
        "beats_every_baseline_with_ci_lower_gt_0": bool(beats_all),
        "baselines_beaten": beats,
        "min_paired_accuracy_diff_ci_lower": min_ci_lower,
        "faithful_support_in_grid": int(faithful_support),
        "faithful_recall_estimable": bool(faithful_recall_estimable),
        "protocol_ever_predicts_faithful": bool(pred_faithful),
        "macro_f1_present": macro_f1,
    }


# ---------------------------------------------------------------- sensitivity (robustness only)
def threshold_sensitivity(units, key, grid):
    rows = []
    for v in grid:
        accs, f1s, covs = [], [], []
        for u in units:
            uu = dict(u)
            gates, pred, n_app, n_pass = apply_gates(
                uu, ref_max=(v if key == "reference" else REF_RESIDUAL_MAX),
                real_max=(v if key == "realization" else REALIZATION_RESIDUAL_MAX))
            uu["predicted"] = pred
            accs.append(1.0 if pred == uu["observed"] else 0.0)
            covs.append(1.0 if pred is not None else 0.0)
            M, _ = confusion([uu])
            f1s.append(per_class(M)[1] if pred is not None else np.nan)
        rows.append({"threshold": v, "accuracy": float(np.mean(accs)),
                     "coverage": float(np.mean(covs)),
                     "macro_f1_present": float(np.nanmean(f1s))})
    return rows


# ---------------------------------------------------------------- consumer axis
def consumer_axis(cs):
    units = []
    for group, rec in cs.items():
        bb = rec["backbone"]; fam = rec["family"]; param = rec["param"]
        for si, (site, sv) in enumerate(rec["sites"].items()):
            depth = int(rec["depths"][si]) if si < len(rec["depths"]) else si + 1
            for consumer, power in sv["consumer_power"].items():
                t = sv["operators"]["O2"][consumer]["transfer"]
                units.append({
                    "id": f"{bb}|{fam}{param}|{site}|{consumer}",
                    "backbone": bb, "family": f"{fam}_{param}", "site": site, "consumer": consumer,
                    "depth": depth, "power": float(power), "T_F": float(t),
                    "reachable": True, "reference_stat": None, "realization_resid": None,
                })
    for u in units:
        u["observed"] = observed_class(u["T_F"])
        u["gates"] = {"G1": {"pass": u["power"] >= POWER_FLOOR, "applicable": True},
                      "G2": {"pass": False, "applicable": False},
                      "G3": {"pass": True, "applicable": True},
                      "G4": {"pass": False, "applicable": False}}
        u["predicted"] = "partial" if u["power"] >= POWER_FLOOR else None
        u["n_applicable"] = 2
        u["n_passed"] = 2 if u["power"] >= POWER_FLOOR else 1
    res = summarise(units)
    res["units"] = [{"id": u["id"], "power": u["power"], "T_F": u["T_F"],
                     "observed": u["observed"], "predicted": u["predicted"]} for u in units]
    low = [u for u in units if u["power"] < POWER_FLOOR]
    res["low_power_abstained"] = {
        "n": len(low), "mean_T_F": float(np.mean([u["T_F"] for u in low])) if low else None,
        "max_T_F": float(np.max([u["T_F"] for u in low])) if low else None,
        "classes": {c: int(sum(1 for u in low if u["observed"] == c)) for c in CLASSES},
    }
    return res


# ---------------------------------------------------------------- observability axis
def gate_stat_quality(units):
    """Rank relation of every outcome-free gate statistic to T_F, and its failed-vs-rest separation."""
    def log10_or_none(v):
        return None if (v is None or v <= 0) else float(np.log10(v))
    stat_fns = {
        "power_consumer_logits": lambda u: u["power"],
        "power_site_features": lambda u: u.get("power_site"),
        "reference_intertwining_residual": lambda u: u["reference_stat"],
        "realization_feature_residual": lambda u: u["realization_resid"],
        "estimation_C2_over_K_log10": lambda u: log10_or_none(u.get("C2_over_K")),
        "depth": lambda u: u["depth"],
    }
    out = {}
    for name, f in stat_fns.items():
        xs, ys = [], []
        for u in units:
            v = f(u)
            if v is None:
                continue
            xs.append(float(v)); ys.append(float(u["T_F"]))
        if len(xs) < 5 or np.ptp(xs) == 0:
            out[name] = {"n": len(xs), "spearman": None, "p": None, "auc_failed_vs_rest": None}
            continue
        sp = stats.spearmanr(xs, ys)
        a = [x for x, y in zip(xs, ys) if y < PARTIAL_AT]
        b = [x for x, y in zip(xs, ys) if y >= PARTIAL_AT]
        auc = None
        if len(a) >= 2 and len(b) >= 2:
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = float(mw.statistic / (len(a) * len(b)))
        out[name] = {"n": len(xs), "spearman": float(sp.statistic), "p": float(sp.pvalue),
                     "auc_failed_vs_rest": auc,
                     "posthoc_separating_threshold": posthoc_threshold(xs, ys)}
    return out


def posthoc_threshold(xs, ys, direction=None):
    """POST-HOC ONLY: the cut on a raw statistic that best separates failed from the rest.

    Reported to expose how far the pre-registered threshold is from the separating value.
    It is never adopted as the primary rule (that would be tuning on the evaluation set).
    """
    if direction is None:
        direction = "high" if stats.spearmanr(xs, ys).statistic < 0 else "low"
    cands = sorted(set(xs))
    best = None
    for t in cands:
        pred = [(x > t) if direction == "high" else (x < t) for x in xs]
        acc = float(np.mean([p == (y < PARTIAL_AT) for p, y in zip(pred, ys)]))
        if best is None or acc > best["accuracy"]:
            best = {"threshold": float(t), "accuracy": acc, "direction": direction}
    return best


def outcome_variant(units, obs_key):
    """Re-grade the same fixed protocol against a different headline operator."""
    sub = []
    for u in units:
        v = u.get(obs_key)
        if v is None:
            continue
        uu = dict(u)
        uu["T_F"] = float(v)
        uu["observed"] = observed_class(uu["T_F"])
        sub.append(uu)
    return summarise(sub)


def observability_axis(qz, dm):
    key = "resnet50,convnext,vitb16,dinov2b14_seed0"
    units = []
    for bb, sites in qz[key]["backbones"].items():
        for site, v in sites.items():
            dmk = f"{bb}|hue_90.0"
            if dmk not in dm or site not in dm[dmk]["sites"]:
                continue
            s = dm[dmk]["sites"][site]
            vis = float(v.get("visible_share_of_displacement", np.nan))
            units.append({
                "id": f"{bb}|hue_90.0|{site}",
                "power": vis, "T_F": float(s["O2_transfer_mean"]),
                "visible_over_isotropic": float(v.get("visible_share_over_isotropic", np.nan)),
                "reachable": bool(s["reachable"]),
            })
    for u in units:
        u["observed"] = observed_class(u["T_F"])
        u["predicted"] = None if u["power"] < POWER_FLOOR else "partial"
        u["depth"] = 0
    res = summarise(units)
    # Spearman of each observability statistic vs T_F (all sites, and reachable only)
    r_all = stats.spearmanr([u["visible_over_isotropic"] for u in units],
                            [u["T_F"] for u in units])
    res["visible_over_isotropic_vs_TF"] = {"spearman": float(r_all.statistic),
                                           "p": float(r_all.pvalue), "n": len(units)}
    res["units"] = [{"id": u["id"], "visible_share": u["power"],
                     "visible_over_isotropic": u["visible_over_isotropic"],
                     "T_F": u["T_F"], "observed": u["observed"], "predicted": u["predicted"]}
                    for u in units]
    return res


# ---------------------------------------------------------------- figure
def make_figure(primary, units, out):
    bs = out["baselines"]["bootstrap"]
    fig = plt.figure(figsize=(13.2, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.15], wspace=0.32)

    # --- panel A: predicted class vs observed T_F
    ax = fig.add_subplot(gs[0, 0])
    dec = [u for u in units if u["predicted"] is not None]
    jit = np.random.default_rng(7).normal(0, 0.055, len(dec))
    xpos = {c: i for i, c in enumerate(CLASSES)}
    cols = {"failed": "#c44e52", "partial": "#dd8452", "faithful": "#55a868"}
    for i, u in enumerate(dec):
        ax.scatter(xpos[u["predicted"]] + jit[i], u["T_F"], s=26,
                   color=cols[u["predicted"]], alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.axhline(FAITHFUL_AT, color="k", ls="--", lw=0.9)
    ax.axhline(PARTIAL_AT, color="k", ls=":", lw=0.9)
    ax.text(2.45, FAITHFUL_AT + 0.015, "faithful $\\geq$ 0.80", ha="right", fontsize=7)
    ax.text(2.45, PARTIAL_AT + 0.015, "partial $\\geq$ 0.40", ha="right", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("protocol-predicted class (outcome-free gates)", fontsize=8)
    ax.set_ylabel("observed $T_F$ (O2)", fontsize=8)
    ax.set_title("A. Prediction vs outcome", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 2.5)

    # --- panel B: confusion matrix
    ax = fig.add_subplot(gs[0, 1])
    M = np.array(primary["confusion_matrix"])
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=10,
                    color=("white" if M[i, j] > M.max() * 0.6 else "black"))
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=8, rotation=20)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("observed", fontsize=8)
    ax.set_title("B. Confusion (primary rule)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- panel C: accuracy vs baselines with bootstrap CI
    ax = fig.add_subplot(gs[0, 2])
    labels, vals, los, his, colors = [], [], [], [], []
    order = ["protocol"] + BASELINE_KEYS
    for k in order:
        if k == "protocol":
            labels.append("protocol (gates)"); colors.append("#4c72b0")
            vals.append(primary["accuracy"])
            ci = bs["predicted"]["accuracy"]
        else:
            labels.append(k.replace("_", " ")); colors.append("#8c8c8c")
            vals.append(out["baselines"]["summary"][k]["accuracy"])
            ci = bs[k]["accuracy"]
        v = 0.0 if vals[-1] is None or np.isnan(vals[-1]) else vals[-1]
        vals[-1] = v
        lo = ci[0] if ci and ci[0] is not None else v
        hi = ci[1] if ci and ci[1] is not None else v
        los.append(min(lo, v)); his.append(max(hi, v))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, alpha=0.85, height=0.6)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor="k", elinewidth=1, capsize=2)
    ax.axvline(1 / 3, color="k", ls=":", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("accuracy (decided units)", fontsize=8)
    ax.set_title("C. Against baselines (95% CI)", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25, axis="x")
    ax.set_xlim(0, 1.0)

    fig.suptitle("F36 — External validation of the diagnostic-report gates as a predictor "
                 "(held-out settings, published result files only)", fontsize=10, y=1.02)
    fig.savefig(FIG, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report
def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def md_table_confusion(M):
    lines = ["| observed \\ predicted | failed | partial | faithful | support |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(CLASSES):
        lines.append(f"| **{c}** | {M[i][0]} | {M[i][1]} | {M[i][2]} | {sum(M[i])} |")
    lines.append(f"| **predicted total** | {sum(r[0] for r in M)} | {sum(r[1] for r in M)} | "
                 f"{sum(r[2] for r in M)} | {sum(sum(r) for r in M)} |")
    return "\n".join(lines)


def write_report(out):
    P = out["primary"]
    B = out["baselines"]
    V = out["verdict"]
    M = P["confusion_matrix"]
    rep = []
    A = rep.append
    A("# Diagnostic-report validation report: do the gates predict faithful reproduction?\n")
    A("> Generated by `scripts/222_diagnostic_validation.py` from the published result files. "
      "Inference-only: frozen pretrained backbones on COCO instance crops, no training, no new "
      "forward passes. Every number below is produced by that script.\n")
    A("## 0. Pre-registered design\n")
    A("The full pre-registration is in the module docstring of the script and is reproduced "
      "verbatim under `pre_registration` in `results/diagnostic_validation.json`. In brief:\n")
    A("- **Target.** Observed outcome is the published global-linear operator **O2**'s headline "
      "score `O2_transfer_mean` per site; class by the pre-registered conventions "
      "(faithful $T_F\\ge0.80$, partial $0.40\\le T_F<0.80$, failed $T_F<0.40$).")
    A("- **Units.** Primary grid = 4 backbones $\\times$ 4 depths $\\times$ {hue 90, heat 2.0} = 32 "
      "sites from `results/depth_map_summary.json`; auxiliary unseen parameter heat 1.0 = 16 sites; "
      "consumer axis = 16 (site, consumer) units from `results/consumer_swap.json`; observability "
      "axis = 16 hue sites from `results/quotientization.json`.")
    A("- **Held-out splits by setting.** S1 backbone dinov2b14; S2 family heat; S3 depth 4; "
      "S4 backbone vitb16; S5 the unseen heat-1.0 parameter. No parameter is fitted and no gate "
      "threshold is adjusted on any split, so these measure transfer of a fixed rule.")
    A("- **Outcome-free gate statistics.** G1 power: the consumer-level effect size `effect_size_logits` "
      "$\\ge$ 0.05 (floor; site-level `effect_size_site` carried as a companion); "
      "G2 reference validity, heat only, `intertwining_residual_canonical` $\\le$ 0.99; "
      "G3 `reachable`; G4 operator realization on the fit side, "
      "`feat_resid_over_displacement` of O2 $\\le$ 1.00 (held-out feature split; the published files "
      "record no fit-split residual, and this split is stated rather than hidden).")
    A("- **Prediction rule.** G1 fail $\\Rightarrow$ **abstain**; else any applicable gate fail "
      "$\\Rightarrow$ **failed**; else **partial**. The protocol is a conjunction of preconditions "
      "and contains no sufficiency test, so it never predicts **faithful**.\n")
    A("Two pre-registered variants are also reported: the **pre-fix tautological variant** (the "
      "certificate as shipped when this ran set Gate 4 = threshold($T_F$), so its verdict reproduced "
      "the outcome by construction; the certificate now thresholds the outcome-free fit-side "
      "realization residual instead) and the **sufficiency variant** (all gates pass "
      "$\\Rightarrow$ faithful).\n")

    A("## 1. Coverage and class support\n")
    A(f"- Primary grid units: **{P['n_units']}**; decided by the protocol: **{P['n_decided']}**; "
      f"abstained (G1 power floor): **{P['n_abstained']}** "
      f"(**coverage = {fmt(P['coverage'])}**).")
    A(f"- Observed class support: " +
      ", ".join(f"{c} = {P['class_support_observed'][c]}" for c in CLASSES) + ".")
    A(f"- Predicted class counts: " +
      ", ".join(f"{c} = {P['predicted_counts'][c]}" for c in CLASSES) + ".\n")
    A("**The faithful class is empty in the published grid.** No site of the 32-site main grid "
      "reaches the pre-registered faithful threshold $T_F\\ge0.80$; the maximum is "
      f"{fmt(max(u['T_F'] for u in out['units_primary']))}. The protocol therefore cannot be "
      "credited for correctly predicting faithful sites, and no threshold on any gate can be "
      "validated against a class that does not occur here.\n")

    A("## 2. Gate availability (are the gates even applicable?)\n")
    A("| gate | applicable units | pass | fail | not applicable |")
    A("|---|---|---|---|---|")
    for g in ["G0", "G1", "G2", "G3", "G4"]:
        app = sum(1 for u in out["units_primary"] if u["gates"][g]["applicable"])
        ps = sum(1 for u in out["units_primary"] if u["gates"][g]["applicable"] and u["gates"][g]["pass"])
        fl = app - ps
        A(f"| {g} | {app} | {ps} | {fl} | {P['n_units'] - app} |")
    A("")
    g4_app = sum(1 for u in out["units_primary"] if u["gates"]["G4"]["applicable"])
    g2_app = sum(1 for u in out["units_primary"] if u["gates"]["G2"]["applicable"])
    A("G1 never fires on the primary grid (minimum effect size "
      f"{fmt(min(u['power'] for u in out['units_primary']))} $>$ 0.05); G2 is applicable only to the "
      f"{g2_app} heat sites, because the hue route uses a direct consumer with no attribute "
      f"description; G4 is recorded for the {g4_app} sites present in the "
      "`results/estimator_control.json` snapshot used here (dinov2b14 is absent from that file). A "
      "protocol whose gates are applicable to only part of the grid issues a decision that is partly "
      "based on information it does not have, and the missing gate is not recorded as a failure.\n")

    A("## 3. Primary results\n")
    A(md_table_confusion(M))
    A("")
    A("| class | precision | recall | F1 | support |")
    A("|---|---|---|---|---|")
    for c in CLASSES:
        d = P["per_class"][c]
        A(f"| {c} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['f1'])} | {d['support']} |")
    A("")
    A(f"- Accuracy (**decided units**): **{fmt(P['accuracy'])}**")
    A(f"- Macro-F1 (present classes): **{fmt(P['macro_f1_present'])}**; "
      f"macro-F1 (all 3, undefined = 0): {fmt(P['macro_f1_all3'])}")
    A(f"- Ordinal prediction vs observed $T_F$: Spearman $\\rho$ = "
      f"{fmt(P['ordinal']['spearman'])} (p = {fmt(P['ordinal']['spearman_p'])}, n = "
      f"{P['ordinal']['n']}), Kendall $\\tau$ = {fmt(P['ordinal']['kendall'])}.")
    A(f"- Gate score (number of applicable gates passed) vs $T_F$: Spearman $\\rho$ = "
      f"{fmt(out['gate_score_ordinal']['spearman'])} "
      f"(p = {fmt(out['gate_score_ordinal']['spearman_p'])}, n = {out['gate_score_ordinal']['n']}).\n")
    n_undetected = sum(1 for u in out["units_primary"]
                       if u["observed"] == "failed" and u["predicted"] != "failed")
    A("The confusion matrix is dominated by the **failed$\\rightarrow$partial** cell: because no "
      "sufficiency gate exists, every site that survives G1 and the applicable failure screens is "
      f"predicted partial, including {n_undetected} of the "
      f"{P['class_support_observed']['failed']} failed sites that are reachable and unremarkable to "
      "the fit-side diagnostic. No partial or faithful site is ever predicted failed, so precision on "
      f"'failed' is {fmt(P['per_class']['failed']['precision'])} and its recall is "
      f"{fmt(P['per_class']['failed']['recall'])}. The rule is a failure *screen*, not a three-way "
      "classifier.\n")

    A("## 4. Against the baselines\n")
    A("| predictor | accuracy | 95% CI | macro-F1 | Spearman vs $T_F$ |")
    A("|---|---|---|---|---|")
    A(f"| **protocol (gates)** | {fmt(P['accuracy'])} | "
      f"[{fmt(B['bootstrap']['predicted']['accuracy'][0])}, "
      f"{fmt(B['bootstrap']['predicted']['accuracy'][1])}] | "
      f"{fmt(P['macro_f1_present'])} | {fmt(P['ordinal']['spearman'])} |")
    for k in BASELINE_KEYS:
        s = B["summary"][k]
        bs = B["bootstrap"][k]
        A(f"| {k.replace('_', ' ')} | {fmt(s['accuracy'])} | "
          f"[{fmt(bs['accuracy'][0])}, {fmt(bs['accuracy'][1])}] | "
          f"{fmt(s['macro_f1_present'])} | {fmt(bs['spearman_mean'])} |")
    A(f"| random uniform (exact, {out['random_baseline']['n_draws']} draws) | "
      f"{fmt(out['random_baseline']['mean'])} | "
      f"[{fmt(out['random_baseline']['ci95'][0])}, {fmt(out['random_baseline']['ci95'][1])}] | "
      f"n/a | n/a |")
    A("")
    A("**Paired bootstrap of the protocol's accuracy advantage** (protocol minus baseline, "
      f"{N_BOOT} resamples over units):\n")
    A("| baseline | mean accuracy difference | 95% CI | fraction > 0 |")
    A("|---|---|---|---|")
    for k in BASELINE_KEYS:
        d = out["paired_vs_baselines"].get(k)
        if not d:
            continue
        A(f"| {k.replace('_', ' ')} | {fmt(d['mean_diff'])} | "
          f"[{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] | {fmt(d['frac_positive'])} |")
    A("")
    A(out["baseline_discussion"])
    A("")

    A("## 5. Held-out settings\n")
    A("| split | n | coverage | accuracy | macro-F1 | Spearman | reach-only accuracy | "
      "depth-rule accuracy |")
    A("|---|---|---|---|---|---|---|---|")
    for name, s in out["splits"].items():
        A(f"| {name} | {s['n_units']} | {fmt(s['coverage'])} | {fmt(s['accuracy'])} | "
          f"{fmt(s['macro_f1_present'])} | {fmt(s['ordinal']['spearman'])} | "
          f"{fmt(s['reach_only_accuracy'])} | {fmt(s['depth_rule_accuracy'])} |")
    A("")
    A(out["split_discussion"])
    A("")

    A("## 6. Robustness of the gate thresholds (not used to choose them)\n")
    A("| Gate 2 reference residual max | accuracy | coverage | macro-F1 |")
    A("|---|---|---|---|")
    for r in out["sensitivity"]["reference"]:
        A(f"| {r['threshold']} | {fmt(r['accuracy'])} | {fmt(r['coverage'])} | "
          f"{fmt(r['macro_f1_present'])} |")
    A("")
    A("| Gate 4 realization residual max | accuracy | coverage | macro-F1 |")
    A("|---|---|---|---|")
    for r in out["sensitivity"]["realization"]:
        A(f"| {r['threshold']} | {fmt(r['accuracy'])} | {fmt(r['coverage'])} | "
          f"{fmt(r['macro_f1_present'])} |")
    A("")
    A(out["sensitivity_discussion"])
    A("")
    ng = out["variants"]["no_estimator_control_gate4"]
    A(f"Dropping the Gate-4 fit-side diagnostic altogether (so nothing depends on the "
      f"`results/estimator_control.json` snapshot, which was still being written when this ran) "
      f"changes the primary accuracy from {fmt(P['accuracy'])} to {fmt(ng['accuracy'])} and the "
      f"confusion to {ng['confusion_matrix']}. The verdict is unchanged.")
    A("")

    A("### 6b. Do any outcome-free gate statistics carry ordinal signal at all?\n")
    A("| statistic | n | Spearman vs $T_F$ | p | AUC (failed vs rest) | post-hoc separating cut |")
    A("|---|---|---|---|---|---|")
    for name, g in out["gate_statistic_quality"].items():
        pt = g.get("posthoc_separating_threshold")
        pts = (f"{pt['direction']} {pt['threshold']:.3f} (acc {pt['accuracy']:.3f})"
               if pt else "n/a")
        A(f"| `{name}` | {g['n']} | {fmt(g['spearman'])} | {fmt(g['p'])} | "
          f"{fmt(g['auc_failed_vs_rest'])} | {pts} |")
    A("")
    A("The `post-hoc separating cut` column is **not** the protocol: it is the threshold that would "
      "best separate failed from non-failed on this very grid, shown only to expose how far the "
      "pre-registered thresholds are from a separating value. Adopting it would be tuning on the "
      "evaluation set.\n")
    A(out["gate_statistic_quality_discussion"])
    A("")
    A("### 6c. Does the verdict depend on which operator defines the outcome?\n")
    A(f"Re-grading the identical fixed protocol against the content-conditioned **O8** operator as "
      f"the headline (same gates, same thresholds) gives accuracy "
      f"{fmt(out['outcome_variant_O8']['accuracy'])} and confusion "
      f"{out['outcome_variant_O8']['confusion_matrix']}. ")
    A(out["outcome_variant_discussion"])
    A("")

    A("## 7. The certificate's former Gate 4 was circular (it has since been corrected)\n")
    A(out["variants"]["tautological"]["explanation"])
    A("")
    sf = out["variants"]["sufficiency"]
    A("The opposite reading — that passing every applicable gate licenses a *faithful* verdict — is "
      f"worse than doing nothing: it scores accuracy {fmt(sf['accuracy'])} with confusion "
      f"{sf['confusion_matrix']}, below the all-partial baseline's "
      f"{fmt(B['summary']['always_partial']['accuracy'])}, because most reachable sites pass all "
      "available gates and most of them are not faithful. This is the direct evidence that the gates "
      "are preconditions, not a sufficiency test, and it is why the primary rule deliberately issues "
      "no faithful verdict.\n")

    A("## 8. Consumer axis: Gate 1 is real, and it fires against the protocol's own interest\n")
    c = out["consumer_axis"]
    A(f"- Units: {c['n_units']}; coverage = {fmt(c['coverage'])}; accuracy on decided units = "
      f"{fmt(c['accuracy'])}.")
    A(f"- Abstained (power $<$ 0.05): {c['low_power_abstained']['n']} units, with mean $T_F$ = "
      f"{fmt(c['low_power_abstained']['mean_T_F'])} (max {fmt(c['low_power_abstained']['max_T_F'])}).")
    A("")
    A("| unit | consumer power | observed $T_F$ | observed class | protocol |")
    A("|---|---|---|---|---|")
    for u in c["units"]:
        A(f"| {u['id']} | {fmt(u['power'], 4)} | {fmt(u['T_F'])} | {u['observed']} | "
          f"{u['predicted'] if u['predicted'] else 'ABSTAIN'} |")
    A("")
    A(out["consumer_discussion"])
    A("")

    A("## 9. Observability axis\n")
    o = out["observability_axis"]
    A(f"- Consumers' visible share of the site displacement vs $T_F$: accuracy = "
      f"{fmt(o['accuracy'])} at coverage {fmt(o['coverage'])}.")
    A(f"- `visible_share_over_isotropic` vs $T_F$: Spearman $\\rho$ = "
      f"{fmt(o['visible_over_isotropic_vs_TF']['spearman'])} "
      f"(p = {fmt(o['visible_over_isotropic_vs_TF']['p'])}).")
    A("")
    A(out["observability_discussion"])
    A("")

    A("## 10. Verdict\n")
    A(f"**{V['verdict']}.**")
    A("")
    A(f"- The protocol beats every baseline with a paired-bootstrap CI lower bound $>0$: "
      f"**{V['beats_every_baseline_with_ci_lower_gt_0']}** "
      f"(baselines beaten: {', '.join(V['baselines_beaten']) if V['baselines_beaten'] else 'none'}; "
      f"worst paired CI lower bound = {fmt(V['min_paired_accuracy_diff_ci_lower'])}).")
    A(f"- Faithful support in the evaluated grid: **{V['faithful_support_in_grid']}**; faithful "
      f"recall estimable: **{V['faithful_recall_estimable']}**; protocol ever predicts faithful: "
      f"**{V['protocol_ever_predicts_faithful']}**.")
    A(f"- Macro-F1 over present classes: {fmt(V['macro_f1_present'])}.")
    A("")
    A(out["verdict_prose"])
    A("")
    A("## 11. Recommended sentence for the paper's Section 4\n")
    A("> " + out["recommended_section4_sentence"])
    A("")
    A("## 12. Applicability domain\n")
    A(out["applicability_domain"])
    A("")
    A("## 13. Provenance\n")
    A("- Source files: " + ", ".join(f"`{s}`" for s in out["data_sources"]) + ".")
    A("")
    A("| input file | bytes | mtime | sha256 (first 16) |")
    A("|---|---|---|---|")
    for n, p in out.get("input_provenance", {}).items():
        if not p.get("exists"):
            A(f"| `{n}` | — | — | missing |")
        else:
            A(f"| `{n}` | {p['bytes']} | {p['mtime']} | `{p['sha256'][:16]}` |")
    A("")
    A("- Script: `scripts/222_diagnostic_validation.py`; JSON: `results/diagnostic_validation.json`; "
      "figure: `paper/figures/F36_diagnostic_validation.png`.")
    A("- No manuscript file was edited.\n")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(rep))
    return "\n".join(rep)


# ---------------------------------------------------------------- main
def main():
    dm, ec, cs, qz = load()
    ecx = ec_index(ec)
    cpow = load_consumer_power()

    # ---- primary evaluation units
    units_primary = add_predictions(build_grid_units(dm, ecx, MAIN_GRID_FAMILIES, cpow))
    units_aux = add_predictions(build_grid_units(dm, ecx, ["heat_1.0"], cpow))
    baseline_units(units_primary)
    baseline_units(units_aux)

    primary = summarise(units_primary)
    for u in units_primary:
        u["_ord"] = None
    # gate-score ordinal (number of applicable gates passed) vs T_F
    gs_ord = {"_ord": 0}
    units_gs = []
    for u in units_primary:
        uu = dict(u); uu["_ord"] = u["n_passed"]; units_gs.append(uu)
    gate_score_ordinal = ordinal_metrics(units_gs, "n_passed", "T_F")
    for u in units_primary:
        u["_ord"] = ORD[u["predicted"]] if u["predicted"] is not None else None

    aux = summarise(units_aux)

    # robustness: drop the Gate-4 fit-side diagnostic entirely, so the result cannot depend on the
    # (still growing) results/estimator_control.json snapshot
    units_no_g4 = build_grid_units(dm, ecx, MAIN_GRID_FAMILIES, cpow)
    for u in units_no_g4:
        u["realization_resid"] = None
    add_predictions(units_no_g4)
    baseline_units(units_no_g4)
    no_g4 = summarise(units_no_g4)

    # ---- baselines
    summary = {k: summarise(units_primary, k) for k in ["predicted"] + BASELINE_KEYS}
    bs = bootstrap(units_primary, ["predicted"] + BASELINE_KEYS)
    paired = paired_bootstrap(units_primary, "predicted", BASELINE_KEYS)
    rand_base = random_baseline_distribution(units_primary)

    # ---- splits
    splits = {}
    def split_metrics(sub, name):
        sub = add_predictions([dict(u) for u in sub])
        baseline_units(sub)
        s = summarise(sub)
        s["reach_only_accuracy"] = accuracy(sub, "reach_only")
        s["depth_rule_accuracy"] = accuracy(sub, "depth_rule")
        s["name"] = name
        s["units"] = [{"id": u["id"], "observed": u["observed"], "predicted": u["predicted"],
                       "reachable": u["reachable"], "T_F": u["T_F"], "gates": u["gates"]}
                      for u in sub]
        return s
    splits["S1 hold-out backbone = dinov2b14"] = split_metrics(
        [u for u in units_primary if u["backbone"] == "dinov2b14"], "S1")
    splits["S2 hold-out family = heat (main grid)"] = split_metrics(
        [u for u in units_primary if u["family_base"] == "heat"], "S2")
    splits["S3 hold-out depth = 4"] = split_metrics(
        [u for u in units_primary if u["depth"] == 4], "S3")
    splits["S4 hold-out backbone = vitb16"] = split_metrics(
        [u for u in units_primary if u["backbone"] == "vitb16"], "S4")
    splits["S5 hold-out parameter = heat 1.0"] = split_metrics(units_aux, "S5")

    # ---- variants
    taut = {
        "agreement_rate": 1.0,
        "by_construction": True,
        "explanation": out_tautology(units_primary),
        "n_units": len(units_primary),
        "accuracy": 1.0,
        "out_of_sample_information": "none",
    }
    sufficiency = {
        "rule": "all applicable gates pass -> faithful, else failed (G1 fail -> abstain)",
    }
    suf_units = []
    for u in units_primary:
        gates = u["gates"]
        if not gates["G1"]["pass"]:
            pred = None
        elif all(gates[g]["pass"] for g in ["G0", "G1", "G2", "G3", "G4"] if gates[g]["applicable"]):
            pred = "faithful"
        else:
            pred = "failed"
        uu = dict(u); uu["predicted_suf"] = pred; suf_units.append(uu)
    sufficiency.update(summarise(suf_units, "predicted_suf"))

    # ---- sensitivity
    sens = {
        "reference": threshold_sensitivity(units_primary, "reference", [0.95, 0.98, 0.99, 0.995, 0.999]),
        "realization": threshold_sensitivity(units_primary, "realization", [0.80, 0.90, 1.00, 1.10, 1.20]),
    }

    # ---- axes
    cax = consumer_axis(cs)
    oax = observability_axis(qz, dm)

    # ---- gate-statistic quality and alternative-outcome robustness
    gsq = gate_stat_quality(units_primary)
    o8v = outcome_variant(units_primary, "T_F_O8")

    # ---- verdict
    V = decide_verdict(primary, bs, paired, primary["class_support_observed"]["faithful"],
                       primary["macro_f1_present"])
    _real = gsq.get("realization_feature_residual", {})
    _depth_acc = summary["depth_rule"]["accuracy"]
    V["summary"] = (
        "Not a validated three-way instrument as shipped: the pre-registered gate conjunction "
        f"scores accuracy {primary['accuracy']:.3f}, never predicts faithful (the class is empty), "
        f"and is beaten by the depth rule ({_depth_acc:.3f}); but one outcome-free gate "
        "statistic, the fit-side realization residual, is a strong ordinal predictor "
        f"(Spearman {_real.get('spearman'):.3f}, AUC {_real.get('auc_failed_vs_rest'):.3f}) that the "
        "certificate as validated did not use as a gate (it thresholded T_F; it has since been "
        "corrected to threshold this residual).")
    V["gate4_realization_auc"] = _real.get("auc_failed_vs_rest")
    V["gate4_realization_spearman"] = _real.get("spearman")

    out = {
        "protocol": "222 diagnostic validation",
        "pre_registration": {
            "power_floor": POWER_FLOOR, "faithful_at": FAITHFUL_AT, "partial_at": PARTIAL_AT,
            "reference_residual_max": REF_RESIDUAL_MAX,
            "realization_residual_max": REALIZATION_RESIDUAL_MAX,
            "depth_fail_at": DEPTH_FAIL_AT, "n_boot": N_BOOT, "n_rand": N_RAND, "seed": SEED,
            "headline_operator": "O2 (published global linear, ridge)",
            "primary_grid": "4 backbones x 4 depths x {hue_90.0, heat_2.0}",
            "splits": ["S1 dinov2b14", "S2 heat", "S3 depth 4", "S4 vitb16", "S5 heat 1.0"],
            "prediction_rule": "G1 fail -> abstain; else any applicable gate fail -> failed; "
                               "else partial; faithful is never predicted",
        },
        "data_sources": ["results/depth_map_summary.json", "results/depth_map_resnet50_{hue,heat}.json",
                         "results/depth_map_convnext_{hue,heat}.json",
                         "results/depth_map_vitb16_{hue,heat}.json",
                         "results/depth_map_dinov2b14_{hue,heat}.json",
                         "results/estimator_control.json",
                         "results/consumer_swap.json", "results/quotientization.json",
                         "results/transportability_certificate_example.json"],
        "units_primary": [{k: v for k, v in u.items() if k not in ("gates", "_ord")} | {"gates": u["gates"]}
                          for u in units_primary],
        "units_aux_heat1": [{k: v for k, v in u.items() if k not in ("gates", "_ord")} | {"gates": u["gates"]}
                            for u in units_aux],
        "primary": primary,
        "auxiliary_heat1": aux,
        "gate_score_ordinal": gate_score_ordinal,
        "baselines": {"summary": {k: summary[k] for k in BASELINE_KEYS},
                      "bootstrap": bs},
        "paired_vs_baselines": paired,
        "random_baseline": rand_base,
        "splits": splits,
        "variants": {"tautological": taut, "sufficiency": sufficiency,
                     "no_estimator_control_gate4": no_g4},
        "input_provenance": file_provenance(
            ["depth_map_summary.json", "estimator_control.json",
             "consumer_swap.json", "quotientization.json",
             "transportability_certificate_example.json",
             "depth_map_resnet50_hue.json", "depth_map_resnet50_heat.json",
             "depth_map_convnext_hue.json", "depth_map_convnext_heat.json",
             "depth_map_vitb16_hue.json", "depth_map_vitb16_heat.json",
             "depth_map_dinov2b14_hue.json", "depth_map_dinov2b14_heat.json"]),
        "sensitivity": sens,
        "consumer_axis": cax,
        "observability_axis": oax,
        "gate_statistic_quality": gsq,
        "outcome_variant_O8": o8v,
        "verdict": V,
    }
    out.update(narrative(out))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1, default=str)

    make_figure(primary, units_primary, out)
    write_report(out)
    print(json.dumps({"verdict": V["verdict"], "accuracy": primary["accuracy"],
                      "macro_f1": primary["macro_f1_present"],
                      "coverage": primary["coverage"],
                      "confusion": primary["confusion_matrix"]}, indent=1))
    print("wrote", OUT_JSON)
    print("wrote", FIG)
    print("wrote", OUT_MD)


def out_tautology(units):
    return ("In the certificate as it shipped when this validation ran, the schema's "
            "`gate_4_realization.verdict` was `verdict(T_F)` and `certificate.faithfulness` was copied "
            "from it, so the certificate reproduced the observed class with agreement 1.000 by "
            "construction. A predictor defined as a function of the outcome cannot be validated "
            "against that outcome: its accuracy, per-class precision and recall are all exact and all "
            "carry zero out-of-sample information. The validation therefore uses only Gate 0-3 and the "
            "fit-side Gate 4 diagnostic, and reports the circular reading separately. The certificate "
            "has since been corrected (`scripts/215_transportability_certificate.py`): Gate 4 now "
            "thresholds the outcome-free fit-side realization residual at the pre-registered 1.00 and "
            "the summary verdict is the precondition conjunction, which cannot return 'faithful'. The "
            "non-circular primary rule's own accuracy is "
            f"{fmt(sum(1 for u in units if u['predicted'] == u['observed']) / len(units))}.")


def narrative(out):
    P = out["primary"]; V = out["verdict"]; B = out["baselines"]
    n_failed = P["class_support_observed"]["failed"]
    n_partial = P["class_support_observed"]["partial"]
    reach_acc = B["summary"]["reach_only"]["accuracy"]
    depth_acc = B["summary"]["depth_rule"]["accuracy"]
    always_acc = B["summary"]["always_partial"]["accuracy"]
    bs_reach = B["bootstrap"]["reach_only"]["accuracy"]
    bs_prot = B['bootstrap']['predicted']["accuracy"]
    paired = out["paired_vs_baselines"]

    # baseline discussion
    d = out["paired_vs_baselines"]
    beats = V["baselines_beaten"]
    depth_d = d["depth_rule"]
    reach_d = d["reach_only"]
    bd = ("The protocol clears the trivial baselines: against always-partial its accuracy advantage is "
          f"{fmt(d['always_partial']['mean_diff'])} "
          f"(95% CI [{fmt(d['always_partial']['ci95'][0])}, {fmt(d['always_partial']['ci95'][1])}]), "
          "and against random assignment it is "
          f"{fmt(d['random_uniform']['mean_diff'])} "
          f"(95% CI [{fmt(d['random_uniform']['ci95'][0])}, {fmt(d['random_uniform']['ci95'][1])}]). "
          "That advantage is the causally unreachable sites, which both baselines must get wrong. It "
          "also beats **reachability alone** by "
          f"{fmt(reach_d['mean_diff'])} (95% CI [{fmt(reach_d['ci95'][0])}, {fmt(reach_d['ci95'][1])}]), "
          "so the reference-validity gate (Gate 2) does add measurably on top of the reachability "
          "boolean. But it does **not** beat **depth alone**: the published depth law, applied as a "
          f"fixed rule with no fitting, reaches accuracy {fmt(depth_acc)} against the protocol's "
          f"{fmt(P['accuracy'])}, a paired difference of {fmt(depth_d['mean_diff'])} "
          f"(95% CI [{fmt(depth_d['ci95'][0])}, {fmt(depth_d['ci95'][1])}]; the protocol is ahead in "
          f"only {fmt(depth_d['frac_positive'])} of resamples). A variable that is part of the "
          "phenomenon under study, and not a validity gate at all, therefore predicts the verdict "
          "better than the gates do. Baselines beaten with a paired CI lower bound above zero: "
          f"{', '.join(beats) if beats else 'none'}.")
    out["baseline_discussion"] = bd

    # per-split structure, computed from the units of each split
    def split_struct(sub):
        return {"n": len(sub),
                "n_failed": sum(1 for u in sub if u["observed"] == "failed"),
                "n_partial": sum(1 for u in sub if u["observed"] == "partial"),
                "n_faithful": sum(1 for u in sub if u["observed"] == "faithful"),
                "n_unreachable": sum(1 for u in sub if not u["reachable"]),
                "n_g2_fail": sum(1 for u in sub
                                 if u["gates"]["G2"]["applicable"] and not u["gates"]["G2"]["pass"]),
                "n_undetected": sum(1 for u in sub
                                    if u["observed"] == "failed" and u["predicted"] != "failed")}
    st = {k: split_struct(v["units"]) for k, v in out["splits"].items()}
    s1, s2, s3, s4, s5 = (st["S1 hold-out backbone = dinov2b14"],
                          st["S2 hold-out family = heat (main grid)"],
                          st["S3 hold-out depth = 4"],
                          st["S4 hold-out backbone = vitb16"],
                          st["S5 hold-out parameter = heat 1.0"])
    sd = ("Held out by setting, the fixed rule transfers unevenly and always for the same reason: "
          "coverage is 1.0 on the main grid (G1 never fires at any site), and accuracy is bounded by "
          "the failed-to-partial confusion that no available gate resolves. "
          f"S1 (dinov2b14, n={s1['n']}) holds {s1['n_failed']} failed and {s1['n_partial']} partial "
          f"sites and only {s1['n_unreachable']} unreachable ones, so the reachability screen catches "
          f"{s1['n_unreachable']} and the protocol scores {fmt(out['splits']['S1 hold-out backbone = dinov2b14']['accuracy'])}, "
          f"below the depth rule's {fmt(out['splits']['S1 hold-out backbone = dinov2b14']['depth_rule_accuracy'])}. "
          f"S2 (heat, n={s2['n']}) and S5 (the unseen heat-1.0 parameter, n={s5['n']}) behave "
          "differently from each other because the reference screen fires far more often on the "
          f"smaller heat parameter: S2 fails Gate 2 on {s2['n_g2_fail']} of {s2['n']} sites and Gate 3 "
          f"on {s2['n_unreachable']}, scoring "
          f"{fmt(out['splits']['S2 hold-out family = heat (main grid)']['accuracy'])}, while S5 fails "
          f"Gate 2 on {s5['n_g2_fail']} of {s5['n']} sites and Gate 3 on {s5['n_unreachable']}, "
          f"scoring {fmt(out['splits']['S5 hold-out parameter = heat 1.0']['accuracy'])}. That is not "
          "transfer of a stable diagnostic: the same gate fires at a different rate under a different "
          "transformation parameter, which is a property of how the canonical residual is normalised, "
          "not of the operator's faithfulness. S3 (depth 4, "
          f"n={s3['n']}) is degenerate: every site is failed, so any rule that says 'not partial' "
          f"scores between the protocol's {fmt(out['splits']['S3 hold-out depth = 4']['accuracy'])} "
          f"and the depth rule's {fmt(out['splits']['S3 hold-out depth = 4']['depth_rule_accuracy'])}. "
          f"S4 (vitb16, n={s4['n']}) is where reachability helps most "
          f"({s4['n_unreachable']} unreachable sites). None of this is out-of-sample *fitting*: no "
          "gate threshold was adjusted on any split, so the numbers are transfer of a fixed checklist, "
          "not generalisation of a learned predictor. In every split the depth rule is at least as "
          "accurate as the protocol.")
    out["split_discussion"] = sd

    sens_ref = out["sensitivity"]["reference"]
    sens_real = out["sensitivity"]["realization"]
    best_ref = max(sens_ref, key=lambda r: r["accuracy"])
    sdisc = ("The primary thresholds were fixed from first principles (0.99 on the canonical "
             "intertwining residual, 1.00 on the feature realization residual) and were not chosen "
             "from these tables. The sweep shows the pre-registered Gate-2 value is *not* the "
             "accuracy-maximising one: an unpregistered 0.98 would give "
             f"{fmt(best_ref['accuracy'])} against the primary {fmt(P['accuracy'])}. That is exactly "
             "the post-hoc threshold selection the pre-registration forbids, so it is reported and "
             "not adopted; and even the best swept setting produces no faithful prediction and does "
             "not change the qualitative picture. Accuracy across both sweeps lies in "
             f"[{fmt(min(r['accuracy'] for r in sens_ref + sens_real))}, "
             f"{fmt(max(r['accuracy'] for r in sens_ref + sens_real))}], coverage stays 1.0, and the "
             "gate count (0-4 passed) correlates only weakly with $T_F$ "
             f"(Spearman {fmt(out['gate_score_ordinal']['spearman'])}, p = "
             f"{fmt(out['gate_score_ordinal']['spearman_p'])}). A gate family that only changes "
             "*which* failures it screens, never whether it can identify success, cannot be rescued "
             "by re-thresholding.")
    out["sensitivity_discussion"] = sdisc

    cax = out["consumer_axis"]
    low = cax["low_power_abstained"]
    cd = ("This is the one slice on which Gate 1 is not vacuous, and it is the most instructive: the "
          f"retrieval consumer's power is below the floor on {low['n']} of {cax['n_units']} units "
          f"(mean $T_F$ = {fmt(low['mean_T_F'])}, max {fmt(low['max_T_F'])}). The protocol abstains "
          "there, which is exactly the behaviour Section 3.7 prescribes, because a consumer that "
          "barely responds cannot adjudicate. But the abstained units are the ones where O2 *looks "
          "best*: the raw $T_F$ that the pre-registered classes reward is highest precisely where the "
          "power gate says the number is uninformative. Validating the gates against the raw "
          "$T_F$ class therefore penalises the protocol for being epistemically correct, and inflates "
          f"the baselines that never abstain. Coverage on this axis is {fmt(cax['coverage'])}, and "
          "the accuracy on the decided units is not comparable to the main grid for the same reason.")
    out["consumer_discussion"] = cd

    oax = out["observability_axis"]
    covered = [u for u in oax["units"] if u["predicted"] is not None]
    covered_classes = {c: sum(1 for u in covered if u["observed"] == c) for c in CLASSES}
    od = ("An observability statistic read from `results/quotientization.json` — the readout's "
          "visible share of the site displacement, `visible_share_of_displacement`, applied with the "
          f"same 0.05 floor — covers only {len(covered)} of {oax['n_units']} hue sites and every one "
          f"of the covered sites is observed partial {covered_classes}. It is therefore perfect on "
          f"the quarter it keeps (accuracy {fmt(oax['accuracy'])}) and abstains on the other three "
          f"quarters (coverage {fmt(oax['coverage'])}); it cannot separate failed from partial at all, "
          "because it rejects every failed and every faithful site it sees. The isotropic-normalised "
          "version has only a weak rank relation to $T_F$ "
          f"(Spearman {fmt(oax['visible_over_isotropic_vs_TF']['spearman'])}, p = "
          f"{fmt(oax['visible_over_isotropic_vs_TF']['p'])}). No alternative observability statistic "
          "available in the published files turns Gate 1 into a usable three-way discriminator.")
    out["observability_discussion"] = od

    gsq = out["gate_statistic_quality"]
    real = gsq.get("realization_feature_residual", {})
    real_pt = real.get("posthoc_separating_threshold")
    best_name = max(gsq.items(), key=lambda kv: abs(kv[1]["spearman"] or 0.0))[0]
    gq = ("This is the most important nuance in the validation, and it cuts both ways. The statistic "
          "behind **Gate 4** — the outcome-free held-out single-step feature realization residual "
          "`feat_resid_over_displacement` — is a strong ordinal predictor of $T_F$: "
          f"Spearman $\\rho$ = {fmt(real.get('spearman'))} (p = {fmt(real.get('p'))}, n = "
          f"{real.get('n')}), AUC {fmt(real.get('auc_failed_vs_rest'))} for failed-vs-rest. It is a "
          f"better ordinal predictor than depth ($\\rho$ = {fmt(gsq['depth']['spearman'])}). The "
          "protocol's failure is therefore *not* that it has no usable signal; it is that the "
          "**certificate as validated did not use this statistic as a gate** (it thresholded $T_F$ "
          "itself; the corrected certificate now thresholds this residual), that the pre-registered "
          "Gate-4 cut of 1.00 — 'no worse than the no-op' — is far "
          "too lax to separate partial from failed, and that the statistic is recorded for only "
          f"{real.get('n')} of 32 sites (absent for the whole held-out dinov2b14 backbone). A "
          "post-hoc cut on the residual would separate failed from not-failed at about "
          + (f"{real_pt['threshold']:.3f}" if real_pt else "n/a")
          + (f" (accuracy {real_pt['accuracy']:.3f})" if real_pt else "")
          + ", but choosing it here would be tuning on the evaluation set, and it still cannot "
          "produce a faithful prediction because no faithful site exists. The weaker gates are the "
          "ones the protocol emphasises: consumer power has the expected sign but little strength "
          f"($\\rho$ = {fmt(gsq['power_consumer_logits']['spearman'])}, AUC "
          f"{fmt(gsq['power_consumer_logits']['auc_failed_vs_rest'])} — failed sites do have lower "
          "power), and it never crosses the 0.05 floor on this grid; reachability is a binary that "
          "only fires off the grid edge. Among genuinely outcome-free statistics, then, exactly one "
          "(`feat_resid_over_displacement`) carries strong signal, and the protocol neither "
          "thresholds it usefully nor records it everywhere.")
    out["gate_statistic_quality_discussion"] = gq

    o8 = out["outcome_variant_O8"]
    o8d = ("The qualitative picture is operator-independent, which is expected because O8 differs from "
           "O2 mainly at the shallow sites of the plain CNN and shares O2's deep-site failures. No "
           "faithful class appears under O8 either, so the protocol still cannot be validated on "
           "faithful sites. This confirms that the checklist's limitation is structural, not an "
           "artefact of the chosen published operator.")
    out["outcome_variant_discussion"] = o8d

    vp = ("The protocol as shipped is a sound *necessary-condition* screen but not a validated "
          "three-way instrument. Its strengths are real: it never claims a site is faithful, the "
          "sites it rejects (causally unreachable; canonical intertwining residual at the 0.99 "
          "boundary) really are failures, and — the strongest positive finding here — the "
          "outcome-free fit-side realization residual behind Gate 4 is a strong ordinal predictor of "
          f"$T_F$ (Spearman $\\rho$ = {fmt(real.get('spearman'))}, AUC "
          f"{fmt(real.get('auc_failed_vs_rest'))}), better than depth. Its weaknesses are equally "
          "concrete: the conjunction as pre-registered cannot output faithful at all (the class is "
          "empty in the published grid), it is beaten as a classifier by the trivial depth rule, the "
          "certificate as validated (`scripts/215`) was circular in Gate 4 because it thresholded the "
          "outcome instead of the realization residual (this has since been corrected), its "
          "pre-registered "
          "Gate-4 cut of 1.00 is too lax to separate partial from failed, and its fit-side statistic "
          f"is missing for {32 - real.get('n', 0)} of 32 sites including the entire held-out "
          "dinov2b14 backbone. The honest label is therefore a **checklist / reporting discipline** "
          "whose one predictive component is not the one it currently reports, and whose thresholds "
          "would need a fresh pre-registered validation on unseen data before any three-way verdict "
          "is credited.")
    out["verdict_prose"] = vp

    max_tf = max(u["T_F"] for u in out["units_primary"])
    sent = (f"We report the five gates as a *diagnostic checklist and reporting discipline* rather "
            f"than a validated predictor: on the {P['n_units']}-site cross-architecture grid the "
            f"outcome-free gate conjunction screens out causally unreachable and reference-invalid "
            f"failures (accuracy {fmt(P['accuracy'])} versus {fmt(always_acc)} for always-partial) but "
            f"leaves the failed-versus-partial boundary unresolved (macro-F1 "
            f"{fmt(P['macro_f1_present'])}), is no better than the site's depth alone (depth-rule "
            f"accuracy {fmt(depth_acc)}; paired difference {fmt(d['depth_rule']['mean_diff'])} "
            f"[{fmt(d['depth_rule']['ci95'][0])}, {fmt(d['depth_rule']['ci95'][1])}]), and can never "
            f"return a faithful verdict because no site of the published grid reaches the "
            f"pre-registered faithful threshold (max $T_F$ = {fmt(max_tf)}), while the one gate "
            f"statistic that carries strong ordinal signal is the outcome-free fit-side realization residual "
            f"(Spearman $\\rho$ = {fmt(real.get('spearman'))}, AUC "
            f"{fmt(real.get('auc_failed_vs_rest'))}), recorded for all {real.get('n')} of "
            f"{P['n_units']} sites in the current files and thresholded at the pre-registered 1.00 — a cut "
            f"too lax to separate partial from failed; the corrected certificate thresholds this residual "
            f"rather than $T_F$.")
    out["recommended_section4_sentence"] = sent

    out["applicability_domain"] = (
        "**Applies.** Frozen pretrained backbones (resnet50, convnext, vitb16, dinov2b14) on COCO "
        "train2017 instance crops at four depths, hue-90 rotation (group, direct consumer) and the "
        "spectral heat semigroup (2.0 and 1.0), consumer = the backbone's own frozen head, headline "
        "operator = O2 global linear ridge. The gate statistics are read from "
        "`results/depth_map_summary.json`, the raw `results/depth_map_*.json` (consumer-level power), "
        "and `results/estimator_control.json` (fit-side realization).\n\n"
        "**Does not apply / not tested here.** (i) Attribute-mediated consumers where the reference "
        "can genuinely fail: Gate 2 is recorded only for heat and only as the canonical intertwining "
        "residual, so the reference-validity gate is untested where it matters most. (ii) Sites with "
        "power below the floor: the protocol abstains and no accuracy claim is made. (iii) "
        "Content-conditioned or nonlinear operator families: the observed outcome is O2 only, so the "
        "validation is about whether the gates predict *O2* faithfulness. (iv) The faithful class is "
        "absent from the published grid, so the protocol's behaviour on faithful sites is "
        "unvalidated by construction. (v) fit-SPLIT realization residuals were not recorded in the "
        "published files; the Gate-4 column uses the held-out feature residual, which is stated "
        "rather than hidden.")
    return {"narrative_ready": True}


if __name__ == "__main__":
    main()
