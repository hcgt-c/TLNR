# -*- coding: utf-8 -*-
"""215_transportability_certificate.py — the transportability certificate: schema + one worked example.

Writes, strictly from existing result files (no new runs):
  * results/transportability_certificate_schema.json  — the artifact's fields and allowed values
  * results/transportability_certificate_example.json — one populated certificate

Thresholds used to turn continuous numbers into certificate verdicts are REPORTING CONVENTIONS, not results:
a certificate records both the convention and the numbers it was applied to, so a reader can re-threshold.

Gate 4 is deliberately OUTCOME-FREE: it thresholds the fit-side realization residual
`feat_resid_over_displacement` of the fitted operator (from results/estimator_control.json) at REALIZATION_AT,
not the downstream transfer T_F. The transfer is carried in the record as a companion outcome, never as the
gate. The summary verdict is the conjunction of the preconditions and therefore never returns "faithful"; an
earlier version of this script thresholded T_F in Gate 4 and copied that verdict into the certificate, which
made the certificate reproduce the observed class by construction (see reports/DIAGNOSTIC_VALIDATION_REPORT.md).

Example cell: resnet50 | hue_90.0 | layer1 (the shallowest reachable site), operator O2 (global linear).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")

POWER_FLOOR = 0.05      # convention: below this the test cannot adjudicate
FAITHFUL_AT = 0.80      # convention: transfer >= 0.80 -> faithful (reported, never a gate)
PARTIAL_AT = 0.40       # convention: 0.40 <= transfer < 0.80 -> partial (reported, never a gate)
REALIZATION_AT = 1.00   # convention: fit-side realization residual <= 1.0 -> no worse than the no-op

SCHEMA = {
    "artifact": "transportability certificate",
    "purpose": "record, for one (backbone, transformation, site, operator family, consumer), whether a "
               "feature-space intervention passes the preconditions for reproducing the real transformation; "
               "the certificate screens failures and never certifies faithfulness",
    "conventions": {
        "power_floor": POWER_FLOOR,
        "faithful_at": FAITHFUL_AT,
        "partial_at": PARTIAL_AT,
        "realization_at": REALIZATION_AT,
        "note": "thresholds are reporting conventions, not results; the raw numbers are always recorded. "
                "Gate 4 thresholds the outcome-free fit-side realization residual, not the transfer; the "
                "summary verdict is a conjunction of preconditions and never returns 'faithful'."
    },
    "fields": {
        "site_id": "str: backbone|family|site",
        "gate_0_transformation": {"family": "str", "exact": "bool",
                                  "algebra": "group | semigroup | monoid"},
        "gate_1_observability": {"powered": "bool", "effect_size": "float", "threshold": "float"},
        "gate_2_reference": {"route": "direct | attribute-mediated",
                             "reference_valid": "bool",
                             "evidence": "str"},
        "gate_3_reachability": {"reachable": "bool", "evidence": "str"},
        "gate_4_realization": {"operator_family": "str", "realization_residual": "float",
                               "threshold": "float", "verdict": "pass | fail",
                               "outcome_transfer_companion": "float", "win_rate_companion": "float"},
        "composition": {"assessed": "bool", "verdict": "composable | lossy | not-assessed"},
        "certificate": {"powered": "bool", "reachable": "bool", "reference_valid": "bool",
                        "realized": "bool",
                        "recommended_family": "orthogonal | global-linear | conditioned | none-found",
                        "verdict": "uninformative | failed | partial",
                        "observed_transfer_companion": "float",
                        "compositionality": "composable | lossy | not-assessed",
                        "recommended_depth": "str"},
        "provenance": {"source_files": "list[str]", "generated_by": "str"}
    }
}


def class_of(t):
    """Reporting convention for the observed outcome; never used as a gate."""
    if t >= FAITHFUL_AT:
        return "faithful"
    if t >= PARTIAL_AT:
        return "partial"
    return "failed"


def main():
    d = json.load(open(os.path.join(RES, "depth_map_summary.json"), encoding="utf-8"))
    key = "resnet50|hue_90.0"
    rec = d[key]
    site = "layer1"
    s = rec["sites"][site]
    eff = float(s["effect_size_site"])
    reachable = bool(s["reachable"])
    o2 = float(s["O2_transfer_mean"])
    o8 = float(s.get("O8_transfer_mean", float("nan")))
    o1 = float(s["O1_transfer_mean"])
    win = float(s["O2_win_rate_mean"])
    powered = eff >= POWER_FLOOR
    # Gate 4 is the outcome-free fit-side realization residual, not the downstream transfer
    ec = json.load(open(os.path.join(RES, "estimator_control.json"), encoding="utf-8"))
    resid = float(ec["cells"]["resnet50|hue|depth1"]["estimators"]["O2_pub"]["feat_resid_over_displacement"])
    realized = resid <= REALIZATION_AT
    reference_valid = True
    # conjunction of preconditions; never returns faithful
    if not powered:
        summary = "uninformative"
    elif not (reachable and reference_valid and realized):
        summary = "failed"
    else:
        summary = "partial"
    fam_recommended = "orthogonal" if class_of(o2) == "faithful" else "conditioned"
    cert = {
        "site_id": f"resnet50|hue_90.0|{site}",
        "gate_0_transformation": {
            "family": "hue_90.0", "exact": True, "algebra": "group",
            "evidence": "controlled render at a specified hue, exact by construction (Section 3.1)"},
        "gate_1_observability": {
            "powered": powered, "effect_size": round(eff, 4), "threshold": POWER_FLOOR},
        "gate_2_reference": {
            "route": "attribute-mediated (direct route available; controlled render, exact ground truth)",
            "reference_valid": reference_valid,
            "evidence": "controlled render: hue is specified per object, so the description is a number for "
                        "these inputs (Section 5)"},
        "gate_3_reachability": {
            "reachable": reachable,
            "evidence": "intervention at this site moves the consumer's own output by %.3f relative "
                        "(effect size)" % eff},
        "gate_4_realization": {
            "operator_family": "O2 (unconstrained global linear, ridge)",
            "realization_residual": round(resid, 4), "threshold": REALIZATION_AT,
            "verdict": "pass" if realized else "fail",
            "outcome_transfer_companion": round(o2, 4),
            "win_rate_companion": round(win, 4),
            "companions": {"O1_orthogonal_transfer": round(o1, 4),
                           "O8_conditioned_transfer": (None if o8 != o8 else round(o8, 4))}},
        "composition": {"assessed": False, "verdict": "not-assessed"},
        "certificate": {
            "powered": powered, "reachable": reachable, "reference_valid": reference_valid,
            "realized": realized,
            "recommended_family": fam_recommended,
            "verdict": summary,
            "observed_transfer_companion": round(o2, 4),
            "compositionality": "not-assessed",
            "recommended_depth": site},
        "provenance": {"source_files": ["results/depth_map_summary.json",
                                        "results/estimator_control.json"],
                       "generated_by": "scripts/215_transportability_certificate.py"}
    }
    json.dump(SCHEMA, open(os.path.join(RES, "transportability_certificate_schema.json"), "w",
                           encoding="utf-8"), indent=1)
    json.dump(cert, open(os.path.join(RES, "transportability_certificate_example.json"), "w",
                         encoding="utf-8"), indent=1)
    print("wrote schema and example certificate")
    print(json.dumps(cert["certificate"], indent=1))


if __name__ == "__main__":
    main()
