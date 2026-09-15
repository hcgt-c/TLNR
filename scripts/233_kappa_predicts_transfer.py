# -*- coding: utf-8 -*-
"""233_kappa_predicts_transfer.py — does the kappa law PREDICT transport on real, nonlinear backbones?

The paper's kappa law, 1 - T_F^2 = kappa / sqrt(1 + kappa^2), was verified in two regimes:
  * synthetic sites with known geometry (closed form tracked to 0.009, script 206);
  * trained LINEAR networks (rank correlation rho(kappa, T_F) = -0.97, script 207).
On real nonlinear backbones only the depth curve of T_F was measured. The law was therefore exact
where we control the geometry and merely correlated where we do not. This script closes that gap.

WHAT IS PREDICTED, AND FROM WHAT
  The measured target is `O2_transfer_mean` from `results/depth_map_summary.json`: the DOWNSTREAM,
  logit-level transfer of the fitted global linear operator on the HELD-OUT images. It is not
  recomputed here; it is read from the audited summary so that the prediction is out of sample in
  both the fitting and the evaluation sense.

  The predictor is computed from the FIT split only, in feature space, and uses NO free parameter:
    M                the global linear action z -> D, fitted by ridge on the fit rows (the O2 family);
    A - I = P M P    its action within the retained subspace R;
    C     = P_perp M P   its mixing OUT of R;
    kappa = ||C||_F / ||A - I||_F;
    T_hat = sqrt(max(0, 1 - kappa / sqrt(1 + kappa^2))).
  No constant is fitted to T_F. The prediction either lands on the measured value or it does not.

THREE PRE-DECLARED VARIANTS (all reported; none selected post hoc)
  R_pca    top-r right singular directions of the fit activations: what the representation keeps,
           Euclidean norm.
  R_cons   top-r right singular directions of the consumer probe (pooled site -> consumer output):
           the subspace the CONSUMER reads, which is the theory's own P, Euclidean norm.
  R_consJ  the same retained subspace, but both norms taken in the consumer-induced pullback
           seminorm ||J . ||_F of Section 3.6 -- the metric the criterion actually measures.

Baselines that must lose for the law to be worth stating: the site effect size, the effective rank of
the activations, and the depth index. Every predictor is scored twice: across all cells, and as a
partial correlation controlling for depth, because within one depth curve every monotone quantity
correlates perfectly and the within-curve correlation therefore tests nothing.

Usage
  python scripts/233_kappa_predicts_transfer.py                     # all four backbones, three configs
  python scripts/233_kappa_predicts_transfer.py --backbones resnet50 --configs hue_90.0
Outputs results/kappa_predicts_transfer.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT
RES = os.path.join(WORK, "results")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


h = load(os.path.join(WORK, "scripts", "188_depth_map_harness.py"), "h188")
# The depth grid was run on the GPU. This script is a CPU job on purpose: the user's training owns the
# card. Overriding the module global makes every helper below (build_backbone, to01, norm, capture)
# allocate on CPU without touching 188 itself.
h.DEV = "cpu"
c136 = h.c136

CONFIGS = {
    "hue_90.0": ("hue", 90.0),
    "heat_2.0": ("heat", 2.0),
    "heat_1.0": ("heat", 1.0),
}
R_GRID = [4, 8, 16, 32, 64]


def top_basis(gram, r):
    """Top-r eigenvectors of a symmetric PSD matrix, as a (d, r) orthonormal basis."""
    r = min(r, gram.shape[0])
    w, V = torch.linalg.eigh(gram.double())
    return V[:, -r:].flip(-1).float()


def kappa_from(M, V, J=None):
    """kappa = ||P_perp M P||_F / ||P M P||_F for the orthonormal basis V of the retained subspace.

    With J supplied, both norms are taken in the consumer-induced (pullback) seminorm ||J . ||_F
    of Section 3.6, which is the metric the criterion actually measures. Reporting the Euclidean and
    the pullback versions side by side is deliberate: the law is derived in a norm, and which norm
    the consumer induces is exactly the question Section 3.6 raises.
    """
    MP = M @ V                       # (C, r)
    PMP = V.T @ MP                   # (r, r)
    perp = MP - V @ PMP              # (C, r): P_perp M P
    within = V @ PMP                 # (C, r): P M P
    if J is not None:
        perp = J @ perp
        within = J @ within
    num = float(perp.norm())
    den = float(within.norm())
    return num / (den + 1e-12), den, num


def predict(kappa):
    return float(np.sqrt(max(0.0, 1.0 - kappa / np.sqrt(1.0 + kappa * kappa))))


def effective_rank(svals):
    s = svals.clamp_min(0)
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet50,convnext,vitb16,dinov2b14")
    ap.add_argument("--configs", default="hue_90.0,heat_2.0,heat_1.0")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_held", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row_cap", type=int, default=150_000)
    a = ap.parse_args()

    assert h.DEV == "cpu", "this script must stay on CPU"
    summary = json.load(open(os.path.join(RES, "depth_map_summary.json")))
    out = {
        "script": "233_kappa_predicts_transfer.py",
        "device": "cpu",
        "target": ("O2_transfer_mean from results/depth_map_summary.json: downstream logit-level transfer "
                   "of the fitted global linear operator, measured on HELD-OUT images"),
        "predictor": "T_hat = sqrt(1 - kappa/sqrt(1+kappa^2)), kappa = ||P_perp M P||_F / ||P M P||_F, "
                     "M fitted by ridge on the FIT split only; no parameter is fitted to T_F",
        "n_fit": a.n_fit, "n_held": a.n_held, "seed": a.seed, "row_cap": a.row_cap,
        "r_grid": R_GRID,
        "cells": {},
    }

    backbones = [b for b in a.backbones.split(",") if b]
    configs = [c for c in a.configs.split(",") if c]
    t0 = time.time()

    for bb in backbones:
        model, sites, names, kind = h.build_backbone(bb)
        crops = h.load_crops(a.n_fit + a.n_held, seed=a.seed)
        x01 = h.to01(crops)
        x = h.norm(x01)
        y = h.logits_plain(model, x)                  # consumer output; 1000 logits, or 768-d CLS for DINOv2
        Fb = h.capture(model, sites, x)               # base features at all depths, one pass
        print(f"[{bb}] base captured, consumer dim {y.shape[1]} ({time.time()-t0:.0f}s)")

        for cfg in configs:
            fam, p = CONFIGS[cfg]
            key = f"{bb}|{cfg}"
            if key not in summary:
                print(f"  skip {key}: not in depth_map_summary.json")
                continue
            xb = h.hue_shift(x01, p) if fam == "hue" else h.heat(x01, p ** 2 / 2)
            xb = h.norm(xb)
            Fr = h.capture(model, sites, xb)
            out["cells"].setdefault(key, {"backbone": bb, "config": cfg, "family": fam, "param": p,
                                          "sites": {}})
            for d in range(1, len(names) + 1):
                i = d - 1
                site = names[i]
                srow = summary[key]["sites"].get(site, {})
                if "O2_transfer_mean" not in srow:
                    print(f"  skip {key}/{site}: no O2_transfer_mean")
                    continue
                measured = float(srow["O2_transfer_mean"])
                reachable = bool(srow.get("reachable", True))

                fb, fr = Fb[i], Fr[i]
                B, C, Hh, Ww = fb.shape
                rows_b = fb.permute(0, 2, 3, 1).reshape(-1, C)
                rows_r = fr.permute(0, 2, 3, 1).reshape(-1, C)
                n_fit_rows = a.n_fit * Hh * Ww
                K = min(n_fit_rows, a.row_cap)
                g = torch.Generator().manual_seed(a.seed)
                sel = torch.randperm(n_fit_rows, generator=g)[:K]
                Xf, Yf = rows_b[:n_fit_rows][sel], rows_r[:n_fit_rows][sel]
                M = c136.fit_ridge(Xf, Yf)             # D ~ M z ; the same family as O2

                # ---- retained subspace, definition 1: representational (PCA of the fit activations)
                gram = Xf.T @ Xf
                svals = torch.linalg.svdvals(gram.double()).float()
                erank = effective_rank(svals)

                # ---- retained subspace, definition 2: consumer-read (right singular space of the probe)
                pooled = fb[:a.n_fit].mean(dim=(2, 3))            # (n_fit, C)
                Wp = c136.fit_ridge(pooled, y[:a.n_fit], lam=1e-3)  # (m, C) pooled -> consumer output
                _, _, Vhp = torch.linalg.svd(Wp.double(), full_matrices=False)
                Vhp = Vhp.float()

                rec = {"depth": d, "feature_shape": [int(B), int(C), int(Hh), int(Ww)],
                       "measured_O2_transfer": measured, "reachable": reachable,
                       "measured_O2_transfer_squared": srow.get("O2_transfer_squared_mean"),
                       "effective_rank_activations": erank,
                       "effect_size_site": srow.get("effect_size_site"),
                       "kappa": {}, "T_hat": {}}
                for r in R_GRID:
                    variants = (("pca", top_basis(gram, r), None),
                                ("cons", Vhp[:min(r, Vhp.shape[0])].T, None),
                                ("consJ", Vhp[:min(r, Vhp.shape[0])].T, Wp))
                    for tag, V, J in variants:
                        k, den, num = kappa_from(M, V, J)
                        rec["kappa"][f"{tag}_r{r}"] = k
                        rec["T_hat"][f"{tag}_r{r}"] = predict(k)
                out["cells"][key]["sites"][site] = rec
                best = " ".join(f"{t}_r{r}:k={rec['kappa'][f'{t}_r{r}']:.2f}/T={rec['T_hat'][f'{t}_r{r}']:.2f}"
                                for t in ("pca", "cons", "consJ") for r in (8, 32))
                print(f"  {key} {site:10s} T_F={measured:+.3f} erank={erank:5.1f} {best}")
            json.dump(out, open(os.path.join(RES, "kappa_predicts_transfer.json"), "w"), indent=1)

    # ---------------- aggregate: does the law predict, and do the naive baselines?
    # Within a single depth curve every monotone quantity correlates perfectly (the smoke test showed
    # effect size, effective rank and kappa all at rho = -1.00 on one backbone), so the test is run
    # ACROSS cells -- four backbones x three configurations x depths -- and each predictor is also
    # scored against the obvious confound, the depth index itself.
    cells = []
    for key, cv in out["cells"].items():
        for site, rec in cv["sites"].items():
            if rec["reachable"]:
                cells.append((key, site, rec))
    meas = np.array([rec["measured_O2_transfer"] for _, _, rec in cells])
    depth = np.array([rec["depth"] for _, _, rec in cells], dtype=float)
    out["aggregate"] = {
        "n_cells": int(len(cells)),
        "n_backbones": len({k.split("|")[0] for k, _, _ in cells}),
        "n_configs": len({k.split("|")[1] for k, _, _ in cells}),
    }
    for r in R_GRID:
        for tag in ("pca", "cons", "consJ"):
            pred = np.array([rec["T_hat"][f"{tag}_r{r}"] for _, _, rec in cells])
            kap = np.array([rec["kappa"][f"{tag}_r{r}"] for _, _, rec in cells])
            out["aggregate"][f"{tag}_r{r}"] = {
                "mae": float(np.abs(pred - meas).mean()),
                "rmse": float(np.sqrt(((pred - meas) ** 2).mean())),
                "bias_mean_pred_minus_meas": float((pred - meas).mean()),
                "pearson": _safe(np.corrcoef(pred, meas)[0, 1], len(cells)),
                "spearman_kappa_vs_TF": _safe(_spearman(kap, meas), len(cells)),
                "partial_spearman_kappa_TF_given_depth": _partial_spearman(kap, meas, depth),
            }
    for tag, getter in (("effect_size_site", lambda rec: rec.get("effect_size_site")),
                        ("effective_rank", lambda rec: rec.get("effective_rank_activations"))):
        v = np.array([getter(rec) for _, _, rec in cells], dtype=float)
        ok = np.isfinite(v)
        out["aggregate"][f"baseline_{tag}"] = {
            "n_cells": int(ok.sum()),
            "spearman_vs_TF": _safe(_spearman(v[ok], meas[ok]), int(ok.sum())),
            "partial_spearman_given_depth": _partial_spearman(v[ok], meas[ok], depth[ok])}
    out["aggregate"]["baseline_depth_index"] = {
        "spearman_vs_TF": _safe(_spearman(depth, meas), len(cells))}
    json.dump(out, open(os.path.join(RES, "kappa_predicts_transfer.json"), "w"), indent=1)

    print(f"\n=== aggregate over {len(cells)} reachable cells "
          f"({out['aggregate']['n_backbones']} backbones x {out['aggregate']['n_configs']} configs) ===")
    for k, v in out["aggregate"].items():
        if not isinstance(v, dict):
            continue
        if "mae" in v:
            print(f"  {k:10s} MAE {v['mae']:.3f}  bias {v['bias_mean_pred_minus_meas']:+.3f}  "
                  f"pearson {v['pearson']:+.3f}  rho(kappa,T_F) {v['spearman_kappa_vs_TF']:+.3f}  "
                  f"partial|depth {v['partial_spearman_kappa_TF_given_depth']:+.3f}")
        elif "spearman_vs_TF" in v:
            ps = v.get("partial_spearman_given_depth")
            print(f"  {k:24s} rho vs T_F {v['spearman_vs_TF']:+.3f}"
                  + (f"  partial|depth {ps:+.3f}" if ps is not None else ""))
    print(f"done in {time.time()-t0:.0f}s -> results/kappa_predicts_transfer.json")
    return 0


def _safe(x, n):
    return float(x) if (n > 2 and np.isfinite(x)) else None


def _partial_spearman(x, y, z):
    """Spearman of x and y after regressing both on the rank of the control z."""
    from scipy.stats import rankdata
    if len(x) < 4:
        return None
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rz = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(rz, rx, rcond=None)[0]
    by = np.linalg.lstsq(rz, ry, rcond=None)[0]
    ex, ey = rx - rz @ bx, ry - rz @ by
    return _safe(np.corrcoef(ex, ey)[0, 1], len(x))


def _spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).statistic


if __name__ == "__main__":
    raise SystemExit(main())
