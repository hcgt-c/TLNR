# -*- coding: utf-8 -*-
"""218_estimator_control.py — is the depth law a representational finding or an estimation artifact?

One protocol, five estimators, three disjoint splits. For every reachable site of every frozen pretrained
backbone (4 backbones x 4 depths x {hue, heat}, seed 0):

  fits and evaluates, on the SAME fit rows and the SAME held-out images,
    (a) O2_pub    : the published global linear operator, ridge with an ABSOLUTE lambda = 1e-3
                    (exactly what c136.fit_ridge does, and what the depth map reported),
    (b) ridge_tun : the same family and the same fit rows, lambda chosen on a THIRD split (validation)
                    by held-out FEATURE error, so no evaluation or downstream information leaks in,
    (c) rank1_ls  : a genuine rank-1 least-squares operator (transformation-aligned via the dominant
                    direction of the cross-covariance, power iteration),
    (d) rank1_rnd : rank-1 operators with random input directions, n_rand draws, median reported,
    (e) O1        : Procrustes (orthogonal) as published,
  and additionally rank1_mean, the rank-1 operator aligned with the mean displacement.

Per (site, estimator) it reports TOGETHER:
    single-step feature residual (relative),
    downstream T_F = 1 - mean||y_int - y_real|| / mean||y_noop - y_real|| on the held-out images,
    projection coefficient, win rate, top-1 agreement with the real route,
plus the estimation-limit diagnostic n_rows, C and C^2/n_rows.

Composition is NOT defined at a depth-map site under the existing protocol (one transformation parameter per
cell); where a family composes (heat), a feature-space composition defect is reported instead, and that is
stated in the output. Nothing is fabricated: a site that cannot be run is recorded as missing.

Usage
  smoke: python scripts/218_estimator_control.py --backbones resnet50 --families heat --depths 1,4 \
             --K 2000 --n_eval 8 --n_val 8
  full : python scripts/218_estimator_control.py \\
             --backbones resnet50,convnext,vitb16,dinov2b14 --families heat,hue --seed 0
  agg  : python scripts/218_estimator_control.py --aggregate
Outputs results/estimator_control.json, paper/figures/F35_estimator_control.png,
        paper/targets/TPAMI/reports/ESTIMATOR_CONTROL_REPORT.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", rp.MPLCONFIGDIR)
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
HEAT_P, HUE_P = 2.0, 90.0

c136 = None
H = None


def load_mods():
    global c136, H
    if c136 is None:
        def lm(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m
        c136 = lm("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
        H = lm("harness188", os.path.join(WORK, "scripts", "188_depth_map_harness.py"))
        H.c136 = c136


# ---------------------------------------------------------------- estimators
def solve_from_gram(G, Cxy, lam_abs=0.0, lam_rel=0.0):
    """W (row convention, X @ W.T ~= Y) from G = X^T X and Cxy = X^T Y."""
    C = G.shape[0]
    A = G.clone()
    if lam_abs:
        A = A + lam_abs * torch.eye(C, dtype=A.dtype)
    if lam_rel:
        A = A + lam_rel * torch.eye(C, dtype=A.dtype) * (torch.diagonal(G).mean() + 1e-12)
    return torch.linalg.solve(A, Cxy).T.float()


def rank1_from_direction(X, Y, u):
    """Best rank-1 operator W = v u^T for a FIXED unit input direction u (v by least squares)."""
    Xu = X @ u
    denom = float((Xu * Xu).sum()) + 1e-12
    v = (Y.T @ Xu) / denom
    return torch.outer(v, u).float()


def rank1_ls(X, Y, iters=25, seed=0):
    """Dominant direction u of X^T Y Y^T X by power iteration (transformation-aligned)."""
    C = X.shape[1]
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(C, generator=g)
    u = u / u.norm()
    for _ in range(iters):
        z1 = X @ u
        z2 = Y.T @ z1
        z3 = Y @ z2
        u = X.T @ z3
        n = u.norm()
        if float(n) < 1e-30:
            break
        u = u / n
    return rank1_from_direction(X, Y, u)


def rank1_mean(X, Y):
    d = (Y - X).mean(0)
    n = d.norm()
    u = d / (n + 1e-12)
    return rank1_from_direction(X, Y, u)


def rank1_random(X, Y, draws=3, seed=0):
    C = X.shape[1]
    g = torch.Generator().manual_seed(seed + 977)
    Ws = []
    for _ in range(draws):
        u = torch.randn(C, generator=g); u = u / u.norm()
        Ws.append(rank1_from_direction(X, Y, u))
    return Ws


# ---------------------------------------------------------------- metrics
def feat_metrics(W, Xe, Ye):
    pred = Xe @ W.T
    num = float((pred - Ye).norm(dim=1).mean())
    disp = float((Ye - Xe).norm(dim=1).mean())
    to_y = float((pred - Ye).norm(dim=1).mean())
    return {"feat_resid_over_displacement": num / (disp + 1e-12),
            "feat_err_to_real": to_y,
            "feat_displacement_norm": disp,
            "pred_norm_over_real_norm": float(pred.norm() / (Ye.norm() + 1e-12))}


def downstream(model, site, kind, W, x_eval, y_null, y_real):
    if hasattr(W, "apply4"):
        fn = lambda t, W=W: H._apply_callable(t, W.apply4, kind)
    else:
        fn = lambda t, W=W: H._apply_matrix(t, W.to(t.device), kind)
    y_i = H.logits_intervened(model, site, fn, x_eval)
    d_int = torch.norm(y_i - y_real, dim=1)
    d_null = torch.norm(y_null - y_real, dim=1)
    cos, proj, _ = c136.align_proj(y_i - y_null, y_real - y_null)
    return {"transfer": 1.0 - float(d_int.mean() / (d_null.mean() + 1e-12)),
            "transfer_squared": 1.0 - float(np.sqrt(float((d_int ** 2).mean()) /
                                                    (float((d_null ** 2).mean()) + 1e-12))),
            "win_rate": float((d_int < d_null).float().mean()),
            "proj_coef_agg": float(proj), "align_cos": float(cos),
            "top1_agree_real": float((c136.top1(y_i) == c136.top1(y_real)).float().mean()),
            "mean_d_int": float(d_int.mean()), "mean_d_noop": float(d_null.mean())}


def subsample_rows(X, K, seed):
    n = X.shape[0]
    if n <= K:
        return X
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(n, generator=g)[:K]
    return X[sel]


def rows_of(feat, idx):
    """(B,C,H,W) -> (B*H*W, C) for the given image indices."""
    f = feat[idx]
    B, C, Hh, Ww = f.shape
    return f.permute(0, 2, 3, 1).reshape(-1, C)


def published_values(backbone, family):
    p = os.path.join(WORK, "results", f"depth_map_{backbone}_{family}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    keys = ([f"{family}_{HEAT_P}", f"{family}_{int(HEAT_P)}"] if family == "heat"
            else [f"{family}_{HUE_P}", f"{family}_{int(HUE_P)}"])
    for key in keys:
        try:
            return d["results"][key]["seed"]["0"]["sites"]
        except KeyError:
            continue
    return {}


# ---------------------------------------------------------------- one (backbone, family)
def run_config(backbone, family, depths, seed, n_fit, n_eval, n_val, K, n_rand, out):
    load_mods()
    model, sites, names, kind = H.build_backbone(backbone)
    total = n_fit + n_eval + n_val
    crops = H.load_crops(total, seed=seed)
    x01 = H.to01(crops)
    x = H.norm(x01)
    xb01 = H.heat(x01, HEAT_P ** 2 / 2) if family == "heat" else H.hue_shift(x01, HUE_P)
    xb = H.norm(xb01)
    pub = published_values(backbone, family)

    t0 = time.time()
    Fb = H.capture(model, sites, x)
    Fr = H.capture(model, sites, xb)
    print(f"[{backbone}/{family}] captured {total} crops x {len(sites)} sites in {time.time()-t0:.0f}s", flush=True)

    fit_idx = list(range(0, n_fit))
    eval_idx = list(range(n_fit, n_fit + n_eval))
    val_idx = list(range(n_fit + n_eval, total))

    xe = x[eval_idx]
    y_null = H.logits_plain(model, xe)
    y_real = H.logits_plain(model, xb[eval_idx])

    for d in depths:
        i = d - 1
        site = sites[i]
        fb, fr = Fb[i], Fr[i]
        B, C, Hh, Ww = fb.shape
        n_rows_total = n_fit * Hh * Ww
        Xf = subsample_rows(rows_of(fb, fit_idx), K, seed)
        Yf = subsample_rows(rows_of(fr, fit_idx), K, seed)
        Xv = subsample_rows(rows_of(fb, val_idx), K, seed + 1)
        Yv = subsample_rows(rows_of(fr, val_idx), K, seed + 1)
        Xe = subsample_rows(rows_of(fb, eval_idx), min(20000, n_eval * Hh * Ww), seed + 2)
        Ye = subsample_rows(rows_of(fr, eval_idx), min(20000, n_eval * Hh * Ww), seed + 2)

        # Gram matrices in float32 (a double copy of 150k x 2048 would be 2.4 GB), cast after the product
        G = (Xf.T @ Xf).double()
        Cxy = (Xf.T @ Yf).double()
        Gv = (Xv.T @ Xv).double()
        Cxyv = (Xv.T @ Yv).double()

        rec = {"backbone": backbone, "family": family, "depth": d, "site": names[i],
               "feature_shape": [int(B), int(C), int(Hh), int(Ww)],
               "kind": kind, "seed": seed, "n_fit_images": n_fit, "n_eval_images": n_eval,
               "n_val_images": n_val, "K_rows_used": int(Xf.shape[0]),
               "n_rows_total_fit": int(n_rows_total),
               "channel_dim_C": int(C),
               "C2_over_K": float(C * C / max(1, Xf.shape[0])),
               "C2_over_n_rows_total": float(C * C / max(1, n_rows_total)),
               "effect_size_site": float(c136.rel_l2(fr[eval_idx].reshape(len(eval_idx), -1),
                                                     fb[eval_idx].reshape(len(eval_idx), -1))),
               "published_O2_transfer": (pub.get(names[i], {}) or {}).get("O2", {}).get("transfer"),
               "estimators": {}, "missing": None}

        # (a) published O2: absolute lambda 1e-3, identical rows to the published fit
        W_pub = solve_from_gram(G, Cxy, lam_abs=1e-3)
        # (b) tuned ridge: relative lambda selected on the validation split by feature error
        cand = {}
        for lam in LAMBDAS:
            W = solve_from_gram(G, Cxy, lam_rel=lam)
            pred = Xv @ W.T
            cand[lam] = float((pred - Yv).norm(dim=1).mean() / ((Yv - Xv).norm(dim=1).mean() + 1e-12))
        best_lam = min(cand, key=cand.get)
        W_tun = solve_from_gram(G, Cxy, lam_rel=best_lam)
        # (c) genuine rank-1 LS, (c') mean-displacement rank-1, (d) random rank-1
        W_r1 = rank1_ls(Xf, Yf)
        W_r1m = rank1_mean(Xf, Yf)
        W_rnd = rank1_random(Xf, Yf, draws=n_rand, seed=seed)
        # (e) orthogonal
        W_o1 = c136.fit_procrustes(Xf, Yf)
        rec["lambda_grid_val_feat_err"] = {str(k): v for k, v in cand.items()}
        rec["best_lambda_rel"] = best_lam

        est = {"O2_pub": W_pub, "ridge_tun": W_tun, "rank1_ls": W_r1,
               "rank1_mean": W_r1m, "O1": W_o1}
        for name, W in est.items():
            rec["estimators"][name] = dict(feat_metrics(W, Xe, Ye),
                                           **downstream(model, site, kind, W, xe, y_null, y_real))
        # random rank-1: median over draws on every reported quantity
        rnd = [dict(feat_metrics(W, Xe, Ye), **downstream(model, site, kind, W, xe, y_null, y_real))
               for W in W_rnd]
        rec["estimators"]["rank1_rnd"] = {k: float(np.median([r[k] for r in rnd])) for k in rnd[0]}
        rec["estimators"]["rank1_rnd"]["draws"] = len(rnd)

        rec["composition"] = ("not defined at a depth-map site under the existing protocol "
                              "(one transformation parameter per cell); see the separate composition study")

        key = f"{backbone}|{family}|depth{d}"
        out["cells"][key] = rec
        json.dump(out, open(os.path.join(WORK, "results", "estimator_control.json"), "w"), indent=1)
        line = " ".join(f"{n}:T{v['transfer']:+.3f}" for n, v in rec["estimators"].items())
        print(f"  {key:34s} C={C:4d} K={Xf.shape[0]:6d} pubO2="
              f"{rec['published_O2_transfer'] if rec['published_O2_transfer'] is None else round(rec['published_O2_transfer'],3)} "
              f"lam*={best_lam} {line}", flush=True)
        del G, Cxy, Gv, Cxyv, Xf, Yf, Xv, Yv, Xe, Ye
        import gc; gc.collect()
    del model, Fb, Fr
    return out


# ---------------------------------------------------------------- aggregation
def spearman(x, y):
    """Tie-averaged Spearman. Relative depth takes only four tied values, so the
    double-argsort (ordinal) form used earlier is not the standard statistic."""
    x = np.asarray(x, float); y = np.asarray(y, float)

    def avg_rank(v):
        order = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return r

    rx = avg_rank(x); ry = avg_rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


def backfill_published(out):
    """Fill published_O2_transfer and the reachability flag where the running process used a stale key."""
    for k, v in out["cells"].items():
        if not isinstance(v, dict) or "backbone" not in v:
            continue
        pub = published_values(v["backbone"], v["family"])
        site = pub.get(v["site"], {}) or {}
        if v.get("published_O2_transfer") is None:
            got = (site.get("O2", {}) or {}).get("transfer")
            if got is not None:
                v["published_O2_transfer"] = got
        if "reachable" not in v or v.get("reachable") is None:
            if site.get("intervention_reaches_consumer") is not None:
                v["reachable"] = bool(site["intervention_reaches_consumer"])
            elif site.get("random_operator_logit_change") is not None:
                v["reachable"] = bool(site["random_operator_logit_change"] > 1e-3)
            else:
                o6 = site.get("O6", {}) or {}
                if o6.get("random_operator_logit_change") is not None:
                    v["reachable"] = bool(o6["random_operator_logit_change"] > 1e-3)
                else:
                    v["reachable"] = None
    return out


def aggregate():
    load_mods()
    path = os.path.join(WORK, "results", "estimator_control.json")
    out = json.load(open(path, encoding="utf-8"))
    out = backfill_published(out)
    cells = out["cells"]
    names = ["O2_pub", "ridge_tun", "rank1_ls", "rank1_mean", "rank1_rnd", "O1"]
    agg = {"n_cells": len(cells), "by_family": {},
           "note": "relative depth = depth/4; pooled over backbones; causally unreachable sites excluded"}
    usable = {k: v for k, v in cells.items()
              if isinstance(v, dict) and "estimators" in v and v.get("reachable") is not False}
    agg["n_unreachable_excluded"] = len(cells) - len(usable)
    for fam in sorted({v["family"] for v in usable.values()}):
        rec = {}
        for name in names:
            xs, ys, deep = [], [], []
            for v in usable.values():
                if v["family"] != fam:
                    continue
                t = v["estimators"][name]["transfer"]
                xs.append(v["depth"] / 4.0); ys.append(t)
                if v["depth"] == 4:
                    deep.append(t)
            rec[name] = {"rho_vs_rel_depth": spearman(xs, ys), "n": len(xs),
                         "median_deepest_transfer": float(np.median(deep)) if deep else None,
                         "mean_transfer": float(np.mean(ys))}
        # per backbone trend for the headline estimators
        per_b = {}
        for b in sorted({v["backbone"] for v in usable.values()}):
            sub = {n: [] for n in names}
            for v in usable.values():
                if v["family"] != fam or v["backbone"] != b:
                    continue
                for n in names:
                    sub[n].append((v["depth"] / 4.0, v["estimators"][n]["transfer"]))
            per_b[b] = {n: spearman([a for a, _ in sub[n]], [t for _, t in sub[n]]) for n in names}
        agg["by_family"][fam] = {"pooled": rec, "per_backbone_rho": per_b}
    out["aggregate"] = agg
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(agg, indent=1))
    return out


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = out["cells"]
    usable = {k: v for k, v in cells.items()
              if isinstance(v, dict) and "estimators" in v and v.get("reachable") is not False}
    fams = sorted({v["family"] for v in usable.values()})
    names = ["O2_pub", "ridge_tun", "rank1_ls", "rank1_rnd", "O1"]
    lbl = {"O2_pub": "published O2 ($\\lambda$=1e-3 abs)", "ridge_tun": "tuned ridge ($\\lambda$ on val)",
           "rank1_ls": "rank-1 LS (aligned)", "rank1_rnd": "rank-1 random (median)",
           "rank1_mean": "rank-1 mean-displacement", "O1": "O1 orthogonal"}
    fig, axes = plt.subplots(2, len(fams), figsize=(6.2 * len(fams), 8.2), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for n in names:
            xs, ys = [], []
            for v in usable.values():
                if v["family"] == fam:
                    xs.append(v["depth"] / 4.0); ys.append(v["estimators"][n]["transfer"])
            if xs:
                ax.plot(xs, ys, "o-", label=lbl[n], ms=4, lw=1.2, alpha=0.9)
        agg = out.get("aggregate", {}).get("by_family", {}).get(fam, {}).get("pooled", {})
        txt = "\n".join(f"{n}: rho={agg[n]['rho_vs_rel_depth']:+.2f}, deep={agg[n]['median_deepest_transfer']:+.2f}"
                        for n in names if n in agg)
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(fc="white", ec="0.7", alpha=0.8))
        ax.set_title(f"{fam}: T_F vs relative depth (pooled over backbones)")
        ax.set_xlabel("relative depth"); ax.set_ylabel("downstream transfer $T_F$")
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
    # bottom row: the estimation-limit diagnostic C^2/K (constraints per fitted parameter), log scale
    bbs = sorted({v["backbone"] for v in usable.values()})
    cmap = {b: c for b, c in zip(bbs, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}
    for ax, fam in zip(axes[1], fams):
        for b in bbs:
            xs, ys = [], []
            for v in usable.values():
                if v["family"] == fam and v["backbone"] == b:
                    xs.append(v["depth"] / 4.0); ys.append(v["C2_over_K"])
            if xs:
                ax.plot(xs, ys, "o-", color=cmap[b], label=b, ms=4, lw=1.2)
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"{fam}: estimation limit $C^2/K$ (higher = fewer rows per parameter)")
        ax.set_xlabel("relative depth"); ax.set_ylabel("$C^2/K$ (log scale)")
        ax.grid(alpha=0.25, which="both")
        ymax = max([v["C2_over_K"] for v in usable.values() if v["family"] == fam] or [1.0])
        ax.annotate(f"max {ymax:.0f}", xy=(1.0, ymax), xytext=(0.55, ymax * 1.05),
                    fontsize=7, ha="right", arrowprops=dict(arrowstyle="->", lw=0.6))
    axes[1][0].legend(fontsize=7, loc="upper left", title="backbone")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    figp = os.path.join(WORK, "paper", "figures", "F35_estimator_control.png")
    os.makedirs(os.path.dirname(figp), exist_ok=True)
    fig.savefig(figp, dpi=200)
    print("saved", figp)
    return figp


# ---------------------------------------------------------------- report
def report(out):
    agg = out.get("aggregate") or aggregate()
    cells = out["cells"]
    names = ["O2_pub", "ridge_tun", "rank1_ls", "rank1_mean", "rank1_rnd", "O1"]
    lines = ["# Estimator-control report: is the depth law representational or an estimation artifact?",
             "",
             "> Generated by `scripts/218_estimator_control.py` from `results/estimator_control.json`. "
             "Inference-only: frozen pretrained backbones on COCO train2017 instance crops, no training, no new data.",
             "",
             "## Protocol recovered from the published harness (188/198/200)",
             "",
             "- Splits (seed 0, 350 crops): fit = images 0-199 **exactly the published fit set**; "
             "held-out evaluation = images 200-249 (a fixed subset of the published 200-299 held-out set); "
             "validation = images 300-349 (used only to choose the ridge strength).",
             "- Consumer: the backbone's own frozen ImageNet head logits; no labels. "
             "`T_F = 1 - mean||y_int - y_real|| / mean||y_noop - y_real||` on the held-out images.",
             "- O2 published = `c136.fit_ridge` with an **absolute** lambda = 1e-3 on raw activations.",
             "- Deviation, uniform across estimators: the fit rows are subsampled to K (see table) rather than "
             "150,000, for CPU feasibility; the published values are quoted alongside from "
             "`results/depth_map_*.json`. The comparison between estimators is internally consistent.",
             "- Composition is not defined at a depth-map site under the existing protocol (one transformation "
             "parameter per cell); it is measured separately in the composition study and is not claimed here.",
             "",
             "## Per-site table",
             "",
             "| cell | C | K | C^2/K | reach | published O2 | O2_pub | ridge_tun (lam*) | rank1_ls | rank1_mean | rank1_rnd | O1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(cells):
        v = cells[k]
        if not isinstance(v, dict) or "estimators" not in v:
            lines.append(f"| {k} | — | — | — | — | — | MISSING: {v.get('missing') if isinstance(v, dict) else 'n/a'} | | | | | |")
            continue
        e = v["estimators"]
        pub = v.get("published_O2_transfer")
        reach = {True: "yes", False: "no", None: "?"}.get(v.get("reachable"), "?")
        lines.append(f"| {k} | {v['channel_dim_C']} | {v['K_rows_used']} | {v['C2_over_K']:.0f} | {reach} | "
                     f"{'n/a' if pub is None else f'{pub:+.3f}'} | {e['O2_pub']['transfer']:+.3f} | "
                     f"{e['ridge_tun']['transfer']:+.3f} ({v['best_lambda_rel']}) | "
                     f"{e['rank1_ls']['transfer']:+.3f} | {e['rank1_mean']['transfer']:+.3f} | "
                     f"{e['rank1_rnd']['transfer']:+.3f} | {e['O1']['transfer']:+.3f} |")
    lines += ["", "## Pooled trend per estimator (relative depth = depth/4)",
              "", "| family | estimator | Spearman rho vs depth | median deepest-site T_F | mean T_F |",
              "|---|---|---|---|---|"]
    for fam, a in agg["by_family"].items():
        for n in names:
            r = a["pooled"].get(n)
            if r:
                deep = r["median_deepest_transfer"]
                deep_s = "n/a" if deep is None else f"{deep:+.3f}"
                lines.append(f"| {fam} | {n} | {r['rho_vs_rel_depth']:+.3f} | {deep_s} | "
                             f"{r['mean_transfer']:+.3f} |")
    # verdict
    lines += ["", "## Verdict logic", "",
              "Representational reading holds if the depth decline survives under the best estimator "
              "(tuned ridge and rank-1 LS): rho stays clearly negative and the deepest-site median transfer "
              "stays low. Estimation-artifact reading holds if a low-complexity estimator recovers the deep "
              "sites and the trend flattens or reverses.", ""]
    for fam, a in agg["by_family"].items():
        p = a["pooled"]
        lines.append(f"- **{fam}**: published O2 rho = {p['O2_pub']['rho_vs_rel_depth']:+.3f} "
                     f"(deep median {p['O2_pub']['median_deepest_transfer']:+.3f}); "
                     f"tuned ridge rho = {p['ridge_tun']['rho_vs_rel_depth']:+.3f} "
                     f"(deep median {p['ridge_tun']['median_deepest_transfer']:+.3f}); "
                     f"rank-1 LS rho = {p['rank1_ls']['rho_vs_rel_depth']:+.3f} "
                     f"(deep median {p['rank1_ls']['median_deepest_transfer']:+.3f}).")
    lines += ["",
              "## Manuscript wording under each outcome",
              "",
              "*If representational:* keep the depth law but replace the ceiling language with "
              "\"under the published fitting protocol, and it survives at matched regularisation and rank\"; "
              "report tuned-ridge and rank-1 controls in the main figure next to O2.",
              "",
              "*If estimation artifact:* the headline changes to \"common fitting protocols mistake estimation "
              "difficulty for representational failure\"; the abstract's depth sentence, the §8 title and the "
              "ceiling proposition must be rewritten, and the practical recommendation becomes low-rank or "
              "λ-tuned estimators rather than a claim about the representation.",
              ""]
    p = os.path.join(WORK, "paper", "targets", "TPAMI", "reports", "ESTIMATOR_CONTROL_REPORT.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print("saved", p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet50,convnext,vitb16,dinov2b14")
    ap.add_argument("--families", default="heat,hue")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--K", type=int, default=150000)
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    load_mods()
    if a.aggregate or a.figure or a.report:
        out = json.load(open(os.path.join(WORK, "results", "estimator_control.json"), encoding="utf-8"))
        if a.aggregate:
            out = aggregate()
        if a.figure:
            figure(out if "aggregate" in out else aggregate())
        if a.report:
            report(out if "aggregate" in out else aggregate())
        return 0
    path = os.path.join(WORK, "results", "estimator_control.json")
    out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {
        "protocol": "218 estimator control", "cells": {}}
    for b in a.backbones.split(","):
        for fam in a.families.split(","):
            depths = [int(d) for d in a.depths.split(",")]
            try:
                out = run_config(b.strip(), fam.strip(), depths, a.seed, a.n_fit, a.n_eval,
                                 a.n_val, a.K, a.n_rand, out)
            except Exception as ex:
                key = f"{b.strip()}|{fam.strip()}"
                out["cells"][key] = {"missing": f"{type(ex).__name__}: {ex}"}
                json.dump(out, open(path, "w"), indent=1)
                print(f"  !! {key} FAILED: {type(ex).__name__}: {ex}", flush=True)
    arr = aggregate()
    json.dump(arr, open(path, "w"), indent=1)
    figure(arr)
    report(arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
