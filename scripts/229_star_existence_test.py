# -*- coding: utf-8 -*-
"""229_star_existence_test.py — direct test of the existence condition (star).

(star):  F(z1) = F(z2)  =>  F(g_tau z1) = F(g_tau z2),   z = f_l(x),  F = tail from the site.

Because the tail composed with the site equals the network, F(z) = logits(x) for a frozen backbone whose
site activation determines the rest of the forward pass. So the consumer-level test is

    logits(x1) = logits(x2)   =>   logits(tau x1) = logits(tau x2),

i.e. two inputs the consumer maps to the same output must stay equal after the same physical transformation.
We discretise the continuous equality by a tolerance quantile q chosen on the BASE pair-distance
distribution, and report the fraction of base-equal pairs that remain equal after the transformation
(preservation), against the random-pair baseline q.

We also run the representation-level version at each depth on the site's spatially pooled activation:
    z1 = z2  =>  g_tau z1 = g_tau z2.

Design choices for honesty:
  * one forward pass per (image set) captures every depth via hooks; base pass shared across families;
  * tolerance is a quantile, so the random-pair preservation rate is q by construction (the baseline);
  * effect size is reported beside every number (a low-power transformation cannot be read);
  * CPU-only and thread-limited: the box is running someone else's training and the GPU is busy.

Usage:
  OMP_NUM_THREADS=4 nice -n 15 python scripts/229_star_existence_test.py --backbones resnet50,vitb16
Outputs results/star_existence_test.json  (merged across backbones)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT


def load188():
    spec = importlib.util.spec_from_file_location("h188", os.path.join(WORK, "scripts", "188_depth_map_harness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def forward_all(h, model, sites, x, bs=16):
    """One pass: returns (logits, {site_index: pooled feature (N, C)}) with pooled = spatial mean."""
    buf = {k: [] for k in range(len(sites))}
    hs = []
    for k, layer in enumerate(sites):
        def mk(k):
            def hook(m, i, o):
                g = h.to_grid(o, None).detach().float()
                buf[k].append(g.mean(dim=(2, 3)).cpu())      # pooled over space
            return hook
        hs.append(layer.register_forward_hook(mk(k)))
    outs = []
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                o = model(x[s:s + bs])
                outs.append((o[0] if isinstance(o, (tuple, list)) else o).detach().float().cpu())
    finally:
        for hh in hs:
            hh.remove()
    Z = {k: torch.cat(v, 0).numpy() for k, v in buf.items()}
    if Z[0].shape[0] != x.shape[0]:
        raise RuntimeError(f"captured {Z[0].shape[0]} activations for {x.shape[0]} images")
    return torch.cat(outs).numpy(), Z


def pair_stats(Y0, Y1, q):
    """Preservation of consumer/site equivalence under the transformation.

    Returns the fraction of base-equal pairs (base distance <= q-quantile) whose transformed distance is
    also <= the q-quantile of the transformed distances, plus the median-distance ratio and the rank
    correlation between the two distance matrices.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    d0 = pdist(Y0)
    d1 = pdist(Y1)
    e0 = float(np.quantile(d0, q))
    e1 = float(np.quantile(d1, q))
    same = d0 <= e0
    keep = d1[same] <= e1
    out = {
        "q": q,
        "eps_base": e0,
        "eps_transformed": e1,
        "n_base_equal_pairs": int(same.sum()),
        "preservation_rate": float(keep.mean()) if same.sum() else float("nan"),
        "baseline_rate": float(q),
        "median_transformed_dist_base_equal": float(np.median(d1[same])) if same.sum() else float("nan"),
        "median_transformed_dist_all": float(np.median(d1)),
        "median_ratio": float(np.median(d1[same]) / (np.median(d1) + 1e-12)) if same.sum() else float("nan"),
        "spearman_base_vs_transformed_dist": float(spearmanr(d0, d1).statistic),
    }
    return out


def noise_set(h, x01, sigma, seed):
    """Additive Gaussian pixel noise at level sigma, clipped to the valid image range."""
    g = torch.Generator().manual_seed(int(seed))
    n = torch.randn(x01.shape, generator=g)
    return h.norm(torch.clamp(x01 + sigma * n, 0.0, 1.0))


def match_noise(h, model, sites, x01, x, y0, target_eff, seed=0, evals=5, tol=0.04):
    """Calibrate the noise level so its logit effect size matches `target_eff`.

    The transformation is a *driven control* in the paper's sense: it moves the consumer by the same amount
    as the real transformation, but it is not a physical transformation of the scene, so it has no reason to
    respect the consumer's equivalence classes. A real transformation that preserves classes better than this
    control does so for a reason beyond "it is a smooth, weak perturbation".

    The effect size is monotone in sigma, so we evaluate a small ladder, use a secant step, and keep the
    evaluation closest to the target; the whole history is stored so the calibration is auditable.
    """
    hist = []

    def ev(sig, tag):
        xc = noise_set(h, x01, sig, seed)
        yc, Zc = forward_all(h, model, sites, xc)
        e = float(np.linalg.norm(yc - y0) / (np.linalg.norm(y0) + 1e-12))
        hist.append({"sigma": float(sig), "effect": e})
        return sig, e, yc, Zc

    # The noise response is steep: on [0,1] images, sigma ~ 3e-2 already moves the ImageNet head by an
    # effect size of order 1. Two small probes fix the (near-linear) slope through the origin, then the
    # secant steps refine it. Five evaluations at most; the history is stored.
    pts = [ev(0.004, 1), ev(0.016, 2)]
    s1, e1, _, _ = pts[0]
    s2, e2, _, _ = pts[1]
    k = (s1 * e1 + s2 * e2) / (s1 * s1 + s2 * s2 + 1e-12)
    cand = float(np.clip(target_eff / max(k, 1e-9), 1e-5, 0.5))
    pts.append(ev(cand, 3))
    for it in range(evals - 3):
        near = sorted(pts, key=lambda p: abs(p[1] - target_eff))[:2]
        (sa, ea, _, _), (sb, eb, _, _) = near
        if abs(ea - eb) < 1e-9:
            break
        s_new = float(np.clip(sa + (target_eff - ea) * (sb - sa) / (eb - ea), 1e-5, 0.5))
        if all(abs(s_new - p[0]) < 1e-7 for p in pts):
            break
        pts.append(ev(s_new, 4 + it))
    best = min(pts, key=lambda p: abs(p[1] - target_eff))
    return best[0], best[1], best[2], best[3], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet50,vitb16")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hue", default="90")
    ap.add_argument("--heat", default="2.0")
    ap.add_argument("--q", default="0.01,0.05")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--controls", action="store_true",
                    help="run only the matched-effect-size noise controls (targets read from the existing JSON)")
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    torch.set_num_threads(a.threads)
    h = load188()
    h.DEV = "cpu"                      # never touch the busy GPU; the box is training
    DEV = "cpu"
    h.DEV = DEV

    hues = [float(v) for v in a.hue.split(",")]
    heats = [float(v) for v in a.heat.split(",")]
    qs = [float(v) for v in a.q.split(",")]

    out_path = os.path.join(WORK, "results", "star_existence_test.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {
        "protocol": "(star) direct test: base-equal pairs must stay equal after the transformation; "
                    "tolerance = quantile q of the base pair-distance distribution; baseline = q",
        "device": "cpu", "n_crops": a.n, "seed": a.seed, "results": {}}

    for bb in a.backbones.split(","):
        t0 = time.time()
        model, sites, names, kind = h.build_backbone(bb)
        crops = h.load_crops(a.n, seed=a.seed)
        x01 = h.to01(crops)
        x = h.norm(x01)
        y0, Z0 = forward_all(h, model, sites, x)
        rec = res["results"].get(bb) or {"kind": kind, "site_names": names, "families": {}}
        rec.setdefault("kind", kind); rec.setdefault("site_names", names); rec.setdefault("families", {})
        rec.setdefault("controls", {})
        print(f"[{bb}] base pass done in {time.time()-t0:.0f}s; sites {names}", flush=True)

        if not a.controls:
            for fam, params, make in (("hue", hues, lambda p: h.norm(h.hue_shift(x01, p))),
                                      ("heat", heats, lambda p: h.norm(h.heat(x01, p ** 2 / 2)))):
                fam_rec = rec["families"].get(fam, {})
                for p in params:
                    xr = make(p)
                    yr, Zr = forward_all(h, model, sites, xr)
                    eff = float(np.linalg.norm(yr - y0) / (np.linalg.norm(y0) + 1e-12))
                    entry = {"effect_size_logits": eff, "param": p, "consumer": {}, "sites": {}}
                    for q in qs:
                        entry["consumer"][f"q={q}"] = pair_stats(y0, yr, q)
                    for k, nm in enumerate(names):
                        entry["sites"][nm] = {f"q={q}": pair_stats(Z0[k], Zr[k], q) for q in qs}
                    fam_rec[str(p)] = entry
                    c = entry["consumer"][f"q={qs[0]}"]
                    print(f"[{bb}/{fam} p={p}] effect {eff:.3f} | consumer preservation {c['preservation_rate']:.3f} "
                          f"(baseline {c['baseline_rate']:.3f}, median-ratio {c['median_ratio']:.3f}, "
                          f"spearman {c['spearman_base_vs_transformed_dist']:.3f})", flush=True)
                rec["families"][fam] = fam_rec
        else:
            for fi, (fam, p0) in enumerate((("hue", hues[0]), ("heat", heats[0]))):
                src = rec.get("families", {}).get(fam, {}).get(str(p0))
                if src is None:
                    print(f"[{bb}/noise@{fam}] no target effect in {out_path} for {fam} p={p0}; skipped", flush=True)
                    continue
                tgt = src["effect_size_logits"]
                sig, eff, yr, Zr, hist = match_noise(h, model, sites, x01, x, y0, tgt,
                                                     seed=a.seed + 101 * (fi + 1))
                entry = {"effect_size_logits": eff, "param": sig, "target_effect": tgt,
                         "calibration": hist, "consumer": {}, "sites": {}}
                for q in qs:
                    entry["consumer"][f"q={q}"] = pair_stats(y0, yr, q)
                for k, nm in enumerate(names):
                    entry["sites"][nm] = {f"q={q}": pair_stats(Z0[k], Zr[k], q) for q in qs}
                rec["controls"][fam] = entry
                c = entry["consumer"][f"q={qs[0]}"]
                print(f"[{bb}/noise@{fam}] target {tgt:.3f} -> effect {eff:.3f} at sigma {sig:.4f} "
                      f"({len(hist)} passes) | consumer preservation {c['preservation_rate']:.3f} "
                      f"(baseline {c['baseline_rate']:.3f}, spearman "
                      f"{c['spearman_base_vs_transformed_dist']:.3f})", flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        res["results"][bb] = rec
        json.dump(res, open(out_path, "w"), indent=1)
        del model
        print(f"[{bb}] done in {rec['seconds']}s; saved {out_path}", flush=True)

    print("saved", out_path)


if __name__ == "__main__":
    main()
