# -*- coding: utf-8 -*-
"""187_heat_semigroup_pilot.py — is a dissipative transformation semigroup carried by a representation?

The second-transformation-family proposal is to use the discrete heat semigroup B_t = exp(-t L) instead of a
second group action, because (i) it is exact, (ii) it is non-invertible, so it exercises the semigroup branch
of the algebra rather than the group branch again, and (iii) blur is a large event for ordinary vision
consumers, so the test has power on externally defined tasks.

This pilot answers the four questions that decide whether the full experiment is worth running, all on an
existing plain-CNN checkpoint:

  1. EXACTNESS. Is B_{t1} B_{t2} = B_{t1+t2} satisfied to machine precision by the implementation? We use the
     spectral (DCT/Neumann) form, which is exact by construction; a truncated Gaussian kernel is not.
  2. THE FREE POSITIVE CONTROL. The proposal notes that a convolution commutes with a convolution, so at the
     first linear layer f_1(B_t x) = B_t f_1(x) should hold exactly. That argument assumes the boundary
     condition of the convolution matches the one of the heat operator. A standard convolutional layer uses
     *zero* padding while our heat operator uses reflecting boundaries, so exactness should hold in the
     interior and fail in a border band whose width grows with t. We measure the border fraction explicitly.
  3. THE DEPTH CURVE. The intertwining residual R_l(t) = ||f_l(B_t x) - B_{l,t} f_l(x)|| /
     ||f_l(B_t x) - f_l(x)|| (0 = the canonical action explains the change, 1 = it explains none of it) at
     pre-ReLU, stage 1 and stage 2, and after a fitted global correction at each site.
  4. POWER. The effect size of B_t at each site and at the logits, since R is meaningless when the site's
     effect size is at the noise floor (the paper's own precondition 3).

Usage: python scripts/187_heat_semigroup_pilot.py [--arm z2 --seed 0 --n 64 --sigmas 0.5,1.0,1.5,2.0]
Outputs results/heat_semigroup_pilot.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, argparse, importlib.util
import numpy as np
import torch
from scipy.fft import dctn, idctn

WORK = rp.REPO_ROOT


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
rbt = c136.rbt
DEV = c136.DEV


# ---------------------------------------------------------------- discrete heat semigroup
def heat_eigs(H, W):
    """Eigenvalues of the 2D Neumann Laplacian, matching the DCT-II basis used below."""
    la = 2 - 2 * np.cos(np.pi * np.arange(H) / H)
    lb = 2 - 2 * np.cos(np.pi * np.arange(W) / W)
    return (la[:, None] + lb[None, :]).astype(np.float32)


def heat(x, t, eigs):
    """B_t x by spectral multiplication: exact, and a semigroup in t by construction."""
    xn = x.detach().cpu().numpy().astype(np.float32)
    a = dctn(xn, type=2, norm="ortho", axes=(-2, -1))
    a = a * np.exp(-t * eigs)[None, None]
    y = idctn(a, type=2, norm="ortho", axes=(-2, -1))
    return torch.from_numpy(y.astype(np.float32)).to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sigmas", default="0.5,1.0,1.5,2.0")
    a = ap.parse_args()
    sigmas = [float(s) for s in a.sigmas.split(",")]

    x_tr_u8, y_tr, x_te_u8, y_te = rbt.load_cifar()
    x01 = torch.from_numpy(x_te_u8[:a.n]).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(y_te[:a.n]).long().to(DEV)

    model, kind = rbt.make_arm(a.arm)
    sd = torch.load(os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{a.seed}.pt"), map_location="cpu")
    model.load_state_dict(sd); model = model.to(DEV).eval()
    base = model.base if hasattr(model, "base") else model

    x = (x01 - rbt.MEAN) / rbt.STD
    eigs32 = heat_eigs(32, 32)
    eigs16 = heat_eigs(16, 16)

    def rel(a_, b_):
        return float((a_ - b_).flatten(1).norm(dim=1).mean() / (b_.flatten(1).norm(dim=1).mean() + 1e-12))

    out = {"arm": a.arm, "seed": a.seed, "n": a.n, "device": DEV,
           "note": "t = sigma^2 / 2 for the unit-spacing Laplacian; the site action uses t_f = t * (S/32)^2",
           "semigroup": {}, "sites": {}}

    # ---- 1. exactness of the implemented semigroup
    xs = x[:8]
    worst = 0.0
    for s1, s2 in ((0.5, 1.0), (1.0, 1.5), (0.25, 2.0), (2.0, 2.0)):
        t1, t2 = s1 ** 2 / 2, s2 ** 2 / 2
        lhs = heat(heat(xs, t1, eigs32), t2, eigs32)
        rhs = heat(xs, t1 + t2, eigs32)
        rel_err = float((lhs - rhs).norm() / (rhs.norm() + 1e-12))   # not `rel`: that name is the helper
        worst = max(worst, rel_err)
    out["semigroup"]["max_rel_error_Bt1Bt2_vs_Bt1t2"] = worst
    # and the same for a truncated Gaussian kernel, to show why the spectral form is used
    def gauss_kernel(sigma, r=4):
        k = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-(k ** 2) / (2 * sigma ** 2)); g = g / g.sum()
        return g
    g1, g2 = gauss_kernel(sigmas[0]), gauss_kernel(sigmas[-1])
    def sep_blur(t, g):
        k = g.to(t.device)
        t = torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1).repeat(t.shape[1], 1, 1, 1), padding=(len(k)//2, 0), groups=t.shape[1])
        t = torch.nn.functional.conv2d(t, k.view(1, 1, 1, -1).repeat(t.shape[1], 1, 1, 1), padding=(0, len(k)//2), groups=t.shape[1])
        return t
    trunc = sep_blur(sep_blur(xs, gauss_kernel(1.0)), gauss_kernel(1.0))
    exact = heat(xs, 1.0 ** 2 / 2, eigs32)
    out["semigroup"]["truncated_gaussian_vs_spectral_rel_diff"] = float(
        (trunc - exact).norm() / (exact.norm() + 1e-12))

    # ---- 1b. WHY the "free positive control" is not exact: boundary conditions, not depth.
    # Convolution commutes with convolution on the infinite lattice. On a finite image the commutation holds
    # only if both operators extend the signal the same way, so we compare the pair the model actually uses
    # (zero padding + reflecting/Neumann heat) against a boundary-matched pair (circular padding + periodic
    # heat). The matched pair should reach machine precision, which isolates the mismatch as a boundary term
    # whose size grows with t; that is what makes the first layer an *approximate* rather than exact control.
    def periodic_eigs(H, W):
        la = 2 - 2 * np.cos(2 * np.pi * np.arange(H) / H)
        lb = 2 - 2 * np.cos(2 * np.pi * np.arange(W) / W)
        return (la[:, None] + lb[None, :]).astype(np.float32)

    def heat_fft(xin, t, eigs):
        a = torch.fft.fft2(xin) * torch.from_numpy(np.exp(-t * eigs)).to(xin.device)
        return torch.fft.ifft2(a).real

    kk = torch.randn(1, 1, 3, 3, device=DEV)
    xk = torch.randn(1, 1, 32, 32, device=DEV)
    ep = periodic_eigs(32, 32)
    bc = {}
    for tt in (0.125, 0.5, 2.0):
        def conv_pad(z, mode):
            return torch.nn.functional.conv2d(torch.nn.functional.pad(z, (1, 1, 1, 1), mode=mode), kk)
        mismatched = rel(conv_pad(heat(xk, tt, eigs32), "constant"),
                         heat(conv_pad(xk, "constant"), tt, eigs32))
        matched = rel(conv_pad(heat_fft(xk, tt, ep), "circular"),
                      heat_fft(conv_pad(xk, "circular"), tt, ep))
        bc[f"t_{tt}"] = {"model_pair_zero_pad_vs_neumann": mismatched,
                         "matched_pair_circular_vs_periodic": matched,
                         "periodic_heat_semigroup_rel_error": rel(
                             heat_fft(heat_fft(xk, tt / 2, ep), tt / 2, ep), heat_fft(xk, tt, ep))}
    out["boundary_condition_diagnostic"] = bc

    # ---- 2/3/4. sites, effect size, intertwining residual, and a fitted global correction
    def site_features(xin):
        store = {}
        h1 = base.bn1.register_forward_hook(lambda m, i, o: store.__setitem__("pre_relu", o.detach()))
        h2 = base.layers[0].register_forward_hook(lambda m, i, o: store.__setitem__("stage1", c136._flat(o).detach()))
        h3 = base.layers[1].register_forward_hook(lambda m, i, o: store.__setitem__("stage2", c136._flat(o).detach()))
        with torch.no_grad():
            logits = c136.run_plain(model, xin)
        for h in (h1, h2, h3):
            h.remove()
        return store, logits

    def fit_global(ref, tgt):
        """Least-squares per-channel gain and full channel map from the canonical action to the truth."""
        R = ref.permute(0, 2, 3, 1).reshape(-1, ref.shape[1])
        T = tgt.permute(0, 2, 3, 1).reshape(-1, tgt.shape[1])
        gain = (R * T).sum(0) / ((R * R).sum(0) + 1e-9)
        C = R.shape[1]
        lam = 1e-3 * float((R * R).sum(0).mean())
        W = torch.linalg.solve(R.T @ R + lam * torch.eye(C, device=R.device), R.T @ T)
        return gain, W

    f0, logits0 = site_features(x)
    for s in sigmas:
        t = s ** 2 / 2
        xb = heat(x, t, eigs32)
        fb, logitsb = site_features(xb)
        rec = {"sigma": s, "t": t,
               "logits_effect_size": c136.rel_l2(logitsb, logits0),
               "logits_flip_rate": float((c136.top1(logitsb) != c136.top1(logits0)).float().mean()),
               "acc_null": float((c136.top1(logits0) == y).float().mean()),
               "acc_blur": float((c136.top1(logitsb) == y).float().mean()),
               "sites": {}}
        for name, S in (("pre_relu", 32), ("stage1", 32), ("stage2", 16)):
            fs, fsb = f0[name], fb[name]
            eigs_s = eigs32 if S == 32 else eigs16
            tf = t * (S / 32.0) ** 2                      # the canonical action at this site's resolution
            canon = heat(fs, tf, eigs_s)
            eff = rel(fsb, fs)                            # how large the transformation is at this site
            resid = rel(fsb, canon)                       # 0 = canonical action explains it, 1 = it explains none
            gain, W = fit_global(canon, fsb)
            g = canon.permute(0, 2, 3, 1) * gain
            resid_gain = rel(fsb, g.permute(0, 3, 1, 2))
            R = canon.permute(0, 2, 3, 1).reshape(-1, canon.shape[1])
            T = fsb.permute(0, 2, 3, 1).reshape(-1, fsb.shape[1])
            pred = (R @ W).reshape(fsb.shape[0], fsb.shape[2], fsb.shape[3], fsb.shape[1]).permute(0, 3, 1, 2)
            resid_full = rel(fsb, pred)
            # border sensitivity at this site: interior versus a border band of width ceil(2*sigma*S/32)
            w = max(8, int(np.ceil(3 * s * S / 32.0)) + 2)
            if w < S // 2:
                sl = (slice(None), slice(None), slice(w, S - w), slice(w, S - w))
                inter_eff = rel(fsb[sl], fs[sl])
                inter_res = rel(fsb[sl], canon[sl])
                inter_res_gain = rel(fsb[sl], (canon.permute(0, 2, 3, 1) * gain).permute(0, 3, 1, 2)[sl])
            else:
                inter_eff = inter_res = inter_res_gain = None
            rec["sites"][name] = {
                "effect_size": eff,
                "intertwining_residual_canonical": resid,
                "intertwining_residual_after_channel_gain": resid_gain,
                "intertwining_residual_after_global_linear": resid_full,
                "interior_effect_size": inter_eff, "interior_residual": inter_res,
                "interior_residual_after_channel_gain": inter_res_gain,
                "border_band_width_cells": w, "feature_size": S,
                "t_feature": tf, "n_channel_map_params": int(W.numel())}
        out["sites"][f"sigma_{s}"] = rec

    # ---- margin sweep at pre-ReLU for one sigma: is the commutation exact away from the border?
    s0 = sigmas[len(sigmas) // 2]
    t0 = s0 ** 2 / 2
    f_m, fb_m = f0["pre_relu"], fb["sites"][f"sigma_{s0}"]["sites"] if False else None
    xb0 = heat(x, t0, eigs32)
    fb0, _ = site_features(xb0)
    sweep = {}
    for w in (2, 4, 8, 12):
        if w < 16:
            sl = (slice(None), slice(None), slice(w, 32 - w), slice(w, 32 - w))
            sweep[str(w)] = rel(fb0["pre_relu"][sl], heat(f_m, t0, eigs32)[sl])
    out["pre_relu_margin_sweep_sigma_%.2f" % s0] = sweep
    out["pre_relu_full_map_residual_sigma_%.2f" % s0] = rel(fb0["pre_relu"], heat(f_m, t0, eigs32))

    fn = os.path.join(WORK, "results", "heat_semigroup_pilot.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"{a.arm} s{a.seed}, n={a.n}")
    print(f"  semigroup max rel error  {out['semigroup']['max_rel_error_Bt1Bt2_vs_Bt1t2']:.3e}"
          f"   | truncated Gaussian vs spectral {out['semigroup']['truncated_gaussian_vs_spectral_rel_diff']:.3e}")
    for key, rec in out["sites"].items():
        p = rec["sites"]["pre_relu"]; s1 = rec["sites"]["stage1"]; s2 = rec["sites"]["stage2"]
        print(f"  sigma={rec['sigma']:.2f}  logits eff {rec['logits_effect_size']:.3f} flip {rec['logits_flip_rate']:.2f} "
              f"acc {rec['acc_null']:.3f}->{rec['acc_blur']:.3f}")
        print(f"      pre-ReLU  eff {p['effect_size']:.4f}  R {p['intertwining_residual_canonical']:.3f}"
              f"  R|gain {p['intertwining_residual_after_channel_gain']:.3f}  interior R {p['interior_residual']}")
        print(f"      stage1    eff {s1['effect_size']:.4f}  R {s1['intertwining_residual_canonical']:.3f}"
              f"  R|gain {s1['intertwining_residual_after_channel_gain']:.3f}  interior R {s1['interior_residual']}")
        print(f"      stage2    eff {s2['effect_size']:.4f}  R {s2['intertwining_residual_canonical']:.3f}"
              f"  R|gain {s2['intertwining_residual_after_channel_gain']:.3f}")
    print("saved", fn)


if __name__ == "__main__":
    main()
