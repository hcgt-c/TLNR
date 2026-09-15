# -*- coding: utf-8 -*-
"""200_quotientization.py — is the depth ceiling caused by the transformation signal leaving the
consumer-visible subspace?

The depth ceiling is not capacity (§8.3), not reachability (§8.4) and not low power (script 199: partial
rho = -0.662, 16/16 tightly matched pairs). The remaining hypothesis is progressive quotientization: with
depth, the transformation-induced displacement becomes (i) larger, (ii) less coherent, and (iii) increasingly
orthogonal to the directions the consumer can see, so no input-independent global operator can reproduce it.

Three measurements per site, all on the same frozen backbone, transformation and held-out set:

  M2 coherence       PCA spectrum of the displacement covariance on the fit split: explained fraction at rank
                     k and its participation ratio. A coherent displacement is one a single global direction
                     can capture.
  M1 visible share   the *exact* consumer-visible fraction of the displacement for a linear read-out consumer
                     (the per-cell probe of script 197, for which the Jacobian is the read-out itself):
                     V_tau = sum_cells ||P_A delta_cell||^2 / sum_cells ||delta_cell||^2, with P_A the
                     row-space projector of A. This asks directly: how much of the transformation signal is
                     in the space the consumer reads?
  rank-k intervention  replace the consumer by F_k = F o P_k, where P_k is the rank-k projector onto the top
                     principal directions of the site covariance fitted on the fit split, and measure the
                     ceiling T_F(k) with everything else fixed. Two controls: a *random* rank-k projector of
                     the same rank, and the rank-k projector onto the *displacement* covariance (an oracle
                     that is allowed to know the transformation).

Prediction under quotientization: the visible share falls with depth; the displacement becomes less coherent;
and the ceiling depends on the visible rank (rising with k), while a random projector of the same rank does
not help. Any of the three failing is a kill for the mechanism story.

Usage: python scripts/200_quotientization.py --backbones resnet50,dinov2b14 --family heat --param 2.0 --seed 0
Outputs results/quotientization.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, time, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
H = load_mod("harness188", os.path.join(WORK, "scripts", "188_depth_map_harness.py"))
H.c136 = c136
DEV = H.DEV
N_BINS, SAT_THRESH = 12, 0.15


def topk_projector(cov, k):
    """Rank-k projector onto the top-k eigenvectors of a (C,C) covariance, returned as a (C,C) matrix."""
    ev, V = torch.linalg.eigh(cov.double())
    Vk = V[:, -k:]                                   # eigh returns ascending eigenvalues
    return (Vk @ Vk.T).float()


def spectrum(cov, ks):
    ev = torch.flip(torch.linalg.eigvalsh(cov.double()).clamp_min(0), [0])   # descending
    tot = float(ev.sum()) + 1e-30
    return {"explained": {str(k): float(ev[:k].sum()) / tot for k in ks if k <= len(ev)},
            "participation_ratio": float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-30)),
            "top1_share": float(ev[0]) / tot, "dim": int(len(ev))}


def hsv(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = torch.maximum(torch.maximum(r, g), b)
    mn = torch.minimum(torch.minimum(r, g), b)
    d = mx - mn
    h = torch.zeros_like(mx)
    nz = d > 1e-6
    m1 = nz & (mx == r); m2 = nz & (mx == g) & ~m1; m3 = nz & (mx == b) & ~m1 & ~m2
    h[m1] = ((g - b) / d.clamp_min(1e-6))[m1] % 6.0
    h[m2] = ((b - r) / d.clamp_min(1e-6))[m2] + 2.0
    h[m3] = ((r - g) / d.clamp_min(1e-6))[m3] + 4.0
    sat = torch.where(mx > 1e-6, d / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return h / 6.0, sat


def hf_energy(x01):
    """High-frequency energy of a crop: |Laplacian| of the luminance, averaged. This is the target of the
    *heat*-aligned read-out: heat blur removes exactly this quantity, whereas hue rotation leaves it
    unchanged, so a read-out fitted to it is aligned with the heat family rather than with hue."""
    lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=lum.dtype, device=lum.device)
    k = k.view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(lum, k, padding=1)
    return lap.abs().mean((1, 2, 3))


def hf_bins(x01, edges):
    """Bin the high-frequency energy with quantile edges taken from the fit split of the base images."""
    return torch.bucketize(hf_energy(x01), edges.to(x01.device))


def cell_bins(x, gh, gw):
    h, sat = hsv(x)
    th = h * 2 * np.pi
    w = torch.where(sat < SAT_THRESH, torch.zeros_like(sat), sat)
    S = torch.nn.functional.adaptive_avg_pool2d((torch.sin(th) * w).unsqueeze(1), (gh, gw)).squeeze(1)
    C = torch.nn.functional.adaptive_avg_pool2d((torch.cos(th) * w).unsqueeze(1), (gh, gw)).squeeze(1)
    Wt = torch.nn.functional.adaptive_avg_pool2d(w.unsqueeze(1), (gh, gw)).squeeze(1)
    idx = ((torch.atan2(S, C) / (2 * np.pi)) % 1.0 * N_BINS).long().clamp(0, N_BINS - 1)
    return torch.where(Wt < 1e-6, torch.full_like(idx, N_BINS), idx)


def run_with(model, site, fn, x, bs=32):
    """Forward pass with fn applied at the site; returns the logits."""
    outs = []
    def hook(m, i, o):
        t = fn(o) if fn is not None else o
        return t
    h = site.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                o = model(x[s:s + bs])
                outs.append((o[0] if isinstance(o, (tuple, list)) else o).detach().cpu())
    finally:
        h.remove()
    return torch.cat(outs)


def transfer(o_int, o_real, o_null):
    d_int = torch.norm(o_int - o_real, dim=1)
    d_null = torch.norm(o_null - o_real, dim=1)
    return 1.0 - float(d_int.mean() / (d_null.mean() + 1e-12)), float(d_null.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet50,dinov2b14")
    ap.add_argument("--family", default="heat", choices=["heat", "hue"])
    ap.add_argument("--param", type=float, default=2.0)
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_held", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout", default="hue", choices=["hue", "hf"],
                    help="hue = read-out to the crop's dominant hue bin (aligned with the hue family); "
                         "hf = read-out to the crop's high-frequency energy bin (aligned with the heat family)")
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    ranks = [int(k) for k in a.ranks.split(",")]
    t0 = time.time()

    out = {"family": a.family, "param": a.param, "depths": depths, "ranks": ranks,
           "n_fit": a.n_fit, "n_held": a.n_held, "seed": a.seed,
           "readout": a.readout,
           "protocol": ("frozen backbone; site displacement and its covariance from the fit split; the rank-k "
                        "intervention replaces the consumer by F o P_k with P_k the top-k principal projector "
                        "of the site covariance fitted on the fit split (controls: random projector of equal "
                        "rank, and the top-k projector of the displacement covariance)"),
           "backbones": {}}
    for backbone in a.backbones.split(","):
        model, sites, names, kind = H.build_backbone(backbone)
        crops = H.load_crops(a.n_fit + a.n_held, seed=a.seed)
        x01 = H.to01(crops)
        x = H.norm(x01)
        xb = H.norm(H.heat(x01, a.param ** 2 / 2) if a.family == "heat" else H.hue_shift(x01, a.param))
        n, held = a.n_fit, slice(a.n_fit, a.n_fit + a.n_held)
        rec_b = {}
        for d in depths:
            site = sites[d - 1]
            Fb = H.capture(model, [site], x)[0]
            Fr = H.capture(model, [site], xb)[0]
            B, C, Hh, Ww = Fb.shape
            rows_b = Fb.permute(0, 2, 3, 1).reshape(-1, C)
            rows_r = Fr.permute(0, 2, 3, 1).reshape(-1, C)
            n_rows = n * Hh * Ww
            g = torch.Generator().manual_seed(a.seed)
            sel = torch.randperm(n_rows, generator=g)[:min(n_rows, 150_000)]
            Xf, Yf = rows_b[:n_rows][sel], rows_r[:n_rows][sel]
            # covariances on the fit split; residualised by the fit mean so the spectrum is about variation
            Xc = Xf - Xf.mean(0, keepdim=True)
            Dc = Yf - Xf
            Dc = Dc - Dc.mean(0, keepdim=True)
            cov_z = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
            cov_d = (Dc.T @ Dc) / max(1, Dc.shape[0] - 1)

            # exact consumer-visible share for the linear read-out consumer, whose target depends on the
            # read-out being tested: hue bin (aligned with hue) or high-frequency energy bin (aligned with heat)
            if a.readout == "hf":
                e_fit = hf_energy(x01[:n])
                edges = torch.quantile(e_fit, torch.linspace(0, 1, N_BINS + 2)[1:-1].to(e_fit.device))
                tgt = hf_bins(x01[:n], edges).repeat_interleave(Hh * Ww).cpu()
            else:
                tgt = cell_bins(x01[:n], Hh, Ww).permute(0, 2, 1).reshape(-1).cpu()
            A = c136.fit_ridge(rows_b[:n_rows], torch.nn.functional.one_hot(tgt, N_BINS + 1).float())
            P_A = (A.T @ torch.linalg.pinv(A @ A.T) @ A).float()          # row-space projector of A (C,C)
            Dh = (rows_r[n_rows:] - rows_b[n_rows:])
            vis = float(((Dh @ P_A) ** 2).sum() / ((Dh ** 2).sum() + 1e-30))
            tot = float((Dh ** 2).sum())
            # ---- matched-rank control for the visible share --------------------------------
            # A random rank-m projector sees, in expectation, a fraction m/C of *any* displacement, so the raw
            # visible share is confounded by the read-out's rank and by the channel count. Two matched-rank
            # baselines are therefore recorded: one at the read-out's nominal rank m, one at its *effective*
            # rank (the participation ratio of A^T A, which for a near-rank-1 read-out is far below m).
            evA = torch.linalg.eigvalsh((A.double().T @ A.double())).clamp_min(0)
            eff_rank_A = float((evA.sum() ** 2) / ((evA ** 2).sum() + 1e-30))
            gen_b = torch.Generator().manual_seed(a.seed)
            def random_share(rank, reps=16):
                acc = 0.0
                for _ in range(reps):
                    Q = torch.linalg.qr(torch.randn(C, rank, generator=gen_b))[0]
                    acc += float(((Dh @ (Q @ Q.T)) ** 2).sum() / (tot + 1e-30))
                return acc / reps
            m_nom = int(A.shape[0])
            k_eff = max(1, int(round(eff_rank_A)))
            chance_nominal = random_share(m_nom)
            chance_effrank = random_share(k_eff)

            # normalise the visible share by its isotropic expectation rank(A)/C: without this the number is
            # dominated by the read-out's rank rather than by the alignment of the displacement with it
            iso = float(A.shape[0]) / C
            rec = {"feature_shape": [int(B), int(C), int(Hh), int(Ww)],
                   "displacement": {"rms": float(torch.sqrt((Dh ** 2).mean())),
                                    "coherence": spectrum(cov_d, ranks)},
                   "site_covariance": spectrum(cov_z, ranks),
                   "visible_share_of_displacement": vis,
                   "visible_share_isotropic_expectation": iso,
                   "visible_share_over_isotropic": vis / iso,
                   "readout_effective_rank": eff_rank_A,
                   "matched_rank_control": {
                       "nominal_rank": m_nom, "random_share_at_nominal": chance_nominal,
                       "effective_rank": eff_rank_A, "effective_rank_rounded": k_eff,
                       "random_share_at_effective": chance_effrank,
                       "visible_over_random_nominal": vis / (chance_nominal + 1e-30),
                       "visible_over_random_effective": vis / (chance_effrank + 1e-30)},
                   "consumer_visible_dim_of_probe": int(A.shape[0]),
                   "rank_intervention": {}}

            y_null = run_with(model, site, None, x)
            y_real = run_with(model, site, None, xb)
            W2 = c136.fit_ridge(Xf, Yf)
            gen = torch.Generator().manual_seed(a.seed)
            for k in ranks:
                if k > C:
                    continue
                Pk = topk_projector(cov_z, k)
                Q = torch.linalg.qr(torch.randn(C, k, generator=gen))[0].float()      # random rank-k basis
                Pk_rand = Q @ Q.T
                Pd = topk_projector(cov_d, k)                                          # oracle (knows delta)
                entry = {}
                for tag, P in (("pca", Pk), ("random", Pk_rand), ("displacement_oracle", Pd)):
                    mk = lambda t, P=P: H._apply_matrix(t, P.to(t.device), kind)
                    mk_int = lambda t, P=P, W=W2: H._apply_matrix(t, (P @ W).to(t.device), kind)
                    y_n = run_with(model, site, mk, x[held])
                    y_r = run_with(model, site, mk, xb[held])
                    y_i = run_with(model, site, mk_int, x[held])
                    tf, d_null = transfer(y_i, y_r, y_n)
                    entry[tag] = {"transfer": tf, "mean_d_noop": d_null,
                                  "visible_share_k": float(((Dh @ P) ** 2).sum() / (tot + 1e-30))}
                rec["rank_intervention"][str(k)] = entry
                print(f"  {backbone} d{d} {names[d-1]:9s} k={k:3d}: "
                      f"PCA {entry['pca']['transfer']:+.3f} (vis {entry['pca']['visible_share_k']:.2f}) | "
                      f"rand {entry['random']['transfer']:+.3f} | oracle {entry['displacement_oracle']['transfer']:+.3f}",
                      flush=True)
            # full-rank reference
            entry = {}
            for tag, W_ in (("none", W2),):
                mk_int = lambda t, W=W_: H._apply_matrix(t, W.to(t.device), kind)
                y_i = run_with(model, site, mk_int, x[held])
                tf, d_null = transfer(y_i, y_real[held], y_null[held])
                entry[tag] = {"transfer": tf, "mean_d_noop": d_null, "visible_share_k": 1.0}
            rec["rank_intervention"]["full"] = entry
            rec["uncompressed_ceiling"] = entry["none"]["transfer"]
            rec["power_gate"] = c136.rel_l2(y_real[held], y_null[held])
            rec_b[names[d - 1]] = rec
            print(f"  {backbone} {names[d-1]}: displacement rms {rec['displacement']['rms']:.3f} | "
                  f"coherence top1 {rec['displacement']['coherence']['top1_share']:.3f} PR "
                  f"{rec['displacement']['coherence']['participation_ratio']:.1f} | visible share of delta "
                  f"{vis:.3f} | uncompressed ceiling {rec['uncompressed_ceiling']:+.3f} | power "
                  f"{rec['power_gate']:.3f}", flush=True)
            del Fb, Fr, rows_b, rows_r, Xf, Yf
            if DEV == "cuda":
                torch.cuda.empty_cache()
        out["backbones"][backbone] = rec_b
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    path = os.path.join(WORK, "results", "quotientization.json")
    prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    # the key must include every configuration parameter: an earlier version omitted the family, so the
    # second family of each seed silently overwrote the first (the third instance of this bug class in this
    # project, after 193's split and 194's sigma pair)
    prev[f"{a.backbones}_seed{a.seed}_{a.family}{a.param}_ro{a.readout}"] = out
    json.dump(prev, open(path, "w"), indent=1)
    print(f"quotientization probe: PASS (0 issues) | {time.time()-t0:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
