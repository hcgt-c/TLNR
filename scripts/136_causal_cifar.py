# -*- coding: utf-8 -*-
"""136_causal_cifar.py — causal activation intervention on colour-sensitive CIFAR arms.

Motivation (reviewer P0 #2): script 133 intervened on a frozen YOLO detector whose outputs are
nearly hue-insensitive (real recolour changes object-anchor logits by 0.04-0.06 rel L2; box AP50
stays 1.0), so the downstream equivalence test there is low-power. Here we repeat the test on the
CIFAR-10 ResNet-44 arms of Route B (results/c10_routeB/*.pt), where colour IS a computational
variable (plain CNN: 0.905 clean -> ~0.79 under hue shift), so effect sizes are large.

Protocol (no training, weights untouched):
  split at layers[0] output (stage-1, 64ch); transport the activation with a global channel-mixing
  operator W_delta, then run the remaining layers.
    real route : y_real = model( T_delta x )
    null       : y_null = model( x )
    intervention: y_int = model_{l>0}( W_delta f_1(x) )
  Operators fitted on a TRAIN split only (per-cell rows), evaluated on HELD-OUT images:
    O1 procrustes   : global orthogonal (133 convention: row f -> f W^T, W = U V^T of Y^T X)
    O2 ridge-LS     : unconstrained linear (capacity upper bound)
    O4 block-rot    : per-channel-pair 2x2 rotation (amplitude preserving, harmonic-like constraint)
    O6 random-orth  : specificity control
Metrics: feature fidelity (rel L2, angle), downstream equivalence (rel L2/KL/top-1 agreement vs
real), direction alignment and projection coefficient of (y_int - y_null) onto (y_real - y_null),
and an explicit power report (||y_real - y_null||, top-1 flip rate).

Usage: python scripts/136_causal_cifar.py --arm z2 --seed 0 --n_train 300 --n_held 200
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = rp.REPO_ROOT
spec = importlib.util.spec_from_file_location("rbt", os.path.join(WORK, "scripts", "c10_routeB_train.py"))
rbt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbt)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- operators ----------------
def fit_procrustes(X, Y):
    """W (d,d) with row convention: X @ W.T ~= Y."""
    M = Y.T @ X
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def fit_ridge(X, Y, lam=1e-3):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y).T


def fit_blockrot(X, Y):
    """Per channel-pair 2x2 orthogonal blocks."""
    d = X.shape[1]
    W = torch.zeros(d, d, device=X.device, dtype=X.dtype)
    for i in range(0, d - 1, 2):
        Xp, Yp = X[:, i:i + 2], Y[:, i:i + 2]
        M = Yp.T @ Xp
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        W[i:i + 2, i:i + 2] = U @ Vh
    if d % 2 == 1:  # last channel: sign only
        i = d - 1
        W[i, i] = torch.sign((X[:, i] * Y[:, i]).sum())
    return W


def make_op5(X, Y, epochs=400, hidden=128, lr=2e-3, seed=0):
    """Content-conditioned operator: W_ridge f + MLP_residual(f), trained on the same train rows."""
    torch.manual_seed(seed)
    d = X.shape[1]
    W = fit_ridge(X, Y)
    R = Y - X @ W.T
    net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                        nn.Linear(hidden, d)).to(X.device)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(X @ W.T + net(X), Y)
        loss.backward(); opt.step()
    def apply(F_rows):
        with torch.no_grad():
            return F_rows @ W.T + net(F_rows)
    return apply


def fit_lowrank(X, Y, r=16, lam=1e-3):
    """Rank-r truncation of the ridge/LS solution (the displacement lives in a low-rank subspace)."""
    W = fit_ridge(X, Y, lam)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(r, S.shape[0])
    return (U[:, :r] * S[:r]) @ Vh[:r]


def make_op7(X, Y, r=16, epochs=800, hidden=256, lr=1e-3, seed=0):
    """O7 = low-rank base + larger MLP residual: a compact content-conditioned operator."""
    torch.manual_seed(seed)
    d = X.shape[1]
    W = fit_lowrank(X, Y, r)
    R = Y - X @ W.T
    net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                        nn.Linear(hidden, d)).to(X.device)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(X @ W.T + net(X), Y)
        loss.backward(); opt.step()
    def apply(F_rows):
        with torch.no_grad():
            return F_rows @ W.T + net(F_rows)
    return apply


class SpatialOp:
    """Callable operator that consumes 4D (B,C,H,W) features (ridge 1x1 + conv residual)."""
    def __init__(self, apply4):
        self.apply4 = apply4


def make_op8(Fx4, Fy4, epochs=800, hidden=32, lr=1e-3, seed=0):
    """O8 = ridge 1x1 mixing + 3x3 conv residual: content- and neighbourhood-conditioned."""
    torch.manual_seed(seed)
    B, C, H, Wd = Fx4.shape
    Xr = Fx4.permute(0, 2, 3, 1).reshape(-1, C)
    Yr = Fy4.permute(0, 2, 3, 1).reshape(-1, C)
    W = fit_ridge(Xr, Yr)
    target = Fy4 - torch.einsum("bchw,dc->bdhw", Fx4, W)
    conv = nn.Sequential(nn.Conv2d(C, hidden, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(hidden, C, 3, padding=1)).to(Fx4.device)
    nn.init.zeros_(conv[-1].weight); nn.init.zeros_(conv[-1].bias)
    opt = torch.optim.Adam(conv.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(conv(Fx4), target)
        loss.backward(); opt.step()
    def apply4(f4):
        with torch.no_grad():
            return torch.einsum("bchw,dc->bdhw", f4, W) + conv(f4)
    return SpatialOp(apply4)


def random_orth(d, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))
    return Q.to(DEV)


# ---------------- feature / intervention hooks ----------------
def get_layer(model, arm, stage=1):
    base = model.base if hasattr(model, "base") else model
    return base.layers[stage - 1]


def _flat(t):
    """(B,X,Y,H,W) group-indexed activations -> (B, X*Y, H, W); 4D unchanged."""
    if t.dim() == 5:
        B, X, Y, H, Wd = t.shape
        return t.reshape(B, X * Y, H, Wd)
    return t


def collect_features(model, layer, x, bs=250):
    feats = []
    def hook(m, i, o):
        feats.append(_flat(o).detach())
    h = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                model(x[s:s + bs])
    finally:
        h.remove()
    return torch.cat(feats, 0)


def run_intervened(model, layer, x, W, bs=250):
    outs = []
    def apply_rows(rows):
        return W(rows) if callable(W) else rows @ W.T
    def hook(m, i, o):
        orig = o.shape
        f = _flat(o)
        B, C, H, Wd = f.shape
        if isinstance(W, SpatialOp):
            ff = W.apply4(f)
        else:
            ff = apply_rows(f.permute(0, 2, 3, 1).reshape(-1, C)).reshape(B, H, Wd, C).permute(0, 3, 1, 2).contiguous()
        return ff.reshape(orig) if len(orig) == 5 else ff
    h = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                outs.append(model(x[s:s + bs]).detach())
    finally:
        h.remove()
    return torch.cat(outs, 0)


def run_plain(model, x, bs=250):
    outs = []
    with torch.no_grad():
        for s in range(0, x.shape[0], bs):
            outs.append(model(x[s:s + bs]).detach())
    return torch.cat(outs, 0)


# ---------------- metrics ----------------
def rel_l2(a, b):
    return (torch.norm(a - b, dim=1) / (torch.norm(b, dim=1) + 1e-9)).mean().item()


def mean_angle(a, b):
    cos = F.cosine_similarity(a, b, dim=1).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos)).mean().item()


def align_proj(delta, ref):
    """Direction alignment (mean per-sample cos) and AGGREGATE projection coefficient
    sum(delta.ref)/sum(ref.ref) -- a stable scalar; per-sample ratios are unstable when ref is small."""
    num = (delta * ref).sum(1)
    den = (ref * ref).sum(1) + 1e-12
    cos = num / (torch.norm(delta, dim=1) * torch.norm(ref, dim=1) + 1e-12)
    agg = (num.sum() / (den.sum() + 1e-12)).item()
    med = (num / den).median().item()
    return cos.mean().item(), agg, med


def boot_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, 1)
    q = F.softmax(q_logits, 1)
    return F.kl_div(p, q, reduction="batchmean").item()


def top1(logits):
    return logits.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_held", type=int, default=200)
    ap.add_argument("--deltas", default="30,60,90")
    ap.add_argument("--cell_subsample", type=int, default=40, help="max spatial cells per image used for fitting")
    ap.add_argument("--stage", type=int, default=1, help="stage index for the split (1 or 2)")
    ap.add_argument("--op5", action="store_true", help="also fit the content-conditioned MLP-residual operator")
    ap.add_argument("--op7", action="store_true", help="also fit O7 = low-rank base + larger MLP residual")
    ap.add_argument("--op7_rank", type=int, default=16)
    ap.add_argument("--op7_epochs", type=int, default=800)
    ap.add_argument("--op8", action="store_true", help="also fit O8 = ridge 1x1 + 3x3 conv residual")
    ap.add_argument("--op8_epochs", type=int, default=800)
    ap.add_argument("--op8_hidden", type=int, default=32)
    ap.add_argument("--op5_epochs", type=int, default=400)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    deltas = [float(d) for d in a.deltas.split(",")]

    # data
    x_tr_u8, y_tr, x_te_u8, y_te = rbt.load_cifar()
    rng = np.random.default_rng(0)
    tr_idx = rng.permutation(len(x_te_u8))[:a.n_train]
    he_idx = rng.permutation(len(x_te_u8))[:a.n_held]
    def to01(u8):
        return torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0

    # model (kind decides the input lifting used by norm)
    model, kind = rbt.make_arm(a.arm)
    ckpt = os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{a.seed}.pt")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(DEV).eval()
    layer = get_layer(model, a.arm, a.stage)

    def norm(x01):
        if kind == "lift8":
            return (rbt.lift_hue(x01, 8) - rbt.MEAN8) / rbt.STD8
        return (x01 - rbt.MEAN) / rbt.STD

    x_tr01 = to01(x_tr_u8[tr_idx]); y_tr_t = torch.from_numpy(y_tr[tr_idx]).long().to(DEV)
    x_he01 = to01(x_te_u8[he_idx]); y_he = torch.from_numpy(y_te[he_idx]).long().to(DEV)
    x_tr = norm(x_tr01); x_he = norm(x_he01)

    def shifted_plain(x01, deg):
        xs = rbt.rgb_hue_shift_t(x01, torch.full((x01.shape[0],), float(deg), device=DEV))
        return norm(xs)

    out = {"arm": a.arm, "seed": a.seed, "stage": a.stage, "n_train": a.n_train, "n_held": a.n_held,
           "device": DEV, "group_axis_flattened": a.arm in ("ce8", "lcer8"),
           "input_kind": kind, "operators": {}, "deltas": {}}

    # null / real baseline on held-out
    y_null = run_plain(model, x_he)
    acc_null = (top1(y_null) == y_he).float().mean().item()

    for delta in deltas:
        t0 = time.time()
        d = int(delta)
        x_tr_sh = shifted_plain(x_tr01, d)
        # features (fit on train)
        Fx = collect_features(model, layer, x_tr, bs=100)
        Fy = collect_features(model, layer, x_tr_sh, bs=100)
        B, C, H, Wd = Fx.shape
        step = max(1, (H * Wd) // a.cell_subsample)
        Fxr = Fx.permute(0, 2, 3, 1).reshape(-1, C)[::step]
        Fyr = Fy.permute(0, 2, 3, 1).reshape(-1, C)[::step]
        ops = {
            "O1_procrustes": fit_procrustes(Fxr, Fyr),
            "O2_ridge": fit_ridge(Fxr, Fyr),
            "O4_blockrot": fit_blockrot(Fxr, Fyr),
            "O6_random": random_orth(C, seed=a.seed),
        }
        if a.op5:
            ops["O5_mlp"] = make_op5(Fxr, Fyr, epochs=a.op5_epochs, seed=a.seed)
        if a.op7:
            ops["O7_lowrank_mlp"] = make_op7(Fxr, Fyr, r=a.op7_rank, epochs=a.op7_epochs, seed=a.seed)
        if a.op8:
            ops["O8_conv_residual"] = make_op8(Fx, Fy, epochs=a.op8_epochs, hidden=a.op8_hidden, seed=a.seed)
            if DEV == "cuda":
                torch.cuda.empty_cache()
        # held-out routes
        x_he_sh = shifted_plain(x_he01, d)
        y_real = run_plain(model, x_he_sh)
        # feature fidelity on held-out
        fh = collect_features(model, layer, x_he, bs=100)
        fh_real = collect_features(model, layer, x_he_sh, bs=100)
        floor = (x_he01 - rbt.rgb_hue_shift_t(x_he01, torch.zeros(x_he.shape[0], device=DEV))).abs().max().item()
        rec = {"power": {
                   "rel_l2_y_real_vs_null": rel_l2(y_real, y_null),
                   "top1_flip_rate_real_vs_null": (top1(y_real) != top1(y_null)).float().mean().item(),
                   "acc_null": acc_null,
                   "acc_real": (top1(y_real) == y_he).float().mean().item(),
                   "hsv_roundtrip_max_abs": floor,
               },
               "feature_fidelity": {}, "downstream": {}}
        fh_rows = fh.permute(0, 2, 3, 1).reshape(-1, C)
        fr_rows = fh_real.permute(0, 2, 3, 1).reshape(-1, C)

        def sample_dist(p, q):
            return (torch.norm(p - q, dim=1) / (torch.norm(q, dim=1) + 1e-9)).cpu().numpy()

        d_null = sample_dist(y_null, y_real)
        for name, W in ops.items():
            if isinstance(W, SpatialOp):
                tW = W.apply4(fh).permute(0, 2, 3, 1).reshape(-1, C)
            else:
                tW = W(fh_rows) if callable(W) else (fh_rows @ W.T)
            rec["feature_fidelity"][name] = {
                "rel_l2": (torch.norm(tW - fr_rows, dim=1) / (torch.norm(fr_rows, dim=1) + 1e-9)).mean().item(),
                "angle_deg": mean_angle(tW, fr_rows),
            }
            y_int = run_intervened(model, layer, x_he, W)
            cos, agg_proj, med_proj = align_proj(y_int - y_null, y_real - y_null)
            d_int = sample_dist(y_int, y_real)
            gain = d_null - d_int            # >0: intervention is closer to the real recolour than null
            lo, hi = boot_ci(gain)
            rec["downstream"][name] = {
                "rel_l2_vs_real": rel_l2(y_int, y_real),
                "rel_l2_vs_null": rel_l2(y_int, y_null),
                "kl_real_int": kl(y_int, y_real),
                "top1_agree_real": (top1(y_int) == top1(y_real)).float().mean().item(),
                "acc_int": (top1(y_int) == y_he).float().mean().item(),
                "align_cos": cos,
                "proj_coef_agg": agg_proj,
                "proj_coef_median": med_proj,
                "vs_null_win_rate": float((d_int < d_null).mean()),
                "vs_null_mean_gain": float(gain.mean()),
                "vs_null_gain_ci95": [lo, hi],
            }
        rec["feature_fidelity"]["null"] = {
            "rel_l2": (torch.norm(fh_rows - fr_rows, dim=1) / (torch.norm(fr_rows, dim=1) + 1e-9)).mean().item(),
            "angle_deg": mean_angle(fh_rows, fr_rows),
        }
        rec["downstream"]["null"] = {
            "rel_l2_vs_real": rel_l2(y_null, y_real),
            "rel_l2_vs_null": 0.0, "kl_real_int": kl(y_null, y_real),
            "top1_agree_real": (top1(y_null) == top1(y_real)).float().mean().item(),
            "acc_int": acc_null, "align_cos": 0.0,
            "proj_coef_agg": 0.0, "proj_coef_median": 0.0,
            "vs_null_win_rate": 0.0, "vs_null_mean_gain": 0.0, "vs_null_gain_ci95": [0.0, 0.0],
        }
        rec["runtime_sec"] = round(time.time() - t0, 1)
        out["deltas"][str(d)] = rec
        print(f"[{a.arm} s{a.seed}] delta={d} power(relL2 real-null)={rec['power']['rel_l2_y_real_vs_null']:.4f} "
              f"flip={rec['power']['top1_flip_rate_real_vs_null']:.3f} | "
              + " ".join(f"{k}:{v['rel_l2_vs_real']:.3f}/cos{v['align_cos']:.2f}/p{v['proj_coef_agg']:.2f}/win{v['vs_null_win_rate']:.2f}"
                         for k, v in rec['downstream'].items() if k != 'null'), flush=True)
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    suffix = "" if a.stage == 1 else f"_stage{a.stage}"
    fn = os.path.join(WORK, "results", f"causal_cifar_{a.arm}_s{a.seed}{suffix}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
