# -*- coding: utf-8 -*-
"""146_unseen_delta_cifar.py — generalisation to an unseen hue shift on CIFAR arms.

Fit operators only at delta = 30 and 60 (train split), compose them, and evaluate at delta = 90
without ever fitting 90. Compare with (i) the null (no intervention), (ii) operators fitted directly
at 90 (reference upper bound), and (iii) a random orthogonal control. Operators: O1 (global
Procrustes) and O8 (ridge 1x1 + 3x3 conv residual, i.e. content- and neighbourhood-conditioned).

Usage: python scripts/146_unseen_delta_cifar.py --arm z2 --seed 0 --n_train 200 --n_held 200
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_held", type=int, default=200)
    ap.add_argument("--op8_epochs", type=int, default=400)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if a.dev == "cuda" else ""
    C = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
    rbt = C.rbt
    DEV = "cuda" if (a.dev == "cuda" and torch.cuda.is_available()) else "cpu"
    C.DEV = DEV

    torch.manual_seed(0); np.random.seed(0)
    x_tr_u8, y_tr, x_te_u8, y_te = rbt.load_cifar()
    rng = np.random.default_rng(0)
    tr = rng.permutation(len(x_te_u8))[:a.n_train]
    he = rng.permutation(len(x_te_u8))[:a.n_held]

    model, kind = rbt.make_arm(a.arm)
    sd = torch.load(os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{a.seed}.pt"), map_location="cpu")
    model.load_state_dict(sd); model = model.to(DEV).eval()
    layer = C.get_layer(model, a.arm, a.stage)

    def to01(u8):
        return torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0

    def norm(x01):
        if kind == "lift8":
            return (rbt.lift_hue(x01, 8) - rbt.MEAN8) / rbt.STD8
        return (x01 - rbt.MEAN) / rbt.STD

    def shift(x01, deg):
        return norm(rbt.rgb_hue_shift_t(x01, torch.full((x01.shape[0],), float(deg), device=DEV)))

    x_tr01, x_he01 = to01(x_tr_u8[tr]), to01(x_te_u8[he])
    y_he = torch.from_numpy(y_te[he]).long().to(DEV)
    xtr, xhe = norm(x_tr01), norm(x_he01)

    # fit at 30 and 60
    F30x = C.collect_features(model, layer, xtr, bs=100)
    F30y = C.collect_features(model, layer, shift(x_tr01, 30), bs=100)
    F60y = C.collect_features(model, layer, shift(x_tr01, 60), bs=100)
    B, Cc, H, Wd = F30x.shape
    Xr = F30x.permute(0, 2, 3, 1).reshape(-1, Cc)
    W30 = C.fit_procrustes(Xr, F30y.permute(0, 2, 3, 1).reshape(-1, Cc))
    W60 = C.fit_procrustes(F30x.permute(0, 2, 3, 1).reshape(-1, Cc), F60y.permute(0, 2, 3, 1).reshape(-1, Cc))
    W90_unseen = W60 @ W30
    o8_30 = C.make_op8(F30x, F30y, epochs=a.op8_epochs, seed=a.seed)
    o8_60 = C.make_op8(F30x, F60y, epochs=a.op8_epochs, seed=a.seed)
    o8_unseen = C.SpatialOp(lambda f4: o8_60.apply4(o8_30.apply4(f4)))

    # reference: fit directly at 90
    F90y = C.collect_features(model, layer, shift(x_tr01, 90), bs=100)
    W90_ref = C.fit_procrustes(Xr, F90y.permute(0, 2, 3, 1).reshape(-1, Cc))
    o8_90ref = C.make_op8(F30x, F90y, epochs=a.op8_epochs, seed=a.seed)
    Wrand = C.random_orth(Cc, seed=a.seed)

    # held-out evaluation at delta=90
    x_real = shift(x_he01, 90)
    y_real = C.run_plain(model, x_real)
    y_null = C.run_plain(model, xhe)
    fh = C.collect_features(model, layer, xhe, bs=100)
    fh_real = C.collect_features(model, layer, x_real, bs=100)
    rows = fh.permute(0, 2, 3, 1).reshape(-1, Cc)
    rrows = fh_real.permute(0, 2, 3, 1).reshape(-1, Cc)

    def feat_metrics(apply_rows=None, apply4=None):
        if apply4 is not None:
            t = apply4(fh).permute(0, 2, 3, 1).reshape(-1, Cc)
        else:
            t = apply_rows(rows)
        return {"rel_l2": ((torch.norm(t - rrows, dim=1) / (torch.norm(rrows, dim=1) + 1e-9)).mean().item()),
                "angle_deg": C.mean_angle(t, rrows)}

    out = {"arm": a.arm, "seed": a.seed, "stage": a.stage, "n_train": a.n_train, "n_held": a.n_held,
           "target_delta": 90, "fitted_deltas": [30, 60], "device": DEV,
           "power": {"rel_l2_real_vs_null": C.rel_l2(y_real, y_null),
                     "top1_flip_rate": (C.top1(y_real) != C.top1(y_null)).float().mean().item(),
                     "acc_null": (C.top1(y_null) == y_he).float().mean().item(),
                     "acc_real": (C.top1(y_real) == y_he).float().mean().item()},
           "variants": {}}

    def eval_variant(name, apply_rows=None, apply4=None, is_null=False):
        # `is_null` marks the reference application: routing the identity through the intervention hook
        # leaves a ~1e-7 relative round-off, which would break ties and fabricate a non-zero win rate and a
        # spurious cosine against the real displacement. For the reference these statistics are 0 and
        # undefined by construction, and the round-off is recorded as a numerical check instead.
        fe = feat_metrics(apply_rows, apply4)
        if apply4 is not None:
            y_int = C.run_intervened(model, layer, xhe, C.SpatialOp(apply4))
        else:
            y_int = C.run_intervened(model, layer, xhe, apply_rows)
        cos, agg, med = C.align_proj(y_int - y_null, y_real - y_null)
        d_int = (torch.norm(y_int - y_real, dim=1) / (torch.norm(y_real, dim=1) + 1e-9)).cpu().numpy()
        d_null = (torch.norm(y_null - y_real, dim=1) / (torch.norm(y_real, dim=1) + 1e-9)).cpu().numpy()
        gain = d_null - d_int
        lo, hi = C.boot_ci(gain)
        out["variants"][name] = {
            "feature": fe,
            "rel_l2_vs_real": C.rel_l2(y_int, y_real),
            "rel_l2_vs_null": C.rel_l2(y_int, y_null),
            "top1_agree_real": (C.top1(y_int) == C.top1(y_real)).float().mean().item(),
            "acc_int": (C.top1(y_int) == y_he).float().mean().item(),
            "align_cos": (None if is_null else cos), "proj_coef_agg": (0.0 if is_null else agg),
            "proj_coef_median": (None if is_null else med),
            "vs_null_win_rate": (0.0 if is_null else float((d_int < d_null).mean())),
            "null_path_roundoff": float(C.rel_l2(y_int, y_null)),
            "vs_null_mean_gain": float(gain.mean()), "vs_null_gain_ci95": [lo, hi],
        }

    eval_variant("null_apply", apply_rows=lambda r: r, is_null=True)
    eval_variant("O1_unseen_compose", apply_rows=lambda r: r @ W90_unseen.T)
    eval_variant("O1_ref_fit90", apply_rows=lambda r: r @ W90_ref.T)
    eval_variant("O6_random", apply_rows=lambda r: r @ Wrand.T)
    eval_variant("O8_unseen_compose", apply4=lambda f4: o8_unseen.apply4(f4))
    eval_variant("O8_ref_fit90", apply4=lambda f4: o8_90ref.apply4(f4))

    fn = os.path.join(WORK, "results", f"causal_unseen_{a.arm}_s{a.seed}_stage{a.stage}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)
    for k, v in out["variants"].items():
        print(f"  {k:22s} feat {v['feature']['rel_l2']:.3f} | out relL2 {v['rel_l2_vs_real']:.3f} "
              f"| cos {v['align_cos']} | proj {v['proj_coef_agg']:.2f} | win {v['vs_null_win_rate']:.2f}")


if __name__ == "__main__":
    main()
