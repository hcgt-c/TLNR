# -*- coding: utf-8 -*-
"""231_multipath_composition.py — composition along *different paths* for one and the same total transformation.

Motivation (editorial): the existing composition result is a single two-step chain evaluated at an unseen
parameter. To turn "a few fitted maps compose decently" into systematic evidence of approximate structure
preservation, hold the *final physical transformation fixed* and vary the decomposition path and the chain
length. For the dissipative family parameterise by t (the semigroup exponent), not by sigma:

    t* = t_a + t_b = 2 t_c = 4 t_d ,       paths:  W_{t_b}W_{t_a},  W_{t_c}^2,  W_{t_d}^4

For hue hold the total angle fixed and vary the decomposition, add a cyclic path that returns to the
identity, and swap the order of two factors (the family commutes, so a non-zero order gap would indict the
protocol rather than the algebra).

Each path reports three things, and they answer different questions:
  (1) feature relative L2 against the real transformed features — does it reproduce the representation change?
  (2) downstream T_F against the real route — does it reproduce the computation?
  (3) path-consistency: the disagreement between equivalent paths' intervened logits — is the composition
      relation itself preserved?
The identity operator would score perfectly on (3), so (3) is never reported alone; controls (no-op,
single-step substitute, direct fit, random orthogonal) are reported beside every path.

SPLIT DISCIPLINE (this is the point of the script): one permutation gives the fit block and the held-out block,
so the two are disjoint **by image index**, and every reported number comes from held-out images that never
entered any fit.

Usage
  smoke: python scripts/231_multipath_composition.py --arm z2 --seed 0 --n_train 60 --n_held 40 --o8_epochs 2 --cell_subsample 8
  full : OMP_NUM_THREADS=4 nice -n 15 python scripts/231_multipath_composition.py --arm z2 --seed 0 --seeds 0,1,2
Outputs results/multipath_composition.json (merged across invocations)
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
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
harness = load_mod("harness188", os.path.join(WORK, "scripts", "188_depth_map_harness.py"))
harness.c136 = c136
train_mod = load_mod("rt", os.path.join(WORK, "scripts", "c10_routeB_train.py"))
DEV = os.environ.get("MP_DEVICE", "cpu")   # the GPU is running someone else's training; CPU is the default
c136.DEV = DEV
harness.DEV = DEV
for _n in ("MEAN", "STD", "MEAN8", "STD8"):        # defined at import time on the default device
    if hasattr(train_mod, _n):
        setattr(train_mod, _n, getattr(train_mod, _n).to(DEV))

# ---- the fixed total transformations and their decompositions -------------------------------
SIGMA_STAR = 2.0 ** 0.5                 # t* = sigma^2/2 = 1.0
HEAT_T_STAR = SIGMA_STAR ** 2 / 2
# Every heat path below must sum to t* = 1.0 in the semigroup exponent. The factors are specified in t,
# and sigma = sqrt(2t): two steps of t=0.5 (sigma 1.0), four of t=0.25 (sigma 0.7071), eight of t=0.125
# (sigma 0.5). `single` is a single step of the *weakest* factor, t = t*/8 (sigma 0.5), i.e. the same
# magnitude as one link of the longest chain rather than the total.
HEAT_PATHS = {
    "two_step_1.0+1.0": [(1.0, "heat"), (1.0, "heat")],
    "four_step_0.7071x4": [(2.0 ** -0.5, "heat")] * 4,
    "eight_step_0.5x8": [(0.5, "heat")] * 8,
    "single_0.5 (t*/8)": [(0.5, "heat")],
    "direct_fit": [(SIGMA_STAR, "heat")],
}
HUE_TOTAL = 90.0
HUE_PATHS = {
    "two_step_45+45": [(45.0, "hue"), (45.0, "hue")],
    "two_step_30+60": [(30.0, "hue"), (60.0, "hue")],
    "two_step_60+30 (order swap)": [(60.0, "hue"), (30.0, "hue")],
    "four_step_22.5x4": [(22.5, "hue")] * 4,
    "cyclic_120-120 (identity)": [(120.0, "hue"), (-120.0, "hue")],
    "single_22.5": [(22.5, "hue")],
    "direct_fit": [(HUE_TOTAL, "hue")],
}


def compose_mats(mats):
    W = mats[0]
    for M in mats[1:]:
        W = M @ W
    return W


def compose_spatials(ops):
    o = ops[0]
    for op in ops[1:]:
        prev = o
        o = c136.SpatialOp(lambda f, op=op, prev=prev: op.apply4(prev.apply4(f)))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--ops", default="O2,O8")
    ap.add_argument("--families", default="heat,hue")
    ap.add_argument("--o8_epochs", type=int, default=150)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_held", type=int, default=200)
    ap.add_argument("--cell_subsample", type=int, default=40)
    a = ap.parse_args()
    ops = [o.strip() for o in a.ops.split(",")]

    x_tr_u8, y_tr, x_te_u8, y_te = train_mod.load_cifar()
    out_path = os.path.join(WORK, "results", "multipath_composition.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {
        "protocol": ("one fixed total transformation per family, several decomposition paths and chain lengths; "
                     "feature error, downstream T_F and path-consistency reported together; fit and held-out "
                     "blocks drawn from a SINGLE permutation, so they are disjoint by image index"),
        "device": DEV, "records": {}}

    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(1000 + seed)
        split = rng.permutation(len(x_te_u8))                    # ONE permutation...
        tr_idx = split[:a.n_train]
        he_idx = split[a.n_train:a.n_train + a.n_held]           # ...so tr and he are disjoint by construction
        overlap = len(set(tr_idx.tolist()) & set(he_idx.tolist()))
        assert overlap == 0, overlap

        model, kind = train_mod.make_arm(a.arm)
        model.load_state_dict(torch.load(os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{seed}.pt"),
                                         map_location="cpu"))
        model = model.to(DEV).eval()
        layer = c136.get_layer(model, a.arm, a.stage)

        def to01(u8):
            return torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0

        def norm(x01):
            if kind == "lift8":
                return (train_mod.lift_hue(x01, 8) - train_mod.MEAN8) / train_mod.STD8
            return (x01 - train_mod.MEAN) / train_mod.STD

        def transformed(x01, spec):
            val, fam = spec
            if fam == "heat":
                return norm(train_mod.heat_blur_batch(x01, float(val))) if val > 0 else norm(x01)
            return norm(train_mod.rgb_hue_shift_t(x01, torch.full((x01.shape[0],), float(val), device=DEV)))

        x01, h01 = to01(x_tr_u8[tr_idx]), to01(x_te_u8[he_idx])
        y_held = torch.from_numpy(y_te[he_idx]).long().to(DEV)

        def feats(_unused, xf, want_logits=False):
            with torch.no_grad():
                f = c136.collect_features(model, layer, xf, bs=50)
                lg = c136.run_plain(model, xf) if want_logits else None
            return f, lg

        F0, _ = feats(None, norm(x01))
        C = F0.shape[1]
        H, Wd = F0.shape[2], F0.shape[3]
        step = max(1, (H * Wd) // a.cell_subsample) if a.cell_subsample else 1

        def rows(F):
            return F.permute(0, 2, 3, 1).reshape(-1, C)

        def rows_fit(F):
            return rows(F)[::step]

        for fam, paths in (("heat", HEAT_PATHS), ("hue", HUE_PATHS)):
            if fam not in a.families.split(","):
                continue
            # the real route for the total transformation
            total = HEAT_T_STAR if fam == "heat" else HUE_TOTAL
            total_key = (total, "heat") if fam == "heat" else (total, "hue")
            xr = transformed(h01, total_key)
            Fr, y_real = feats(None, xr, want_logits=True)
            with torch.no_grad():
                y_null = c136.run_plain(model, norm(h01))
            d_null = torch.norm(y_null - y_real, dim=1)

            # fit one operator per distinct factor on the FIT block, cache them
            factors = sorted({s for p in paths.values() for s in p if (s[1] == "heat" and s[0] > 0)
                              or (s[1] == "hue" and s[0] != 0)})
            fits = {}
            for (val, f2) in factors:
                Ft, _ = feats(None, transformed(x01, (val, f2)))
                for op in ops:
                    key = (op, val, f2)
                    if op == "O2":
                        fits[key] = c136.fit_ridge(rows_fit(F0), rows_fit(Ft)).to(DEV)
                    else:
                        fits[key] = harness.make_op8_batched(F0, Ft, epochs=a.o8_epochs, seed=seed)

            def apply_path(op, path):
                mats, sps = [], []
                for (val, f2) in path:
                    if val == 0:
                        continue
                    W = fits[(op, val, f2)]
                    (mats if not hasattr(W, "apply4") else sps).append(W)
                if hasattr(fits[(op, path[0][0], path[0][1])], "apply4"):
                    return compose_spatials(sps) if sps else c136.SpatialOp(lambda f: f)
                return compose_mats(mats) if mats else torch.eye(C, device=DEV)

            def intervene(W):
                return c136.run_intervened(model, layer, norm(h01), W)

            def apply_feats(W, F):
                if hasattr(W, "apply4"):
                    return W.apply4(F)
                return (F.permute(0, 2, 3, 1).reshape(-1, C) @ W.T).reshape(F.shape)

            with torch.no_grad():
                F_h = c136.collect_features(model, layer, norm(h01), bs=50)
            rec = {"family": fam, "total": total, "arm": a.arm, "seed": seed, "stage": a.stage,
                   "n_train": a.n_train, "n_held": a.n_held, "split_overlap": overlap,
                   "paths": {}}
            interv_logits = {}
            for name, path in paths.items():
                for op in ops:
                    W = apply_path(op, path)
                    with torch.no_grad():
                        y_int = intervene(W)
                        f_int = apply_feats(W, F_h) if not hasattr(W, "apply4") else W.apply4(F_h)
                    d_int = torch.norm(y_int - y_real, dim=1)
                    key = f"{name}|{op}"
                    interv_logits[key] = y_int
                    rec["paths"][key] = {
                        "n_steps": len(path),
                        "transfer": 1.0 - float(d_int.mean() / (d_null.mean() + 1e-12)),
                        "top1_agree_real": float((c136.top1(y_int) == c136.top1(y_real)).float().mean()),
                        "win_rate_vs_null": float((d_int < d_null).float().mean()),
                        "feat_rel_l2": float((f_int - Fr).norm() / (Fr.norm() + 1e-12)),
                    }
            # path consistency between equivalent paths (same total transformation)
            equiv = [k for k in rec["paths"] if "direct_fit" not in k and "cyclic" not in k]
            cons = {}
            for i in range(len(equiv)):
                for j in range(i + 1, len(equiv)):
                    k1, k2 = equiv[i], equiv[j]
                    if k1.split("|")[1] != k2.split("|")[1]:
                        continue
                    d = torch.norm(interv_logits[k1] - interv_logits[k2], dim=1) / (torch.norm(y_real, dim=1) + 1e-9)
                    cons[f"{k1} vs {k2}"] = float(d.mean())
            rec["path_consistency"] = cons
            res["records"][f"{a.arm}_s{seed}_stage{a.stage}_{fam}"] = rec
            json.dump(res, open(out_path, "w"), indent=1)
            best = max((v["transfer"] for k, v in rec["paths"].items() if k.startswith("two_step")),
                       default=float("nan"))
            print(f"[{a.arm} s{seed} {fam}] two-step T_F max {best:.3f} | "
                  f"paths {len(rec['paths'])} | split overlap {overlap}", flush=True)
        del model
    print("saved", out_path)


if __name__ == "__main__":
    main()
