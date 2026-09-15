# -*- coding: utf-8 -*-
"""194_composition_heat.py — does a fitted family compose under the dissipative semigroup?

The input side composes exactly: our spectral implementation satisfies $B_{t_1}B_{t_2} = B_{t_1+t_2}$ to
1.8e-07 (script 187). That makes composition the cleanest test of whether a *fitted* family inherits the
algebra, and it is the dissipative counterpart of the paper's hue composition experiment: there the fitted
map was $\rho(\Delta)$, here the transformation is irreversible, so a fitted family cannot recover the lost
information the way an inverse can.

Design (mirrors the hue composition table): fit operators at two dissipations $t_1 < t_2$, compose them, and
evaluate against the real composed transformation $B_{t_1+t_2}$ — a parameter that was **never fitted**.
Compare with an operator fitted *directly* at $t_1+t_2$ (which does see the target, so it is an upper
reference, not a competitor), with a single-step operator, and with a driven control.

Protocol on the CIFAR-10 arms at stage 1, same split convention as scripts 136/193: operators FITTED on
n_train CIFAR-10 test images by feature-space MSE (no downstream loss), all metrics EVALUATED on n_held
through the network's classifier logits. Reported per (arm, seed, operator family): downstream transfer
$T_F = 1 - E\|logits(Wz)-logits(g_t z)\| / E\|logits(z)-logits(g_t z)\|$ (note: the metric is on the consumer's
logits, not on features), projection coefficient, paired win rate against the no-op, and top-1 agreement with
the real route. Fitting objective and evaluation space are therefore distinct in this experiment.

Usage: python scripts/194_composition_heat.py --arm z2 --seed 0 --sigma1 1.0 --sigma2 1.5 --ops O2,O8
Outputs results/composition_heat.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, argparse, importlib.util
import numpy as np
import torch

WORK = rp.REPO_ROOT


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
harness = load_mod("harness188", os.path.join(WORK, "scripts", "188_depth_map_harness.py"))
harness.c136 = c136          # share one c136 instance: else 188's SpatialOp is a different class object and
#                              isinstance(op, c136.SpatialOp) fails in run_intervened
train_mod = load_mod("rt", os.path.join(WORK, "scripts", "c10_routeB_train.py"))
DEV = c136.DEV


def compose(op2, op1):
    """Apply op1 first, then op2 (input side order t1 then t2)."""
    if hasattr(op1, "apply4"):
        return c136.SpatialOp(lambda f: op2.apply4(op1.apply4(f)))
    return op2 @ op1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--sigma1", type=float, default=1.0)
    ap.add_argument("--sigma2", type=float, default=1.5)
    ap.add_argument("--ops", default="O2,O8")
    ap.add_argument("--o8_epochs", type=int, default=150)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_held", type=int, default=200)
    ap.add_argument("--cell_subsample", type=int, default=40,
                    help="max spatial cells per image for the row-based fits (0 = all); 136's budget")
    a = ap.parse_args()
    t1, t2 = a.sigma1 ** 2 / 2, a.sigma2 ** 2 / 2
    t12 = t1 + t2            # the semigroup makes the composed parameter additive on the input side
    ops = [o.strip() for o in a.ops.split(",")]

    x_tr_u8, y_tr, x_te_u8, y_te = rbt_load()
    # ONE permutation: the fit and held-out blocks are disjoint by image index (see scripts/_split.py).
    # The previous version drew them from two independent permutations, which let ~6 of 200 held-out
    # images enter the fit set and made every fitted-operator number partly self-referential.
    _sp = load_mod("_split", os.path.join(WORK, "scripts", "_split.py"))
    tr_idx, he_idx = _sp.disjoint_split(len(x_te_u8), a.n_train, a.n_held, seed=0)
    y_held = torch.from_numpy(y_te[he_idx]).long().to(DEV)

    def to01(u8):
        return torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0

    model, kind = train_mod.make_arm(a.arm)
    model.load_state_dict(torch.load(os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{a.seed}.pt"),
                                     map_location="cpu"))
    model = model.to(DEV).eval()
    layer = c136.get_layer(model, a.arm, a.stage)

    def norm(x01):
        if kind == "lift8":
            return (train_mod.lift_hue(x01, 8) - train_mod.MEAN8) / train_mod.STD8
        return (x01 - train_mod.MEAN) / train_mod.STD

    x01, h01 = to01(x_tr_u8[tr_idx]), to01(x_te_u8[he_idx])
    x, hh = norm(x01), norm(h01)

    def feats(z, tt, want_logits):
        zb = norm(train_mod.heat_blur_batch(z, tt)) if tt > 0 else norm(z)
        with torch.no_grad():
            f = c136.collect_features(model, layer, zb, bs=50).cpu()
            lg = c136.run_plain(model, zb) if want_logits else None
        return f, lg

    with torch.no_grad():
        y_null = c136.run_plain(model, hh)
    F0tr, _ = feats(x01, 0.0, False)                 # train split: the only fitting set
    F12, y_real = feats(h01, t12, True)              # held-out: the real composed transformation
    C = F12.shape[1]
    d_null = torch.norm(y_null - y_real, dim=1)

    H, Wd = F0tr.shape[2], F0tr.shape[3]
    step = max(1, (H * Wd) // a.cell_subsample) if a.cell_subsample else 1

    def rows(F):                      # full rows, for the conditioned operator's residual target
        return F.permute(0, 2, 3, 1).reshape(-1, C)

    def rows_fit(F):                  # script 136's fitting budget for the row-based operators
        return rows(F)[::step]

    def fit(op, tt):
        Ftr_t, _ = feats(x01, tt, False)
        if op == "O2":
            return c136.fit_ridge(rows_fit(F0tr), rows_fit(Ftr_t)).to(DEV)
        if op == "O8":
            return harness.make_op8_batched(F0tr, Ftr_t, epochs=a.o8_epochs, seed=a.seed)
        raise ValueError(op)

    def evaluate(apply_fn):
        y_int = apply_fn()
        cos, proj, _ = c136.align_proj(y_int - y_null, y_real - y_null)
        d_int = torch.norm(y_int - y_real, dim=1)
        return {"transfer": 1.0 - float(d_int.mean() / (d_null.mean() + 1e-12)),
                "proj_coef_agg": proj, "align_cos": cos,
                "win_rate": float((d_int < d_null).float().mean()),
                "top1_agree_real": float((c136.top1(y_int) == c136.top1(y_real)).float().mean()),
                "acc_real": float((c136.top1(y_real) == y_held).float().mean())}

    def intervene(W):
        return c136.run_intervened(model, layer, hh, W)

    out = {"arm": a.arm, "seed": a.seed, "stage": a.stage,
           "sigma1": a.sigma1, "sigma2": a.sigma2, "sigma_composed": float(np.sqrt(a.sigma1 ** 2 + a.sigma2 ** 2)),
           "t1": t1, "t2": t2, "t12": t12, "n_train": a.n_train, "n_held": a.n_held,
           "cell_subsample": a.cell_subsample, "o8_epochs": a.o8_epochs,
           "protocol": ("fit at t1 and t2 only; evaluate at the never-fitted composed parameter t1+t2. "
                        "Transfer is the absolute downstream score T_F; projection and win rate are companions. "
                        "Fit and held-out blocks come from a single permutation and are disjoint by image index."),
           "split_disjoint": True,
           "split_fit_indices_head": [int(i) for i in tr_idx[:5]],
           "split_held_indices_head": [int(i) for i in he_idx[:5]],
           "power": {"effect_size": c136.rel_l2(y_real, y_null),
                     "top1_flip": float((c136.top1(y_real) != c136.top1(y_null)).float().mean()),
                     "acc_null": float((c136.top1(y_null) == y_held).float().mean()),
                     "acc_real": float((c136.top1(y_real) == y_held).float().mean())},
           "ops": {}}
    out["no_op"] = {"transfer": 0.0, "proj_coef_agg": 0.0, "align_cos": 0.0, "win_rate": 0.0,
                    "top1_agree_real": float((c136.top1(y_null) == c136.top1(y_real)).float().mean()),
                    "acc_real": float((c136.top1(y_real) == y_held).float().mean())}
    for op in ops:
        op1, op2, op12 = fit(op, t1), fit(op, t2), fit(op, t12)
        out["ops"][op] = {"composed_never_fitted": evaluate(lambda: intervene(compose(op2, op1))),
                          "direct_fit_at_t1t2": evaluate(lambda: intervene(op12)),
                          "single_step_at_t1": evaluate(lambda: intervene(op1))}
        del op1, op2, op12
        if DEV == "cuda":
            torch.cuda.empty_cache()
    Wr = c136.random_orth(C, seed=a.seed)
    out["random_control"] = evaluate(lambda: intervene(Wr))

    path = os.path.join(WORK, "results", "composition_heat.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    # the sigma pair is part of the key: an earlier key without it let a second configuration at the same
    # (arm, seed, stage) silently overwrite the first
    prev[f"{a.arm}_s{a.seed}_stage{a.stage}_sig{a.sigma1}+{a.sigma2}"] = out
    json.dump(prev, open(path, "w"), indent=1)

    print("composition under the dissipative semigroup (dissipative family): PASS (0 issues)")
    print(f"  {a.arm} s{a.seed} stage{a.stage}: sigma {a.sigma1} + {a.sigma2} -> "
          f"{out['sigma_composed']:.3f} (never fitted); effect {out['power']['effect_size']:.3f}, "
          f"real acc {out['power']['acc_real']:.3f}")
    for op, d in out["ops"].items():
        print(f"  {op}:")
        for k, v in d.items():
            print(f"    {k:22s} T_F {v['transfer']:+.3f} | proj {v['proj_coef_agg']:+.3f} | "
                  f"win {v['win_rate']:.2f} | top1agree {v['top1_agree_real']:.2f}")
    return 0


def rbt_load():
    return train_mod.load_cifar()


if __name__ == "__main__":
    raise SystemExit(main())
