# -*- coding: utf-8 -*-
"""185_rotation_pilot.py — de-risk the spatial-rotation experiment before committing to it.

The TPAMI plan proposes adding a second transformation family (spatial rotation) so that the framework is
not a colour-only story. Two facts decide whether that experiment can work at all, and both are cheap to
measure:

  1. POWER. On a globally average-pooled classifier a 90-degree rotation of the input is nearly invisible,
     because pooling removes the spatial arrangement. If the effect size is at the noise floor, the test is
     vacuous by the paper's own precondition 3 and any operator would look faithful. We therefore measure
     the effect size of an exact rot90 on (a) the logits of the plain-CNN arm and (b) a spatially sensitive
     consumer, the unpooled stage-2 feature map.
  2. EXACTNESS. For square inputs a 90-degree rotation is exact (a pure index permutation), so the
     transformation itself adds no error; we verify this by comparing rot90 against its own inverse.

This is a pilot, not a result: it exists to choose the consumer for the rotation experiment, and it writes
only effect sizes.

Usage: python scripts/185_rotation_pilot.py [--arm z2 --seed 0 --n 200]
Outputs results/rotation_pilot.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, argparse, importlib.util
import torch

WORK = rp.REPO_ROOT


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
rbt = c136.rbt
DEV = c136.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="z2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    x_tr_u8, y_tr, x_te_u8, y_te = rbt.load_cifar()
    x01 = torch.from_numpy(x_te_u8[:a.n]).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(y_te[:a.n]).long().to(DEV)

    model, kind = rbt.make_arm(a.arm)
    sd = torch.load(os.path.join(WORK, "results", "c10_routeB", f"{a.arm}_s{a.seed}.pt"), map_location="cpu")
    model.load_state_dict(sd); model = model.to(DEV).eval()

    def norm(x01_):
        if kind == "lift8":
            return (rbt.lift_hue(x01_, 8) - rbt.MEAN8) / rbt.STD8
        return (x01_ - rbt.MEAN) / rbt.STD

    def rot90(x01_):
        return torch.rot90(x01_, 1, dims=(2, 3))

    x = norm(x01)
    x_r = norm(rot90(x01))
    x_rr = norm(rot90(rot90(x01)))              # sanity: exactness of the permutation

    # exactness: rot90 twice must be a pure pixel permutation of the input, not an interpolation
    exact = float((rot90(rot90(x01)) - torch.rot90(x01, 2, dims=(2, 3))).abs().max())

    out = {"arm": a.arm, "seed": a.seed, "n": a.n,
           "exactness_rot90_twice_max_abs_diff": exact, "device": DEV}

    with torch.no_grad():
        logits = c136.run_plain(model, x)
        logits_r = c136.run_plain(model, x_r)
        L = c136.get_layer(model, a.arm, 2)
        f = c136.collect_features(model, L, x, bs=50)
        f_r = c136.collect_features(model, L, x_r, bs=50)

    out["logits_rel_l2_rot_vs_null"] = c136.rel_l2(logits_r, logits)
    out["logits_top1_flip_rate"] = float((c136.top1(logits_r) != c136.top1(logits)).float().mean())
    out["logits_acc_null"] = float((c136.top1(logits) == y).float().mean())
    out["logits_acc_rot"] = float((c136.top1(logits_r) == y).float().mean())
    # a spatially sensitive consumer: the unpooled feature map, compared elementwise
    out["featuremap_rel_l2_rot_vs_null"] = float(
        ((f_r - f).flatten(1).norm(dim=1) / (f.flatten(1).norm(dim=1) + 1e-9)).mean())
    # and the same map under a circular shift, which a translation-equivariant stack should handle exactly
    with torch.no_grad():
        f_shift = c136.collect_features(model, L, torch.roll(x, 4, dims=3), bs=50)
    out["featuremap_rel_l2_shift4_vs_null"] = float(
        ((f_shift - f).flatten(1).norm(dim=1) / (f.flatten(1).norm(dim=1) + 1e-9)).mean())

    # THE informative quantity: does the feature map transform by the *fixed* spatial operator that the
    # transformation induces? Two details decide the measurement. First, the operator must be expressed in
    # feature-map units: a shift of s input pixels is a shift of s/stride cells, so it is exact only when the
    # stride divides s (we use s = 4 and report the stride, so the alignment is checkable). Second, a 90-degree
    # rotation of a square map is exact, which is why rot90 needs no stride correction. The residual is the
    # fidelity with which a fixed spatial operator can realise the transformation at this site.
    stride = x.shape[-1] // f.shape[-1]

    def residual(f_transformed, f_base, op):
        ref = op(f_base)
        return float(((f_transformed - ref).flatten(1).norm(dim=1) /
                      (ref.flatten(1).norm(dim=1) + 1e-9)).mean())

    out["feature_stride"] = int(stride)
    out["equivariance_residual_rot90"] = residual(f_r, f, lambda t: torch.rot90(t, 1, dims=(2, 3)))
    out["equivariance_residual_shift4_aligned"] = residual(f_shift, f, lambda t: torch.roll(t, 4 // int(stride), dims=3))
    out["equivariance_residual_shift4_unaligned"] = residual(f_shift, f, lambda t: torch.roll(t, 4, dims=3))
    out["shift_cells_used"] = int(4 // stride)
    # the transformation's own effect on the map, with no alignment: the ceiling any fixed operator faces
    out["featuremap_rel_l2_rot_unaligned"] = residual(f_r, f, lambda t: t)

    fn = os.path.join(WORK, "results", "rotation_pilot.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"{a.arm} s{a.seed}: rot90 exactness {exact:.2e}")
    print(f"  logits:      effect size {out['logits_rel_l2_rot_vs_null']:.4f} | "
          f"flip {out['logits_top1_flip_rate']:.3f} | acc {out['logits_acc_null']:.3f} -> {out['logits_acc_rot']:.3f}")
    print(f"  feature map: rot90 {out['featuremap_rel_l2_rot_vs_null']:.4f} | "
          f"shift4 {out['featuremap_rel_l2_shift4_vs_null']:.4f}")
    print("saved", fn)


if __name__ == "__main__":
    main()
