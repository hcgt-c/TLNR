# -*- coding: utf-8 -*-
"""157_multiattribute_intervention.py — multi-attribute intervention on a frozen transformer.

Paper B needs more than one continuous attribute. This script builds a controlled multi-attribute
scene family (hue = circular, saturation = monotone scalar, value = monotone scalar, quantity =
discrete count) and repeats the input-equivalence intervention protocol (script 155) for each
attribute on the SAME frozen ViT-B/16 and the SAME operator families (O1 Procrustes / O5 per-token
MLP / O8 3x3 conv residual / O6 random orthogonal).

Protocol per attribute a with input transform T_a(., delta):
  base input x          -> block-k tokens f_k(x)
  real input T_a(x)     -> block-k tokens f_k(T_a x)      (target)
  operator op fitted on TRAIN pairs (f_k(x), f_k(T_a x))
  intervention route: f_k(x) replaced by op(f_k(x)), tail + frozen attribute probe re-run
Downstream: one Ridge probe per attribute on pooled FINAL features, trained on base+variant of that
attribute only and shared by every route, so no operator gets an advantage.

Reported per attribute/operator: block-k input power (rel L2 variant vs base), representation and
final-feature fidelity to the real route, alignment cosine, aggregate projection coefficient,
paired win rate, and the downstream attribute-transfer fraction (err_null - err_int)/(err_null -
err_real) with err from the probe readout against the variant's true attribute value.

Usage: python scripts/157_multiattribute_intervention.py --backbone vitb16 --block 4 --n_scenes 180
Outputs results/multiattr_intervention_<backbone>_b<block>_s<seed>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math, colorsys, argparse, importlib.util
import numpy as np
from PIL import Image

WORK = rp.REPO_ROOT
SHAPES = ["triangle", "rectangle", "circle", "star", "pentagon", "hexagon"]
SIZE = 224
ATTRS = ["hue", "saturation", "value", "quantity"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(WORK, "scripts", name))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


gen = load_script("01_gen_synthetic.py")
vit = load_script("155_vit_intervention.py")


# ---------------------------------------------------------------- scene family
def sample_object(rs, objs):
    for _ in range(300):
        R = rs.uniform(24.0, 42.0)
        cx = rs.uniform(R + 12, SIZE - R - 12)
        cy = rs.uniform(R + 12, SIZE - R - 12)
        if all(math.hypot(cx - o["cx"], cy - o["cy"]) >= (R + o["R"]) * 0.92 for o in objs):
            return dict(shape=SHAPES[rs.randint(len(SHAPES))], cx=cx, cy=cy, R=R,
                        rot=rs.uniform(0, 2 * math.pi), hue=float(rs.uniform(0, 360)),
                        sat=float(rs.uniform(0.30, 0.50)), val=float(rs.uniform(0.40, 0.55)))
    return None


def sample_scene(rs, nmin=1, nmax=3):
    objs = []
    for _ in range(rs.randint(nmin, nmax + 1)):
        o = sample_object(rs, objs)
        if o is not None:
            objs.append(o)
    return objs or [sample_object(rs, [])]


def render(objs, hue_shift=0.0, sat_scale=1.0, val_scale=1.0, keep=None):
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    for o in (objs if keep is None else objs[:keep]):
        h = (o["hue"] + hue_shift) % 360.0
        s = min(1.0, max(0.05, o["sat"] * sat_scale))
        v = min(1.0, max(0.05, o["val"] * val_scale))
        rgb = colorsys.hsv_to_rgb(h / 360.0, s, v)
        gen.draw_shape(img, o["shape"], (o["cx"], o["cy"]), o["R"], o["rot"],
                       (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return np.asarray(img)


def attr_value(objs, attr, var):
    """Ground-truth attribute of the (base or variant) scene."""
    if attr == "hue":
        # circular mean over objects (a linear mean of hue angles is not well defined)
        hc = float(np.mean([math.cos(math.radians(o["hue"])) for o in objs]))
        hs = float(np.mean([math.sin(math.radians(o["hue"])) for o in objs]))
        h = (math.degrees(math.atan2(hs, hc)) + (90.0 if var else 0.0)) % 360.0
        return np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])
    if attr == "saturation":
        return float(min(1.0, np.mean([o["sat"] for o in objs]) * (2.0 if var else 1.0)))
    if attr == "value":
        return float(min(1.0, np.mean([o["val"] for o in objs]) * (1.8 if var else 1.0)))
    if attr == "quantity":
        return float(min(len(objs) + (1 if var else 0), 4))
    raise ValueError(attr)


def variant_image(objs, attr):
    if attr == "hue":
        return render(objs, hue_shift=90.0)
    if attr == "saturation":
        return render(objs, sat_scale=2.0)
    if attr == "value":
        return render(objs, val_scale=1.8)
    if attr == "quantity":
        k = min(len(objs) + 1, 4)
        if k > len(objs):
            seed = 1000 + int(objs[0]["cx"] * 7 + objs[0]["cy"] * 13)
            o = sample_object(np.random.RandomState(seed), objs)
            objs = objs + ([o] if o is not None else [])
        return render(objs, keep=k)
    raise ValueError(attr)


def attr_err(pred, true, attr):
    if attr == "hue":
        d = math.degrees(math.atan2(pred[1], pred[0]) - math.atan2(true[1], true[0]))
        return abs((d + 180.0) % 360.0 - 180.0)
    return float(abs(float(pred) - float(true)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitb16")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_scenes", type=int, default=180)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--fit_imgs", type=int, default=100)
    ap.add_argument("--op8_epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=30)
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import Ridge

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rs = np.random.RandomState(a.seed)

    scenes = [sample_scene(rs) for _ in range(a.n_scenes)]
    perm = rs.permutation(a.n_scenes)
    te = np.sort(perm[:a.n_test]); tr = np.sort(perm[a.n_test:])
    fit_i = tr[:min(a.fit_imgs, len(tr))]

    model, kind = vit.build_backbone(a.backbone, dev)
    blocks, norm = vit.blocks_of(model, kind)
    patch = model.patch_size if kind == "dinov2" else 16
    grid = (SIZE // patch, SIZE // patch)
    D = model.embed_dim if kind == "dinov2" else model.hidden_dim
    mean = vit.MEAN.to(dev); std = vit.STD.to(dev)

    def prep(u8):
        x = torch.from_numpy(np.asarray(u8)).float().permute(0, 3, 1, 2).to(dev) / 255.0
        return (x - mean) / std

    def feats(u8):
        fb, ff = [], []
        with torch.no_grad():
            x = prep(u8)
            for s in range(0, x.shape[0], a.bs):
                t = vit.tokens_of(model, kind, x[s:s + a.bs])
                for blk in blocks[:a.block + 1]:
                    t = blk(t)
                fb.append(t.cpu())
                for blk in blocks[a.block + 1:]:
                    t = blk(t)
                ff.append(vit.pool(norm(t)).cpu().numpy())
        return torch.cat(fb, 0), np.concatenate(ff, 0)

    def tail_pool(fk):
        with torch.no_grad():
            t = fk.to(dev)
            for blk in blocks[a.block + 1:]:
                t = blk(t)
            return vit.pool(norm(t)).cpu().numpy()

    base_u8 = np.stack([render(o) for o in scenes])
    Bk_all, Bf_all = feats(base_u8)

    Q, _ = np.linalg.qr(np.random.RandomState(a.seed + 7).normal(size=(D, D)).astype(np.float32))
    W6 = torch.from_numpy(Q.astype(np.float32)).to(dev)

    out = {"backbone": a.backbone, "block": a.block, "seed": a.seed, "n_scenes": a.n_scenes,
           "n_fit": int(len(fit_i)), "n_test": int(len(te)), "attr_order": ATTRS,
           "grid": list(grid), "dim": D, "attributes": {}}

    for attr in ATTRS:
        var_u8 = np.stack([variant_image(o, attr) for o in scenes])
        Vk_all, Vf_all = feats(var_u8)

        Xp = np.concatenate([Bf_all[tr], Vf_all[tr]], 0)
        yp = np.stack([attr_value(scenes[i], attr, j) for j in (0, 1) for i in tr])
        probe = Ridge(alpha=1.0).fit(Xp, yp)

        def probe_of(nf):
            p = probe.predict(nf)
            if attr == "hue":
                p = np.atleast_2d(p)
                p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
            return p

        def mat(v):
            return np.asarray(v).reshape(len(te), -1)

        # 155's fit_ops strips the CLS token itself, so pass the full sequence
        Fx = Bk_all[fit_i].to(dev)
        Fy = Vk_all[fit_i].to(dev)
        ops = dict(vit.fit_ops(Fx, Fy, grid, a.seed, a.op8_epochs))
        ops["O6_random_orthogonal"] = lambda t: t @ W6.T

        power_b = float(np.linalg.norm((Vk_all[te][:, 1:] - Bk_all[te][:, 1:]).numpy(), axis=(1, 2)).mean() /
                        (np.linalg.norm(Vk_all[te][:, 1:].numpy(), axis=(1, 2)).mean() + 1e-9))
        p_base, p_var = probe_of(Bf_all[te]), probe_of(Vf_all[te])
        y_true = np.stack([attr_value(scenes[i], attr, True) for i in te])
        err_null = float(np.mean([attr_err(p_base[j], y_true[j], attr) for j in range(len(te))]))
        err_real = float(np.mean([attr_err(p_var[j], y_true[j], attr) for j in range(len(te))]))
        rec = {"power_block_rel_l2": power_b, "probe_err_null": err_null, "probe_err_real": err_real, "ops": {}}
        for name, op in ops.items():
            with torch.no_grad():
                Fint = op(Bk_all[te].to(dev))
            Ff_int = tail_pool(Fint)
            repr_rel = float(np.linalg.norm((Fint[:, 1:] - Vk_all[te][:, 1:].to(dev)).cpu().numpy(), axis=(1, 2)).mean() /
                             (np.linalg.norm(Vk_all[te][:, 1:].numpy(), axis=(1, 2)).mean() + 1e-9))
            fin_rel = float(np.linalg.norm(Ff_int - Vf_all[te], axis=1).mean() /
                            (np.linalg.norm(Vf_all[te], axis=1).mean() + 1e-9))
            p_int = probe_of(Ff_int)
            Pi, Pb, Pv = mat(p_int), mat(p_base), mat(p_var)
            num = ((Pi - Pb) * (Pv - Pb)).sum(1)
            den = ((Pv - Pb) ** 2).sum(1) + 1e-12
            cos = num / (np.linalg.norm(Pi - Pb, axis=1) * np.linalg.norm(Pv - Pb, axis=1) + 1e-12)
            err_int = float(np.mean([attr_err(p_int[j], y_true[j], attr) for j in range(len(te))]))
            gap = err_null - err_real
            rec["ops"][name] = {
                "repr_rel_l2_vs_real": repr_rel,
                "final_rel_l2_vs_real": fin_rel,
                "align_cos": float(cos.mean()),
                "proj_coef_agg": float(num.sum() / den.sum()),
                "win_rate": float((np.linalg.norm(Pi - Pv, axis=1) <
                                   np.linalg.norm(Pb - Pv, axis=1)).mean()),
                "probe_err_int": err_int,
                "attr_transfer": float((err_null - err_int) / gap) if abs(gap) > 1e-6 else None,
            }
        out["attributes"][attr] = rec
        print(f"[{attr}] power {power_b:.3f} err_null {err_null:.3f} err_real {err_real:.3f} | " +
              " ".join(f"{n.split('_')[0]} {v['attr_transfer'] if v['attr_transfer'] is None else round(v['attr_transfer'], 2)}"
                       for n, v in rec["ops"].items()), flush=True)

    fn = os.path.join(WORK, "results", f"multiattr_intervention_{a.backbone}_b{a.block}_s{a.seed}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
