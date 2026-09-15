# -*- coding: utf-8 -*-
"""159_realmultiattr_intervention.py — multi-attribute intervention on REAL image regions.

The synthetic multi-attribute family (script 157) shows the protocol is not hue-specific, but a TPAMI
reviewer will ask for a real-domain version. Here the attributes are exact per-pixel transforms on real
COCO instance crops, so ground truth is exact without any renderer:
  * hue:        H -> H + delta           (delta sampled from 30/60/90/120 deg)
  * saturation: S -> clip(scale * S)     (scale sampled from 1.25/1.5/1.75/2.0)
  * value:      V -> clip(scale * V)     (scale sampled from 1.25/1.5/1.75)
Protocol is identical to scripts 155/157: freeze ViT-B/16 (or DINOv2-B/14), split after block k, fit
O1/O5/O8 + random-orthogonal O6 on TRAIN pairs at block k, and evaluate on held-out regions with a
per-attribute Ridge probe (trained on base+variant pooled final features of that attribute only and
shared by every route). Metrics: block-k input power, representation/final fidelity, cosine, aggregate
projection coefficient, paired win rate, and the downstream attribute-transfer fraction.

Usage: python scripts/159_realmultiattr_intervention.py --backbone vitb16 --block 4 --n_regions 240
Outputs results/realmultiattr_intervention_<backbone>_b<block>_s<seed>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math, argparse, importlib.util
import numpy as np
from PIL import Image

WORK = rp.REPO_ROOT
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
SIZE = 224
ATTRS = ["hue", "saturation", "value"]
HUE_STEPS = [30.0, 60.0, 90.0, 120.0]
SAT_STEPS = [1.25, 1.5, 1.75, 2.0]
VAL_STEPS = [1.25, 1.5, 1.75]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(WORK, "scripts", name))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


vit = load_script("155_vit_intervention.py")


# ---------------------------------------------------------------- exact HSV transforms
def rgb_to_hsv_np(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    m = nz & (mx == r); h[m] = (((g - b)[m] / d[m]) % 6.0)
    m = nz & (mx == g); h[m] = ((b - r)[m] / d[m]) + 2.0
    m = nz & (mx == b); h[m] = ((r - g)[m] / d[m]) + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


def hsv_to_rgb_np(h, s, v):
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    out = np.zeros(h.shape + (3,), np.float32)
    for k, (R, G, B) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        if m.any():
            out[m] = np.stack([R[m], G[m], B[m]], -1)
    return out


def transform(u8, attr, param):
    x = u8.astype(np.float32) / 255.0
    h, s, v = rgb_to_hsv_np(x)
    if attr == "hue":
        h = (h + param / 360.0) % 1.0
    elif attr == "saturation":
        s = np.clip(s * param, 0.0, 1.0)
    elif attr == "value":
        v = np.clip(v * param, 0.0, 1.0)
    return (hsv_to_rgb_np(h, s, v) * 255.0).round().clip(0, 255).astype(np.uint8)


def region_stats(u8):
    """Pixel-derived attribute of a region: saturation-weighted circular-mean hue, mean S, mean V,
    and the dominant share m of the saturated colour weight within +/-30 deg of that mean."""
    x = u8.astype(np.float32) / 255.0
    h, s, v = rgb_to_hsv_np(x)
    h = h.reshape(-1); s = s.reshape(-1); v = v.reshape(-1)
    keep = (s > 0.15) & (v > 0.10)
    if keep.sum() < 200:
        keep = v > 0.05
        hue_ok = False
    else:
        hue_ok = True
    w = s[keep]
    ang = h[keep] * 2 * np.pi
    cx = float((w * np.cos(ang)).sum()); sy = float((w * np.sin(ang)).sum())
    hue = (math.degrees(math.atan2(sy, cx)) % 360.0) if (hue_ok and (abs(cx) + abs(sy) > 1e-6)) else None
    if hue is not None:
        d = np.abs(((h[keep] * 360.0 - hue + 180.0) % 360.0) - 180.0)
        m = float(w[d <= 30.0].sum() / (w.sum() + 1e-9))
    else:
        m = 0.0
    return {"hue": hue, "saturation": float(s[keep].mean()), "value": float(v[keep].mean()),
            "m": m, "n_keep": int(keep.sum()), "hue_ok": hue_ok}


def target_of(stats, attr):
    if attr == "hue":
        if stats["hue"] is None:
            return None
        return np.array([math.cos(math.radians(stats["hue"])), math.sin(math.radians(stats["hue"]))])
    return float(stats[attr])


def param_target(attr, param):
    """Target of the probe: the applied transform, in a scale-free parameterisation."""
    if attr == "hue":
        return np.array([math.cos(math.radians(param)), math.sin(math.radians(param))])
    return float(math.log(param))          # param is a multiplicative scale (1 = identity)


def identity_target(attr):
    return np.array([1.0, 0.0]) if attr == "hue" else 0.0


def attr_err(pred, true, attr):
    if attr == "hue":
        d = math.degrees(math.atan2(pred[1], pred[0]) - math.atan2(true[1], true[0]))
        return abs((d + 180.0) % 360.0 - 180.0)
    return abs(float(pred) - float(true))


# ---------------------------------------------------------------- real regions
def select_regions(n_regions, per_cat=20):
    ann = json.load(open(ANN))
    imgs = {i["id"]: i for i in ann["images"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    rng = np.random.RandomState(0)
    pool = {}
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w * h < 20000 or w < 80 or h < 80:
            continue
        pool.setdefault(cats[a["category_id"]], []).append(a)
    order = sorted(pool.keys()); rng.shuffle(order)
    out = []
    for c in order:
        lst = pool[c][:]; rng.shuffle(lst)
        out.extend(lst[:per_cat])
        if len(out) >= n_regions:
            break
    return out[:n_regions], imgs, cats


def crop_region(a, imgs, cache={}):
    im = imgs[a["image_id"]]
    path = os.path.join(IMG, im["file_name"])
    if not os.path.exists(path):
        return None
    if im["file_name"] not in cache:
        I = Image.open(path); I.draft("RGB", (max(1, I.size[0] // 2), max(1, I.size[1] // 2)))
        cache.clear(); cache[im["file_name"]] = I.convert("RGB")
    I = cache[im["file_name"]]
    sc = I.size[0] / im["width"]
    x, y, w, h = [v * sc for v in a["bbox"]]
    W, H = I.size
    mx, my = 0.10 * w, 0.10 * h
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(W, int(x + w + mx)), min(H, int(y + h + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.asarray(I.crop((x0, y0, x1, y1)).resize((SIZE, SIZE), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitb16")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_regions", type=int, default=240)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--fit_imgs", type=int, default=150)
    ap.add_argument("--op8_epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--probe", default="ridge", choices=["ridge", "krr", "ridge2"])
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rs = np.random.RandomState(a.seed)

    regs, imgs, cats = select_regions(a.n_regions)
    crops, keep = [], []
    for r in regs:
        c = crop_region(r, imgs)
        if c is not None:
            crops.append(c); keep.append(r)
    crops = np.stack(crops)
    n = len(crops)
    cats_kept = [cats[r["category_id"]] for r in keep]
    # Split by SOURCE IMAGE, not by region. A region-level permutation puts two crops of the same COCO
    # photo on opposite sides of the fit/held-out split; the crops then share appearance and annotation
    # context, so the held-out score is partly a fit-set score. Grouping is cheap and removes the leak:
    # every region of an image travels with that image. The assertion below makes a reintroduced leak
    # impossible to miss.
    img_of = np.array([r["image_id"] for r in keep])
    uniq_img = np.unique(img_of)
    te_parts, tr_parts, n_te = [], [], 0
    for u in uniq_img[rs.permutation(len(uniq_img))]:
        idx = np.where(img_of == u)[0]
        if n_te < a.n_test:
            te_parts.append(idx); n_te += len(idx)
        else:
            tr_parts.append(idx)
    TE0 = np.sort(np.concatenate(te_parts))
    TR0 = np.sort(np.concatenate(tr_parts))
    if set(img_of[TE0]) & set(img_of[TR0]):
        raise RuntimeError("fit/held-out split shares a source image: the split is not grouped")

    # per-region parameters, deterministic
    params = {}
    for attr in ATTRS:
        steps = HUE_STEPS if attr == "hue" else (SAT_STEPS if attr == "saturation" else VAL_STEPS)
        params[attr] = np.array([steps[rs.randint(len(steps))] for _ in range(n)])

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

    Bstat_all = [region_stats(c) for c in crops]
    # variant statistics are computed after rendering the variant crops (below); placeholders here
    Bk_all, Bf_all = feats(crops)
    Q, _ = np.linalg.qr(np.random.RandomState(a.seed + 7).normal(size=(D, D)).astype(np.float32))
    W6 = torch.from_numpy(Q.astype(np.float32)).to(dev)

    out = {"backbone": a.backbone, "block": a.block, "seed": a.seed, "domain": "real COCO train2017 instance crops",
           "split_grouped_by_image": True,
           "n_regions": n, "n_test_target": int(len(TE0)), "attr_order": ATTRS,
           "grid": list(grid), "dim": D, "probe": a.probe, "steps": {"hue": HUE_STEPS, "saturation": SAT_STEPS, "value": VAL_STEPS},
           "categories": sorted(set(cats_kept)), "attributes": {}}

    for attr in ATTRS:
        p = params[attr]
        var_u8 = np.stack([transform(crops[i], attr, float(p[i])) for i in range(n)])
        Vstat = [region_stats(c) for c in var_u8]
        Bstat = Bstat_all
        Vk_all, Vf_all = feats(var_u8)

        # pixel-derived targets: base rows carry the region's own attribute, variant rows the transformed
        # region's attribute (both read from pixels), so the probe has real signal and the null route's
        # error is the true attribute displacement of the transform
        ok = {i for i in range(n)
              if target_of(Bstat_all[i], attr) is not None and target_of(Vstat[i], attr) is not None}
        tr = np.array([i for i in TR0 if i in ok])
        te = np.array([i for i in TE0 if i in ok])
        fit_i = tr[:min(a.fit_imgs, len(tr))]
        Xp = np.concatenate([Bf_all[tr], Vf_all[tr]], 0)
        yp = np.stack([target_of(Bstat[i] if j == 0 else Vstat[i], attr)
                       for j in (0, 1) for i in tr])
        if a.probe == "krr":
            probe = KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / Xp.shape[1]).fit(Xp, yp)
        elif a.probe == "ridge2":
            Xa = np.concatenate([Xp, Xp ** 2], 1)
            probe = Ridge(alpha=1.0).fit(Xa, yp)
        else:
            probe = Ridge(alpha=1.0).fit(Xp, yp)

        def probe_of(nf):
            if a.probe == "ridge2":
                nf = np.concatenate([nf, nf ** 2], 1)
            q = probe.predict(nf)
            if attr == "hue":
                q = np.atleast_2d(q)
                q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            return q

        def mat(v):
            return np.asarray(v).reshape(len(te), -1)

        Fx = Bk_all[fit_i].to(dev)
        Fy = Vk_all[fit_i].to(dev)
        ops = dict(vit.fit_ops(Fx, Fy, grid, a.seed, a.op8_epochs))
        ops["O6_random_orthogonal"] = lambda t: t @ W6.T

        power_b = float(np.linalg.norm((Vk_all[te][:, 1:] - Bk_all[te][:, 1:]).numpy(), axis=(1, 2)).mean() /
                        (np.linalg.norm(Vk_all[te][:, 1:].numpy(), axis=(1, 2)).mean() + 1e-9))
        p_base, p_var = probe_of(Bf_all[te]), probe_of(Vf_all[te])
        y_true = np.stack([target_of(Vstat[i], attr) for i in te])
        err_null = float(np.mean([attr_err(p_base[j], y_true[j], attr) for j in range(len(te))]))
        err_real = float(np.mean([attr_err(p_var[j], y_true[j], attr) for j in range(len(te))]))
        mvals = np.array([Bstat_all[i]["m"] for i in te])
        rec = {"power_block_rel_l2": power_b, "probe_err_null": err_null, "probe_err_real": err_real,
               "n_eval": int(len(te)), "m_median": float(np.median(mvals)),
               "frac_m_ge_0.70": float((mvals >= 0.70).mean()),
               "hue_ok_frac": float(np.mean([Bstat_all[i]["hue_ok"] for i in te])), "ops": {}}
        strata = {grp: sel for grp, sel in [("m_ge_0.70", mvals >= 0.70), ("m_lt_0.70", mvals < 0.70)]
                  if sel.sum() >= 5}
        for grp, sel in strata.items():
            idx = np.where(sel)[0]
            rec.setdefault("strata", {})[grp] = {
                "n": int(sel.sum()),
                "probe_err_null": float(np.mean([attr_err(p_base[j], y_true[j], attr) for j in idx])),
                "probe_err_real": float(np.mean([attr_err(p_var[j], y_true[j], attr) for j in idx])),
                "ops": {}}
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
                "repr_rel_l2_vs_real": repr_rel, "final_rel_l2_vs_real": fin_rel,
                "align_cos": float(cos.mean()), "proj_coef_agg": float(num.sum() / den.sum()),
                "win_rate": float((np.linalg.norm(Pi - Pv, axis=1) < np.linalg.norm(Pb - Pv, axis=1)).mean()),
                "probe_err_int": err_int,
                "attr_transfer": float((err_null - err_int) / gap) if abs(gap) > 1e-6 else None,
                "per_sample_probe_err_int": [float(attr_err(p_int[j], y_true[j], attr)) for j in range(len(te))],
                "per_sample_transfer": [float((attr_err(p_base[j], y_true[j], attr) -
                                               attr_err(p_int[j], y_true[j], attr)) /
                                              max(abs(attr_err(p_base[j], y_true[j], attr) -
                                                      attr_err(p_var[j], y_true[j], attr)), 1e-6))
                                        for j in range(len(te))],
            }
            for grp, sel in strata.items():
                idx = np.where(sel)[0]
                e_n = float(np.mean([attr_err(p_base[j], y_true[j], attr) for j in idx]))
                e_r = float(np.mean([attr_err(p_var[j], y_true[j], attr) for j in idx]))
                e_i = float(np.mean([attr_err(p_int[j], y_true[j], attr) for j in idx]))
                g = e_n - e_r
                rec["strata"][grp]["ops"][name] = {
                    "probe_err_int": e_i,
                    "attr_transfer": float((e_n - e_i) / g) if abs(g) > 1e-6 else None,
                    "win_rate": float((np.linalg.norm(Pi[idx] - Pv[idx], axis=1) <
                                       np.linalg.norm(Pb[idx] - Pv[idx], axis=1)).mean()),
                    "proj_coef_agg": float(((Pi[idx] - Pb[idx]) * (Pv[idx] - Pb[idx])).sum() /
                                           (((Pv[idx] - Pb[idx]) ** 2).sum() + 1e-12)),
                }
        out["attributes"][attr] = rec
        print(f"[{attr}] power {power_b:.3f} err_null {err_null:.3f} err_real {err_real:.3f} | " +
              " ".join(f"{nm.split('_')[0]} {v['attr_transfer'] if v['attr_transfer'] is None else round(v['attr_transfer'], 2)}"
                       for nm, v in rec["ops"].items()), flush=True)

    fn = os.path.join(WORK, "results", f"realmultiattr_intervention_{a.backbone}_b{a.block}_s{a.seed}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
