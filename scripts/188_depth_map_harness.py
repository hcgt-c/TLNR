# -*- coding: utf-8 -*-
"""188_depth_map_harness.py — the architecture x depth map for faithful intervention.

One site, five numbers. This harness fits the paper's operator families at a mid-layer site of a *frozen
pretrained backbone* on *real images*, and reports, per (backbone, depth, transformation family, parameter):

    effect size     how large the real transformation is for this consumer (the power gate, Section 3.7)
    O1, O2, O8, O6  projection coefficient of each fitted operator against the real route
    win rate        paired fraction on which the intervention is closer to the real route than the no-op
    R               (heat family only) the intertwining residual: how much of the site's change the
                    *canonical* heat action explains, 0 = all of it, 1 = none of it

Two design points make this cheap and honest:
  * all depths are captured in ONE forward pass per (image set, transformation) via hooks, so the cost is
    feature extraction, not per-depth passes;
  * the criterion needs no labels, so the consumer is the backbone's own frozen ImageNet head on real COCO
    crops. We report no accuracy: the effect size is what licenses reading an operator result.

Usage
  screen: python scripts/188_depth_map_harness.py --backbone resnet50 --family heat --params 0.5,1.0,2.0 --screen
  full:   python scripts/188_depth_map_harness.py --backbone resnet50 --family heat --params 1.0 --depths 1,2,3,4
Outputs results/depth_map_<backbone>_<family>.json  (merged across invocations)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.fft import dctn, idctn

WORK = rp.REPO_ROOT
HUB = rp.DINOV2_HUB
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
SIZE = 224
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_c136():
    spec = importlib.util.spec_from_file_location("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c136 = load_c136()


# ------------------------------------------------------------------ backbones and their sites
def build_backbone(name):
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        sites = [m.layer1, m.layer2, m.layer3, m.layer4]
        names = ["layer1", "layer2", "layer3", "layer4"]
        return m.to(DEV).eval(), sites, names, "cnn"
    if name == "convnext":
        import torchvision
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feats = list(m.features)
        idx = [1, 3, 5, 7]
        return m.to(DEV).eval(), [feats[i] for i in idx], [f"features{i}" for i in idx], "cnn"
    if name == "vitb16":
        import torchvision
        m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        layers = list(m.encoder.layers)
        idx = [2, 5, 8, 11]
        return m.to(DEV).eval(), [layers[i] for i in idx], [f"block{i}" for i in idx], "vit"
    if name == "dinov2b14":
        try:
            m = torch.hub.load(HUB, "dinov2_vitb14", source="local")
        except Exception:
            sys.path.insert(0, HUB)
            from dinov2.models.vision_transformer import DinoVisionTransformer
            from dinov2.configs import dinov2_default_config as cfgmod
            cfg = cfgmod.get_config("dinov2_vitb14")
            m = DinoVisionTransformer(**cfg)
            ck = torch.load(rp.DINOV2_WEIGHTS,
                            map_location="cpu")
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck["model"].items()}
            m.load_state_dict(sd, strict=False)
        blocks = list(m.blocks)
        idx = [2, 5, 8, 11]
        return m.to(DEV).eval(), [blocks[i] for i in idx], [f"block{i}" for i in idx], "dinov2"
    raise ValueError(name)


def to_grid(t, kind):
    """Normalise a site activation to (B, C, H, W), dropping a class token if present."""
    if isinstance(t, (tuple, list)):
        t = t[0]
    if t.dim() == 4:
        return t
    if t.dim() == 3:                                   # (B, N(+1), C) tokens
        B, N, C = t.shape
        n = N - 1 if int(round(np.sqrt(N - 1))) ** 2 == N - 1 else N
        p = int(round(np.sqrt(n)))
        patch = t[:, N - n:, :] if n != N else t
        return patch.transpose(1, 2).reshape(B, C, p, p)
    raise ValueError(t.shape)



def _apply_matrix(t, W, kind):
    """Apply a channel map at every location, preserving the site's layout (tokens keep the class token)."""
    g = to_grid(t, kind)
    B, C, H, Wd = g.shape
    rows = g.permute(0, 2, 3, 1).reshape(-1, C) @ W.T
    out = rows.reshape(B, H, Wd, C).permute(0, 3, 1, 2)
    if t.dim() == 3:
        return torch.cat([t[:, :1], out.flatten(2).transpose(1, 2)], 1)
    return out


def _apply_callable(t, fn, kind):
    """Same, for operators that act on the 4-D map (e.g. the 3x3 residual)."""
    g = to_grid(t, kind)
    out = fn(g)
    if t.dim() == 3:
        return torch.cat([t[:, :1], out.flatten(2).transpose(1, 2)], 1)
    return out


# ------------------------------------------------------------------ transformations
def heat_eigs(H, W):
    la = 2 - 2 * np.cos(np.pi * np.arange(H) / H)
    lb = 2 - 2 * np.cos(np.pi * np.arange(W) / W)
    return (la[:, None] + lb[None, :]).astype(np.float32)


_HEAT_CACHE = {}


def heat(x, t):
    """Exact discrete heat semigroup by DCT (Neumann) spectral multiplication, per channel."""
    H, W = x.shape[-2:]
    if (H, W) not in _HEAT_CACHE:
        _HEAT_CACHE[(H, W)] = heat_eigs(H, W)
    e = _HEAT_CACHE[(H, W)]
    a = dctn(x.detach().cpu().numpy().astype(np.float32), type=2, norm="ortho", axes=(-2, -1))
    a = a * np.exp(-t * e)[None, None]
    return torch.from_numpy(idctn(a, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)).to(x.device)


def hue_shift(x01, deg):
    """Exact per-pixel hue rotation in HSV, vectorised (a per-pixel Python loop is ~4 orders too slow)."""
    x = x01.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx
    s_ = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    rc = np.where(d > 1e-9, (mx - r) / np.maximum(d, 1e-9), 0.0)
    gc = np.where(d > 1e-9, (mx - g) / np.maximum(d, 1e-9), 0.0)
    bc = np.where(d > 1e-9, (mx - b) / np.maximum(d, 1e-9), 0.0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = ((h / 6.0) + deg / 360.0) % 1.0
    # hsv -> rgb (standard six-sector formula)
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p_ = v * (1.0 - s_)
    q_ = v * (1.0 - f * s_)
    t_ = v * (1.0 - (1.0 - f) * s_)
    out = np.empty_like(x)
    for k, (cr, cg, cb) in enumerate([(v, t_, p_), (q_, v, p_), (p_, v, t_),
                                      (p_, q_, v), (t_, p_, v), (v, p_, q_)]):
        m = i == k
        out[..., 0][m], out[..., 1][m], out[..., 2][m] = cr[m], cg[m], cb[m]
    return torch.from_numpy(out.transpose(0, 3, 1, 2)).to(x01.device)


# ------------------------------------------------------------------ data
def load_crops(n, seed=0):
    with open(ANN) as f:
        ann = json.load(f)
    imgs = {im["id"]: im for im in ann["images"]}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ann["annotations"]))
    out, cache = [], {}
    for i in order:
        if len(out) >= n:
            break
        a = ann["annotations"][i]
        im = imgs.get(a["image_id"])
        if im is None or a.get("iscrowd", 0):
            continue
        path = os.path.join(IMG, im["file_name"])
        if not os.path.exists(path):
            continue
        if im["file_name"] not in cache:
            I = Image.open(path); I.draft("RGB", (max(1, I.size[0] // 2), max(1, I.size[1] // 2)))
            cache.clear(); cache[im["file_name"]] = I.convert("RGB")
        I = cache[im["file_name"]]
        sc = I.size[0] / im["width"]
        x, y, w, h = [v * sc for v in a["bbox"]]
        W_, H_ = I.size
        mx, my = 0.10 * w, 0.10 * h
        x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
        x1, y1 = min(W_, int(x + w + mx)), min(H_, int(y + h + my))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        out.append(np.asarray(I.crop((x0, y0, x1, y1)).resize((SIZE, SIZE), Image.BILINEAR)))
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} of {n} requested crops could be built; the fit/held-out split "
                           f"would be silently wrong")
    return np.stack(out)


def to01(u8):
    return torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0


def norm(x01):
    return (x01 - MEAN.to(DEV)) / STD.to(DEV)


# ------------------------------------------------------------------ capture and intervention

def make_op8_batched(fb, fr, epochs=150, hidden=32, lr=1e-3, seed=0, bs=16):
    """O8 with the full fitting set: the 1x1 ridge is exact and the 3x3 residual is trained by minibatch
    gradient steps over every fit image, so the operator is not handicapped by a memory cap."""
    torch.manual_seed(seed)
    C = fb.shape[1]
    Xr = fb.permute(0, 2, 3, 1).reshape(-1, C)
    Yr = fr.permute(0, 2, 3, 1).reshape(-1, C)
    W = c136.fit_ridge(Xr, Yr)                      # stays on CPU; moved per batch inside apply4
    target = fr - torch.einsum("bchw,dc->bdhw", fb, W)
    conv = torch.nn.Sequential(torch.nn.Conv2d(C, hidden, 3, padding=1), torch.nn.SiLU(),
                               torch.nn.Conv2d(hidden, C, 3, padding=1)).to(DEV)
    torch.nn.init.zeros_(conv[-1].weight); torch.nn.init.zeros_(conv[-1].bias)
    opt = torch.optim.Adam(conv.parameters(), lr=lr)
    B = fb.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(B)
        for i in range(0, B, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(conv(fb[idx].to(DEV)), target[idx].to(DEV))
            loss.backward(); opt.step()
    def apply4(x4):
        with torch.no_grad():
            return torch.einsum("bchw,dc->bdhw", x4, W.to(x4.device)) + conv(x4)
    return c136.SpatialOp(apply4)


def capture(model, sites, x, bs=32):
    """Features at every site in ONE pass, as 4-D maps on CPU.

    The hook must *append* per batch; an earlier version assigned, which silently kept only the last batch
    (12 of 300 images) and therefore fitted every operator on 12 images. The assertion at the end makes that
    failure impossible to reintroduce silently.
    """
    buf = {k: [] for k in range(len(sites))}
    hs = []
    for k, layer in enumerate(sites):
        def mk(k):
            def hook(m, i, o):
                buf[k].append(to_grid(o, None).detach().cpu())
            return hook
        hs.append(layer.register_forward_hook(mk(k)))
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                model(x[s:s + bs])
    finally:
        for h in hs:
            h.remove()
    store = {k: torch.cat(v, 0) for k, v in buf.items()}
    if store[0].shape[0] != x.shape[0]:
        raise RuntimeError(f"captured {store[0].shape[0]} activations for {x.shape[0]} images")
    return store


def logits_plain(model, x, bs=32):
    outs = []
    with torch.no_grad():
        for s in range(0, x.shape[0], bs):
            o = model(x[s:s + bs])
            outs.append((o[0] if isinstance(o, (tuple, list)) else o).detach().cpu())
    return torch.cat(outs)


def logits_intervened(model, site, apply_fn, x, bs=32):
    """Run the model with the site activation replaced by apply_fn(activation)."""
    outs = []

    def hook(m, i, o):
        t = o
        if isinstance(t, tuple):
            head, rest = t[0], t[1:]
            return (apply_fn(head),) + rest
        return apply_fn(t)

    h = site.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, x.shape[0], bs):
                o = model(x[s:s + bs])
                outs.append((o[0] if isinstance(o, (tuple, list)) else o).detach().cpu())
    finally:
        h.remove()
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--family", default="heat", choices=["heat", "hue"])
    ap.add_argument("--params", default="1.0")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_held", type=int, default=100)
    ap.add_argument("--ops", default="O1,O2,O8,O6")
    ap.add_argument("--o8_epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screen", action="store_true", help="effect sizes only, no operator fitting")
    a = ap.parse_args()

    params = [float(p) for p in a.params.split(",")]
    depths = [int(d) for d in a.depths.split(",")]
    ops = [o.strip() for o in a.ops.split(",")]

    model, sites, names, kind = build_backbone(a.backbone)
    crops = load_crops(a.n_fit + a.n_held, seed=a.seed)
    x01 = to01(crops)
    x = norm(x01)

    out_path = os.path.join(WORK, "results", f"depth_map_{a.backbone}_{a.family}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {
        "backbone": a.backbone, "family": a.family, "device": DEV, "kind": kind,
        "consumer": "frozen pretrained ImageNet head (no labels used; criterion is label-free)",
        "n_fit": a.n_fit, "n_held": a.n_held, "site_names": names, "results": {}}

    t0 = time.time()
    if a.family == "heat":
        xb_all = {p: norm(heat(x01, p ** 2 / 2)) for p in params}
    else:
        xb_all = {p: norm(hue_shift(x01, p)) for p in params}

    y0 = logits_plain(model, x)
    eff = {}
    for p in params:
        yr = logits_plain(model, xb_all[p])
        eff[p] = {"effect_size_logits": c136.rel_l2(yr, y0),
                  "top1_flip_rate": float((c136.top1(yr) != c136.top1(y0)).float().mean())}
    print(f"{a.backbone}/{a.family}: effect sizes " +
          ", ".join(f"{p}: {eff[p]['effect_size_logits']:.3f}/flip {eff[p]['top1_flip_rate']:.2f}" for p in params))

    if a.screen:
        out["results"]["screen"] = eff
        json.dump(out, open(out_path, "w"), indent=1)
        print("saved", out_path)
        return 0

    Fb = capture(model, sites, x)                     # base features, all depths
    for p in params:
        Fr = capture(model, sites, xb_all[p])
        yr = logits_plain(model, xb_all[p])
        key = f"{a.family}_{p}"
        out["results"].setdefault(key, {})
        out["results"][key].setdefault("seed", {})
        out["results"][key]["seed"].setdefault(str(a.seed), {
            "effect_size_logits": eff[p]["effect_size_logits"],
            "top1_flip_rate": eff[p]["top1_flip_rate"], "n_fit": int(a.n_fit),
            "n_held": int(a.n_held), "sites": {}})
        for d in depths:
            i = d - 1
            # site-level tensors stay on CPU: layer1 of a 300-image set is (300,256,56,56) = ~1 GB, and
            # four such copies do not fit in the 6 GB card. Only the O8 training subset goes to the GPU.
            fb, fr = Fb[i], Fr[i]
            B, C, H, W = fb.shape
            rec = {"feature_shape": [int(B), int(C), int(H), int(W)],
                   "operator_scope": ("patch tokens only; the class token is preserved" if kind in ("vit", "dinov2")
                                      else "all spatial locations")}
            rows_b = fb.permute(0, 2, 3, 1).reshape(-1, C)
            rows_r = fr.permute(0, 2, 3, 1).reshape(-1, C)
            n_rows_fit = a.n_fit * H * W
            # a fixed row subsample bounds the Gram computation at high-resolution sites; where the full fit
            # is affordable the subsample covers every row
            K = min(n_rows_fit, 150_000)
            g = torch.Generator().manual_seed(a.seed)
            sel = torch.randperm(n_rows_fit, generator=g)[:K]
            Xf, Yf = rows_b[:n_rows_fit][sel], rows_r[:n_rows_fit][sel]
            if a.family == "heat":
                # canonical action at this site: t_site = t * (H / SIZE)^2
                tf = (p ** 2 / 2) * (H / float(SIZE)) ** 2
                rec["t_feature"] = tf
                # R is accumulated in chunks so that a second full copy of the maps is never materialised
                num = den = 0.0
                for s0 in range(0, B, 16):
                    cb = heat(fb[s0:s0 + 16], tf)
                    num += float((fr[s0:s0 + 16] - cb).flatten(1).norm(dim=1).sum())
                    den += float((fr[s0:s0 + 16] - fb[s0:s0 + 16]).flatten(1).norm(dim=1).sum())
                rec["intertwining_residual_canonical"] = num / (den + 1e-12)
                rec["n_rows_used_for_fit"] = int(K)
                rec["n_images_fit"] = int(a.n_fit)
                rec["n_held_out"] = int(B - a.n_fit)
            rec["effect_size_site"] = c136.rel_l2(fr.reshape(B, -1), fb.reshape(B, -1))
            ref = (yr - y0)
            for op in ops:
                if op == "O1":
                    W_ = c136.fit_procrustes(Xf, Yf)
                    apply_fn = lambda t, W_=W_: _apply_matrix(t, W_.to(t.device), kind)
                elif op == "O2":
                    W_ = c136.fit_ridge(Xf, Yf)
                    apply_fn = lambda t, W_=W_: _apply_matrix(t, W_.to(t.device), kind)
                elif op == "O6":
                    W_ = c136.random_orth(C, seed=a.seed)
                    apply_fn = lambda t, W_=W_: _apply_matrix(t, W_.to(t.device), kind)
                elif op == "O8":
                    # full fitting set, batched: no memory cap, so O2 vs O8 is apples-to-apples
                    op8 = make_op8_batched(fb[:a.n_fit], fr[:a.n_fit], epochs=a.o8_epochs, seed=a.seed)
                    if DEV == "cuda":
                        torch.cuda.empty_cache()
                    apply_fn = lambda t, op8=op8: _apply_callable(t, op8.apply4, kind)
                else:
                    continue
                yi = logits_intervened(model, sites[i], apply_fn, x[a.n_fit:])
                if op == "O6":
                    rec["random_operator_logit_change"] = c136.rel_l2(yi, y0[a.n_fit:])
                    rec["intervention_reaches_consumer"] = bool(rec["random_operator_logit_change"] > 1e-3)
                cos, proj, _ = c136.align_proj(yi - y0[a.n_fit:], ref[a.n_fit:])
                d_int = torch.norm(yi - yr[a.n_fit:], dim=1)
                d_null = torch.norm(y0[a.n_fit:] - yr[a.n_fit:], dim=1)
                # downstream transfer: 1 = the intervention matches the real route, 0 = no better than the
                # no-op, < 0 = worse than doing nothing. This is the only reported quantity that is an
                # absolute faithfulness score; projection and win rate are companion statistics.
                transfer = 1.0 - float(d_int.mean() / (d_null.mean() + 1e-12))
                # the same quantity under the *squared* criterion. The published operator families are fitted
                # by least squares, i.e. they are optimal for this squared version and not necessarily for the
                # norm version above; recording both is what makes that mismatch measurable rather than
                # rhetorical (see THEORY_CRITERION_CONSISTENCY.md).
                transfer_sq = 1.0 - float(np.sqrt(float((d_int ** 2).mean()) /
                                                  (float((d_null ** 2).mean()) + 1e-12)))
                rec[op] = {"transfer": transfer, "transfer_squared": transfer_sq,
                           "proj_coef_agg": proj, "align_cos": cos,
                           "win_rate": float((d_int < d_null).float().mean()),
                           "mean_d_int": float(d_int.mean()), "mean_d_noop": float(d_null.mean()),
                           "top1_agree_real": float((c136.top1(yi) == c136.top1(yr[a.n_fit:])).float().mean())}
            out["results"][key]["seed"][str(a.seed)]["sites"][names[i]] = rec
            print(f"  {key} s{a.seed} {names[i]:10s} eff_site {rec['effect_size_site']:.3f} " +
                  " ".join(f"{op}:T{rec[op]['transfer']:.2f}/p{rec[op]['proj_coef_agg']:.2f}"
                           for op in ops if op in rec))
        json.dump(out, open(out_path, "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
