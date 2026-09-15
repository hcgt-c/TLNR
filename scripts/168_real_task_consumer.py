# -*- coding: utf-8 -*-
"""168_real_task_consumer.py — mid-layer intervention against a real-image task whose output depends
on a continuous attribute.

Paper B's real-domain evidence so far is a Ridge attribute probe; the synthetic hue-bin detector is the
only task consumer. This script supplies a second, real-image consumer: a small convolutional classifier
trained end to end on pixel inputs to identify the *value level* of a COCO instance crop, where the level
is produced by an exact value transform (v x 0.60 / 1.00 / 1.45). The classifier therefore consumes the
attribute by construction, and the transform has exact ground truth.

Protocol:
  * 240 COCO instance crops (the same deterministic selection as scripts 159/169), 150 regions train /
    90 regions test, each rendered at three value levels;
  * a small CNN task head (four conv blocks + linear) is trained on the training regions only;
  * the intervention protocol is then run on that task's stage-2 feature map for the base -> bright pair
    (v x 1.45): operators O1 (global Procrustes), O2 (ridge), O5 (ridge + per-cell MLP residual),
    O8 (ridge 1x1 + 3x3 residual, content + neighbourhood) and O6 (random orthogonal) are fitted on
    training pairs and evaluated on held-out regions;
  * metrics: effect size of the real transform, projection coefficient, direction cosine, paired win
    rate against the no-op, top-1 agreement with the real route, task accuracy of each route, and the
    feature fidelity of each operator.

Usage: python scripts/168_real_task_consumer.py [--epochs 20 --n_regions 240]
Outputs results/real_task_consumer.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WORK = rp.REPO_ROOT
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
SIZE = 224
LEVELS = [0.60, 1.00, 1.45]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def select_regions(n_target, per_cat=20):
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
        if len(out) >= n_target:
            break
    return out[:n_target], imgs, cats


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
    x0, y0 = max(0, int(x - 0.1 * w)), max(0, int(y - 0.1 * h))
    x1, y1 = min(W, int(x + w + 0.1 * w)), min(H, int(y + h + 0.1 * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.asarray(I.crop((x0, y0, x1, y1)).resize((SIZE, SIZE), Image.BILINEAR))


def value_scale(u8, k):
    x = u8.astype(np.float32) / 255.0
    h, s, v = rgb_to_hsv(x)
    v = np.clip(v * k, 0.0, 1.0)
    return (hsv_to_rgb(h, s, v) * 255.0).round().clip(0, 255).astype(np.uint8)


def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    h = np.zeros_like(mx); nz = d > 1e-9
    m = nz & (mx == r); h[m] = (((g - b)[m] / d[m]) % 6.0)
    m = nz & (mx == g); h[m] = ((b - r)[m] / d[m]) + 2.0
    m = nz & (mx == b); h[m] = ((r - g)[m] / d[m]) + 4.0
    return h / 6.0, np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0), mx


def hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    out = np.zeros(h.shape + (3,), np.float32)
    for k, (R, G, B) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        sel = i == k
        if sel.any():
            out[sel] = np.stack([R[sel], G[sel], B[sel]], -1)
    return out


class TaskNet(nn.Module):
    """Small convolutional task head: four blocks, the 2nd block's output is the intervention site."""

    def __init__(self, n_cls=3, w=(32, 64, 128, 256)):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, w[0], 3, 2, 1), nn.BatchNorm2d(w[0]), nn.SiLU(),
                                nn.Conv2d(w[0], w[0], 3, 1, 1), nn.BatchNorm2d(w[0]), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv2d(w[0], w[1], 3, 2, 1), nn.BatchNorm2d(w[1]), nn.SiLU(),
                                nn.Conv2d(w[1], w[1], 3, 1, 1), nn.BatchNorm2d(w[1]), nn.SiLU())
        self.b3 = nn.Sequential(nn.Conv2d(w[1], w[2], 3, 2, 1), nn.BatchNorm2d(w[2]), nn.SiLU(),
                                nn.Conv2d(w[2], w[2], 3, 1, 1), nn.BatchNorm2d(w[2]), nn.SiLU())
        self.b4 = nn.Sequential(nn.Conv2d(w[2], w[3], 3, 2, 1), nn.BatchNorm2d(w[3]), nn.SiLU())
        self.head = nn.Linear(w[3], n_cls)

    def forward(self, x, stop_after=None):
        x = self.b1(x)
        x = self.b2(x)
        if stop_after == 2:
            return x
        x = self.b3(x); x = self.b4(x)
        return self.head(x.mean((2, 3))), x


def prep(u8, mean, std, dev):
    x = torch.from_numpy(u8).float().permute(0, 3, 1, 2).to(dev) / 255.0
    return (x - mean) / std


def run_tail(model, feat, mean, std):
    with torch.no_grad():
        x = model.b3(feat); x = model.b4(x)
        return model.head(x.mean((2, 3)))


class SpatialOp:
    def __init__(self, f): self.f = f

    def __call__(self, f4): return self.f(f4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_regions", type=int, default=240)
    ap.add_argument("--n_test", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--op8_epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    c136 = load_mod("c136", os.path.join(WORK, "scripts", "136_causal_cifar.py"))
    regs, imgs, cats = select_regions(a.n_regions)
    crops, keep = [], []
    for r in regs:
        c = crop_region(r, imgs)
        if c is not None:
            crops.append(c); keep.append(r)
    crops = np.stack(crops)
    n = len(crops)
    perm = np.random.RandomState(a.seed).permutation(n)
    te = np.sort(perm[:a.n_test]); tr = np.sort(perm[a.n_test:])
    print(f"regions {n} (train {len(tr)} / test {len(te)}), device {DEV}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)

    # ---- data: three value levels per region, labels from the level
    X, Y, RID = [], [], []
    for lv_i, k in enumerate(LEVELS):
        X.append(np.stack([value_scale(crops[i], k) for i in range(n)]))
        Y.append(np.full(n, lv_i)); RID.append(np.arange(n))
    X = np.concatenate(X); Y = np.concatenate(Y); RID = np.concatenate(RID)
    tr_mask = np.isin(RID, tr)
    model = TaskNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Xt = prep(X[tr_mask], mean, std, DEV); Yt = torch.from_numpy(Y[tr_mask]).long().to(DEV)
    for ep in range(a.epochs):
        perm2 = torch.randperm(len(Xt), device=DEV)
        tot = 0.0
        for s in range(0, len(Xt), a.bs):
            idx = perm2[s:s + a.bs]
            logits, _ = model(Xt[idx])
            loss = F.cross_entropy(logits, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{a.epochs} loss {tot/len(Xt):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        te_all = prep(X[np.isin(RID, te)], mean, std, DEV)
        y_te = torch.from_numpy(Y[np.isin(RID, te)]).long().to(DEV)
        acc = float((model(te_all)[0].argmax(1) == y_te).float().mean())
    print(f"task accuracy on held-out regions (three levels): {acc:.3f}", flush=True)

    # ---- intervention on the base -> bright pair
    k_base, k_bright = LEVELS[1], LEVELS[2]
    base_u8 = np.stack([value_scale(crops[i], k_base) for i in range(n)])
    var_u8 = np.stack([value_scale(crops[i], k_bright) for i in range(n)])
    xb, xv = prep(base_u8, mean, std, DEV), prep(var_u8, mean, std, DEV)

    def feats(x):
        outs = []
        with torch.no_grad():
            for s in range(0, x.shape[0], 25):
                outs.append(model(x[s:s + 25], stop_after=2).cpu())
        return torch.cat(outs, 0)

    FB, FV = feats(xb), feats(xv)

    def route(feat4):
        with torch.no_grad():
            outs = []
            for s in range(0, feat4.shape[0], 40):
                outs.append(run_tail(model, feat4[s:s + 40].to(DEV), mean, std).cpu())
        return torch.cat(outs, 0)

    y_null, y_real = route(FB), route(FV)
    Y3 = Y.reshape(len(LEVELS), n)
    y_tb = torch.from_numpy(Y3[2][te]).long()      # true label of the BRIGHT variant (level index 2)
    y_tbase = torch.from_numpy(Y3[1][te]).long()   # true label of the base variant (level index 1)
    power = {
        "effect_rel_l2": float(torch.norm(y_real[te] - y_null[te]) / (torch.norm(y_null[te]) + 1e-9)),
        "flip_rate": float((y_real[te].argmax(1) != y_null[te].argmax(1)).float().mean()),
        "acc_null_vs_bright_label": float((y_null[te].argmax(1) == y_tb).float().mean()),
        "acc_real_vs_bright_label": float((y_real[te].argmax(1) == y_tb).float().mean()),
        "acc_null_vs_base_label": float((y_null[te].argmax(1) == y_tbase).float().mean()),
        "acc_real_vs_base_label": float((y_real[te].argmax(1) == y_tbase).float().mean()),
        "task_acc_three_levels": acc,
    }

    # operator fitting on train pairs
    Fx = FB[tr].to(DEV); Fy = FV[tr].to(DEV)
    C = Fx.shape[1]
    xr = Fx.permute(0, 2, 3, 1).reshape(-1, C); yr = Fy.permute(0, 2, 3, 1).reshape(-1, C)
    ops = {"O1_procrustes": c136.fit_procrustes(xr, yr), "O2_ridge": c136.fit_ridge(xr, yr),
           "O5_mlp": c136.make_op5(xr, yr, epochs=300, seed=a.seed),
           "O8_conv_residual": c136.make_op8(Fx, Fy, epochs=a.op8_epochs, hidden=32, seed=a.seed),
           "O6_random": c136.random_orth(C, seed=a.seed)}
    fr = FV[te].permute(0, 2, 3, 1).reshape(-1, C)

    def dist(p, q):
        return (torch.norm(p - q, dim=1) / (torch.norm(q, dim=1) + 1e-9)).cpu().numpy()

    d_null = dist(y_null[te], y_real[te])
    out = {"protocol": ("small CNN task head trained end to end on real COCO crops to identify the value "
                        "level (x0.60 / x1.00 / x1.45, exact transform); intervention on the stage-2 feature "
                        "map for the base -> bright pair; 150 training regions / 90 held-out regions"),
           "n_regions": n, "n_test": int(len(te)), "task_acc_three_levels": acc, "power": power, "ops": {}}
    for name, op in ops.items():
        B, Cc, H, W = FB[te].shape
        Ft = FB[te].to(DEV)
        if isinstance(op, c136.SpatialOp):
            Fint = op.apply4(Ft)
            tW = Fint.permute(0, 2, 3, 1).reshape(-1, Cc)
        else:
            rows = Ft.permute(0, 2, 3, 1).reshape(-1, Cc)
            lin = op(rows) if callable(op) else rows @ op.T
            tW = lin
            Fint = lin.reshape(len(te), H, W, Cc).permute(0, 3, 1, 2)
        y_int = route(Fint.cpu())
        cos, proj, _ = c136.align_proj(y_int - y_null[te], y_real[te] - y_null[te])
        d_int = dist(y_int, y_real[te])
        gain = d_null - d_int
        lo, hi = c136.boot_ci(gain)
        out["ops"][name] = {
            "feature_rel_l2": float((torch.norm(tW - fr.to(tW.device), dim=1) /
                                     (torch.norm(fr.to(tW.device), dim=1) + 1e-9)).mean()),
            "proj_coef_agg": proj, "align_cos": cos,
            "win_rate": float((d_int < d_null).mean()), "mean_gain": float(gain.mean()),
            "gain_ci95": [lo, hi],
            "top1_agree_real": float((y_int.argmax(1) == y_real[te].argmax(1)).float().mean()),
            "acc_int_vs_bright_label": float((y_int.argmax(1) == y_tb).float().mean()),
        }
        print(f"  {name:20s} proj {proj:.2f} cos {cos:.2f} win {out['ops'][name]['win_rate']:.2f} "
              f"top1 {out['ops'][name]['top1_agree_real']:.2f}", flush=True)
        if DEV == "cuda":
            torch.cuda.empty_cache()
    fn = os.path.join(WORK, "results", "real_task_consumer.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
