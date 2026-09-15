# -*- coding: utf-8 -*-
"""156_detector_o8_multiseed.py — detector-domain intervention with the O8 operator, three splits.

Complements script 142 (matrix operators, single split) with:
  * O8 = ridge 1x1 + 3x3 conv residual fitted on the stride-8 feature map (content+neighbourhood);
  * three train/held splits (seeds 0/1/2, 40 train / 80 held each);
  * box-level agreement (F1@IoU0.5 and 11-point AP50) plus a tensor-level equivalence proxy.

Detector: runs/detect/runs_det_hue/hue_cls8/weights/best.pt (classes = hue bins).
Data: data/yolodet_hue/val (120 images); hue shifts applied inside GT boxes only.

Usage: python scripts/156_detector_o8_multiseed.py --seed 0
Outputs results/detector_o8_seed<seed>.json and a printed summary.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, math, time, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.nms import non_max_suppression

WORK = rp.REPO_ROOT
WT = os.path.join(WORK, "runs/detect/runs_det_hue/hue_cls8/weights/best.pt")
IMD = os.path.join(WORK, "data/yolodet_hue/val/images")
LBD = os.path.join(WORK, "data/yolodet_hue/val/labels")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONF, IOU_NMS, IOU_MATCH = 0.05, 0.7, 0.5
STRIDE, FCH, FH, FW = 8, 64, 28, 28
DELTAS = [30.0, 60.0, 90.0]


def load_det():
    net = torch.load(WT, map_location=DEV, weights_only=False)
    if isinstance(net, dict):
        net = net.get("model", net)
    net.to(DEV).eval().float()
    return net


def gt_boxes(fn):
    p = os.path.join(LBD, fn[:-4] + ".txt")
    return [] if not os.path.exists(p) else np.loadtxt(p).reshape(-1, 5)


def box_mask(fn, size=224):
    m = np.zeros((size, size), bool)
    bs = gt_boxes(fn)
    if len(bs) == 0:
        return np.ones((size, size), bool)
    for _, cx, cy, w, h in bs:
        x0 = int(max(0, (cx - w / 2) * size)); x1 = int(min(size, math.ceil((cx + w / 2) * size)))
        y0 = int(max(0, (cy - h / 2) * size)); y1 = int(min(size, math.ceil((cy + h / 2) * size)))
        m[y0:y1, x0:x1] = True
    return m


def cell_mask(fn):
    m = np.zeros((FH, FW), bool)
    for _, cx, cy, w, h in gt_boxes(fn):
        gx0 = int(max(0, math.floor((cx - w / 2) * 224 / STRIDE))); gx1 = int(min(FW, math.ceil((cx + w / 2) * 224 / STRIDE)))
        gy0 = int(max(0, math.floor((cy - h / 2) * 224 / STRIDE))); gy1 = int(min(FH, math.ceil((cy + h / 2) * 224 / STRIDE)))
        m[gy0:gy1, gx0:gx1] = True
    return m.reshape(-1)


def shift_u8(u8, deg, mask):
    a = u8.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(r, np.maximum(g, b)); mn = np.minimum(r, np.minimum(g, b)); d = mx - mn
    l = (mx + mn) / 2.0
    s = np.zeros_like(d); nz = d > 1e-6
    s[nz] = np.where(l[nz] < 0.5, d[nz] / (mx[nz] + mn[nz]), d[nz] / (2.0 - mx[nz] - mn[nz]))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(r == mx, (g - b) / d, np.where(g == mx, 2.0 + (b - r) / d, 4.0 + (r - g) / d))
        h = ((h % 6.0) * 60.0).copy(); h[~nz] = 0.0
    hn = (h + deg) % 360.0
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = hn / 60.0
    xv = c * (1.0 - np.abs(hp % 2.0 - 1.0)); mm = l - c / 2.0
    k = hp.astype(np.int64) % 6
    R = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [c, xv, 0, 0, xv, c])
    G = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [xv, c, c, xv, 0, 0])
    B = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [0, 0, xv, c, c, xv])
    out = (np.clip(np.stack([R + mm, G + mm, B + mm], -1), 0, 1) * 255).astype(np.uint8)
    res = u8.copy(); res[mask] = out[mask]
    return res


def to_t(u8):
    return torch.tensor(u8.astype(np.float32) / 255.0, device=DEV).permute(2, 0, 1)[None]


def dets(out):
    pred = out[0] if isinstance(out, (list, tuple)) else out
    r = non_max_suppression(pred.clone(), conf_thres=CONF, iou_thres=IOU_NMS, max_det=30)[0]
    return r.cpu().numpy() if r is not None and len(r) else np.zeros((0, 6), np.float32)


def match(hyp, ref):
    tp = np.zeros(len(hyp), bool)
    if len(hyp) == 0 or len(ref) == 0:
        return tp, np.zeros((len(hyp), len(ref)), np.float32)
    iou = np.zeros((len(hyp), len(ref)), np.float32)
    for a in range(len(hyp)):
        ax0, ay0, ax1, ay1 = hyp[a, :4]
        for b in range(len(ref)):
            if hyp[a, 5] != ref[b, 5]:
                continue
            bx0, by0, bx1, by1 = ref[b, :4]
            iw = max(0.0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0.0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih
            ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
            iou[a, b] = inter / ua if ua > 0 else 0.0
    used = set()
    for o in np.argsort(-iou.ravel()):
        a, b = int(o // len(ref)), int(o % len(ref))
        if iou[a, b] < IOU_MATCH or tp[a] or b in used:
            continue
        tp[a] = True; used.add(b)
    return tp, iou


def box_metrics(hyp, ref):
    tp, iou = match(hyp, ref)
    prec = tp.sum() / max(1, len(hyp)); rec = tp.sum() / max(1, len(ref))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    order = np.argsort(-hyp[:, 4]) if len(hyp) else np.array([], int)
    tp_s = tp[order]
    if len(tp_s) and rec > 0:
        cum = np.cumsum(tp_s); precs = cum / (np.arange(len(tp_s)) + 1)
        recs = cum / max(1, len(ref))
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precs[recs >= t]
            ap += (p.max() if len(p) else 0.0) / 11
    else:
        ap = 0.0
    miou = float(iou[tp].max(1).mean()) if tp.any() and len(ref) else 0.0
    return {"n_hyp": int(len(hyp)), "n_ref": int(len(ref)), "tp": int(tp.sum()),
            "precision": float(prec), "recall": float(rec), "f1_iou50": float(f1), "ap50_11pt": float(ap),
            "mean_iou_matched": float(miou)}


def tensor_of(out):
    pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_held", type=int, default=80)
    ap.add_argument("--op8_epochs", type=int, default=400)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(f for f in os.listdir(IMD) if f.endswith(".jpg"))
    perm = rng.permutation(len(files))
    tr = [files[i] for i in perm[:a.n_train]]
    he = [files[i] for i in perm[a.n_train:a.n_train + a.n_held]]
    net = load_det()
    mid = nn.Sequential(*list(net.model[:4])).to(DEV).eval()
    tail = list(net.model[4:])

    def fmap(u8):
        with torch.no_grad():
            return mid(to_t(u8))

    def full(u8):
        with torch.no_grad():
            return net(to_t(u8))

    def forward_hooked(u8, op):
        h = net.model[3].register_forward_hook(lambda m, i, o: op(o))
        try:
            with torch.no_grad():
                out = net(to_t(u8))
        finally:
            h.remove()
        return out

    out = {"weights": os.path.basename(WT), "seed": a.seed, "n_train": a.n_train, "n_held": a.n_held,
           "deltas": {}}
    for delta in DELTAS:
        t0 = time.time()
        # ---- fit on train: rows/maps of (x, T_delta x) restricted to object cells
        X, Y, Xm, Ym = [], [], [], []
        for fn in tr:
            u8 = np.asarray(Image.open(os.path.join(IMD, fn)).convert("RGB"))
            mk = box_mask(fn)
            u8s = shift_u8(u8, delta, mk)
            f0, f1 = fmap(u8), fmap(u8s)
            cm = torch.tensor(cell_mask(fn), device=DEV)
            X.append(f0.reshape(FCH, -1)[:, cm]); Y.append(f1.reshape(FCH, -1)[:, cm])
            Xm.append(f0[0]); Ym.append(f1[0])
        Xr = torch.cat(X, 1).T.contiguous(); Yr = torch.cat(Y, 1).T.contiguous()
        Xm = torch.stack(Xm); Ym = torch.stack(Ym)
        # O1 procrustes
        U, S, Vh = torch.linalg.svd((Yr.T @ Xr).float())
        W1 = (U @ Vh)
        # O2 ridge
        W2 = torch.linalg.solve((Xr.T @ Xr + 1e-3 * torch.eye(FCH, device=DEV)), Xr.T @ Yr).T
        # O8 ridge + conv residual on the full 28x28 maps
        tgt = Ym - torch.einsum("bchw,dc->bdhw", Xm, W2)
        conv = nn.Sequential(nn.Conv2d(FCH, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, FCH, 3, padding=1)).to(DEV)
        nn.init.zeros_(conv[-1].weight); nn.init.zeros_(conv[-1].bias)
        opt = torch.optim.Adam(conv.parameters(), lr=1e-3)
        for _ in range(a.op8_epochs):
            opt.zero_grad(); F.mse_loss(conv(Xm), tgt).backward(); opt.step()
        Wr = torch.randn(FCH, FCH, device=DEV); Q, R = torch.linalg.qr(Wr)

        def apply_W(W, f):
            return torch.einsum("ij,jn->in", W, f.reshape(FCH, -1)).reshape(1, FCH, FH, FW)

        def apply_O8(f):
            with torch.no_grad():
                return torch.einsum("bchw,dc->bdhw", f, W2) + conv(f)

        ops = {"O1_procrustes": lambda f: apply_W(W1, f), "O2_ridge": lambda f: apply_W(W2, f),
               "O8_conv_residual": apply_O8, "O6_random": lambda f: apply_W(Q, f)}
        rec = {"power": {}, "routes": {}}
        # ---- held-out: real and null
        treal, tnull, t_int = {}, {}, {}
        real_dets, null_dets = [], []
        for fn in he:
            u8 = np.asarray(Image.open(os.path.join(IMD, fn)).convert("RGB"))
            u8s = shift_u8(u8, delta, box_mask(fn))
            o_real = full(u8s); o_null = full(u8)
            real_dets.append(dets(o_real)); null_dets.append(dets(o_null))
            treal[fn] = tensor_of(o_real); tnull[fn] = tensor_of(o_null)
        rec["power"]["tensor_rel_l2_real_vs_null"] = float(np.mean([
            (torch.norm(treal[f] - tnull[f]) / (torch.norm(treal[f]) + 1e-9)).item() for f in he]))
        m = {"n_hyp": 0, "n_ref": 0, "tp": 0, "precision": 0.0, "recall": 0.0, "f1_iou50": 0.0,
             "ap50_11pt": 0.0, "mean_iou_matched": 0.0}
        for i, fn in enumerate(he):
            r = box_metrics(null_dets[i], real_dets[i])
            for k in m:
                m[k] += r[k]
        for k in ("precision", "recall", "f1_iou50", "ap50_11pt", "mean_iou_matched"):
            m[k] /= max(1, len(he))
        rec["routes"]["null"] = {"box": m, "tensor_rel_l2_vs_real": float(np.mean([
            (torch.norm(treal[f] - tnull[f]) / (torch.norm(treal[f]) + 1e-9)).item() for f in he]))}
        for name, op in ops.items():
            boxacc = {"n_hyp": 0, "n_ref": 0, "tp": 0, "precision": 0.0, "recall": 0.0, "f1_iou50": 0.0,
                      "ap50_11pt": 0.0, "mean_iou_matched": 0.0}
            rl, cosl, num, den = [], [], 0.0, 0.0
            for i, fn in enumerate(he):
                u8 = np.asarray(Image.open(os.path.join(IMD, fn)).convert("RGB"))
                y = forward_hooked(u8, op)
                r = box_metrics(dets(y), real_dets[i])
                for k in boxacc:
                    boxacc[k] += r[k]
                t_int = tensor_of(y)
                rl.append((torch.norm(t_int - treal[fn]) / (torch.norm(treal[fn]) + 1e-9)).item())
                d_i, d_r = t_int - tnull[fn], treal[fn] - tnull[fn]
                cosl.append(F.cosine_similarity(d_i[None], d_r[None]).item())
                num += float((d_i * d_r).sum()); den += float((d_r * d_r).sum())
            for k in ("precision", "recall", "f1_iou50", "ap50_11pt", "mean_iou_matched"):
                boxacc[k] /= max(1, len(he))
            rec["routes"][name] = {"box": boxacc, "tensor_rel_l2_vs_real": float(np.mean(rl)),
                                   "align_cos": float(np.mean(cosl)), "proj_coef_agg": num / (den + 1e-12)}
        rec["runtime_sec"] = round(time.time() - t0, 1)
        out["deltas"][str(int(delta))] = rec
        print(f"[seed {a.seed} delta {int(delta)}] power relL2 {rec['power']['tensor_rel_l2_real_vs_null']:.3f} | "
              + " ".join(f"{k}:F1 {v['box']['f1_iou50']:.3f}/AP {v['box']['ap50_11pt']:.3f}/proj {v.get('proj_coef_agg', 0):.2f}"
                         for k, v in rec["routes"].items()), flush=True)
    fn = os.path.join(WORK, "results", f"detector_o8_seed{a.seed}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
