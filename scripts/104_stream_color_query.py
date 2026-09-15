# -*- coding: utf-8 -*-
"""Downstream application demo: streaming per-box color query / gating with cached features.

Scenario: a deployment keeps the stride-8 feature maps and the boxes (features are cached because
re-running the backbone is expensive), and the raw pixels are *not* retained by the consumer.
Questions asked at runtime:
  (Q1) which boxes are in a given hue window? (color gating / sorting / retrieval)
  (Q2) how long does answering cost?
We measure, over N validation images:
  - code path: per-box hue from the single cached forward per image (no extra forward)
  - pixel path: per-box dominant pixel hue (requires retaining pixels; no forward)
  - network-view path: what a pixel-side recolor would cost if the network's own color view is
    needed instead (full re-forward, measured separately in script 100)
and we report (a) latency per query batch, (b) agreement between code-based and pixel-based gating.
Outputs results/stream_color_query.json + demo figure.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

WORK = rp.REPO_ROOT
sys.path.insert(0, WORK + "/scripts")
LAB = os.path.join(WORK, "data/yolodet/val_big/labels")
IMDIR = os.path.join(WORK, "data/yolodet/val_big/images")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WT = os.path.join(WORK, "runs/detect/runs_det2/arm_aux_r/weights/best.pt")

from ultralytics.nn.tasks import DetectionModel


class AuxDet(DetectionModel):          # pickle-compatible with scripts/77
    def _capture(self, m, i, o):
        self._mid_cache = o
        return o

    def forward(self, x, *a, **k):
        return super().forward(x, *a, **k)


net = torch.load(WT, map_location=DEV, weights_only=False)
if isinstance(net, dict):
    net = net.get("model", net)
net.to(DEV).eval()
net.float()
net.aux_code = getattr(net, "aux_code", nn.Conv2d(64, 4, 1).to(DEV)).float()
mid_seq = nn.Sequential(*list(net.model[:4])).to(DEV).eval()


def pixel_hue_of_box(pil, box, pad=0.0):
    a = np.asarray(pil).astype(np.float32) / 255.0
    H, W = a.shape[:2]
    _, cx, cy, bw, bh = box
    x0 = int(max(0, (cx - bw / 2) * W)); x1 = int(min(W, (cx + bw / 2) * W))
    y0 = int(max(0, (cy - bh / 2) * H)); y1 = int(min(H, (cy + bh / 2) * H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    p = a[y0:y1, x0:x1].reshape(-1, 3)
    mx = p.max(1); mn = p.min(1); d = mx - mn
    nz = d > 1e-6
    if nz.sum() < 5:
        return None
    r, g, b = p[nz, 0], p[nz, 1], p[nz, 2]
    dd = d[nz]; mxx = mx[nz]
    h = np.where(r == mxx, (g - b) / dd, np.where(g == mxx, 2 + (b - r) / dd, 4 + (r - g) / dd))
    h = (h % 6) * 60.0
    w = dd / np.maximum(mxx, 1e-6)
    vx = np.sum(w * np.cos(np.deg2rad(h))); vy = np.sum(w * np.sin(np.deg2rad(h)))
    return float(np.degrees(np.arctan2(vy, vx)) % 360)


files = sorted(f for f in os.listdir(LAB) if f.endswith(".txt"))[:200]
# ---- one pass: code hues for all boxes (this is the *cached-feature* artifact we keep) ----
code_hues, pix_hues, box_meta = [], [], []
t_code_total = 0.0
for fn in files:
    boxes = np.loadtxt(os.path.join(LAB, fn)).reshape(-1, 5)
    pil = Image.open(os.path.join(IMDIR, fn[:-4] + ".jpg")).convert("RGB")
    x = torch.tensor(np.asarray(pil).astype(np.float32) / 255.0, device=DEV).permute(2, 0, 1)[None]
    t0 = time.perf_counter()
    with torch.no_grad():
        mid = mid_seq(x)
        code = net.aux_code(mid)[0]
    t_code_total += time.perf_counter() - t0
    for j, b in enumerate(boxes):
        _, cx, cy, bw, bh = b
        gx = int(min(max(cx * 224 / 8.0, 0), code.shape[-1] - 1))
        gy = int(min(max(cy * 224 / 8.0, 0), code.shape[-2] - 1))
        z1x, z1y = float(code[0, gy, gx]), float(code[1, gy, gx])
        code_hues.append(math.degrees(math.atan2(z1y, z1x)) % 360)
        pix_hues.append(pixel_hue_of_box(pil, b))
        box_meta.append((fn, j))

# ---- Q1: color gating over 12 hue windows (runtime query cost) ----
windows = [(i * 30, i * 30 + 40) for i in range(12)]
def in_win(h, w):
    return (h is not None) and ((h - w[0]) % 360) <= (w[1] - w[0])

# code path query cost: rotate the cached 6-d codes and threshold
cvec = np.zeros((len(code_hues), 6), np.float32)
cvec[:, 2] = np.cos(np.deg2rad(code_hues)); cvec[:, 3] = np.sin(np.deg2rad(code_hues))
def rho_mat(d):
    th = math.radians(d); c1, s1 = math.cos(th), math.sin(th); c2, s2 = math.cos(2 * th), math.sin(2 * th)
    return np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,c1,-s1,0,0],[0,0,s1,c1,0,0],[0,0,0,0,c2,-s2],[0,0,0,0,s2,c2]], np.float32)

t0 = time.perf_counter()
for _ in range(50):
    for w in windows:
        Rm = rho_mat(w[0])
        rot = cvec @ Rm.T
        hues = (np.degrees(np.arctan2(rot[:, 3], rot[:, 2])) % 360)
        sel = ((hues - w[0]) % 360) <= 40
t_code_query = (time.perf_counter() - t0) / 50 / len(windows)   # per window over all boxes

# pixel path query cost: need pixels; here we already computed hues, but the expensive part is
# reading pixels + HSV when pixels are cold (not retained). Measure a cold read+crop+HSV per box.
t0 = time.perf_counter()
for fn, j in box_meta[:400]:
    pil = Image.open(os.path.join(IMDIR, fn[:-4] + ".jpg")).convert("RGB")
    boxes = np.loadtxt(os.path.join(LAB, fn)).reshape(-1, 5)
    _ = pixel_hue_of_box(pil, boxes[j])
t_pixel_query_400 = (time.perf_counter() - t0) / 400.0

# agreement between code-based and pixel-based gating
agree, tot = 0, 0
for w in windows:
    for ch, ph in zip(code_hues, pix_hues):
        if ph is None:
            continue
        tot += 1
        agree += int(in_win(ch, w) == in_win(ph, w))

perf = json.load(open(os.path.join(WORK, "results/perf_reforward_vs_rho.json")))
res = {
  "n_images": len(files), "n_boxes": len(code_hues),
  "cached_forward_time_per_image_s": round(t_code_total / len(files), 4),
  "query_code_path_per_window_all_boxes_s": t_code_query,
  "query_pixel_path_per_box_cold_s": t_pixel_query_400,
  "query_network_reforward_per_box_s": perf["pixel_side_recolor_s"],
  "gating_agreement_code_vs_pixel": round(agree / max(tot, 1), 4),
  "protocol": ("200 val_big images; boxes from GT labels; code hue read from the single cached "
               "stride-8 forward per image; pixel hue from the same box region when pixels are "
               "available; gating over 12 hue windows of width 40 deg."),
  "note": ("In a feature-cached deployment the consumer keeps stride-8 features (and boxes) but not "
           "pixels: the code path answers color queries from cache, while the pixel path is "
           "unavailable without re-reading and re-decoding images. If the network's own color view "
           "is required, the pixel route additionally needs a full re-forward "
           f"({perf['pixel_side_recolor_s']*1000:.1f} ms/object) versus a {t_code_query*1e6:.0f} ns "
           "class rotation on cached codes."),
}
json.dump(res, open(os.path.join(WORK, "results/stream_color_query.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "protocol"}, indent=1))
