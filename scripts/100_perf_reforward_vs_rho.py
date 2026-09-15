# -*- coding: utf-8 -*-
"""Time-cost contrast: pixel-side hue recolor (requires full backbone re-forward)
vs closed-form code rotation rho(delta) (a sparse 6x6 linear op on the code vector).
No retraining; measures the real per-operation cost on the deployment GPU.
Outputs results/perf_reforward_vs_rho.json (+ rough figures for the paper).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, time, math
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = rp.REPO_ROOT
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

# --- backbone forward cost (YOLO11n, 224x224, batch=1, fp32) ---
base = YOLO(rp.YOLO_WEIGHTS).model.to(DEV).eval()
x = torch.randn(1, 3, 224, 224, device=DEV)
with torch.no_grad():
    for _ in range(10):
        base(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        base(x)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) / 100.0   # seconds per full forward

# pixel hue shift helper cost (numpy HSV rotate on 224x224) - include as input-side extra
def shift_np(a01, deg):
    r, g, b = a01[..., 0], a01[..., 1], a01[..., 2]
    mx = np.maximum(r, np.maximum(g, b)); mn = np.minimum(r, np.minimum(g, b))
    d = mx - mn; l = (mx + mn) / 2.0
    s = np.zeros_like(d); nz = d > 1e-6
    s[nz] = np.where(l[nz] < 0.5, d[nz] / (mx[nz] + mn[nz]), d[nz] / (2.0 - mx[nz] - mn[nz]))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(r == mx, (g - b) / d, np.where(g == mx, 2.0 + (b - r) / d, 4.0 + (r - g) / d))
        h = (h % 6.0) * 60.0; h = h.copy(); h[~nz] = 0.0
    hn = (h + deg) % 360.0
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = hn / 60.0; xv = c * (1.0 - np.abs(hp % 2.0 - 1.0)); mm = l - c / 2.0
    k = hp.astype(np.int64) % 6
    R = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [c, xv, 0, 0, xv, c])
    G = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [xv, c, c, xv, 0, 0])
    B = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [0, 0, xv, c, c, xv])
    return np.stack([R + mm, G + mm, B + mm], -1)

img = (np.random.rand(224, 224, 3).astype(np.float32))
for _ in range(10):
    shift_np(img, 30.0)
t0 = time.perf_counter()
for _ in range(200):
    shift_np(img, 30.0)
t_shift = (time.perf_counter() - t0) / 200.0

# --- closed-form code rotation cost: rho(delta) on the 6-d code ---
# rho = diag(1,1,R(d),R(2d)) -> implemented as precomputed 6x6 matmul
def rho_mat(d):
    th = math.radians(d)
    c1, s1 = math.cos(th), math.sin(th)
    c2, s2 = math.cos(2 * th), math.sin(2 * th)
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, c1, -s1, 0, 0],
                     [0, 0, s1, c1, 0, 0],
                     [0, 0, 0, 0, c2, -s2],
                     [0, 0, 0, 0, s2, c2]], np.float32)
cvec = np.random.rand(100000, 6).astype(np.float32)   # batch of code vectors (many objects at once)
Rm = rho_mat(30.0)
t0 = time.perf_counter()
for _ in range(50):
    out = cvec @ Rm.T
t_rho_batch = (time.perf_counter() - t0) / 50.0 / len(cvec)  # per object
# single-vector version (per-object loop scenario)
t0 = time.perf_counter()
for _ in range(20000):
    _ = cvec[:1] @ Rm.T
t_rho_1 = (time.perf_counter() - t0) / 20000.0

res = {
  "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
  "backbone_full_forward_224_batch1_s": t_fwd,
  "pixel_hsv_shift_224_s": t_shift,
  "pixel_side_recolor_s": t_shift + t_fwd,          # shift pixels + full re-forward
  "code_rotation_per_object_s_batched": t_rho_batch, # amortized over 100k objects
  "code_rotation_single_s": t_rho_1,
  "speedup_pixel_vs_code_batched": (t_shift + t_fwd) / max(t_rho_batch, 1e-12),
  "note": ("Pixel-side recoloring of an object = recolor pixels + full backbone forward. "
           "Code-side recoloring = one sparse 6x6 rotation per object on already-extracted "
           "features/code; no forward. 'per-object color view' is what changes; to also change "
           "downstream outputs a code-consuming head is required (e.g., the YOLO per-box head, "
           "Section 9), not a re-forward."),
}
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
json.dump(res, open(os.path.join(WORK, "results/perf_reforward_vs_rho.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
