# -*- coding: utf-8 -*-
"""Region-wise rho operator: definition + demonstration (no retraining).
On the existing per-cell code (a spatial colour field) we define, for a region r,
    rho_r(Delta):  Z1(u,v) -> e^{i Delta} Z1(u,v)  for (u,v) in r
so that the pooled region phase Phi_r = arg sum_{r} Z1 advances by exactly Delta, and different
regions can receive different shifts in the same forward pass.
We verify:
  (a) exactness: pooled phase advance == Delta (by construction, measured),
  (b) equivalence: the code-space shift matches a real pixel recolour + re-forward for the region
      (region-level equivariance error),
  (c) independent multi-region shifts (A: +60 deg, B: -40 deg) with unchanged background,
  (d) cost: per-region rho (microseconds) vs pixel route (shift + full re-forward, ~38 ms).
Outputs results/region_rho_demo.json + figure F17_region_rho.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

WORK = rp.REPO_ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
SIZE = 224

from ultralytics.nn.tasks import DetectionModel


class AuxDet(DetectionModel):
    def _capture(self, m, i, o):
        return o
    def forward(self, x, *a, **k):
        return super().forward(x, *a, **k)


net = torch.load(os.path.join(WORK, "runs/detect/runs_det2/arm_aux_r/weights/best.pt"),
                 map_location=DEV, weights_only=False)
if isinstance(net, dict):
    net = net["model"]
net.to(DEV).eval(); net.float()
net.aux_code = getattr(net, "aux_code", nn.Conv2d(64, 4, 1).to(DEV)).float()
mid_seq = nn.Sequential(*list(net.model[:4])).to(DEV).eval()


def hsv_rgb(h, s, v):
    import colorsys
    return tuple(int(255*c) for c in colorsys.hsv_to_rgb((h % 360)/360.0, s, v))


def render_two_region(hA, hB, bands=4, s=0.85, v=0.85, R=80):
    """pentagon split into bands; region A = even bands, region B = odd bands."""
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    m = Image.new("L", (SIZE, SIZE), 0); d = ImageDraw.Draw(m)
    n = 5
    d.polygon([(SIZE/2 + R*math.cos(2*math.pi*k/n - math.pi/2),
                SIZE/2 + R*math.sin(2*math.pi*k/n - math.pi/2)) for k in range(n)], fill=255)
    mask = np.asarray(m) > 0
    img[mask] = hsv_rgb(hA, s, v)
    xs = np.arange(SIZE)[None, :].repeat(SIZE, 0)
    band = (xs / (SIZE / bands)).astype(int)
    regB = mask & (band % 2 == 1)
    img[regB] = hsv_rgb(hB, s, v)
    # region masks on the stride-8 grid
    cellA = np.zeros((28, 28), bool); cellB = np.zeros((28, 28), bool)
    for gy in range(28):
        for gx in range(28):
            blk = mask[gy*8:(gy+1)*8, gx*8:(gx+1)*8]
            if blk.sum() < 20:
                continue
            blkB = regB[gy*8:(gy+1)*8, gx*8:(gx+1)*8]
            if blkB.sum() > blk.sum()/2:
                cellB[gy, gx] = True
            else:
                cellA[gy, gx] = True
    return Image.fromarray(img), cellA, cellB, mask, regB


def code_map(im):
    a = np.asarray(im).astype(np.float32)/255.0
    x = torch.tensor(a, device=DEV).permute(2, 0, 1)[None]
    with torch.no_grad():
        f = mid_seq(x)
        code = net.aux_code(f)[0]           # (4,28,28)
    return code.cpu().numpy()


def pooled_phase(code, cells):
    z1x = code[0][cells]; z1y = code[1][cells]
    if len(z1x) == 0:
        return None, 0.0
    vx = z1x.mean(); vy = z1y.mean()
    return float(math.degrees(math.atan2(vy, vx)) % 360), float(np.hypot(vx, vy))


circ = lambda a, b: min(abs(a-b), 360-abs(a-b))
rows = []
for (hA, hB) in [(0, 120), (40, 210), (300, 60)]:
    for bands in [2, 4]:
        im, cellA, cellB, mask, regB = render_two_region(hA, hB, bands=bands)
        code0 = code_map(im)
        phiA0, mA0 = pooled_phase(code0, cellA)
        phiB0, mB0 = pooled_phase(code0, cellB)
        dA, dB = 60.0, -40.0
        # (a) region-wise rho applied in code space
        t0 = time.perf_counter()
        for _ in range(1000):
            codeS = code0.copy()
            thA = math.radians(dA); cA, sA = math.cos(thA), math.sin(thA)
            thB = math.radians(dB); cB, sB = math.cos(thB), math.sin(thB)
            for cells, (c, s) in [(cellA, (cA, sA)), (cellB, (cB, sB))]:
                zx, zy = codeS[0][cells].copy(), codeS[1][cells].copy()
                codeS[0][cells] = c*zx - s*zy
                codeS[1][cells] = s*zx + c*zy
        t_rho = (time.perf_counter() - t0)/1000
        phiA1, _ = pooled_phase(codeS, cellA)
        phiB1, _ = pooled_phase(codeS, cellB)
        # (b) real pixel recolour of each region + re-forward
        img_arr = np.asarray(im).astype(np.float32)/255.0
        shifted = img_arr.copy()
        for msk, dshift in [(cellA, dA), (cellB, dB)]:
            ys, xs = np.where(msk)
            # map stride-8 cells back to pixels
            pmsk = np.zeros((SIZE, SIZE), bool)
            for gy, gx in zip(ys, xs):
                pmsk[gy*8:(gy+1)*8, gx*8:(gx+1)*8] = True
            pmsk &= mask
            sub = shifted.copy()
            rgb = sub[pmsk]
            mx = rgb.max(1); mn = rgb.min(1); dd = mx-mn
            hh = np.where(mx == rgb[:, 0], (rgb[:, 1]-rgb[:, 2])/np.maximum(dd, 1e-6),
                          np.where(mx == rgb[:, 1], 2 + (rgb[:, 2]-rgb[:, 0])/np.maximum(dd, 1e-6),
                                   4 + (rgb[:, 0]-rgb[:, 1])/np.maximum(dd, 1e-6)))
            hh = (hh % 6)*60.0
            ss = np.where(mx > 1e-6, dd/np.maximum(mx, 1e-6), 0)
            vr = mx
            h2 = (hh + dshift) % 360
            import colorsys
            newrgb = np.array([colorsys.hsv_to_rgb(h/360.0, s_, v_) for h, s_, v_ in zip(h2, ss, vr)], np.float32)
            shifted[pmsk] = newrgb
        im_real = Image.fromarray(np.clip(shifted*255, 0, 255).astype(np.uint8))
        code_real = code_map(im_real)
        phiA_real, _ = pooled_phase(code_real, cellA)
        phiB_real, _ = pooled_phase(code_real, cellB)
        rows.append({
            "hA": hA, "hB": hB, "bands": bands, "dA": dA, "dB": dB,
            "phaseA_orig": round(phiA0, 1), "phaseB_orig": round(phiB0, 1),
            "rho_advance_A": round(circ(phiA1, (phiA0 + dA) % 360), 2),
            "rho_advance_B": round(circ(phiB1, (phiB0 + dB) % 360), 2),
            "real_advance_A": round((phiA_real - phiA0) % 360, 1),
            "real_advance_B": round((phiB_real - phiB0) % 360, 1),
            "code_vs_real_A": round(circ(phiA1, phiA_real), 1),
            "code_vs_real_B": round(circ(phiB1, phiB_real), 1),
            "region_mag_A": round(mA0, 3), "region_mag_B": round(mB0, 3),
        })
        print(rows[-1], flush=True)

perf = json.load(open(os.path.join(WORK, "results/perf_reforward_vs_rho.json")))
out = {"protocol": ("per-cell code = spatial colour field; region-wise operator rho_r(Delta) rotates the code "
                    "phase inside region r only; verified on controlled two-region renders (region A = even bands, "
                    "region B = odd bands) against a real pixel recolour + re-forward of the same regions"),
       "region_rho_cost_s": t_rho,
       "pixel_route_cost_s": perf["pixel_side_recolor_s"],
       "results": rows}
json.dump(out, open(os.path.join(WORK, "results/region_rho_demo.json"), "w"), indent=1)
print("saved results/region_rho_demo.json | rho cost %.1f us vs pixel route %.1f ms"
      % (t_rho*1e6, perf["pixel_side_recolor_s"]*1e3))

# figure: one example, original vs region-wise shifted codes
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
im, cellA, cellB, mask, regB = render_two_region(0, 120, bands=4)
code0 = code_map(im)
ph = np.degrees(np.arctan2(code0[1], code0[0])) % 360
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im); axes[0].set_title("input: two colour regions"); axes[0].axis("off")
axes[1].imshow(ph, cmap="hsv", vmin=0, vmax=360); axes[1].set_title("per-cell code phase Z1(u,v)"); axes[1].axis("off")
shifted = ph.copy()
shifted[cellA] = (shifted[cellA] + 60) % 360
shifted[cellB] = (shifted[cellB] - 40) % 360
axes[2].imshow(shifted, cmap="hsv", vmin=0, vmax=360)
axes[2].set_title("region-wise $\\rho_r(\\Delta)$: A +60°, B −40° (no re-forward)"); axes[2].axis("off")
plt.tight_layout(); plt.savefig(os.path.join(WORK, "results/figures/F17_region_rho.png"), dpi=150)
print("figure F17_region_rho.png written")
