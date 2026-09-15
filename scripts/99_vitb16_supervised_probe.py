# -*- coding: utf-8 -*-
"""Supervised pretrained ViT (torchvision ViT-B/16, IMAGENET1K_V1) per-block probe on dense5,
same methodology as scripts/94 (DINOv2): pose-averaged hue orbits -> per-block token-mean ->
per-shape DC-removed FFT (k1+k2, winding, norm stability) + cross-shape zero-shot hue readout
on held-out hues {20,260}.
Outputs results/vitb16_supervised_probe.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = rp.REPO_ROOT
H = np.arange(0, 360, 5)
SH = ['triangle', 'rectangle', 'circle', 'star', 'pentagon', 'hexagon']
TR, TE = [0, 1, 2, 3], [4, 5]

d = np.load(os.path.join(WORK, "data", "dense5.npz"))
img, shape, hue = d["img"], d["shape"], d["hue"]

model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
model.to(DEV).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

outs = {}


def make_hook(i):
    def hook(m, inp, out):
        x = out.detach().float()            # (B, 197, 768)
        outs.setdefault(i, []).append(x.mean(dim=1).cpu().numpy())
    return hook


# ViT-B/16: encoder.layers[i]; hooks after each transformer block
handles = [model.encoder.layers[i].register_forward_hook(make_hook(i))
           for i in range(len(model.encoder.layers))]
with torch.no_grad():
    for s0 in range(0, len(img), 16):
        a = img[s0:s0 + 16].astype(np.float32) / 255.0
        x = torch.tensor(a, device=DEV).permute(0, 3, 1, 2)
        x = (x - mean) / std
        model(x)
for h in handles:
    h.remove()
nblocks = len(model.encoder.layers)
print("blocks:", nblocks)

res = {}
for bi in range(nblocks):
    Zraw = np.concatenate(outs[bi])
    D = Zraw.shape[1]
    Z = Zraw.reshape(6, 72, 4, D).mean(2)
    ks, mws, ns = [], [], []
    for si in range(6):
        Zc = Z[si] - Z[si].mean(0)
        E = (np.abs(np.fft.rfft(Zc, axis=0)) ** 2).sum(1)
        Et = E.sum()
        ks.append((E[1] + E[2]) / Et)
        mws.append(float(np.sum(np.arange(len(E)) * E) / Et))
        n = np.linalg.norm(Z[si], axis=1)
        ns.append(n.std() / n.mean())
    # zero-shot readout {20,260}
    Xs, Ys = [], []
    for si in TR:
        Zi = Z[si]
        for hi, h in enumerate(H):
            if h in (20, 260) or h == 0:
                continue
            Xs.append(Zi[hi]); Ys.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    Xt = torch.tensor(np.array(Xs), dtype=torch.float32, device=DEV)
    Yt = torch.tensor(np.array(Ys), dtype=torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(D, 128), nn.SiLU(), nn.Linear(128, 2)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(250):
        p = net(Xt); loss = Fn.mse_loss(p, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
    errs = []
    with torch.no_grad():
        for si in TE:
            Zi = Z[si]
            for hi, h in enumerate(H):
                if h not in (20, 260):
                    continue
                y = net(torch.tensor(Zi[hi], dtype=torch.float32, device=DEV)).cpu().numpy()
                ang = np.degrees(np.arctan2(y[1], y[0])) % 360
                errs.append(min(abs(ang - h), 360 - abs(ang - h)))
    res[bi] = {"k1k2": round(float(np.mean(ks)), 3),
               "per_shape_k1k2": {SH[s]: round(float(ks[s]), 3) for s in range(6)},
               "winding": round(float(np.mean(mws)), 2),
               "norm_std": round(float(np.mean(ns)), 4),
               "read_err_deg": round(float(np.mean(errs)), 2),
               "D": D}
    print(f"block {bi:2d} D={D:4d} k1k2={res[bi]['k1k2']:.3f} wind={res[bi]['winding']:.2f} "
          f"read={res[bi]['read_err_deg']}")

out = {"arch": "torchvision ViT-B/16 (IMAGENET1K_V1, supervised)", "protocol": "same as 94_dinov2_probe",
       "n_blocks": nblocks, "blocks": {str(k): v for k, v in res.items()}}
json.dump(out, open(os.path.join(WORK, "results/vitb16_supervised_probe.json"), "w"), indent=1)
print("saved results/vitb16_supervised_probe.json")
