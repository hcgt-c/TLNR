# -*- coding: utf-8 -*-
"""Is the harmonic colour geometry learned, or a trivial consequence of the input?
Control chain requested by review: pixels -> random-init network -> trained network, measured with
exactly the paper's protocol on the same data (features/dense5: 6 shapes x 72 hues x 4 poses).

Arms
  pixels (hue)     : the input's own saturation-weighted hue distribution is exactly harmonic by
                     construction, so it fixes the trivial baseline (k1 = 1 by definition).
  pixels (rgb)     : mean RGB of the object region, i.e. what a linear readout of the image sees.
  random y3 / y5 / y7 : the SAME YOLO11n architecture with re-initialised weights (no training).
  trained y3 / y5 / y7: the released pretrained YOLO11n.

Metrics, identical for every arm (pose-averaged features, per-shape DC removal):
  k1+k2 share of the centred hue-orbit energy (paper: 0.868 for trained y3);
  cross-shape phase alignment (mean pairwise cosine of unit-normalised per-frequency curves);
  cross-shape zero-shot hue readout: ridge probe fitted on the 4 training shapes, evaluated on the
  2 unseen shapes at held-out hues {20, 260} (same protocol as the paper's code readout).
Outputs results/harmonic_controls.json + results/figures/F24_learned_or_trivial.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

WORK = rp.REPO_ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SH = ['triangle', 'rectangle', 'circle', 'star', 'pentagon', 'hexagon']
TRAIN_SH = [0, 1, 2, 3]          # triangle, rectangle, circle, star
TEST_SH = [4, 5]                 # pentagon, hexagon (unseen)
HELD_OUT_HUES = [20, 260]        # never seen during probe fitting
H = np.arange(0, 360, 5)
NV = 4
torch.manual_seed(0)


def feats_of(model, imgs, hooks_idx=(3, 5, 7)):
    """extract GAP features at the given module indices for a stack of uint8 images"""
    caps = {}
    handles = [model.model[i].register_forward_hook(
        (lambda idx: lambda m, i_, o: caps.__setitem__(idx, o.detach()))(i)) for i in hooks_idx]
    out = {i: [] for i in hooks_idx}
    with torch.no_grad():
        for k in range(len(imgs)):
            a = imgs[k].astype(np.float32) / 255.0
            x = torch.tensor(a, device=DEV).permute(2, 0, 1).unsqueeze(0)
            model(x)
            for i in hooks_idx:
                o = caps[i]
                out[i].append(o.mean(dim=(2, 3))[0].cpu().numpy() if o.dim() == 4 else o[0].cpu().numpy())
    for h in handles:
        h.remove()
    return {i: np.stack(v) for i, v in out.items()}


def standardise(Z):
    """z-score each feature dimension across the hue orbit (per shape), removing the dominance of
    high-variance non-hue directions so the hue-orbit energy share is comparable across arms"""
    out = np.zeros_like(Z)
    for si in range(Z.shape[0]):
        mu = Z[si].mean(0, keepdims=True)
        sd = Z[si].std(0, keepdims=True) + 1e-9
        out[si] = (Z[si] - mu) / sd
    return out


def harmonics(Z):
    """Z: (6,72,d) pose-averaged; returns k1+k2 share per shape and the cross-shape phase alignment"""
    shares = []
    for si in range(6):
        G = Z[si] - Z[si].mean(0)
        E = (np.abs(np.fft.rfft(G, axis=0)) ** 2).sum(1)
        shares.append(float((E[1] + E[2]) / E.sum()))
    # cross-shape phase alignment: unit-normalise each frequency's complex response, then cosine
    cosines = []
    unit = []
    for si in range(6):
        G = Z[si] - Z[si].mean(0)
        Fq = np.fft.rfft(G, axis=0)                       # (36, d)
        n = np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-12
        unit.append(Fq / n)
    for a in range(6):
        for b in range(a + 1, 6):
            v = np.real((unit[a] * np.conj(unit[b])).sum(1))   # per-frequency alignment
            cosines.append(float(np.mean(v[1:4])))             # k=1..3 band
    return shares, cosines


def readout(Z, d):
    """ridge hue probe: fit on training shapes (all hues except held-out), test on unseen shapes
    at the held-out hues; returns median circular error in degrees"""
    X, Y = [], []
    for si in TRAIN_SH:
        for hi, h in enumerate(H):
            if h in HELD_OUT_HUES:
                continue
            X.append(Z[si, hi]); Y.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    X = np.stack(X); Y = np.stack(Y)
    lam = 1e-2 * np.trace(X.T @ X) / X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    errs = []
    for si in TEST_SH:
        for h in HELD_OUT_HUES:
            hi = int(np.where(H == h)[0][0])
            p = Z[si, hi] @ W
            ph = np.degrees(np.arctan2(p[1], p[0])) % 360
            errs.append(min(abs(ph - h), 360 - abs(ph - h)))
    return float(np.median(errs)), float(np.mean(errs))


def mlp_readout(Z, seed=0, steps=3000):
    """same zero-shot protocol as readout(), but with a small MLP probe instead of a ridge, so that
    the trained code head (an MLP) is compared against controls of equal probe capacity"""
    torch.manual_seed(seed)
    X, Y = [], []
    for si in TRAIN_SH:
        for hi, h in enumerate(H):
            if h in HELD_OUT_HUES:
                continue
            X.append(Z[si, hi]); Y.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    X = torch.tensor(np.stack(X), dtype=torch.float32); Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    d = X.shape[1]
    net = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 2)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xd, Yd = X.to(DEV), Y.to(DEV)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(Xd) - Yd) ** 2).sum(1).mean()
        loss.backward(); opt.step()
    net.eval()
    errs = []
    with torch.no_grad():
        for si in TEST_SH:
            for h in HELD_OUT_HUES:
                hi = int(np.where(H == h)[0][0])
                p = net(torch.tensor(Z[si, hi], dtype=torch.float32, device=DEV)[None])[0].cpu().numpy()
                ph = np.degrees(np.arctan2(p[1], p[0])) % 360
                errs.append(min(abs(ph - h), 360 - abs(ph - h)))
    return float(np.median(errs)), float(np.mean(errs))


def main():
    d = np.load(f"{WORK}/data/dense5.npz")
    imgs, shapes, hues = d["img"], d["shape"], d["hue"]
    Zpix_hue, Zpix_rgb = [], []
    for k in range(len(imgs)):
        a = imgs[k].astype(np.float32) / 255.0
        # object = non-white pixels
        m = (a.min(2) < 0.97)
        if m.sum() < 50:
            m = np.ones(a.shape[:2], bool)
        px = a[m]
        mx = px.max(1); mn = px.min(1); dd = mx - mn
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        hh = np.where(r == mx, (g - b) / np.maximum(dd, 1e-6),
                      np.where(g == mx, 2 + (b - r) / np.maximum(dd, 1e-6), 4 + (r - g) / np.maximum(dd, 1e-6)))
        hh = (hh % 6) * 60
        s = np.where(mx > 1e-6, dd / np.maximum(mx, 1e-6), 0)
        w = s + 1e-9; w = w / w.sum()
        c = float((w * np.cos(np.deg2rad(hh))).sum()); sn = float((w * np.sin(np.deg2rad(hh))).sum())
        Zpix_hue.append([c, sn])
        Zpix_rgb.append(px.mean(0))
    shape_idx = {s: i for i, s in enumerate(SH)}

    def pack(rows):
        Z = np.zeros((6, 72, len(rows[0])))
        for k in range(len(rows)):
            Z[shape_idx[shapes[k].item() if hasattr(shapes[k], 'item') else str(shapes[k])],
              int(np.where(H == hues[k])[0][0])] = rows[k]
        return Z

    # pose-average by construction: dense5 order is shape -> hue -> pose
    def poseavg(rows):
        A = np.stack(rows).reshape(6, 72, NV, -1).mean(2)
        return A

    arms = {}
    arms["pixels_hue"] = poseavg(Zpix_hue)
    arms["pixels_rgb"] = poseavg(Zpix_rgb)

    trained = YOLO(rp.YOLO_WEIGHTS).model.to(DEV).eval()
    rand = YOLO(rp.YOLO_WEIGHTS).model.to(DEV).eval()
    for m in rand.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()

    for tag, model in [("trained", trained), ("random", rand)]:
        F = feats_of(model, imgs)
        for idx, name in [(3, "y3"), (5, "y5"), (7, "y7")]:
            arms[f"{tag}_{name}"] = poseavg(list(F[idx]))

    out = {"protocol": ("dense5 (6 shapes x 72 hues x 4 poses, 224px); pose-averaged GAP features with "
                        "per-shape DC removal; k1+k2 share of centred hue-orbit energy; cross-shape phase "
                        "alignment = mean pairwise cosine of unit-normalised per-frequency responses in the "
                        "k=1..3 band; readout = ridge probe (lambda = 1e-2 * trace/n) fitted on the four "
                        "training shapes excluding hues {20,260}, evaluated zero-shot on the two unseen "
                        "shapes at those held-out hues."),
           "held_out_hues": HELD_OUT_HUES, "unseen_shapes": [SH[i] for i in TEST_SH], "arms": {}}
    for name, Z in arms.items():
        shares, cos = harmonics(Z)
        med, mean = readout(Z, Z.shape[2])
        shares_w, _ = harmonics(standardise(Z))
        mlp_med, mlp_mean = mlp_readout(Z)
        out["arms"][name] = {"k1k2_share_mean": round(float(np.mean(shares)), 4),
                             "k1k2_share_mean_standardised": round(float(np.mean(shares_w)), 4),
                             "k1k2_share_range": [round(float(np.min(shares)), 4), round(float(np.max(shares)), 4)],
                             "cross_shape_phase_cos_mean": round(float(np.mean(cos)), 4),
                             "readout_median_deg": round(med, 3), "readout_mean_deg": round(mean, 3),
                             "mlp_readout_median_deg": round(mlp_med, 3), "mlp_readout_mean_deg": round(mlp_mean, 3),
                             "dim": int(Z.shape[2])}
        print(f"{name:12s} k1+k2={out['arms'][name]['k1k2_share_mean']:.3f} "
              f"phase_cos={out['arms'][name]['cross_shape_phase_cos_mean']:.3f} "
              f"ridge={med:.2f} mlp={mlp_med:.2f} deg (dim {Z.shape[2]})", flush=True)

    json.dump(out, open(f"{WORK}/results/harmonic_controls.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(out["arms"].keys())
    ks = [out["arms"][n]["k1k2_share_mean"] for n in names]
    ksw = [out["arms"][n]["k1k2_share_mean_standardised"] for n in names]
    ps = [out["arms"][n]["cross_shape_phase_cos_mean"] for n in names]
    rd = [out["arms"][n]["readout_median_deg"] for n in names]
    rdm = [out["arms"][n]["mlp_readout_median_deg"] for n in names]
    cols = ["#7f8c8d" if n.startswith("pixels") else ("#2980b9" if n.startswith("random") else "#c0392b") for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, vals, title, ylab, logy in [(ax[0], ks, "(A) harmonic concentration", "k1+k2 share", False),
                                       (ax[1], ps, "(B) cross-shape phase sharing", "mean phase cosine (k=1..3)", False),
                                       (ax[2], rd, "(C) cross-shape zero-shot hue readout", "median error (deg)", True)]:
        a.bar(range(len(names)), vals, color=cols, alpha=0.9)
        if title.startswith("(A)"):
            a.plot(range(len(names)), ksw, "k^--", ms=5, lw=1.2, label="after variance normalisation")
            a.legend(fontsize=7.5)
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        a.set_title(title, fontsize=10); a.set_ylabel(ylab)
        if logy:
            a.set_yscale("log")
            a.plot(range(len(names)), rdm, "k^--", ms=5, lw=1.2, label="MLP probe (equal capacity)")
            a.legend(fontsize=7.5)
        a.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(f"{WORK}/results/figures/F24_learned_or_trivial.png", dpi=150)
    print("figure F24_learned_or_trivial.png written")


main()
