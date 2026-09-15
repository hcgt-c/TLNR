# -*- coding: utf-8 -*-
"""Route B: unified CIFAR-10 color-robustness benchmark (fair, same-budget).

Arms (same epochs/batch/optimizer/seed; models from the two official repos):
  z2     : plain ResNet-44 (CEConv-main, rotations=1)              -> CNN baseline
  z2hue  : same + train-time hue jitter (+-90 deg)                 -> augmentation baseline
  z2heat : same + train-time heat blur (sigma ~ U(0, blur_sigma))  -> dissipation-insensitivity control
  ce8    : CEConv ResNet-44 rotations=8 (CEConv-main)              -> CEConv (filter-space hue rotation)
  lcer8  : LCER ResNet-44 G_h=8 (GroupConvHS, input lifted x8)     -> LCER (input-side lifting)
  ocode  : z2 + harmonic color code (l,shat,z1,z2) at stage-1, trained by a label-free
           rho-equivariance pair loss (our method; no per-image color GT)

Protocol: train hue jitter = 0 except z2hue (+-90deg); RandomCrop4 + HFlip; CIFAR-stats norm.
Eval: standard test accuracy + hue-shift sweep over factors in [-0.5, 0.5] (= -180..180 deg).

Usage:
  python scripts/c10_routeB_train.py --arm z2 --seed 0 --epochs 100 [--sanity]
Writes results/c10_routeB/<arm>_s<seed>.json (+ .pt + train log csv).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = rp.REPO_ROOT
LCER = os.path.join(WORK, "external/Learning-Color-Equivariant-Representations-main")
CECV = os.path.join(WORK, "external/CEConv-main")
sys.path.insert(0, LCER)
sys.path.insert(0, CECV)

OUT = os.path.join(WORK, "results", "c10_routeB")
os.makedirs(OUT, exist_ok=True)
CIFAR_ROOT = rp.CIFAR10_ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010], device=DEV).view(1, 3, 1, 1)
MEAN8 = MEAN.repeat(1, 8, 1, 1)
STD8 = STD.repeat(1, 8, 1, 1)


# ---------------- data ----------------
def load_cifar():
    cache = os.path.join(WORK, "data", "cifar10_raw.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_tr"], d["y_tr"], d["x_te"], d["y_te"]
    import torchvision
    tr = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=False)
    te = torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=False)
    x_tr, y_tr = np.asarray(tr.data), np.asarray(tr.targets)
    x_te, y_te = np.asarray(te.data), np.asarray(te.targets)
    os.makedirs(os.path.join(WORK, "data"), exist_ok=True)
    np.savez(cache, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)
    return x_tr, y_tr, x_te, y_te


def to_tensor(batch_u8):
    return torch.from_numpy(batch_u8).float().permute(0, 3, 1, 2).to(DEV) / 255.0


def rgb_hue_shift_t(x, deg):
    """Exact HSV-H shift by per-sample deg (degrees). x in [0,1], (B,3,H,W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    mx = torch.maximum(r, torch.maximum(g, b))
    mn = torch.minimum(r, torch.minimum(g, b))
    d = mx - mn
    eps = 1e-8
    h = torch.zeros_like(d)
    m1 = (mx == r) & (d > eps)
    m2 = (mx == g) & (d > eps)
    m3 = (mx == b) & (d > eps)
    h[m1] = ((g[m1] - b[m1]) / d[m1]) % 6.0
    h[m2] = ((b[m2] - r[m2]) / d[m2]) + 2.0
    h[m3] = ((r[m3] - g[m3]) / d[m3]) + 4.0
    h = (h * 60.0) % 360.0
    hn = (h + deg.view(-1, 1, 1, 1)) % 360.0
    l = (mx + mn) / 2.0
    s = torch.zeros_like(d)
    nz = d > eps
    s[nz] = torch.where(l[nz] < 0.5, d[nz] / (mx[nz] + mn[nz] + 1e-8),
                        d[nz] / (2.0 - mx[nz] - mn[nz] + 1e-8))
    c = (1.0 - torch.abs(2.0 * l - 1.0)) * s
    hp = hn / 60.0
    xv = c * (1.0 - torch.abs(hp % 2.0 - 1.0))
    m = l - c / 2.0
    k = hp.long() % 6
    z = torch.zeros_like(c)
    R = torch.where(k == 0, c, torch.where(k == 1, xv, torch.where(k == 2, z, torch.where(k == 3, z, torch.where(k == 4, xv, c)))))
    G = torch.where(k == 0, xv, torch.where(k == 1, c, torch.where(k == 2, c, torch.where(k == 3, xv, z))))
    B = torch.where(k == 0, z, torch.where(k == 1, z, torch.where(k == 2, xv, torch.where(k == 3, c, c))))
    return torch.cat([R + m, G + m, B + m], 1).clamp(0, 1)


def lift_hue(x, G=8):
    B = x.shape[0]
    outs = []
    for k in range(G):
        outs.append(rgb_hue_shift_t(x, torch.full((B,), float(k * 360.0 / G), device=DEV)))
    return torch.cat(outs, 1)  # (B, G*3, H, W)


_HEAT_EIGS = {}


def heat_blur_batch(x, t):
    """Exact discrete heat semigroup B_t = exp(-t L) applied to a batch (one t for the whole batch).

    Spectral (DCT/Neumann) form, so that the augmentation uses exactly the same operator as the intervention
    protocol: a truncated Gaussian kernel is not a semigroup (see scripts/187).
    """
    from scipy.fft import dctn, idctn
    H, Wd = x.shape[-2:]
    if (H, Wd) not in _HEAT_EIGS:
        la = 2 - 2 * np.cos(np.pi * np.arange(H) / H)
        lb = 2 - 2 * np.cos(np.pi * np.arange(Wd) / Wd)
        _HEAT_EIGS[(H, Wd)] = (la[:, None] + lb[None, :]).astype(np.float32)
    e = _HEAT_EIGS[(H, Wd)]
    a = dctn(x.detach().cpu().numpy().astype(np.float32), type=2, norm="ortho", axes=(-2, -1))
    a = a * np.exp(-t * e)[None, None]
    return torch.from_numpy(idctn(a, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)).to(x.device)


def train_aug(x, jitter_deg=0.0, blur_sigma=0.0):
    """RandomCrop(4)+HFlip; batch-uniform crop offset, per-sample flip & jitter; optional heat blur.

    The blur arm samples one sigma per batch from U(0, blur_sigma) and applies the exact heat semigroup, so
    the training distribution contains the dissipative transformations the intervention test uses.
    """
    B = x.shape[0]
    if blur_sigma > 0:
        s_ = float(torch.rand(1).item()) * blur_sigma
        x = heat_blur_batch(x, s_ ** 2 / 2)
    if jitter_deg > 0:
        f = (torch.rand(B, device=DEV) * 2 - 1) * jitter_deg
        x = rgb_hue_shift_t(x, f)
    xp = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (1,), device=DEV).item()
    dx = torch.randint(0, 9, (1,), device=DEV).item()
    x = xp[:, :, dy:dy + 32, dx:dx + 32].contiguous()
    flip = torch.rand(B, device=DEV) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    return x


# ---------------- models ----------------
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_arm(arm):
    if arm in ("z2", "z2hue", "z2heat"):
        from models.resnet import ResNet44
        return ResNet44(rotations=1, num_classes=10), "rgb"
    if arm == "ce8":
        from models.resnet import ResNet44
        return ResNet44(rotations=8, num_classes=10), "rgb"
    if arm == "lcer8":
        from networks.resnet import ResNet44 as LR
        return LR(num_classes=10, n_groups_hue=8, n_groups_saturation=1, ours=True), "lift8"
    if arm == "ocode":
        from models.resnet import ResNet44
        return OCodeNet(ResNet44(rotations=1, num_classes=10)), "rgb"
    raise ValueError(arm)


class OCodeNet(nn.Module):
    """z2 ResNet-44 + harmonic color code (l, shat, z1, z2) read from stage-1 GAP.
    Trained by label-free rho-equivariance on (x, T_d x) pairs; classification head unchanged.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(
            nn.Linear(32, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 6))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def _stem(self, x):
        return F.relu(self.base.bn1(self.base.conv1(x)))

    def code_of(self, x):
        s1 = self.base.layers[0](self._stem(x))
        g = self.avg(s1).flatten(1)
        o = self.head(g)
        l = F.softplus(o[:, 0])
        sh = torch.sigmoid(o[:, 1])
        return l, sh, o[:, 2:4], o[:, 4:6]

    def cls_of(self, x):
        out = self._stem(x)
        out = self.base.layers[0](out)
        out = self.base.layers[1](out)
        out = self.base.layers[2](out)
        out = self.avg(out).flatten(1)
        return self.base.linear(out)

    def forward(self, x):
        return self.cls_of(x)


def rot_z(z, deg):
    th = torch.deg2rad(deg)
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([z[:, 0] * c - z[:, 1] * s, z[:, 0] * s + z[:, 1] * c], 1)


# ---------------- train / eval ----------------
def run(arm, seed, epochs, jitter_deg, sanity, eval_factors, blur_sigma=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, y_tr, x_te, y_te = load_cifar()
    n_tr = x_tr.shape[0]
    y_tr_t = torch.from_numpy(y_tr).to(DEV)
    y_te_t = torch.from_numpy(y_te).to(DEV)

    net, mode = make_arm(arm)
    net.to(DEV)
    n_params = count_params(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    steps_per_epoch = math.ceil(n_tr / 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    lam, lam_sl, lam_a = 0.5, 0.3, 0.05

    log = []
    t_start = time.time()
    global_step = 0
    idx_all = np.arange(n_tr)
    for ep in range(epochs):
        net.train()
        np.random.shuffle(idx_all)
        ep_loss = ep_ok = ep_n = 0.0
        for s0 in range(0, n_tr, 128):
            ids = idx_all[s0:s0 + 128]
            xb = train_aug(to_tensor(x_tr[ids]), jitter_deg, blur_sigma)
            yb = y_tr_t[ids]
            if mode == "rgb":
                xb = (xb - MEAN) / STD
            elif mode == "lift8":
                xb = lift_hue(xb, 8)
                xb = (xb - MEAN8) / STD8
            opt.zero_grad()
            if arm == "ocode":
                # pair loss: same aug applied to a hue-shifted twin
                ids2 = np.random.choice(ids, len(ids), replace=False)
                xraw = to_tensor(x_tr[ids2])
                dg = (torch.randint(1, 12, (xraw.shape[0],), device=DEV).float() * 15.0)
                xp = rgb_hue_shift_t(train_aug(xraw, 0.0), dg)
                xp = (xp - MEAN) / STD
                logits = net.cls_of(xb)
                l1, sh1, z1a, z2a = net.code_of(xb)
                l2, sh2, z1b, z2b = net.code_of(xp)
                dgz = dg % 360.0
                loss_z = ((z1b - rot_z(z1a, dgz)) ** 2).sum(1).mean() + ((z2b - rot_z(z2a, 2 * dgz)) ** 2).sum(1).mean()
                loss_sl = (((l1 - l2) ** 2) + ((sh1 - sh2) ** 2)).mean()
                anchor = ((torch.abs(z1a).mean(1) - 1.0).clamp(min=0).pow(2)).mean()
                loss = lossf(logits, yb) + lam * (loss_z + lam_sl * loss_sl + lam_a * anchor)
            else:
                loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                ok = (net(xb).argmax(1) == yb).sum().item() if arm != "ocode" else (logits.argmax(1) == yb).sum().item()
            ep_loss += loss.item() * xb.shape[0]
            ep_ok += ok
            ep_n += xb.shape[0]
            global_step += 1
            if sanity and global_step >= 120:
                break
        if ep_n > 0:
            log.append([ep, ep_loss / ep_n, ep_ok / ep_n, time.time() - t_start])
            print(f"[{arm}/s{seed}] ep {ep} loss {ep_loss/ep_n:.3f} acc {ep_ok/ep_n:.3f} wall {time.time()-t_start:.0f}s", flush=True)
        if sanity:
            break

    # ---- eval ----
    net.eval()
    te_t = to_tensor(x_te)
    results = {}
    with torch.no_grad():
        for f in eval_factors:
            xs = rgb_hue_shift_t(te_t, torch.full((te_t.shape[0],), float(f * 360.0), device=DEV))
            if mode == "rgb":
                xs = (xs - MEAN) / STD
            else:
                xs = lift_hue(xs, 8)
                xs = (xs - MEAN8) / STD8
            accs = 0
            for s0 in range(0, te_t.shape[0], 256):
                logits = net(xs[s0:s0 + 256])
                accs += (logits.argmax(1) == y_te_t[s0:s0 + 256]).sum().item()
            results[f] = accs / te_t.shape[0]
    clean = results[0.0]
    out = {
        "arm": arm, "seed": seed, "epochs_run": len(log), "n_params": n_params,
        "clean_acc": clean,
        "sweep": {f"{f:.4f}": results[f] for f in eval_factors},
        "mean_sweep_acc": float(np.mean(list(results.values()))),
        "train_log": log, "wall_seconds": time.time() - t_start,
        "protocol": ("CIFAR-10 50k train / 10k test; jitter +-90deg for z2hue only; "
                     "RandomCrop4+HFlip; Adam 1e-3 wd 1e-5 OneCycle(15%); batch 128; sweep=37 factors in [-0.5,0.5]"),
    }
    fn = os.path.join(OUT, f"{arm}_s{seed}.json")
    json.dump(out, open(fn, "w"), indent=1)
    torch.save(net.state_dict(), os.path.join(OUT, f"{arm}_s{seed}.pt"))
    print("saved", fn, "| clean", round(clean, 4), "| mean_sweep", round(out["mean_sweep_acc"], 4), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--blur_sigma", type=float, default=0.0,
                    help="max heat-blur sigma for train-time augmentation (arm z2heat defaults to 2.0)")
    ap.add_argument("--jitter", type=float, default=None,
                    help="train hue jitter in +-degrees for any arm (default: 90 for z2hue, else 0)")
    ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args()
    if a.jitter is None:
        jitter = 90.0 if a.arm == "z2hue" else 0.0
        blur = a.blur_sigma if a.blur_sigma > 0 else (2.0 if a.arm == "z2heat" else 0.0)
    else:
        jitter = a.jitter
    if a.sanity:
        run(a.arm, a.seed, 2, jitter, True, [-0.5, -0.25, 0.0, 0.25, 0.5])
    else:
        evalf = [round(i, 4) for i in np.linspace(-0.5, 0.5, 37)]
        run(a.arm, a.seed, a.epochs, jitter, False, evalf, blur)


if __name__ == "__main__":
    main()
