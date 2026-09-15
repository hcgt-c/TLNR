# -*- coding: utf-8 -*-
"""Code hue-read domain map: evaluate code3d_v3 zero-shot hue read error over the full
(s,v) grid of dense3d_lattice on the UNSEEN shapes (pentagon, hexagon).
Mirrors the pool/feat_std of scripts/79 so the loaded head operates in its trained scale.
Outputs results/code3d_sv_domain.json  (per test shape x 4s x 4v mean circular error).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = rp.REPO_ROOT
sys.path.insert(0, WORK + "/scripts")

H5 = np.arange(0, 360, 5)


def load_pool_train_feats():
    feats = []
    def add(p, arr, shape):
        feats.append((p, arr))
    d = np.load(os.path.join(WORK, "features/dense5_y3.npz"))
    F = d["feats"].reshape(6, 72, 4, 64).mean(2)
    for si in range(4):
        for hi in range(72):
            feats.append(F[si, hi])
    d = np.load(os.path.join(WORK, "features/sheet_hv_y3.npz"))
    F = d["feats"].reshape(10, 6, 72, 64)
    for vi in range(10):
        for si in range(4):
            for hi in range(72):
                feats.append(F[vi, si, hi])
    d = np.load(os.path.join(WORK, "features/dense3d_rings_y3.npz"))
    F = d["feats"].reshape(6, 5, 36, 4, 64).mean(3)
    for si in range(4):
        for svi in range(5):
            for k in range(36):
                feats.append(F[si, svi, k])
    d = np.load(os.path.join(WORK, "features/dense3d_lattice_y3.npz"))
    F = d["feats"].reshape(6, 12, 4, 4, 4, 64).mean(4)
    for si in range(4):
        for i in range(12):
            for a in range(4):
                for b in range(4):
                    feats.append(F[si, i, a, b])
    d = np.load(os.path.join(WORK, "features/sheet_sv_y3.npz"))
    F = d["feats"].reshape(3, 6, 10, 10, 64)
    for hi in range(3):
        for si in range(4):
            for a in range(10):
                for b in range(10):
                    feats.append(F[hi, si, a, b])
    A = np.stack(feats)
    return float(np.std(A))


class Head6(nn.Module):
    def __init__(self, dh=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, dh), nn.SiLU(), nn.Linear(dh, dh), nn.SiLU(), nn.Linear(dh, 6))

    def forward(self, x):
        o = self.net(x)
        l = F.softplus(o[:, 0:1]); s = torch.sigmoid(o[:, 1:2])
        return torch.cat([l, s, o[:, 2:6]], 1)


def circ(a, b):
    return min(abs(a - b), 360 - abs(a - b))


def main():
    print("compute feat_std...", flush=True)
    feat_std = load_pool_train_feats()
    head = Head6().to(DEV)
    head.load_state_dict(torch.load(os.path.join(WORK, "results/code3d_v3.pt"), map_location=DEV))
    head.eval()

    d = np.load(os.path.join(WORK, "features/dense3d_lattice_y3.npz"))
    F = d["feats"].reshape(6, 12, 4, 4, 4, 64).mean(4)   # shape,12h,4s,4v,64
    SV = [0.25, 0.5, 0.75, 1.0]
    res = {}
    with torch.no_grad():
        for si in [4, 5]:  # pentagon, hexagon (unseen)
            for a, s in enumerate(SV):
                for b, v in enumerate(SV):
                    X = F[si, :, a, b] / feat_std
                    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
                    c = head(Xt).cpu().numpy()
                    hues = (np.degrees(np.arctan2(c[:, 3], c[:, 2])) % 360)
                    gt = np.arange(0, 360, 30)
                    errs = [circ(hues[i], gt[i]) for i in range(12)]
                    res.setdefault(f"s{s:.2f}_v{v:.2f}", {}).setdefault(["pentagon", "hexagon"][si - 4], round(float(np.mean(errs)), 2))
    # also pool overall per test shape across s,v and overall
    per_sv = {k: {"pentagon": v["pentagon"], "hexagon": v["hexagon"]} for k, v in res.items()}
    mean_pe = np.mean([v["pentagon"] for v in res.values()])
    mean_he = np.mean([v["hexagon"] for v in res.values()])
    out = {
        "protocol": "code3d_v3 zero-shot on dense3d_lattice test shapes (pentagon/hexagon), "
                    "12 hues x 4 poses avg, y3 GAP / feat_std; circular hue error (deg) vs (s,v)",
        "feat_std": feat_std,
        "grid_s": SV, "grid_v": SV,
        "per_cell": res,
        "mean_err_pentagon_deg": round(float(mean_pe), 2),
        "mean_err_hexagon_deg": round(float(mean_he), 2),
    }
    fn = os.path.join(WORK, "results/code3d_sv_domain.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)
    # compact print
    for k, v in res.items():
        print(k, v)
    print("mean pentagon %.2f hexagon %.2f" % (mean_pe, mean_he))


if __name__ == "__main__":
    main()
