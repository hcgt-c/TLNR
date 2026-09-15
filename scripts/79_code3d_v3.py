# -*- coding: utf-8 -*-
"""完整 3D 颜色码 v3：(l→v, ŝ→s 独立通道, z1=e^{iθ}, z2=e^{i2θ})。
在合并 (h,s,v) y3 特征池训练（同 71 采样，l/s 全权、z 相位按 m* 降权）。
评测（未见形状）：hue 相位(高/低色度)、l~v、ŝ~s 相关/二分类/解耦、ρ(Δ) 不变性、3D 逐点读出。
"""
import numpy as np
import numpy.linalg as la
import torch, torch.nn as nn, torch.nn.functional as F
import json, math
from collections import defaultdict
import random

torch.manual_seed(0); np.random.seed(0); random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SH = [0, 1, 2, 3]; TEST_SH = [4, 5]
H5 = np.arange(0, 360, 5)

def pool():
    p = []
    d = np.load("features/dense5_y3.npz")
    F = d["feats"].reshape(6, 72, 4, 64).mean(2)
    for si in range(6):
        for hi, h in enumerate(H5):
            p.append((si, h, 1.0, 1.0, F[si, hi]))
    d = np.load("features/sheet_hv_y3.npz")
    F = d["feats"].reshape(10, 6, 72, 64)
    v10 = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    for vi in range(10):
        for si in range(6):
            for hi, h in enumerate(H5):
                p.append((si, h, 1.0, float(v10[vi]), F[vi, si, hi]))
    d = np.load("features/dense3d_rings_y3.npz")
    F = d["feats"].reshape(6, 5, 36, 4, 64).mean(3)
    SV = [(0.3, 1.0), (0.6, 1.0), (1.0, 0.6), (0.6, 0.6), (0.35, 0.35)]
    for si in range(6):
        for svi, (s, v) in enumerate(SV):
            for k, h in enumerate(range(0, 360, 10)):
                p.append((si, h, s, v, F[si, svi, k]))
    d = np.load("features/dense3d_lattice_y3.npz")
    F = d["feats"].reshape(6, 12, 4, 4, 4, 64).mean(4)
    SV2 = [0.25, 0.5, 0.75, 1.0]
    for si in range(6):
        for i, h in enumerate(range(0, 360, 30)):
            for a, s in enumerate(SV2):
                for b, v in enumerate(SV2):
                    p.append((si, h, s, v, F[si, i, a, b]))
    d = np.load("features/sheet_sv_y3.npz")
    F = d["feats"].reshape(3, 6, 10, 10, 64)
    sv10 = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    for hi, h in enumerate([0, 120, 240]):
        for si in range(6):
            for a, s in enumerate(sv10):
                for b, v in enumerate(sv10):
                    p.append((si, h, float(s), float(v), F[hi, si, a, b]))
    return p

print("pool...", flush=True)
POOL = pool()
tr_samp = [p for p in POOL if p[0] in TRAIN_SH]
feat_std = float(np.std([p[4] for p in tr_samp]))
cell = defaultdict(list)
for si, h, s, v, f in tr_samp:
    cell[(si, round(s, 2), round(v, 2))].append((h, f))
CELLS = list(cell.keys())
# m* 表
mraw = {}
for key, items in cell.items():
    Z = np.stack([it[1] for it in items]); Zc = Z - Z.mean(0, keepdims=True)
    mraw[(key[1], key[2])] = float(np.sqrt(np.mean(Zc ** 2)))
ref = mraw.get((1.0, 1.0), max(mraw.values()))
MST = np.array(sorted(mraw.keys()))
mstar = {k: v / ref for k, v in mraw.items()}
def m_star(s, v):
    k = np.array([round(s, 2), round(v, 2)])
    return mstar[tuple(MST[np.argmin(np.sum((MST - k) ** 2, axis=1))])]
cellfeat = {}
for key, items in cell.items():
    hm = {}
    for h, f in items:
        hm.setdefault(int(h), f)
    cellfeat[key] = hm

class Head6(nn.Module):
    def __init__(self, dh=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, dh), nn.SiLU(), nn.Linear(dh, dh), nn.SiLU(), nn.Linear(dh, 6))
    def forward(self, x):
        o = self.net(x)
        l = F.softplus(o[:, 0:1]); s = torch.sigmoid(o[:, 1:2])
        return torch.cat([l, s, o[:, 2:6]], 1)   # l, ŝ, z1x,z1y,z2x,z2y

model = Head6().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

def batch(bs=96):
    Xs, Ys, Ws = [], [], []
    keys = list(cellfeat.keys())
    for _ in range(bs):
        si, s, v = random.choice(keys)
        hm = cellfeat[(si, s, v)]
        hs = sorted(hm.keys())
        if len(hs) < 2:
            continue
        m = m_star(s, v)
        h = random.choice(hs)
        Xs.append(hm[h] / feat_std)
        th = math.radians(h)
        Ys.append([v, s, m * math.cos(th), m * math.sin(th), m * math.cos(2 * th), m * math.sin(2 * th)])
        Ws.append(1.0)
        h2 = random.choice(hs)
        Xs.append(hm[h2] / feat_std)
        th2 = math.radians(h2)
        Ys.append([v, s, m * math.cos(th2), m * math.sin(th2), m * math.cos(2 * th2), m * math.sin(2 * th2)])
        Ws.append(1.0)
    return (np.array(Xs), np.array(Ys), np.array(Ws)) if len(Xs) else None

for ep in range(1500):
    el = 0.0; ntot = 0
    for _ in range(24):
        b_ = batch()
        if b_ is None:
            continue
        X, Y, W = b_
        x = torch.tensor(X, dtype=torch.float32, device=DEV)
        y = torch.tensor(Y, dtype=torch.float32, device=DEV)
        pred = model(x)
        loss = torch.mean((pred[:, 0] - y[:, 0]) ** 2) + torch.mean((pred[:, 1] - y[:, 1]) ** 2)
        zt = y[:, 2:]; zp = pred[:, 2:]
        w = torch.clamp(torch.tensor(np.array([m_star(s, v) for si, h, s, v in [(None,)*4]]), dtype=torch.float32, device=DEV), 0.05, 1.0) if False else None
        loss += torch.mean((zp - zt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        el += loss.item() * len(X); ntot += len(X)
    if ep % 500 == 0:
        print("ep %d loss %.5f" % (ep, el / ntot), flush=True)

# ---------- 评测（未见形状）----------
def circ(a, b): return min(abs(a - b), 360 - abs(a - b))
test_cells = defaultdict(list)
for si, h, s, v, f in POOL:
    if si in TEST_SH:
        test_cells[(round(s, 2), round(v, 2))].append((h, s, v, f))
res = {}
model.eval()
err_hi, err_lo, ls, ss, vs_, lv, ls_ = [], [], [], [], [], [], []
s_bin, s_raw = [], []
raw_X = []
with torch.no_grad():
    for (s, v), items in test_cells.items():
        for h, s0, v0, f in items:
            c = model(torch.tensor(f / feat_std, dtype=torch.float32, device=DEV)[None])[0].cpu().numpy()
            a1 = np.degrees(np.arctan2(c[3], c[2])) % 360
            e = circ(a1, h)
            mref = m_star(s, v)
            if mref > 0.4:
                err_hi.append(e)
            else:
                err_lo.append(e)
            ls.append(c[0]); ss.append(c[1]); vs_.append(v)
            lv.append(c[0]); ls_.append(s0)
            raw_X.append(f)
            s_bin.append(1.0 if (c[1] >= 0.55) == (s0 >= 0.55) else 0.0)
res["hue_err_hi"] = round(float(np.mean(err_hi)), 2)
res["hue_err_lo"] = round(float(np.mean(err_lo)), 2)
res["l_corr_v"] = round(float(np.corrcoef(ls, vs_)[0, 1]), 3)
res["shat_corr_s"] = round(float(np.corrcoef(ss, ls_)[0, 1]), 3)
res["shat_bin_acc"] = round(float(np.mean(s_bin)), 4)
res["shat_corr_v_at_fixed_s"] = None
# 解耦：固定 s 下 ŝ 与 v 的相关（低=解耦好）；固定 v 下 l 与 s 的相关
sub = {}
corr_sv = []
with torch.no_grad():
    for sval in [0.3, 0.6, 1.0]:
        vals = []
        for (s, v), items in test_cells.items():
            if abs(s - sval) > 1e-6:
                continue
            for h, s0, v0, f in items:
                c = model(torch.tensor(f / feat_std, dtype=torch.float32, device=DEV)[None])[0].cpu().numpy()
                vals.append((c[1], v))
        if len(vals) > 4:
            corr_sv.append(np.corrcoef([a for a, _ in vals], [b for _, b in vals])[0, 1])
res["shat_corr_v_fixed_s"] = round(float(np.mean(corr_sv)), 3) if corr_sv else None
# raw 二分类 s 对照（线性探针）
Xs = np.array(raw_X); ys = np.array([1.0 if it[2] >= 0.55 else 0.0 for (s, v), items in test_cells.items() for it in items])
Wr = la.solve(Xs.T @ Xs + 1e-2 * np.eye(Xs.shape[1]), Xs.T @ (2 * ys - 1))
rawacc = np.mean([1.0 if ((f @ Wr >= 0) == (y == 1.0)) else 0.0 for f, y in zip(Xs, ys)])
res["raw_s_bin_acc_ref"] = round(float(rawacc), 4)
# ρ(Δ) 不变性：色相旋转不改 ŝ/l
d5 = np.load("features/dense5_y3.npz")
F5 = d5["feats"].reshape(6, 72, 4, 64).mean(2)
dlt_sl, dlt_ph = [], []
with torch.no_grad():
    for si in TEST_SH:
        cs = model(torch.tensor(F5[si] / feat_std, dtype=torch.float32, device=DEV)).cpu().numpy()
        for h in H5[::6]:
            hi = H5.tolist().index(h)
            if h + 40 in H5:
                hi2 = H5.tolist().index(h + 40)
                a1 = np.degrees(np.arctan2(cs[hi, 3], cs[hi, 2])) % 360
                a2 = np.degrees(np.arctan2(cs[hi2, 3], cs[hi2, 2])) % 360
                dlt_ph.append(circ((a2 - a1) % 360, 40))
                dlt_sl.append(abs(cs[hi, 0] - cs[hi2, 0]) + abs(cs[hi, 1] - cs[hi2, 1]))
res["rho_d40_deg"] = round(float(np.mean(dlt_ph)), 2)
res["rho_sl_drift"] = round(float(np.mean(dlt_sl)), 4)
json.dump(res, open("results/code3d_v3.json", "w"), indent=1)
print(json.dumps(res, indent=1))
torch.save(model.state_dict(), "results/code3d_v3.pt")
print("saved results/code3d_v3.json + .pt")
