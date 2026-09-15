# -*- coding: utf-8 -*-
"""A. ResNet50 泛化探针：dense5 上各 stage GAP 的谱/球壳 + 中层码头 hue 读出（跨形状零样本）。
对照既有 ResNet18 l2 / YOLO y3 数字（dense_continuous_v3/l2_consistency）。"""
import numpy as np, torch, os, json, math, time
import torchvision.models as M
from collections import defaultdict

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(OUT, "data", "dense5.npz"))
img, shape, hue = d["img"], d["shape"], d["hue"]
SH = ['triangle','rectangle','circle','star','pentagon','hexagon']
TR = [0,1,2,3]; TE = [4,5]
H = np.arange(0,360,5)

m = M.resnet50(); m.load_state_dict(torch.load("external/models/resnet50-0676ba61.pth", map_location="cpu"))
m.to(DEV).eval()
hooks = {}
for tag, layer in [("l1", m.layer1), ("l2", m.layer2), ("l3", m.layer3), ("l4", m.layer4)]:
    layer.register_forward_hook(lambda md,i,o,t=tag: hooks.__setitem__(t, o.mean(dim=(2,3)).detach()))
F = defaultdict(list)
t0=time.time()
with torch.no_grad():
    for i in range(len(img)):
        a = img[i].astype(np.float32)/np.float32(255.0)
        a = (a - np.array([0.485,0.456,0.406], np.float32))/np.array([0.229,0.224,0.225], np.float32)
        x = torch.tensor(a, device=DEV).permute(2,0,1).unsqueeze(0)
        m(x)
        for t in ["l1","l2","l3","l4"]:
            F[t].append(hooks[t][0].cpu().numpy())
print("extract %.0fs" % (time.time()-t0))
res = {"protocol": "dense5, resnet50 offline, stage GAP, 4 pose avg"}
for t in ["l1","l2","l3","l4"]:
    Z = np.stack(F[t]).reshape(6,72,4,-1).mean(2)
    kshare=[]; ns=[]; mw=[]
    for si in range(6):
        Zc = Z[si]-Z[si].mean(0)
        E = (np.abs(np.fft.rfft(Zc,axis=0))**2).sum(1); Et=E.sum()
        kshare.append((E[1]+E[2])/Et); mw.append(float(np.sum(np.arange(len(E))*E)/Et))
        n = np.linalg.norm(Z[si],axis=1); ns.append(n.std()/n.mean())
    res[t] = {"k1k2_share_mean": round(float(np.mean(kshare)),3),
              "mean_winding": round(float(np.mean(mw)),2),
              "norm_std_mean": round(float(np.mean(ns)),4)}
print(json.dumps(res, indent=1))
json.dump(res, open("results/resnet50_probe.json","w"), indent=1)

# 码头 hue 读出（l3 1024d 与 l2 512d，M1 式简版 400 步）
import torch.nn as nn, torch.nn.functional as Fn
def readout_test(dim, tag):
    Xs, Ys = [], []
    for si in TR:
        Zi = np.stack(F[tag]).reshape(6,72,4,-1).mean(2)[si]
        for hi,h in enumerate(H):
            if h in (20,260) or h==0: continue
            Xs.append(Zi[hi]); Ys.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    X = torch.tensor(np.array(Xs), dtype=torch.float32, device=DEV); Y = torch.tensor(np.array(Ys), dtype=torch.float32, device=DEV)
    net = nn.Sequential(nn.Linear(dim,128), nn.SiLU(), nn.Linear(128,2)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(400):
        p = net(X); loss = Fn.mse_loss(p, Y); opt.zero_grad(); loss.backward(); opt.step()
    errs=[]
    with torch.no_grad():
        for si in TE:
            Zi = np.stack(F[tag]).reshape(6,72,4,-1).mean(2)[si]
            for hi,h in enumerate(H):
                if h not in (20,260): continue
                y = net(torch.tensor(Zi[hi].astype(np.float32), device=DEV)).cpu().numpy()
                ang = np.degrees(np.arctan2(y[1],y[0]))%360
                errs.append(min(abs(ang-h),360-abs(ang-h)))
    return round(float(np.mean(errs)),2)
r50 = {}
for tag, dim in [("l2",512),("l3",1024)]:
    r50[tag] = readout_test(dim, tag)
print("R50 code-head hue err (cross-shape zero-shot {20,260}):", r50)
json.dump(r50, open("results/resnet50_readout.json","w"), indent=1)
