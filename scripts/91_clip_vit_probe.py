# -*- coding: utf-8 -*-
"""C. 预训练 ViT(CLIP ViT-B/32, TorchScript 离线) 泛化探针：dense5 上 CLS 特征的谱/球壳+码读出。
范围：仅 final CLS(512, encode_image 归一化)；非逐层。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import numpy as np, torch, json, os, math
from collections import defaultdict
import torch.nn as nn, torch.nn.functional as Fn

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(OUT, "data", "dense5.npz"))
img, shape, hue = d["img"], d["shape"], d["hue"]
SH = ['triangle','rectangle','circle','star','pentagon','hexagon']
TR=[0,1,2,3]; TE=[4,5]; H=np.arange(0,360,5)

clip = torch.jit.load(rp.CLIP_WEIGHTS, map_location="cpu").to(DEV).eval()
# 尝试 visual 塔直出未归一化特征
has_visual = hasattr(clip, "visual")
F = []
with torch.no_grad():
    for i in range(0, len(img), 64):
        a = img[i:i+64].astype(np.float32)/np.float32(255.0)
        a = (a - np.array([0.48145466,0.4578275,0.40821073], np.float32))/np.array([0.26862954,0.26130258,0.27577711], np.float32)
        x = torch.tensor(a, device=DEV).permute(0,3,1,2).half()
        if has_visual:
            f = clip.visual(x)
        else:
            f = clip.encode_image(x)
        F.append(f.cpu().numpy().astype(np.float32))
Zraw = np.concatenate(F)
print("feat shape", Zraw.shape, "norm mean", float(np.linalg.norm(Zraw,axis=1).mean()))
res = {"protocol": "CLIP ViT-B/32 offline TorchScript, final CLS, dense5, 4 pose avg, use_visual=%s" % has_visual}
Z = Zraw.reshape(6,72,4,-1).mean(2)
ks=[]; mw=[]; ns=[]
for si in range(6):
    Zc = Z[si]-Z[si].mean(0)
    E=(np.abs(np.fft.rfft(Zc,axis=0))**2).sum(1); Et=E.sum()
    ks.append((E[1]+E[2])/Et); mw.append(float(np.sum(np.arange(len(E))*E)/Et))
    n=np.linalg.norm(Z[si],axis=1); ns.append(n.std()/n.mean())
res.update({"k1k2_share_mean": round(float(np.mean(ks)),3),
            "per_shape": {SH[si]: round(float(ks[si]),3) for si in range(6)},
            "mean_winding": round(float(np.mean(mw)),2),
            "norm_std_mean": round(float(np.mean(ns)),4)})
print(json.dumps(res, indent=1))
json.dump(res, open("results/clip_vit_probe.json","w"), indent=1)
# 码读出（final CLS 上，跨形状零样本 {20,260}）
Xs,Ys=[],[]
for si in TR:
    Zi = Z[si]
    for hi,h in enumerate(H):
        if h in (20,260) or h==0: continue
        Xs.append(Zi[hi]); Ys.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
X=torch.tensor(np.array(Xs),dtype=torch.float32,device=DEV); Y=torch.tensor(np.array(Ys),dtype=torch.float32,device=DEV)
net=nn.Sequential(nn.Linear(Zraw.shape[1],128),nn.SiLU(),nn.Linear(128,2)).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
for _ in range(400):
    p=net(X); loss=Fn.mse_loss(p,Y); opt.zero_grad(); loss.backward(); opt.step()
errs=[]
with torch.no_grad():
    for si in TE:
        Zi=Z[si]
        for hi,h in enumerate(H):
            if h not in (20,260): continue
            y=net(torch.tensor(Zi[hi],dtype=torch.float32,device=DEV)).cpu().numpy()
            ang=np.degrees(np.arctan2(y[1],y[0]))%360
            errs.append(min(abs(ang-h),360-abs(ang-h)))
read = round(float(np.mean(errs)),2)
print("CLIP CLS hue read err {20,260}:", read)
json.dump({"read_err_deg": read}, open("results/clip_vit_readout.json","w"), indent=1)
