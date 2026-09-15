# -*- coding: utf-8 -*-
"""提取 dense5 数据的 y3 特征（区域池化更佳？先用 GAP 与 mask-region 都存）"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from ultralytics import YOLO
mm = YOLO(rp.YOLO_WEIGHTS).model.to(DEV).eval()
hooks = {}
mm.model[3].register_forward_hook(lambda m,i,o: hooks.__setitem__('y3', o.detach() if o.dim()==4 else None))
d = np.load(os.path.join(OUT, "data", "dense5.npz"))
shape, hue, img = d["shape"], d["hue"], d["img"]
feats = []
with torch.no_grad():
    for i in range(len(img)):
        a = img[i].astype(np.float32)/np.float32(255.0)
        x = torch.tensor(a, device=DEV).permute(2,0,1).unsqueeze(0)
        mm(x)
        f = hooks['y3'].mean(dim=(2,3))[0].cpu().numpy()
        feats.append(f)
feats = np.stack(feats)
np.savez(os.path.join(OUT, "features", "dense5_y3.npz"),
         feats=feats, shape=shape, hue=hue)
print("dense5_y3:", feats.shape)
