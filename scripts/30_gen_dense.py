# -*- coding: utf-8 -*-
"""生成细色相数据：hue 5° 步（72 点），s=v=1.0，6 形状 × 每色 4 变体"""
import os, sys, json
import numpy as np
import importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_gen_synthetic.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)

N_VAR = 4
SHAPES = gen.TRAIN_SHAPES + gen.TEST_SHAPES
HUES = list(range(0, 360, 5))   # 72 点
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dense5.npz")
arrays = {"shape": [], "hue": [], "img": []}
import numpy as np
for sh in SHAPES:
    for h in HUES:
        for _ in range(N_VAR):
            img = gen.render(sh, h, 1.0, 1.0)
            arrays["shape"].append(sh)
            arrays["hue"].append(h)
            arrays["img"].append(img)
np.savez(OUT, shape=np.array(arrays["shape"]), hue=np.array(arrays["hue"]),
         img=np.array(arrays["img"]))
print(f"dense5 saved: {len(arrays['img'])} imgs = {len(SHAPES)} shapes x {len(HUES)} hues x {N_VAR} var")
