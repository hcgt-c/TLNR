# -*- coding: utf-8 -*-
"""重算 64 的头条并固化 results/hue_prediction_demo.json（避免手抄）"""
import numpy as np, json, importlib.util, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("m64", os.path.join(os.path.dirname(os.path.abspath(__file__)), "64_hue_prediction_demo.py"))
m = importlib.util.module_from_spec(spec)
# 不执行模块（其顶层会跑评测打印）；改为轻量重实现：直接 import 其函数
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "64_hue_prediction_demo.py")).read()
# 提取函数部分（def 之前只 import）
exec("import numpy as np\nimport numpy.linalg as la\n" +
     src[src.index("H = np.arange"):src.index('print("== dense5')], globals())
out = {"dense5": [], "sheet_hv_10v": {}}
F = load("features/dense5_y3.npz", True)
for r in eval_plane(F, 0):
    out["dense5"].append({"K": r[0], "relMSE_op": round(r[1], 4), "copy": round(r[2], 4),
                          "gain": round(r[2] / r[1], 1), "cos": round(r[3], 3)})
F = load("features/sheet_hv_y3.npz", False)
rows = [r for vi in range(10) for r in eval_plane(F, vi)]
rows = np.array(rows)
for K in (3, 4):
    sel = rows[rows[:, 1] == K]
    g = sel[:, 3] / sel[:, 2]
    out["sheet_hv_10v"]["K%d" % K] = {"gain_mean": round(float((sel[:, 3] / sel[:, 2]).mean()), 1),
                                      "gain_range": [round(float(g.min()), 1), round(float(g.max()), 1)],
                                      "cos_mean": round(float(sel[:, 4].mean()), 3)}
json.dump(out, open("results/hue_prediction_demo.json", "w"), indent=1)
print(json.dumps(out, indent=1))
