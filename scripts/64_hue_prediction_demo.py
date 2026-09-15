# -*- coding: utf-8 -*-
"""B 核心正例 v2：理论结构预测 ≈ 实测（跨形状零样本色相，逐 v 面）。
协议 = 已验证的 35 号：逐形状去 DC；跨 4 train 形状拟合带限算子（排除留出色相）；
锚点 z0 = 测试形状居中特征在 h=0°；评测在留出色相 {20,260}。
指标（35 一致）：rel err = Σ‖ẑ−z‖²/Σ‖z‖²（求和范围=被评测点），gain = copy/op；
余弦；应用指标：NN 色相圆误差（预测特征在 train 库最近邻的色相 vs 真色相）。"""
import numpy as np
import numpy.linalg as la

H = np.arange(0, 360, 5)
HOLD = [20, 260]

def load(path, preavg):
    d = np.load(path)
    if preavg:
        return d["feats"].reshape(6, 72, 4, 64).mean(2)[None]
    return d["feats"].reshape(10, 6, 72, 64)

def eval_plane(F, pi):
    out = []
    for si in (4, 5):                       # test shapes
        Zc = F[pi, si] - F[pi, si].mean(0)
        var = np.sum(Zc ** 2)
        bank = np.concatenate([F[pi, j] - F[pi, j].mean(0) for j in range(4)], 0)
        bank_h = np.tile(H, 4)
        for K in (3, 4):
            Xr, Yr = [], []
            for j in range(4):
                Zj = F[pi, j] - F[pi, j].mean(0)
                for hi, h in enumerate(H):
                    if h in HOLD or h == 0:
                        continue
                    Xr.append(np.concatenate([np.cos(np.deg2rad(k * h)) * Zj[0] for k in range(1, K + 1)] +
                                             [np.sin(np.deg2rad(k * h)) * Zj[0] for k in range(1, K + 1)]))
                    Yr.append(Zj[hi])
            X, Y = np.array(Xr), np.array(Yr)
            W = la.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ Y)
            def apply(z0, h):
                ph = np.concatenate([np.cos(np.deg2rad(k * h)) * z0 for k in range(1, K + 1)] +
                                    [np.sin(np.deg2rad(k * h)) * z0 for k in range(1, K + 1)])
                return ph @ W
            z0 = Zc[0]
            eo = ec = 0.0; cs = []
            nno = nnc = []
            for hi, h in enumerate(H):
                if h not in HOLD:
                    continue
                zt = Zc[hi]
                zp = apply(z0, h)
                eo += np.sum((zp - zt) ** 2)
                ec += np.sum((z0 - zt) ** 2)
                cs.append(float(zp @ zt / (la.norm(zp) * la.norm(zt) + 1e-12)))
                def cerr(a, b):
                    return min(abs(a - b), 360 - abs(a - b))
                nno.append(cerr(bank_h[la.norm(bank - zp, axis=1).argmin()], h))
                nnc.append(cerr(bank_h[la.norm(bank - z0, axis=1).argmin()], h))
            out.append((K, eo / var, ec / var, np.mean(cs), np.mean(nno), np.mean(nnc)))
    return out

print("== dense5 面 (s=v=1)：test pentagon/hexagon ==")
F = load("features/dense5_y3.npz", True)
for K, eo, ec, cs, nno, nnc in eval_plane(F, 0):
    print("  K=%d: relMSE op=%.4f copy=%.4f gain=%.1fx  余弦=%.3f  NN色相误差 op=%.0f° copy=%.0f°" %
          (K, eo, ec, ec / eo, cs, nno, nnc))

print("\n== sheet_hv：10 个 v 面（s=1）汇总 ==")
F = load("features/sheet_hv_y3.npz", False)
rows = []
for vi in range(10):
    for r in eval_plane(F, vi):
        rows.append((vi,) + r)
rows = np.array(rows)
for K in (3, 4):
    sel = rows[rows[:, 1] == K]
    eo, ec, cs = sel[:, 2].mean(), sel[:, 3].mean(), sel[:, 4].mean()
    nno, nnc = sel[:, 5].mean(), sel[:, 6].mean()
    g = sel[:, 3] / sel[:, 2]
    print("  K=%d: relMSE op=%.4f copy=%.4f gain=%.1fx [%.1f,%.1f]  余弦=%.3f  NN op=%.0f° copy=%.0f°"
          % (K, eo, ec, ec / eo, g.min(), g.max(), cs, nno, nnc))

# ---- 固化结果 ----
import json as _json
d5rows = []
F = load("features/dense5_y3.npz", True)
for r in eval_plane(F, 0):
    d5rows.append({"K": int(r[0]), "relMSE_op": round(float(r[1]), 4), "copy": round(float(r[2]), 4),
                   "gain": round(float(r[2] / r[1]), 1), "cos": round(float(r[3]), 3),
                   "nn_op_deg": round(float(r[4]), 0), "nn_copy_deg": round(float(r[5]), 0)})
F = load("features/sheet_hv_y3.npz", False)
rows = [r for vi in range(10) for r in eval_plane(F, vi)]
rows = np.array(rows)                       # (K, eo, ec, cs, nno, nnc)
agg = {}
for K in (3, 4):
    sel = rows[rows[:, 0] == K]
    g = sel[:, 2] / sel[:, 1]
    agg["K%d" % K] = {"gain_mean": round(float(g.mean()), 1), "gain_range": [round(float(g.min()), 1), round(float(g.max()), 1)],
                      "cos_mean": round(float(sel[:, 3].mean()), 3)}
_json.dump({"dense5": d5rows, "sheet_hv_10v": agg,
            "protocol": "cross-shape zero-shot hue {20,260}; per-shape DC removed; anchor z(0); band-linear op fit on 4 train shapes"},
           open("results/hue_prediction_demo.json", "w"), indent=1)
print("saved results/hue_prediction_demo.json")

