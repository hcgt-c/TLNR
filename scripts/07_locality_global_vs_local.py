# -*- coding: utf-8 -*-
"""
07 局部性对照：全局单算子 vs 分弧段局部算子（跨形状）
========================================================
问题：04（全局单算子,混合位移尺度）跨形状≈打平；05（40° 单步）≈1.3×。
假说：CNN 特征中 hue 位移场随位置旋转（D1≈0/-0.2）——全局单 J 不能覆盖全环；
      但"每 40° 弧段的位移"可能是跨形状共享的 → 分段局部算子应接近甚至超过单步全局。

设计（干净设定，只测 hue、s=v=1、40° 单步）：
  对 8 个弧段 k∈{0..7}（起点 h0=40k）：
    训练段内算子：跨 4 训练形状的 (z(h0)→z(h0+40)) 位移，学 rank-r 共享缩放算子
      （对数域线性：对每个 U 通道 i，w_i' = w_i * exp(Δλ_i)，同 04 机制）
    测试：跨 2 测试形状该弧段的位移预测 → rel_err vs copy
对照：
  A 全局单算子（所有弧段共享一个 λ，同 05 D3）
  B 分段算子（每弧段各自 λ，但都用训练形状学）
  C copy
报告：每弧段 + 平均。若 B 平均增益显著 > A → 位移场非全局平移不变，
     方案应采用多局部算子或高阶级数（说明截图公式需"分段参数化"）。
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_train_lie_operator.py")).read().split("if __name__")[0])

OUT = os.path.dirname(os.path.abspath(__file__)) + "/.."
RES_DIR = os.path.join(OUT, "results")


def train_local(cells, segs, r=32, epochs=150, lr=8e-3):
    """每弧段训练独立 (U,λ)。返回 {(h0,h1): (U,a)}。U 每段独立初始化。"""
    d = len(next(iter(cells.values())))
    ops = {}
    for (h0, h1) in segs:
        Z1s, Z2s = [], []
        for sh in TRAIN_SHAPES:
            for s, v in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:
                if (sh, h0, s, v) in cells and (sh, h1, s, v) in cells:
                    Z1s.append(cells[(sh, h0, s, v)]); Z2s.append(cells[(sh, h1, s, v)])
        if len(Z1s) < 4:
            continue
        Z1 = torch.tensor(np.array(Z1s), dtype=torch.float64, device=DEV)
        Z2 = torch.tensor(np.array(Z2s), dtype=torch.float64, device=DEV)
        torch.manual_seed(0)
        U = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64, device=DEV))[0].requires_grad_(True)
        lam = torch.zeros(r, dtype=torch.float64, requires_grad=True, device=DEV)
        opt = torch.optim.Adam([U, lam], lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            W = U.t() @ Z1.t()
            y = Z1 + (U @ ((torch.exp(lam).unsqueeze(1) - 1.0) * W)).t()
            loss = ((y - Z2) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                g = U.grad
                for i in range(r):
                    for j in range(i):
                        g[:, i] -= g[:, j] * torch.dot(U[:, i], U[:, j])
                U.grad = g
            opt.step()
            with torch.no_grad():
                U.data = torch.linalg.qr(U.data)[0]
        ops[(h0, h1)] = (U.detach().cpu().numpy(), lam.detach().cpu().numpy())
    return ops


def local_predict(ops, h0, h1, z):
    U, lam = ops[(h0, h1)]
    w = U.T @ z
    return z + U @ ((np.exp(lam) - 1.0) * w)


def global_ops(cells, r=32, epochs=150, lr=8e-3):
    """全局单算子：所有 (h,h+40) 段合并训练一个 (U,λ)。"""
    d = len(next(iter(cells.values())))
    Z1s, Z2s = [], []
    for (h0, h1) in segs_all():
        for sh in TRAIN_SHAPES:
            for s, v in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:
                if (sh, h0, s, v) in cells and (sh, h1, s, v) in cells:
                    Z1s.append(cells[(sh, h0, s, v)]); Z2s.append(cells[(sh, h1, s, v)])
    Z1 = torch.tensor(np.array(Z1s), dtype=torch.float64, device=DEV)
    Z2 = torch.tensor(np.array(Z2s), dtype=torch.float64, device=DEV)
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64, device=DEV))[0].requires_grad_(True)
    lam = torch.zeros(r, dtype=torch.float64, requires_grad=True, device=DEV)
    opt = torch.optim.Adam([U, lam], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        W = U.t() @ Z1.t()
        y = Z1 + (U @ ((torch.exp(lam).unsqueeze(1) - 1.0) * W)).t()
        loss = ((y - Z2) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            g = U.grad
            for i in range(r):
                for j in range(i):
                    g[:, i] -= g[:, j] * torch.dot(U[:, i], U[:, j])
            U.grad = g
        opt.step()
        with torch.no_grad():
            U.data = torch.linalg.qr(U.data)[0]
    return (U.detach().cpu().numpy(), lam.detach().cpu().numpy())


def segs_all():
    return [(h, (h + 40) % 360) for h in TRAIN_HUES]


def run(bb, layer, r=32):
    cells = load_cells(bb, layer)
    segs = segs_all()
    local = train_local(cells, segs, r=r)
    gU, glam = global_ops(cells, r=r)
    rows = []
    agg = {"local": [], "global": [], "copy": []}
    for (h0, h1) in segs:
        e_loc, e_glo, e_copy = [], [], []
        for sh in TEST_SHAPES:
            for s, v in [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:
                if (sh, h0, s, v) not in cells or (sh, h1, s, v) not in cells:
                    continue
                z1, z2 = cells[(sh, h0, s, v)], cells[(sh, h1, s, v)]
                if (h0, h1) in local:
                    yl = local_predict(local, h0, h1, z1)
                    e_loc.append(rel_err(yl, z2))
                yg = lie_predict_np(gU, glam, np.zeros(r), np.zeros(r), [40.0, 0.0, 0.0], z1)
                e_glo.append(rel_err(yg, z2))
                e_copy.append(rel_err(z1, z2))
        if e_loc:
            rows.append({"seg": f"{h0}→{h1}", "local": float(np.mean(e_loc)),
                         "global": float(np.mean(e_glo)), "copy": float(np.mean(e_copy)),
                         "n": len(e_loc)})
            agg["local"].append(float(np.mean(e_loc)))
            agg["global"].append(float(np.mean(e_glo)))
            agg["copy"].append(float(np.mean(e_copy)))
    out = {"rows": rows,
           "mean": {k: float(np.mean(v)) for k, v in agg.items()},
           "gain_local_over_copy": float(np.mean(agg["copy"]) / (np.mean(agg["local"]) + 1e-30)),
           "gain_global_over_copy": float(np.mean(agg["copy"]) / (np.mean(agg["global"]) + 1e-30))}
    print(f"[{bb} {layer}] 均值: local={out['mean']['local']:.5f} global={out['mean']['global']:.5f} "
          f"copy={out['mean']['copy']:.5f}")
    print(f"  增益 vs copy: 局部={out['gain_local_over_copy']:.3f}× 全局={out['gain_global_over_copy']:.3f}×")
    json.dump(out, open(os.path.join(RES_DIR, f"locality_{bb}_{layer}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
    layer = sys.argv[2] if len(sys.argv) > 2 else "y3"
    run(bb, layer)
