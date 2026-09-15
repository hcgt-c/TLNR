# -*- coding: utf-8 -*-
"""
20c 两阶段：逐段学真实 Ω_k（旋转 log）→ 傅里叶拟合 Ω(θ)（稳定、验证表达力）
========================================================
阶段1：对每个 40° 段独立学反对称 A_k 使 exp(A_k·40°)z1≈z2（同 16 的逐段版）
        —— 得到"逐段真实旋转生成元"A_k（8 个）。
阶段2：把 A_k 序列做傅里叶拟合 Ω(θ)=Ω0+Σ(C_k cos+S_k sin)（对每个矩阵元素，
       8 点拟合 ≤3 阶 → 过参数但最小二乘稳定），评估拟合误差。
       若拟合误差小 ⟹ Ω(θ) 平滑（3 阶傅里叶能表达）→ 层次 A 表达力足够，
       端到端训练不稳定是优化问题而非表达力问题。
输出：阶段1逐段误差；阶段2傅里叶拟合残差（R²）；重评估平滑Ω的跨形状增益+闭合。
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_cells, TRAIN_SHAPES, TEST_SHAPES, TRAIN_HUES, SVPAIRS,
                    hue_delta, rel_err, OUT)

RES_DIR = os.path.join(OUT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 24
SEG = 40.0


def pca(cells, dim_pca):
    allZ = np.array(list(cells.values()))
    mu = allZ.mean(0)
    _, _, Vt = np.linalg.svd(allZ - mu, full_matrices=False)
    return Vt[:dim_pca], mu


def learn_seg_skew(cells, shapes, Wp, mu, h1, epochs=800, lr=5e-3):
    """学 A_k（反对称）使 exp(A_k*40)z1≈z2。返回 A_k。"""
    h2 = (h1 + SEG) % 360
    Z1, Z2 = [], []
    for sh in shapes:
        for (s, v) in SVPAIRS:
            if (sh, h1, s, v) in cells and (sh, h2, s, v) in cells:
                Z1.append(Wp @ (cells[(sh, h1, s, v)] - mu))
                Z2.append(Wp @ (cells[(sh, h2, s, v)] - mu))
    if len(Z1) < 2:
        return None
    Z1t = torch.tensor(np.array(Z1), dtype=torch.float32, device=DEV)
    Z2t = torch.tensor(np.array(Z2), dtype=torch.float32, device=DEV)
    torch.manual_seed(0)
    S = torch.randn(D, D, dtype=torch.float32, device=DEV) * 0.05
    S.requires_grad_(True)
    opt = torch.optim.Adam([S], lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(len(Z1t), device=DEV)[:64]
        opt.zero_grad()
        A = S - S.t()
        E = torch.linalg.matrix_exp(A * SEG)
        y = (E @ Z1t[idx].t()).t()
        loss = ((y - Z2t[idx]) ** 2).mean()
        loss.backward()
        opt.step()
    A = (S.detach() - S.detach().t()).cpu().numpy()
    return A


def fourier_fit(As, K=3):
    """As: (8,D,D) 在 θ_k=TRAIN_HUES[:-1] 采样。拟合 Ω(θ) 傅里叶，返回系数与 R²。"""
    thetas = np.array(TRAIN_HUES[:len(As)], dtype=float)
    # 每矩阵元素独立拟合（D*D 元素 × 8 点）
    A = np.stack(As)                       # (8, D, D)
    n = len(thetas)
    # 设计矩阵：Φ = [1, cos(ωθ), sin(ωθ), cos(2ωθ), sin(2ωθ), ...]  (n, 1+2K)
    w = 2 * np.pi / 360.0
    Phi = np.ones((n, 1 + 2 * K))
    for k in range(1, K + 1):
        Phi[:, 2*k-1] = np.cos(k * w * thetas)
        Phi[:, 2*k] = np.sin(k * w * thetas)
    # 对每个 (i,j) 元素：A[:,i,j] = Phi @ coef
    coef = np.linalg.lstsq(Phi, A.reshape(n, -1), rcond=None)[0]  # (1+2K, D*D)
    pred = (Phi @ coef).reshape(n, D, D)
    # R²
    ss_res = np.sum((A - pred) ** 2)
    ss_tot = np.sum((A - A.mean(0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    # 反对称化预测（保证旋转）
    Om = np.zeros_like(pred)
    for t in range(n):
        M = pred[t]
        Om[t] = (M - M.T) / 2.0
    return coef, r2, Om


def evaluate_omega(Omega_fn, bb, layer, Wp, mu):
    """用 Ω(θ) 函数做跨形状评估 + 闭合。Omega_fn(theta_deg)->(D,D) np 反对称"""
    cells = load_cells(bb, layer)
    e_op, e_copy, e_clo = [], [], []
    for sh in TEST_SHAPES:
        for (s, v) in SVPAIRS:
            for h1 in TRAIN_HUES:
                for h2 in TRAIN_HUES:
                    if h1 == h2:
                        continue
                    dh = hue_delta(h1, h2)
                    if abs(dh) < 20 or abs(dh) > 160:
                        continue
                    if (sh, h1, s, v) not in cells or (sh, h2, s, v) not in cells:
                        continue
                    w1 = Wp @ (cells[(sh, h1, s, v)] - mu)
                    w2 = Wp @ (cells[(sh, h2, s, v)] - mu)
                    # 沿路径分 40° 步
                    z = w1.copy()
                    rem = abs(dh)
                    sgn = 1 if dh > 0 else -1
                    hc = float(h1)
                    while rem > 1e-6:
                        ds = min(SEG, rem)
                        Om = Omega_fn(hc + sgn * ds / 2)
                        Ot = torch.tensor(Om * (sgn * ds), dtype=torch.float32)
                        E = torch.linalg.matrix_exp(Ot).numpy()
                        z = E @ z
                        hc += sgn * ds
                        rem -= ds
                    e_op.append(rel_err(z, w2))
                    e_copy.append(rel_err(w1, w2))
            # 闭合
            if (sh, 0, s, v) in cells:
                w0 = Wp @ (cells[(sh, 0, s, v)] - mu)
                z = w0.copy()
                for hh in range(0, 360, 40):
                    Om = Omega_fn(hh + 20)
                    Ot = torch.tensor(Om * 40.0, dtype=torch.float32)
                    E = torch.linalg.matrix_exp(Ot).numpy()
                    z = E @ z
                e_clo.append(rel_err(z, w0))
    return {"gain": float(np.mean(e_copy) / (np.mean(e_op) + 1e-30)),
            "err_op": float(np.mean(e_op)),
            "closure_err_360": float(np.mean(e_clo)) if e_clo else None,
            "n_test": len(e_op)}


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
    layer = sys.argv[2] if len(sys.argv) > 2 else "y3"
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cells = load_cells(bb, layer)
    Wp, mu = pca(cells, D)
    # 阶段1：逐段独立反对称 A_k
    As = []
    for h1 in TRAIN_HUES[:8]:      # 8 个开段
        A = learn_seg_skew(cells, TRAIN_SHAPES, Wp, mu, h1)
        if A is not None:
            As.append(A)
    print(f"阶段1: 学到 {len(As)} 段旋转生成元")
    # 阶段2：傅里叶拟合
    coef, r2, Om = fourier_fit(As, K)
    print(f"阶段2: 傅里叶拟合 R²={r2:.4f}（1=平滑Ω可表达逐段结构）")
    thetas_fit = np.array(TRAIN_HUES[:len(As)])
    # 用拟合系数构造 Ω(θ) 函数
    w = 2 * np.pi / 360.0
    def Omega_fn(th):
        th = float(th)
        basis = [1.0] + [np.cos(k*w*th) if i % 2 == 1 else np.sin(k*w*th)
                         for k in range(1, K+1) for i in (1, 2)]
        # coef shape (1+2K, D*D)
        M = (np.array(basis) @ coef).reshape(D, D)
        return (M - M.T) / 2.0
    ev = evaluate_omega(Omega_fn, bb, layer, Wp, mu)
    print(f"平滑Ω跨形状增益={ev['gain']:.2f}× err_op={ev['err_op']:.5f} "
          f"闭合={ev['closure_err_360']:.5f}")
    res = {"n_segs": len(As), "fourier_r2": float(r2), "evaluate": ev}
    json.dump(res, open(os.path.join(RES_DIR, f"fourier_fit_{bb}_{layer}_K{K}.json"), "w"), indent=1)
