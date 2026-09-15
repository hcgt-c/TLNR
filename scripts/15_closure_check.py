# -*- coding: utf-8 -*-
"""
15 方向②第一刀：分段算子的闭合一致性检验（沿环复合 vs 直接预测）
========================================================
目标：07 在 40° 段内学跨形状共享算子（每个弧段 h_k→h_k+40 一个 U_k,λ_k）。
本脚本检验分段结构沿色相环的"整体一致性"：
  C1 环上任意两点 (h_i, h_j)（弧长 ≤ 320°）：用分段算子沿路径逐步复合
     (h_i→h_i+40→...+→h_j) 预测 z(h_j)，误差 vs copy —— 分段能否当"联络"用？
  C2 闭合性：从 h=0 走完 8 段到 320，再沿闭合段 320→360≡0 回原点；
     比较"复合绕一圈" vs 恒等，量化闭合误差（真群作用应≈0）。
  C3 段间连接：若闭合误差大，测每段单独预测的段内误差（应该小），
     定位误差来自"段间接口"（曲率未建模处）。
数据：y3（最佳层）跨 4 训练形状学分段算子，2 测试形状验证。
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_cells, TRAIN_SHAPES, TEST_SHAPES, TRAIN_HUES, SVPAIRS,
                    unit, rel_err, OUT)

RES_DIR = os.path.join(OUT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def learn_segments(bb, layer, shapes, r=16, epochs=250, lr=8e-3):
    """每段 (h_k→h_k+40) 学 (U,lam)。返回 {(h_k): (U,lam)}。"""
    cells = load_cells(bb, layer)
    d = len(next(iter(cells.values())))
    ops = {}
    for k in range(len(TRAIN_HUES)):
        h1 = TRAIN_HUES[k]
        h2 = TRAIN_HUES[(k + 1) % len(TRAIN_HUES)]
        Z1s, Z2s = [], []
        for sh in shapes:
            for (s, v) in SVPAIRS:
                if (sh, h1, s, v) in cells and (sh, h2, s, v) in cells:
                    Z1s.append(cells[(sh, h1, s, v)])
                    Z2s.append(cells[(sh, h2, s, v)])
        if len(Z1s) < 4:
            ops[(h1, h2)] = None
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
        ops[(h1, h2)] = (U.detach().cpu().numpy(), lam.detach().cpu().numpy())
    return ops


def seg_apply(op, z):
    if op is None:
        return z
    U, lam = op
    w = U.T @ z
    return z + U @ ((np.exp(lam) - 1.0) * w)


def seg_apply_inv(op, z):
    """逆算子：exp(−λ)（对易共享基可逆）。"""
    if op is None:
        return z
    U, lam = op
    w = U.T @ z
    return z + U @ ((np.exp(-lam) - 1.0) * w)


def run(bb, layer):
    cells = load_cells(bb, layer)
    ops = learn_segments(bb, layer, TRAIN_SHAPES)
    # 段索引（环上位置）
    idx = {h: k for k, h in enumerate(TRAIN_HUES)}
    # C1：任意两点路径复合（测试形状）
    err_path, err_direct_seg, err_copy = [], [], []
    path_err_detail = []
    for sh in TEST_SHAPES:
        for (s, v) in SVPAIRS:
            for i in range(len(TRAIN_HUES)):
                for j in range(len(TRAIN_HUES)):
                    if i == j:
                        continue
                    h_i, h_j = TRAIN_HUES[i], TRAIN_HUES[j]
                    if (sh, h_i, s, v) not in cells or (sh, h_j, s, v) not in cells:
                        continue
                    z1 = cells[(sh, h_i, s, v)]
                    z2 = cells[(sh, h_j, s, v)]
                    # 前向逐段（绕环，取短弧方向）
                    step = 1 if (j - i) % len(TRAIN_HUES) <= len(TRAIN_HUES) // 2 \
                        else -1
                    z = z1.copy()
                    cur = i
                    nstep = 0
                    while cur != j:
                        nxt = (cur + step) % len(TRAIN_HUES)
                        hA, hB = TRAIN_HUES[cur], TRAIN_HUES[nxt]
                        if step == -1:
                            # 逆向：用 (hB→hA) 段算子的逆（exp(−λ)——对易共享基可逆）
                            op = ops.get((hB, hA))
                            if op is None:
                                break
                            z = seg_apply_inv(op, z)
                        else:
                            op = ops.get((hA, hB))
                            if op is None:
                                break
                            z = seg_apply(op, z)
                        cur = nxt
                        nstep += 1
                    if cur == j:
                        err_path.append(rel_err(z, z2))
                        err_copy.append(rel_err(z1, z2))
                        path_err_detail.append({"from": h_i, "to": h_j,
                                                "nstep": nstep,
                                                "err": round(rel_err(z, z2), 4)})
    # C2：闭合——0→320→0 走 8+8 步（绕整环两遍会各向同性吗？只走完整一圈 8 段回 0）
    # 环上 i=0 → 走 8 步回 0（经过 9 个点闭合）
    err_closure = []
    for sh in TEST_SHAPES:
        for (s, v) in SVPAIRS:
            if (sh, TRAIN_HUES[0], s, v) not in cells:
                continue
            z = cells[(sh, TRAIN_HUES[0], s, v)].copy()
            for k in range(len(TRAIN_HUES) - 1):
                op = ops.get((TRAIN_HUES[k], TRAIN_HUES[k + 1]))
                z = seg_apply(op, z)
            # 最后闭合段 320→0（=360）
            op = ops.get((TRAIN_HUES[-1], TRAIN_HUES[0]))
            if op is not None:
                z = seg_apply(op, z)
            err_closure.append(rel_err(z, cells[(sh, TRAIN_HUES[0], s, v)]))
    # C3：单段（40°）误差（参考）
    err_single = []
    for sh in TEST_SHAPES:
        for (s, v) in SVPAIRS:
            for k in range(len(TRAIN_HUES) - 1):
                hA, hB = TRAIN_HUES[k], TRAIN_HUES[k + 1]
                if (sh, hA, s, v) in cells and (sh, hB, s, v) in cells:
                    op = ops.get((hA, hB))
                    y = seg_apply(op, cells[(sh, hA, s, v)])
                    err_single.append(rel_err(y, cells[(sh, hB, s, v)]))
    res = {
        "path_compound_err": float(np.mean(err_path)) if err_path else None,
        "copy_err": float(np.mean(err_copy)) if err_copy else None,
        "path_over_copy_gain": float(np.mean(err_copy) / (np.mean(err_path) + 1e-30)) if err_path else None,
        "closure_err_full_loop": float(np.mean(err_closure)) if err_closure else None,
        "single_seg_err": float(np.mean(err_single)) if err_single else None,
        "n_path": len(err_path), "n_closure": len(err_closure),
        "worst_paths": sorted(path_err_detail, key=lambda x: -x["err"])[:5],
    }
    print(f"[{bb} {layer}]")
    print(f"  C1 路径复合误差={res['path_compound_err']:.5f} (copy={res['copy_err']:.5f}, "
          f"gain={res['path_over_copy_gain']:.1f}×, n={res['n_path']})")
    print(f"  C2 整环闭合误差={res['closure_err_full_loop']:.5f}")
    print(f"  C3 单段误差={res['single_seg_err']:.5f}（对照）")
    json.dump(res, open(os.path.join(RES_DIR, f"closure_check_{bb}_{layer}.json"), "w"), indent=1)


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
    layer = sys.argv[2] if len(sys.argv) > 2 else "y3"
    run(bb, layer)
