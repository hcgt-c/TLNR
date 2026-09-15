# -*- coding: utf-8 -*-
"""B 应用指标：色相圆回归探针（y3，跨形状零样本色相）。
Part1 稠密训练（72h 全用，排除 {20,260}）：探针在 train 形状上拟合 z→e^{ih}，
      测试形状 hold hues {20,260}：real 特征 vs 算子预测(K=4) vs copy(z0) 的圆误差。
Part2 稀疏训练（40° 9 色相）增强对比：real-only vs +算子增强(中间色相) vs +线性插值增强。
跑 dense5 (s=v=1) 与 sheet_hv 10 个 v 面。"""
import numpy as np
import numpy.linalg as la
import json

H = np.arange(0, 360, 5)
HOLD = [20, 260]
TRAIN_HUES40 = [0, 40, 80, 120, 160, 200, 240, 280, 320]

def load(path, preavg):
    d = np.load(path)
    return (d["feats"].reshape(6, 72, 4, 64).mean(2)[None] if preavg
            else d["feats"].reshape(10, 6, 72, 64))

def band_fit(Ztrain, K=4):
    """Ztrain: list of 4 centered (72,64)。返回 apply(z0,h)。"""
    Xr, Yr = [], []
    for Zj in Ztrain:
        for hi, h in enumerate(H):
            if h in HOLD or h == 0:
                continue
            Xr.append(np.concatenate([np.cos(np.deg2rad(k * h)) * Zj[0] for k in range(1, K + 1)] +
                                     [np.sin(np.deg2rad(k * h)) * Zj[0] for k in range(1, K + 1)]))
            Yr.append(Zj[hi])
    X, Y = np.array(Xr), np.array(Yr)
    W = la.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ Y)
    return lambda z0, h: np.concatenate(
        [np.cos(np.deg2rad(k * h)) * z0 for k in range(1, K + 1)] +
        [np.sin(np.deg2rad(k * h)) * z0 for k in range(1, K + 1)]) @ W

def probe_err(Ztrain, Ztest, feats_variant):
    """线性探针 z→[cos h, sin h]（岭），返回各测试点在 hold hues 的圆误差(deg)。
    feats_variant: callable(test_shape_centered, h) -> feature vector"""
    Xr, Yr = [], []
    for Zj in Ztrain:
        for hi, h in enumerate(H):
            if h in HOLD:
                continue
            Xr.append(Zj[hi]); Yr.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    X, Y = np.array(Xr), np.array(Yr)
    W = la.solve(X.T @ X + 1e-4 * np.eye(X.shape[1]), X.T @ Y)
    errs = []
    for Zt in Ztest:
        for hi, h in enumerate(H):
            if h not in HOLD:
                continue
            f = feats_variant(Zt, h)
            y = f @ W
            pred = np.degrees(np.arctan2(y[1], y[0])) % 360
            errs.append(min(abs(pred - h), 360 - abs(pred - h)))
    return np.mean(errs)

def eval_plane(F, pi):
    Ztrain = [F[pi, j] - F[pi, j].mean(0) for j in range(4)]
    Ztest = [F[pi, j] - F[pi, j].mean(0) for j in (4, 5)]
    op = band_fit(Ztrain, 4)
    e_real = probe_err(Ztrain, Ztest, lambda Zt, h: Zt[H.tolist().index(h)])
    e_pred = probe_err(Ztrain, Ztest, lambda Zt, h: op(Zt[0], h))
    e_copy = probe_err(Ztrain, Ztest, lambda Zt, h: Zt[0])
    return e_real, e_pred, e_copy

out = {}
F = load("features/dense5_y3.npz", True)
r = eval_plane(F, 0)
out["dense5_hue_probe_deg"] = {"real": round(r[0], 1), "operator_pred": round(r[1], 1), "copy": round(r[2], 1)}
print("dense5: real=%.1f° pred=%.1f° copy=%.1f°" % r)

F = load("features/sheet_hv_y3.npz", False)
rows = [eval_plane(F, vi) for vi in range(10)]
a = np.array(rows)
out["sheet_hv_10v_hue_probe_deg"] = {"real": round(float(a[:, 0].mean()), 1),
                                     "operator_pred": round(float(a[:, 1].mean()), 1),
                                     "copy": round(float(a[:, 2].mean()), 1)}
print("sheet_hv 10v: real=%.1f° pred=%.1f° copy=%.1f°" % tuple(a.mean(0)))

# ---- Part2 稀疏训练 + 增强（dense5） ----
def sparse_probe_err(Ztrain, Ztest, variant, train_hues):
    Xr, Yr = [], []
    for Zj in Ztrain:
        for h in train_hues:
            Xr.append(Zj[H.tolist().index(h)]); Yr.append([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h))])
    X, Y = np.array(Xr), np.array(Yr)
    W = la.solve(X.T @ X + 1e-4 * np.eye(X.shape[1]), X.T @ Y)
    errs = []
    for Zt in Ztest:
        for hi, h in enumerate(H):
            if h not in HOLD:
                continue
            f = variant(Zt, h)
            y = f @ W
            pred = np.degrees(np.arctan2(y[1], y[0])) % 360
            errs.append(min(abs(pred - h), 360 - abs(pred - h)))
    return np.mean(errs)

F = load("features/dense5_y3.npz", True)
Ztrain = [F[0, j] - F[0, j].mean(0) for j in range(4)]
Ztest = [F[0, j] - F[0, j].mean(0) for j in (4, 5)]
op = band_fit(Ztrain, 4)
def real_at(Zt, h): return Zt[H.tolist().index(h)]
def op_aug_train():
    # 训练集 = 9 真实色相 + 算子预测的 18 个中间色相（train 形状、同锚点 z0）
    hs = sorted(set(TRAIN_HUES40) | {h for h in range(10, 350, 20) if h not in HOLD})
    return hs
e_rs = sparse_probe_err(Ztrain, Ztest, real_at, TRAIN_HUES40)
e_ra = sparse_probe_err(Ztrain, Ztest, real_at, op_aug_train())   # 假增强：用"真值"中间色相当训练（上界）
e_oa = sparse_probe_err(Ztrain, Ztest,
                        lambda Zt, h: Zt[H.tolist().index(h)] if h in TRAIN_HUES40 else op(Zt[0], h),
                        op_aug_train())                            # 增强=真实9色相 + 算子预测中间色相
# 线性插值增强对照：中间色相特征 = 最近两个真实色相特征线性插值
def lin_interp(Zj, h):
    h0 = max([x for x in TRAIN_HUES40 if x <= h] or [0])
    h1 = min([x for x in TRAIN_HUES40 if x > h] or [360])
    a = (h - h0) / (h1 - h0)
    return (1 - a) * Zj[H.tolist().index(h0)] + a * Zj[H.tolist().index(h1 % 360)]
def lin_aug_variant(Zt, h):
    return Zt[H.tolist().index(h)] if h in TRAIN_HUES40 else lin_interp(Zt, h)
e_la = sparse_probe_err(Ztrain, Ztest, lin_aug_variant, op_aug_train())
out["dense5_sparse40_probe_deg"] = {"real_only": round(e_rs, 1), "real+linear_interp_aug": round(e_la, 1),
                                    "real+operator_aug": round(e_oa, 1), "real+true_mid_aug(上界)": round(e_ra, 1)}
print("稀疏40°训练探针(hold 20/260): real-only=%.1f° +lin=%.1f° +op=%.1f° +true=%.1f°" % (e_rs, e_la, e_oa, e_ra))

json.dump(out, open("results/hue_probe_app.json", "w"), indent=1, ensure_ascii=False)
print("saved results/hue_probe_app.json")
