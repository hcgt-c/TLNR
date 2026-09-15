# -*- coding: utf-8 -*-
"""公共工具：数据加载、误差、预测、测试对构造（供 04-09 复用）"""
import os, sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__)) + "/.."
FEAT_DIR = os.path.join(OUT, "features")

TRAIN_SHAPES = ["triangle", "rectangle", "circle", "star"]
TEST_SHAPES = ["pentagon", "hexagon"]
TRAIN_HUES = [0, 40, 80, 120, 160, 200, 240, 280, 320]
TEST_HUES = [20, 260]
SATS = [0.5, 1.0]
VALS = [0.5, 1.0]
SVPAIRS = [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]


def load_cells(bb, layer):
    z = np.load(os.path.join(FEAT_DIR, f"feats_{bb}.npz"), allow_pickle=True)
    F, shape, hue, sat, val = z[f"feats_{layer}"], z["shape"], z["hue"], z["sat"], z["val"]
    cells = {}
    for sh, h, s, v in set(zip(shape, hue, sat, val)):
        m = (shape == sh) & (hue == h) & (sat == s) & (val == v)
        cells[(sh, h, s, v)] = F[m].mean(0)
    return cells


def hue_delta(h1, h2):
    d = (h2 - h1) % 360
    if d > 180:
        d -= 360
    return d


def rel_err(y_hat, y2):
    return float(np.sum((y_hat - y2) ** 2) / (np.sum(y2 ** 2) + 1e-30))


def lie_predict_np(U, a, b, c, dp, z):
    lam = dp[0] * a + dp[1] * b + dp[2] * c
    w = U.T @ z
    return z + U @ ((np.exp(lam) - 1.0) * w)


def segs_all():
    return [(h, (h + 40) % 360) for h in TRAIN_HUES]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v
