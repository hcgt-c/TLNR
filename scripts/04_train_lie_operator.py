# -*- coding: utf-8 -*-
"""
04 颜色李代数算子：训练与系统验证 v2（骨干特征上，色相按环处理）
========================================================
机制：T(p) = z + U·((exp(dp·λ)−1) ⊙ (Uᵀ z))，U∈R^{d×r}，λ=(a,b,c)。
修正 v1 的建模错误：色相是周期变量（S¹）。所有 hue 差取 **mod 差值**
（0..360 环上小弧 ∈ (−180,180]），即 θ 是"沿色相环的弧长增量"（加法参数，
配合指数算子 = 局部线性化的旋转）。这使 320→0 正确编码为 +40 而非 −320。

划分：
    训练：形状 4 × 色相 9（0..320 步 40）× 饱和 2 × 明度 2 —— cell 均值
    T2 跨形状：测试形状 pentagon/hexagon（整族形状训练不可见）
    T3 跨色相：训练形状 × 色相 {20,260}（从最近可见色相 0/40 或 240/280 转移）

基线：copy / globaladd（训练平均单位位移×Δh 的加性共享位移）。
报告：各测试面 rel_err 与相对增益；复合律检查（用学好的算子数值验证）。
"""
import os, sys, json, itertools
import numpy as np
import torch

OUT = os.path.dirname(os.path.abspath(__file__)) + "/.."
FEAT_DIR = os.path.join(OUT, "features")
RES_DIR = os.path.join(OUT, "results")
os.makedirs(RES_DIR, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SHAPES = ["triangle", "rectangle", "circle", "star"]
TEST_SHAPES = ["pentagon", "hexagon"]
TRAIN_HUES = [0, 40, 80, 120, 160, 200, 240, 280, 320]
TEST_HUES = [20, 260]
SATS = [0.5, 1.0]
VALS = [0.5, 1.0]


def load_cells(bb, layer):
    z = np.load(os.path.join(FEAT_DIR, f"feats_{bb}.npz"), allow_pickle=True)
    F, shape, hue, sat, val = z[f"feats_{layer}"], z["shape"], z["hue"], z["sat"], z["val"]
    cells = {}
    for sh, h, s, v in set(zip(shape, hue, sat, val)):
        m = (shape == sh) & (hue == h) & (sat == s) & (val == v)
        cells[(sh, h, s, v)] = F[m].mean(0)
    return cells


def hue_delta(h1, h2):
    """色相环小弧差 ∈ (−180,180]"""
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


def train(bb, layer, r=32, epochs=400, lr=2e-3, seed=0):
    cells = load_cells(bb, layer)
    train = {(sh, h, s, v): cells[(sh, h, s, v)]
             for (sh, h, s, v) in cells if sh in TRAIN_SHAPES and h in TRAIN_HUES}
    Z1s, Z2s, DPs = [], [], []
    n_skip = 0
    for (sh, h1, s1, v1), z1 in train.items():
        for (sh2, h2, s2, v2), z2 in train.items():
            if sh != sh2:
                continue
            dh = hue_delta(h1, h2)
            if abs(dh) < 20 or abs(dh) > 120:
                n_skip += 1
                continue
            dls = float(np.log(s2) - np.log(s1))
            dv = float(v2 - v1)
            Z1s.append(z1); Z2s.append(z2); DPs.append([dh, dls, dv])
    Z1 = torch.tensor(np.array(Z1s), dtype=torch.float64, device=DEV)
    Z2 = torch.tensor(np.array(Z2s), dtype=torch.float64, device=DEV)
    DP = torch.tensor(np.array(DPs), dtype=torch.float64, device=DEV)
    d = Z1.shape[1]
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64, device=DEV))[0]
    U.requires_grad_(True)
    a = torch.zeros(r, dtype=torch.float64, requires_grad=True, device=DEV)
    b = torch.zeros(r, dtype=torch.float64, requires_grad=True, device=DEV)
    c = torch.zeros(r, dtype=torch.float64, requires_grad=True, device=DEV)
    opt = torch.optim.Adam([U, a, b, c], lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        lam = DP[:, 0:1] * a + DP[:, 1:2] * b + DP[:, 2:3] * c
        W = U.t() @ Z1.t()
        y = Z1 + (U @ ((torch.exp(lam).t() - 1.0) * W)).t()
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
        hist.append(loss.item())
    print(f"[{bb} {layer}] pairs={len(Z1s)} skip={n_skip} "
          f"loss_final={loss.item():.6f} min={min(hist):.6f}", flush=True)
    return U.detach().cpu().numpy(), a.detach().cpu().numpy(), \
        b.detach().cpu().numpy(), c.detach().cpu().numpy()


def test_pairs_and_shift(bb, layer):
    cells = load_cells(bb, layer)
    train_cells = {(sh, h, s, v): cells[(sh, h, s, v)]
                   for (sh, h, s, v) in cells if sh in TRAIN_SHAPES and h in TRAIN_HUES}
    # 单位位移（每度）基线：训练形状、同 (s,v)、hue 差 40/80 的真实位移
    per_deg = []
    for (sh, h1, s, v), z1 in train_cells.items():
        for (sh2, h2, s2, v2), z2 in train_cells.items():
            if sh == sh2 and s == s2 and v == v2:
                dh = hue_delta(h1, h2)
                if dh in (40, 80):
                    per_deg.append((z2 - z1) / dh)
    g_shift = np.mean(per_deg, axis=0) if per_deg else np.zeros(cells[(TRAIN_SHAPES[0], TRAIN_HUES[0], 1.0, 1.0)].shape)

    def make(sh, h1, h2, s, v, mode):
        if (sh, h1, s, v) in cells and (sh, h2, s, v) in cells:
            return dict(mode=mode, z1=cells[(sh, h1, s, v)], z2=cells[(sh, h2, s, v)],
                        dh=hue_delta(h1, h2))
        return None
    pairs = []
    for sh in TEST_SHAPES:
        for s in SATS:
            for v in VALS:
                for h1 in TRAIN_HUES:
                    for step in (40, 80, 120):
                        p = make(sh, h1, (h1 + step) % 360, s, v, "T2_cross_shape")
                        if p:
                            pairs.append(p)
    for sh in TRAIN_SHAPES:
        for s in SATS:
            for v in VALS:
                for h2 in TEST_HUES:
                    for h1 in [(h2 - 40) % 360, (h2 - 20) % 360, (h2 + 20) % 360]:
                        if h1 in TRAIN_HUES:
                            p = make(sh, h1, h2, s, v, "T3_cross_hue")
                            if p:
                                pairs.append(p)
    return pairs, g_shift


def evaluate(bb, layer, U, a, b, c):
    pairs, g_shift = test_pairs_and_shift(bb, layer)
    stats = {}
    for mode in ["T2_cross_shape", "T3_cross_hue"]:
        grp = [p for p in pairs if p["mode"] == mode]
        e_lie, e_copy, e_gadd = [], [], []
        for p in grp:
            dp = [p["dh"], 0.0, 0.0]
            y = lie_predict_np(U, a, b, c, dp, p["z1"])
            e_lie.append(rel_err(y, p["z2"]))
            e_copy.append(rel_err(p["z1"], p["z2"]))
            e_gadd.append(rel_err(p["z1"] + g_shift * p["dh"], p["z2"]))
        stats[mode] = {
            "n": len(grp),
            "rel_err_lie": float(np.mean(e_lie)),
            "rel_err_copy": float(np.mean(e_copy)),
            "rel_err_globaladd": float(np.mean(e_gadd)),
            "lie_over_copy_gain": float(np.mean(e_copy) / (np.mean(e_lie) + 1e-30)),
            "lie_over_globaladd_gain": float(np.mean(e_gadd) / (np.mean(e_lie) + 1e-30)),
        }
    return stats


def composition_check(U, a, b, c):
    """复合律：T(dp1)∘T(dp2) vs T(dp1+dp2)，在随机向量上（不依赖数据）"""
    d = U.shape[0]
    rng = np.random.RandomState(0)
    errs = []
    for _ in range(200):
        z = rng.randn(d)
        dp1 = rng.uniform(-60, 60) * np.array([1.0, 0.0, 0.0])
        dp2 = rng.uniform(-60, 60) * np.array([1.0, 0.0, 0.0])
        y_seq = lie_predict_np(U, a, b, c, dp2, lie_predict_np(U, a, b, c, dp1, z))
        y_one = lie_predict_np(U, a, b, c, dp1 + dp2, z)
        errs.append(np.linalg.norm(y_seq - y_one) / (np.linalg.norm(y_one) + 1e-12))
    return float(np.mean(errs))


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    layer = sys.argv[2] if len(sys.argv) > 2 else "l1"
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    U, a, b, c = train(bb, layer, r=r)
    ev = evaluate(bb, layer, U, a, b, c)
    ev["composition_err"] = composition_check(U, a, b, c)
    print(json.dumps(ev, indent=1))
    json.dump(ev, open(os.path.join(RES_DIR, f"mechanism_{bb}_{layer}_r{r}.json"), "w"), indent=1)
    np.savez(os.path.join(RES_DIR, f"op_{bb}_{layer}_r{r}.npz"), U=U, a=a, b=b, c=c)
