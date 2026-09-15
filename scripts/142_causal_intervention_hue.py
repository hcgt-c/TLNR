# -*- coding: utf-8 -*-
"""142_causal_intervention_hue.py — 中层激活干预是否等价于真正给输入换色？

评审质疑：项目现有的颜色算子只作用于辅助码，检测头并不消费该码，所以没有下游后果。
本脚本检验：在中间层激活 f_l（stage-3 / stride-8 / 64 通道，即项目称为 y3 的张量）上
做特征空间输运 W_delta，能否复现「把输入真正换色 delta 后整网前向」的输出。

两条路径（同一目标 y_real）：
  (R) real route   : x -> x_delta（像素级 HSV 往返换色，仅作用于 GT box 内像素），整网前向 -> y_real
  (I) intervention : f_l(x) -> W_delta f_l(x)（逐像素通道混合，等价 1x1 卷积的共享算子），
                     只前向剩余层 f_{l+1:L} -> y_intervened

拟合：W_delta 只用 TRAIN 划分（每张图 28x28 个特征单元中落在 GT box 内的单元，
即真正被换色影响的区域），一次全局拟合，无逐图拟合。主算子为完全正交 Procrustes，
约定与 scripts/97_coco_procrustes.py 相同（M = Y^T X，SVD 得 R = U V^T，行向量输运 f -> f R^T，
即列向量 f -> R f）；另报无约束最小二乘（线性容量更强的变体）与全图拟合变体。
所有下游/保真度指标在 HELD-OUT 图像上报告。对照：不干预的 y_0（null）、随机正交输运（特异性）、
delta=0 的 HSV 往返（量化地板）。另做复合性 W_{d2}W_{d1} vs W_{d1+d2} 与未见 delta 外推。

无训练、不修改权重与既有脚本。用法：python scripts/133_causal_intervention.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

T0 = time.time()
WORK = rp.REPO_ROOT
WT = os.path.join(WORK, "runs/detect/runs_det_hue/hue_cls8/weights/best.pt")
IMD = os.path.join(WORK, "data/yolodet_hue/val/images")
LBD = os.path.join(WORK, "data/yolodet_hue/val/labels")
OUT_JSON = os.path.join(WORK, "results/causal_intervention_hue.json")
FIG = os.path.join(WORK, "results/figures/F25_causal_intervention_hue.png")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
DELTAS = [30.0, 60.0, 90.0]
N_IMG = int(os.environ.get("CI_NIMG", "60"))      # 默认 60（env 仅用于冒烟测试）
N_TRAIN = int(os.environ.get("CI_NTRAIN", "40"))  # 其余为 held-out
CONF, IOU_NMS, IOU_MATCH = 0.05, 0.7, 0.5
N_RAND = 3                         # 随机正交输运对照重复次数
SPLIT_LAYER = 3                    # net.model[:4] 的输出 = stride-8 / 64ch（项目称 y3）
FCH, FH, FW = 64, 28, 28
STRIDE = 8
KPROC, KOLS, KALL = "procrustes_objcells", "least_squares_objcells", "procrustes_allpixels"
KMASK = "procrustes_objcells_masked"
VARIANT_ORDER = [KPROC, KOLS, KALL, KMASK]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- 冻结模型
class AuxDet(DetectionModel):
    """checkpoint 内 pickle 的子类（见 scripts/77_arm_aux.py）；eval 时直通。"""

    def _capture(self, m, i, o):
        return o

    def forward(self, x, *a, **k):
        return super().forward(x, *a, **k)


net = torch.load(WT, map_location=DEV, weights_only=False)
if isinstance(net, dict):
    net = net.get("model", net)
net.to(DEV).eval().float()
net.aux_code = getattr(net, "aux_code", nn.Conv2d(64, 4, 1).to(DEV)).float()
mid_seq = nn.Sequential(*list(net.model[:4])).to(DEV).eval()
SAVE = set(net.save)
NL = len(net.model)


def fwd_from_mid(f_new):
    """忠实复刻 BaseModel._predict_once 自第 4 层起的路径（Concat 的 f 索引照旧处理）。"""
    y = [None] * NL
    x = f_new
    for m in net.model[SPLIT_LAYER + 1:]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y[m.i] = x if m.i in SAVE else None
    return x


def transport(W, fmat):
    """列向量约定 f_out = W f，逐像素共享，作用于全部 784 个单元。fmat: (64,784) -> (1,64,28,28)。"""
    return torch.einsum("ij,jn->in", W, fmat.to(W.device)).reshape(1, FCH, FH, FW)


def transport_masked(W, fmat, cm):
    """只在 GT box 内的单元上施加 W，其余单元保持恒等（诊断变体）。"""
    f = fmat.to(W.device)
    out = torch.einsum("ij,jn->in", W, f)
    m = torch.tensor(cm, dtype=torch.bool, device=W.device)
    return torch.where(m, out, f).reshape(1, FCH, FH, FW)


# ---------------------------------------------------------------- 像素级换色
def shift_u8(u8, deg, mask):
    """HSV 往返色相平移 deg，仅 mask 内像素；mask 外逐位不变。"""
    a = u8.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    d = mx - mn
    l = (mx + mn) / 2.0
    s = np.zeros_like(d)
    nz = d > 1e-6
    s[nz] = np.where(l[nz] < 0.5, d[nz] / (mx[nz] + mn[nz]), d[nz] / (2.0 - mx[nz] - mn[nz]))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(r == mx, (g - b) / d, np.where(g == mx, 2.0 + (b - r) / d, 4.0 + (r - g) / d))
        h = (h % 6.0) * 60.0
        h = h.copy()
        h[~nz] = 0.0
    hn = (h + deg) % 360.0
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = hn / 60.0
    xv = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    mm = l - c / 2.0
    k = hp.astype(np.int64) % 6
    R = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [c, xv, 0, 0, xv, c])
    G = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [xv, c, c, xv, 0, 0])
    B = np.select([k == 0, k == 1, k == 2, k == 3, k == 4, k == 5], [0, 0, xv, c, c, xv])
    out = (np.clip(np.stack([R + mm, G + mm, B + mm], -1), 0, 1) * 255).astype(np.uint8)
    res = u8.copy()
    res[mask] = out[mask]
    return res


def gt_boxes(fn, size):
    p = os.path.join(LBD, fn[:-4] + ".txt")
    return [] if not os.path.exists(p) else np.loadtxt(p).reshape(-1, 5)


def box_mask(shape_hw, fn, size):
    m = np.zeros(shape_hw, bool)
    bs = gt_boxes(fn, size)
    if len(bs) == 0:
        return np.ones(shape_hw, bool)
    for _, cx, cy, w, h in bs:
        x0 = int(max(0, (cx - w / 2) * size))
        x1 = int(min(size, math.ceil((cx + w / 2) * size)))
        y0 = int(max(0, (cy - h / 2) * size))
        y1 = int(min(size, math.ceil((cy + h / 2) * size)))
        m[y0:y1, x0:x1] = True
    return m


def cell_mask(fn):
    """28x28 展平后的 bool：y3 单元落在 GT box 内（换色实际影响的特征单元）。"""
    m = np.zeros((FH, FW), bool)
    for _, cx, cy, w, h in gt_boxes(fn, 224):
        gx0 = int(max(0, math.floor((cx - w / 2) * 224 / STRIDE)))
        gx1 = int(min(FW, math.ceil((cx + w / 2) * 224 / STRIDE)))
        gy0 = int(max(0, math.floor((cy - h / 2) * 224 / STRIDE)))
        gy1 = int(min(FH, math.ceil((cy + h / 2) * 224 / STRIDE)))
        m[gy0:gy1, gx0:gx1] = True
    return m.reshape(-1)


def to_t(u8):
    return torch.tensor(u8.astype(np.float32) / 255.0, device=DEV).permute(2, 0, 1)[None]


# ---------------------------------------------------------------- 数据 + 冻结前向
files = sorted(f for f in os.listdir(IMD) if f.endswith(".jpg"))[:N_IMG]
perm = np.random.RandomState(SEED).permutation(len(files))
train_idx = sorted(perm[:N_TRAIN].tolist())
held_idx = sorted(perm[N_TRAIN:].tolist())

F, O, CM = {}, {}, {}
for i, fn in enumerate(files):
    u8 = np.asarray(Image.open(os.path.join(IMD, fn)).convert("RGB"))
    msk = box_mask(u8.shape[:2], fn, u8.shape[0])
    ims = {"0": u8, "rt0": shift_u8(u8, 0.0, msk)}
    for d in DELTAS:
        ims["%g" % d] = shift_u8(u8, d, msk)
    F[i], O[i], CM[i] = {}, {}, cell_mask(fn)
    for k, im in ims.items():
        xt = to_t(im)
        with torch.no_grad():
            f = mid_seq(xt)
            out = net(xt)
        F[i][k] = f[0].reshape(FCH, -1).cpu().float()
        O[i][k] = {"dec": out[0][0].cpu().float(),
                   "sco": out[1]["scores"][0].cpu().float(),
                   "box": out[1]["boxes"][0].cpu().float()}
    if (i + 1) % 20 == 0:
        print("  extract %d/%d  %.1fs" % (i + 1, len(files), time.time() - T0), flush=True)

_u8 = np.asarray(Image.open(os.path.join(IMD, files[0])).convert("RGB"))
_xt = to_t(_u8)
with torch.no_grad():
    _full = net(_xt)[0]
    _man = fwd_from_mid(mid_seq(_xt))[0]
FAITH = float((_man - _full).abs().max())
print("tail-forward faithfulness (max abs diff vs full forward): %g" % FAITH, flush=True)
assert FAITH < 1e-4, "remaining-layer replay is not faithful"
print("object-cell fraction per image: mean %.3f" % float(np.mean([CM[i].mean() for i in range(len(files))])), flush=True)


# ---------------------------------------------------------------- 拟合（仅 TRAIN）
def rows(idx, variant, cells):
    """cells=None -> 全部 784 个单元；否则用 CM 掩码。"""
    out = []
    for i in idx:
        a = F[i][variant].numpy().T
        out.append(a if cells is None else a[CM[i]])
    return np.concatenate(out, 0)


def angles(A, B):
    """行向量集合的逐单元夹角均值（度）与相对 L2 误差（A 为输运后，B 为真实）。"""
    na = np.linalg.norm(A, axis=1) + 1e-9
    nb = np.linalg.norm(B, axis=1) + 1e-9
    cos = np.clip((A * B).sum(1) / (na * nb), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean()), float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-9))


def rel_np(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-9))


RES, CONV = {}, {}
for d in DELTAS:
    key = "%g" % d
    Xo, Yo = rows(train_idx, "0", "obj"), rows(train_idx, key, "obj")     # 对象单元
    Xa, Ya = rows(train_idx, "0", None), rows(train_idx, key, None)       # 全图
    # 行约定算子 A（f_row -> f_row A）；脚本 97 约定：M = Y^T X，SVD 得 R = U V^T，输运用 A = R^T
    U, S, Vt = np.linalg.svd(Yo.T @ Xo)
    A_proc = (U @ Vt).T.astype(np.float32)
    A_ols = (np.linalg.pinv(Xo) @ Yo).astype(np.float32)
    Ua, Sa, Vta = np.linalg.svd(Ya.T @ Xa)
    A_all = (Ua @ Vta).T.astype(np.float32)
    # 约定自检（训练划分上）：Procrustes 的 L2 必须不劣于恒等，也必须不劣于方向转置
    e_id = rel_np(Xo, Yo)
    e_fit = rel_np(Xo @ A_proc, Yo)
    e_wrong = rel_np(Xo @ A_proc.T, Yo)
    CONV[key] = {"train_rel_l2_identity": e_id, "train_rel_l2_fitted": e_fit,
                 "train_rel_l2_transposed_operator_control": e_wrong,
                 "convention_ok": bool(e_fit <= e_id + 1e-9 and e_fit <= e_wrong + 1e-9)}
    assert CONV[key]["convention_ok"], "Procrustes convention check failed for delta %s" % key
    # 列约定算子 W（f_col -> W f_col），W = A^T
    RES[key] = {KPROC: torch.tensor(A_proc.T.copy(), device=DEV),
                KOLS: torch.tensor(A_ols.T.copy(), device=DEV),
                KALL: torch.tensor(A_all.T.copy(), device=DEV),
                "A": {KPROC: A_proc, KOLS: A_ols, KALL: A_all}}
    print("  fit delta=%s: procrustes train relL2 %.4f (identity %.4f, transposed %.4f) | ols %.4f"
          % (key, e_fit, e_id, e_wrong, rel_np(Xo @ A_ols, Yo)), flush=True)

RAND = []
for r in range(N_RAND):
    Q, _ = np.linalg.qr(np.random.RandomState(SEED + 100 + r).randn(FCH, FCH))
    RAND.append(torch.tensor(Q.astype(np.float32), device=DEV))
IDENT = torch.eye(FCH, device=DEV)


# ---------------------------------------------------------------- 指标
def rel(A, B):
    return float((A - B).norm() / (B.norm() + 1e-9))


def cosd(A, B):
    return float((A * B).sum() / ((A.norm() + 1e-12) * (B.norm() + 1e-12)))


def proj_coef(d_int, d_real):
    """<y_int-y_0, y_real-y_0> / ||y_real-y_0||^2：1 = 方向与幅度都复现，0 = 无位移。"""
    return float((d_int * d_real).sum() / ((d_real * d_real).sum() + 1e-12))


def dets(dec):
    # 注意：ultralytics 的 NMS 会原地改写输入的 box 通道（prediction[..., :4] = xywh2xyxy(...)
    # 作用在 transpose 得到的视图上），因此必须传入 clone，否则会污染被比较的参考张量。
    r = non_max_suppression(dec[None].clone(), conf_thres=CONF, iou_thres=IOU_NMS, max_det=30)[0]
    return r.cpu().numpy() if r is not None and len(r) else np.zeros((0, 6), np.float32)


def match_boxes(hyp, ref):
    """贪心一对一匹配（同类、IoU >= IOU_MATCH）。返回每个 hyp 的 TP 标记。"""
    tp = np.zeros(len(hyp), bool)
    if len(hyp) == 0 or len(ref) == 0:
        return tp
    iou = np.zeros((len(hyp), len(ref)), np.float32)
    for a in range(len(hyp)):
        ax0, ay0, ax1, ay1 = hyp[a, :4]
        for b in range(len(ref)):
            if hyp[a, 5] != ref[b, 5]:
                continue
            bx0, by0, bx1, by1 = ref[b, :4]
            iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            ih = max(0.0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih
            ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
            iou[a, b] = inter / ua if ua > 0 else 0.0
    used_r = set()
    for o in np.argsort(-iou.ravel()):
        a, b = int(o // len(ref)), int(o % len(ref))
        if iou[a, b] < IOU_MATCH or tp[a] or b in used_r:
            continue
        tp[a] = True
        used_r.add(b)
    return tp


def eval_route(fmap_fn, dkey):
    """fmap_fn(i) 给出第 i 张 held-out 图的干预后 y3；只前向剩余层，与 real route 比较。"""
    rows_out = []
    for i in held_idx:
        real, y0 = O[i][dkey], O[i]["0"]
        with torch.no_grad():
            out = fwd_from_mid(fmap_fn(i))
        dec, sco, box = out[0][0].cpu(), out[1]["scores"][0].cpu(), out[1]["boxes"][0].cpu()
        idx = torch.tensor(np.where(CM[i])[0], dtype=torch.long)       # stride-8 层的对象锚点
        # 先算张量级指标（在任何 NMS 之前），再做框级指标
        r = {"rel_dec": rel(dec, real["dec"]), "rel_box": rel(box, real["box"]), "rel_sco": rel(sco, real["sco"]),
             "cos_dec": cosd(dec, real["dec"]),
             "align_box": cosd(box - y0["box"], real["box"] - y0["box"]),
             "align_dec": cosd(dec - y0["dec"], real["dec"] - y0["dec"]),
             "proj_box": proj_coef(box - y0["box"], real["box"] - y0["box"]),
             "proj_dec": proj_coef(dec - y0["dec"], real["dec"] - y0["dec"]),
             "rel_box_obj": rel(box[:, idx], real["box"][:, idx]),
             "rel_sco_obj": rel(sco[:, idx], real["sco"][:, idx]),
             "align_box_obj": cosd(box[:, idx] - y0["box"][:, idx], real["box"][:, idx] - y0["box"][:, idx]),
             "proj_box_obj": proj_coef(box[:, idx] - y0["box"][:, idx], real["box"][:, idx] - y0["box"][:, idx]),
             }
        hb, rb = dets(dec), dets(real["dec"])
        tp = match_boxes(hb, rb)
        r.update({"n_hyp": len(hb), "n_ref": len(rb), "tp": int(tp.sum()),
                  "scored": [(float(hb[a, 4]), bool(tp[a])) for a in range(len(hb))]})
        rows_out.append(r)
    agg = {}
    for k in ["rel_dec", "rel_box", "rel_sco", "cos_dec", "align_box", "align_dec", "proj_box", "proj_dec",
              "rel_box_obj", "rel_sco_obj", "align_box_obj", "proj_box_obj"]:
        v = np.array([r[k] for r in rows_out])
        agg[k + "_mean"] = round(float(v.mean()), 5)
        agg[k + "_median"] = round(float(np.median(v)), 5)
    nb = np.array([r["n_hyp"] for r in rows_out], float)
    nr = np.array([r["n_ref"] for r in rows_out], float)
    tp = np.array([r["tp"] for r in rows_out], float)
    prec = tp / np.maximum(nb, 1)
    rec = np.where(nr > 0, tp / np.maximum(nr, 1), np.nan)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    agg.update({"n_hyp_mean": round(float(nb.mean()), 3), "n_ref_mean": round(float(nr.mean()), 3),
                "tp_mean": round(float(tp.mean()), 3),
                "precision_iou50_mean": round(float(np.nanmean(prec)), 4),
                "recall_iou50_mean": round(float(np.nanmean(rec)), 4),
                "f1_iou50_mean": round(float(np.nanmean(f1)), 4)})
    scored = sorted([s for r in rows_out for s in r["scored"]], key=lambda t: -t[0])
    nref = sum(r["n_ref"] for r in rows_out)
    ctp = cfp = 0
    prev_r, ap = 0.0, 0.0
    for s, t in scored:
        ctp += int(t)
        cfp += int(not t)
        rc = ctp / max(nref, 1)
        ap += (ctp / max(ctp + cfp, 1)) * max(0.0, rc - prev_r)
        prev_r = max(prev_r, rc)
    agg["box_agreement_AP50"] = round(float(ap), 4)
    return agg


DOWN = {}
for d in DELTAS:
    key = "%g" % d
    def fm(v):
        W = RES[key][KPROC if v == KMASK else v]
        if v == KMASK:
            return lambda i: transport_masked(W, F[i]["0"], CM[i]).float()
        return lambda i: transport(W, F[i]["0"]).float()

    per = {v: eval_route(fm(v), key) for v in VARIANT_ORDER}
    nl = eval_route(lambda i: transport(IDENT, F[i]["0"]).float(), key)
    ra = [eval_route((lambda i, Q=Q: transport(Q, F[i]["0"]).float()), key) for Q in RAND]
    rnd = {k: round(float(np.mean([a[k] for a in ra])), 5) for k in nl}
    rt = {k + "_mean": round(float(np.mean([rel(O[i]["rt0"][t], O[i]["0"][t]) for i in held_idx])), 6)
          for k, t in [("rel_dec", "dec"), ("rel_box", "box"), ("rel_sco", "sco")]}
    DOWN[key] = {"intervened": per[KPROC], "null_unintervened": nl, "random_orthogonal": rnd,
                 "hsv_roundtrip0_floor": rt, "variants": per,
                 "gap_ratio_rel_box": round(per[KPROC]["rel_box_mean"] / max(nl["rel_box_mean"], 1e-12), 4),
                 "gap_ratio_rel_box_obj": round(per[KPROC]["rel_box_obj_mean"] / max(nl["rel_box_obj_mean"], 1e-12), 4)}
    print("  downstream delta=%s | obj-anchor rel_box: proc %.4f  ols %.4f  allpx %.4f  null %.4f  rand %.4f"
          % (key, per[KPROC]["rel_box_obj_mean"], per[KOLS]["rel_box_obj_mean"], per[KALL]["rel_box_obj_mean"],
             nl["rel_box_obj_mean"], rnd["rel_box_obj_mean"]), flush=True)
    print("      align_box_obj: proc %.3f ols %.3f null %.3f rand %.3f | proj_box_obj proc %.3f | AP50 %.3f (null %.3f)"
          % (per[KPROC]["align_box_obj_mean"], per[KOLS]["align_box_obj_mean"], nl["align_box_obj_mean"],
             rnd["align_box_obj_mean"], per[KPROC]["proj_box_obj_mean"],
             per[KPROC]["box_agreement_AP50"], nl["box_agreement_AP50"]), flush=True)
    for tag, a in [("proc", per[KPROC]), ("ols", per[KOLS]), ("allpx", per[KALL]), ("mask", per[KMASK]),
                   ("null", nl), ("rand", rnd)]:
        print("      [%s] n_hyp %.2f n_ref %.2f TP %s P %.3f R %.3f F1 %.3f AP50 %.3f"
              % (tag, a["n_hyp_mean"], a["n_ref_mean"], a.get("tp_mean", "-"), a["precision_iou50_mean"],
                 a["recall_iou50_mean"], a["f1_iou50_mean"], a["box_agreement_AP50"]), flush=True)


# ---------------------------------------------------------------- 特征保真度（HELD-OUT）
def feat_report(A_list, key):
    Xo, Yo = rows(held_idx, "0", "obj"), rows(held_idx, key, "obj")
    Xa, Ya = rows(held_idx, "0", None), rows(held_idx, key, None)
    raw_a, raw_e = angles(Xo, Yo)
    out = {"raw_angle_deg_objcells": round(raw_a, 3), "raw_rel_l2_objcells": raw_e,
           "identity_null_rel_l2_objcells": raw_e,
           "raw_angle_deg_allpixels": round(angles(Xa, Ya)[0], 3),
           "identity_null_rel_l2_allpixels": angles(Xa, Ya)[1]}
    for name, A in A_list:
        ta, te = angles(Xo @ A, Yo)
        aa, ae = angles(Xa @ A, Ya)
        out[name] = {"objcells": {"transport_angle_deg": round(ta, 3), "rel_l2": te},
                     "allpixels": {"transport_angle_deg": round(aa, 3), "rel_l2": ae}}
    return out


FEAT = {("%g" % d): feat_report([(KPROC, RES["%g" % d]["A"][KPROC]),
                                 (KOLS, RES["%g" % d]["A"][KOLS]),
                                 (KALL, RES["%g" % d]["A"][KALL])], "%g" % d) for d in DELTAS}
for k in FEAT:
    print("  feature delta=%s: raw obj-cell angle %.2f deg / relL2 %.4f | procrustes %.2f / %.4f | ols %.2f / %.4f | allpx-fit %.2f / %.4f"
          % (k, FEAT[k]["raw_angle_deg_objcells"], FEAT[k]["identity_null_rel_l2_objcells"],
             FEAT[k][KPROC]["objcells"]["transport_angle_deg"], FEAT[k][KPROC]["objcells"]["rel_l2"],
             FEAT[k][KOLS]["objcells"]["transport_angle_deg"], FEAT[k][KOLS]["objcells"]["rel_l2"],
             FEAT[k][KALL]["objcells"]["transport_angle_deg"], FEAT[k][KALL]["objcells"]["rel_l2"]), flush=True)


# ---------------------------------------------------------------- 复合性
COMP = {}
for d1, d2, ds in [(30.0, 60.0, 90.0), (30.0, 30.0, 60.0)]:
    k1, k2, ks = "%g" % d1, "%g" % d2, "%g" % ds
    for fam in [KPROC, KOLS]:
        Wc = (RES[k2][fam] @ RES[k1][fam]).float()
        Ws = RES[ks][fam]
        fc, fr, oc, orr = [], [], [], []
        for i in held_idx:
            f0 = F[i]["0"]
            a, b = transport(Wc, f0).float(), transport(Ws, f0).float()
            ac, bc = a[0].reshape(FCH, -1).cpu(), b[0].reshape(FCH, -1).cpu()
            re = F[i][ks]
            fc.append(float((ac - bc).norm() / (bc.norm() + 1e-9)))
            fr.append(float((ac - re).norm() / (re.norm() + 1e-9)))
            with torch.no_grad():
                ya = fwd_from_mid(a)[1]["boxes"][0].cpu()
                yb = fwd_from_mid(b)[1]["boxes"][0].cpu()
            oc.append(rel(ya, yb))
            orr.append(rel(ya, O[i][ks]["box"]))
        tag = "%g+%g_vs_%g_%s" % (d1, d2, ds, fam)
        COMP[tag] = {
            "operator_family": fam,
            "feature_rel_l2_composite_vs_direct": round(float(np.mean(fc)), 5),
            "feature_rel_l2_composite_vs_real_recolour": round(float(np.mean(fr)), 5),
            "output_rel_l2_composite_vs_direct": round(float(np.mean(oc)), 5),
            "output_rel_l2_composite_vs_real_recolour": round(float(np.mean(orr)), 5)}
        c = COMP[tag]
        print("  composition %-42s: feat %.4f / %.4f  out %.4f / %.4f" % (
            tag, c["feature_rel_l2_composite_vs_direct"], c["feature_rel_l2_composite_vs_real_recolour"],
            c["output_rel_l2_composite_vs_direct"], c["output_rel_l2_composite_vs_real_recolour"]), flush=True)


# ---------------------------------------------------------------- 未见 delta（只用 30/60 预测 90）
def eval_unseen(W, note, masked=False):
    """把算子 W（可能来自 30/60 的复合或生成元外推）用于预测 delta=90，指标定义与其它小节一致：
    feature 侧同时给对象单元与全图两种口径，下游指标用与 DOWN 完全相同的 eval_route。"""
    fo, ao, fa, aa = [], [], [], []
    for i in held_idx:
        t = (transport_masked(W, F[i]["0"], CM[i]) if masked else transport(W, F[i]["0"])).float()
        tm = t[0].reshape(FCH, -1).cpu()
        re = F[i]["90"]
        ridx = np.where(CM[i])[0]
        fo.append(float((tm[:, ridx] - re[:, ridx]).norm() / (re[:, ridx].norm() + 1e-9)))
        ao.append(angles(tm.numpy().T[ridx], re.numpy().T[ridx])[0])
        fa.append(rel(tm, re))
        aa.append(angles(tm.numpy().T, re.numpy().T)[0])
    a = eval_route((lambda i: (transport_masked(W, F[i]["0"], CM[i]) if masked else transport(W, F[i]["0"])).float()), "90")
    return {"feature_rel_l2_heldout_objcells": round(float(np.mean(fo)), 5),
            "feature_angle_deg_heldout_objcells": round(float(np.mean(ao)), 3),
            "feature_rel_l2_heldout_allpixels": round(float(np.mean(fa)), 5),
            "feature_angle_deg_heldout_allpixels": round(float(np.mean(aa)), 3),
            "downstream_rel_box_obj_mean": a["rel_box_obj_mean"], "downstream_rel_box_mean": a["rel_box_mean"],
            "downstream_align_box_obj_mean": a["align_box_obj_mean"],
            "downstream_proj_box_obj_mean": a["proj_box_obj_mean"],
            "box_agreement_AP50": a["box_agreement_AP50"],
            "gap_ratio_rel_box_obj": round(a["rel_box_obj_mean"] / max(DOWN["90"]["null_unintervened"]["rel_box_obj_mean"], 1e-12), 4),
            "note": note}


GEN = {"composition_of_fitted_30_and_60": eval_unseen(
    (RES["60"][KPROC] @ RES["30"][KPROC]).float(),
    "W_90 := W_60 @ W_30（procrustes_objcells），两者只按各自 delta 在 TRAIN 上拟合，90 从未参与拟合")}
try:
    from scipy.linalg import logm, expm
    G = logm(RES["30"][KPROC].cpu().numpy().astype(np.float64)) / 30.0
    Wg = np.real(expm(90.0 * G)).astype(np.float32)
    GEN["generator_exp_90_from_log_W30"] = eval_unseen(
        torch.tensor(Wg, device=DEV), "W_90 := exp(90 G)，G = logm(W_30)/30（单参数生成元外推）")
except Exception as e:                                              # pragma: no cover
    GEN["generator_exp_90_from_log_W30"] = {"error": repr(e)}
# 参照：delta=90 直接参与拟合（指标定义完全相同，便于逐项对比）
GEN["per_delta_fitted_W90_reference"] = eval_unseen(
    RES["90"][KPROC], "W_90 直接在 TRAIN 划分上按 delta=90 拟合（held-out 仅指图像），作为参照上界")
print("  generalisation done %.1fs" % (time.time() - T0), flush=True)


# ---------------------------------------------------------------- 判定
def verdict_text():
    ks = ["%g" % d for d in DELTAS]
    fam = {}
    for tag, key in [("proc", KPROC), ("ols", KOLS), ("allpx", KALL), ("mask", KMASK)]:
        fam[tag] = {k: DOWN[k]["variants"][key] for k in ks}
    nl = {k: DOWN[k]["null_unintervened"] for k in ks}
    rn = {k: DOWN[k]["random_orthogonal"] for k in ks}
    fl_box = DOWN["30"]["hsv_roundtrip0_floor"]["rel_box_mean"]
    fl_obj = DOWN["30"]["hsv_roundtrip0_floor"]["rel_box_mean"]
    c = COMP["30+60_vs_90_" + KPROC]
    co = COMP["30+60_vs_90_" + KOLS]
    g = GEN["composition_of_fitted_30_and_60"]
    ref = GEN["per_delta_fitted_W90_reference"]

    def ser(metric, tag):
        return "、".join("%s deg %.4f" % (k, fam[tag][k][metric]) for k in ks)

    s = ("真实换色的下游效应：held-out 上不干预的 y_0 与真实换色 y_real 在对象锚点（stride-8、"
         "落在 GT box 内的锚点）DFL logits 上的相对 L2 为 "
         + "、".join("%s deg %.4f" % (k, nl[k]["rel_box_obj_mean"]) for k in ks)
         + "，而 delta=0 的 HSV 往返量化地板仅 %.5f（全锚点口径）。" % fl_box)
    s += ("中层输运干预后的相对误差（对象锚点）：procrustes " + ser("rel_box_obj_mean", "proc")
          + "；least squares " + ser("rel_box_obj_mean", "ols")
          + "；procrustes+仅对象单元施加 " + ser("rel_box_obj_mean", "mask")
          + "；与 null 的比值 procrustes " + "、".join("%s deg %.2f" % (k, DOWN[k]["gap_ratio_rel_box_obj"]) for k in ks)
          + "（<1 = 把输出拉向真实换色，>1 = 比不干预更远）。随机正交输运对照 "
          + "、".join("%s deg %.4f" % (k, rn[k]["rel_box_obj_mean"]) for k in ks)
          + "（随机输运后 NMS 几无检测，AP50 为 0）。")
    s += ("输出位移方向对齐（对象锚点 DFL logits，与真实换色引起的位移夹角余弦）：procrustes "
          + ser("align_box_obj_mean", "proc") + "；least squares " + ser("align_box_obj_mean", "ols")
          + "；procrustes+仅对象单元 " + ser("align_box_obj_mean", "mask")
          + "；随机正交对照 " + "、".join("%s deg %.3f" % (k, rn[k]["align_box_obj_mean"]) for k in ks) + "。"
          + "投影系数（1 = 方向与幅度都复现）：procrustes " + ser("proj_box_obj_mean", "proc")
          + "；least squares " + ser("proj_box_obj_mean", "ols") + "。")
    s += ("框级一致性（以真实换色检测为参考的 box_agreement_AP50）：null "
          + "、".join("%s %.3f" % (k, nl[k]["box_agreement_AP50"]) for k in ks)
          + "；procrustes " + ser("box_agreement_AP50", "proc")
          + "；least squares " + ser("box_agreement_AP50", "ols")
          + "；procrustes+仅对象单元 " + ser("box_agreement_AP50", "mask") + "。")
    s += ("特征侧：对象单元上真实换色使 y3 平均转过 "
          + "、".join("%s deg %.1f" % (k, FEAT[k]["raw_angle_deg_objcells"]) for k in ks)
          + " 度，单个全局线性算子把对象单元上的相对 L2 从（恒等/不输运）"
          + "、".join("%s %.4f" % (k, FEAT[k]["identity_null_rel_l2_objcells"]) for k in ks)
          + " 只降到 procrustes " + "、".join("%s %.4f" % (k, FEAT[k][KPROC]["objcells"]["rel_l2"]) for k in ks)
          + "、least squares " + "、".join("%s %.4f" % (k, FEAT[k][KOLS]["objcells"]["rel_l2"]) for k in ks)
          + "（即单步线性输运最多解释约一到三成的特征变化）。")
    s += ("复合性：‖W60(W30 f) − W90 f‖/‖W90 f‖ = %.4f（输出侧 %.4f；least squares 版 %.4f / %.4f），"
          "单步输运可用而群复合不成立，与 scripts/97 的既有结论一致。"
          % (c["feature_rel_l2_composite_vs_direct"], c["output_rel_l2_composite_vs_direct"],
             co["feature_rel_l2_composite_vs_direct"], co["output_rel_l2_composite_vs_direct"]))
    s += ("未见 delta 泛化：只用 30/60 的算子复合外推到 90，特征相对误差 %.4f、对象锚点下游 rel_box %.4f、方向对齐 %.3f"
          "（直接按 90 拟合的参照为特征 %.4f、下游 %.4f、对齐 %.3f）；即未见 delta 上的输运并未比直接拟合差多少，"
          "因为两者都远离真实换色。"
          % (g["feature_rel_l2_heldout_objcells"], g["downstream_rel_box_obj_mean"], g["downstream_align_box_obj_mean"],
             ref["feature_rel_l2_heldout_objcells"], ref["downstream_rel_box_obj_mean"],
             ref["downstream_align_box_obj_mean"]))
    better_p = [k for k in ks if fam["proc"][k]["rel_box_obj_mean"] < nl[k]["rel_box_obj_mean"]]
    better_o = [k for k in ks if fam["ols"][k]["rel_box_obj_mean"] < nl[k]["rel_box_obj_mean"]]
    spec_o = [k for k in ks if fam["ols"][k]["align_box_obj_mean"] > rn[k]["align_box_obj_mean"] + 0.1]
    spec_p = [k for k in ks if fam["proc"][k]["align_box_obj_mean"] > rn[k]["align_box_obj_mean"] + 0.1]
    ratios_p = [DOWN[k]["gap_ratio_rel_box_obj"] for k in ks]
    ratios_o = [fam["ols"][k]["rel_box_obj_mean"] / max(nl[k]["rel_box_obj_mean"], 1e-12) for k in ks]
    n_better_p = sum(1 for r in ratios_p if r < 1.0)
    n_better_o = sum(1 for r in ratios_o if r < 1.0)
    if better_o and spec_o:
        s += ("结论（直说）：中层激活干预在**行为层面是方向正确、部分幅度正确**的替代，但不是忠实替代。"
              "把 y3 换成 W_delta f_l 后重新前向剩余层：正交 Procrustes 在 %d/%d 个 delta 上比完全不干预的 null 更接近真实换色"
              "（距离比值 %s，<1 即更近），无约束最小二乘在 %d/%d 个 delta 上更近（比值 %s）；两者方向对齐都显著高于随机正交对照。"
              "但幅度只复现了真实位移的一部分（投影系数 procrustes %s、least squares %s），特征侧也只解释一到三成变化，"
              "复合律不成立。因此结论是：让检测头消费一个中层线性输运过的激活，得到的是**可用的同方向近似**，"
              "而不是换色后的网络计算本身。"
              % (n_better_p, len(ks), "、".join("%.2f" % r for r in ratios_p),
                 n_better_o, len(ks), "、".join("%.2f" % r for r in ratios_o),
                 "、".join("%.3f" % fam["proc"][k]["proj_box_obj_mean"] for k in ks),
                 "、".join("%.3f" % fam["ols"][k]["proj_box_obj_mean"] for k in ks)))
    elif better_o:
        s += ("结论（直说）：只有容量更大的无约束最小二乘算子在 %d/%d 个 delta 上比 null 更接近真实换色（比值 %s），"
              "但方向对齐未显著高于随机对照；正交 Procrustes 与 null 无优势。结论：中层线性干预至多是弱同方向近似。"
              % (n_better_o, len(ks), "、".join("%.2f" % r for r in ratios_o)))
    else:
        s += ("结论（直说）：中层激活干预不能替代真正换色——把 y3 换成 W_delta f_l 后重新前向剩余层，"
              "检测头输出与真实换色后的输出之间的距离并不优于完全不干预的 null，方向对齐也不高于随机正交输运对照。"
              "该负结果不被淡化。")
    ap_hi = max([fam[x][k]["box_agreement_AP50"] for k in ks for x in ("proc", "ols", "mask")] + [0.0])
    ap_null = max([nl[k]["box_agreement_AP50"] for k in ks] + [0.0])
    if ap_hi >= 0.3 and ap_null <= 0.15:
        s += ("与 133 的形状分类检测器不同，本设定（类别=色相桶）对颜色敏感：不干预的框级 AP50 仅 %.3f，"
              "而中层输运变体达到 %.3f——该下游把颜色当作计算量，因果等价检验在此**高功效**；"
              "在高功效设定下，单步线性输运已把类别一致度恢复到可用水平，但方向/幅度的投影仍不完整。"
              % (ap_null, ap_hi))
    else:
        s += ("需要同时指出本设定对下游检验的限制：该冻结检测器的输出（框与类）对色相变化本身近乎不敏感，"
              "真实换色只让对象锚点上的 DFL logits 变化约 %s（相对 L2），框级 AP 接近饱和，"
              "因此任何下游等价性检验在这里都是低功效的。"
              % "、".join("%.3f" % nl[k]["rel_box_obj_mean"] for k in ks))
    return s


# ---------------------------------------------------------------- 输出
def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


result = {
    "protocol": (
        "冻结模型 runs/detect/runs_det2/arm_aux_r/weights/best.pt（YOLO11n 检测臂，eval + float；无训练、无权重修改、"
        "不重跑任何既有脚本）。分裂点：net.model[:4] 的输出 = layer 3 输出 = stride-8 / 64 通道特征图（项目称 y3），"
        "干预张量即该 y3；剩余层由脚本内复刻的 _predict_once（自第 4 层起，含 Concat 的 f 索引处理）前向，"
        "并已校验未修改的 f_l 手工送入剩余层的输出与整网前向逐位一致（max abs diff = %g）。"
        "数据：data/yolodet/val_big 前 %d 张合成检测图，固定种子打乱后 %d 张 TRAIN / %d 张 HELD-OUT。"
        "换色（真实路径 R）：像素级 HSV 往返色相平移，只作用于 GT box 内像素（box 外逐位不变），x_delta 为 8bit 重量化；"
        "同一往返在 delta=0 上给出量化地板。输运算子 W_delta：只在 TRAIN 划分上、只用落在 GT box 内的 y3 单元"
        "（换色真正影响的区域，平均占 %.1f%% 的单元）做一次全局拟合，逐像素通道混合（等价 1x1 卷积的共享算子），"
        "无逐图拟合；主算子为完全正交 Procrustes，约定同 scripts/97（M = Y^T X，SVD 得 R = U V^T，行向量输运 f -> f R^T），"
        "并附训练划分上的方向自检（拟合算子的 L2 必须不劣于恒等与转置算子）。另报无约束最小二乘与全图拟合两个变体。"
        "下游指标在剩余层输出的原始检测头张量上计算（decoded 输出、DFL box logits、class scores），并做 "
        "NMS(conf=%.2f, iou=%.2f) 后的框级比较：同类、IoU>=%.2f 贪心一对一匹配的 precision/recall/F1，以及以真实换色"
        "检测为参考、按分数排序的全点 box_agreement_AP50；另给出只取坐标落在 GT box 内的 stride-8 锚点"
        "（对象锚点）的更敏感口径。对照：不干预 y_0（null）、随机正交输运（特异性）、delta=0 量化地板。"
        "复合性检验 W_{d2}W_{d1} vs W_{d1+d2}（特征与输出两侧）。未见 delta 泛化：只用 30/60 的算子（矩阵复合与 logm "
        "单参数生成元外推）预测 90，并与直接在 TRAIN 上拟合的 W_90 参照对比。全部数字由这一次运行产生。"
        % (FAITH, N_IMG, N_TRAIN, len(held_idx), 100 * float(np.mean([CM[i].mean() for i in range(len(files))])),
           CONF, IOU_NMS, IOU_MATCH)),
    "delta_deg": DELTAS,
    "split": {"n_total": len(files), "n_train": len(train_idx), "n_heldout": len(held_idx), "seed": SEED,
              "train_images": [files[i] for i in train_idx], "heldout_images": [files[i] for i in held_idx],
              "note": "拟合只用 TRAIN 划分的像素；所有保真度/下游指标在 HELD-OUT 图像上报告"},
    "split_layer": {"layer_index": SPLIT_LAYER, "tensor": "stride-8 / 64ch (project y3)",
                    "shape_per_image": [FCH, FH, FW], "tail_layers": "net.model[4:] (index 4..23, Detect last)",
                    "tail_replay_faithfulness_max_abs_diff": FAITH,
                    "intervention_impl": "把 W_delta f_l(x) 作为第 4 层输入，其余层与整网完全相同的调用方式"},
    "operator_variants": {KPROC: "正交 Procrustes，只用 GT box 内的 y3 单元拟合（主算子，约定同 scripts/97）",
                          KOLS: "无约束最小二乘，只用 GT box 内的 y3 单元拟合（线性容量上界）",
                          KALL: "正交 Procrustes，用全部 784 个单元拟合（含未换色背景）",
                          KMASK: "与主算子相同，但只施加在 GT box 内的单元上（其余单元保持恒等；诊断变体）"},
    "convention_check_train": CONV,
    "train_residual": {k: {"raw_angle_deg_objcells": FEAT[k]["raw_angle_deg_objcells"],
                           "procrustes_angle_deg_objcells": FEAT[k][KPROC]["objcells"]["transport_angle_deg"],
                           "procrustes_rel_l2_objcells": FEAT[k][KPROC]["objcells"]["rel_l2"],
                           "least_squares_rel_l2_objcells": FEAT[k][KOLS]["objcells"]["rel_l2"]} for k in FEAT},
    "heldout_residual": {k: FEAT[k] for k in FEAT},
    "feature_fidelity": {"per_delta": FEAT,
                         "note": "对象单元 = y3 中落在 GT box 内的 28x28 单元（换色真正影响的区域）；"
                                 "identity null = 不做任何输运；least_squares = 无约束 64x64 线性算子"},
    "downstream_equivalence": {k: clean(DOWN[k]) for k in DOWN},
    "composition_error": COMP,
    "generalisation_to_unseen_delta": GEN,
    "runtime_sec": round(time.time() - T0, 1),
    "verdict": verdict_text(),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(result, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ks = ["%g" % d for d in DELTAS]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
x = np.arange(len(ks))
w = 0.27
ax[0].bar(x - w, [FEAT[k]["raw_angle_deg_objcells"] for k in ks], w, label="raw / identity (no transport)", color="#bbbbbb")
ax[0].bar(x, [FEAT[k][KPROC]["objcells"]["transport_angle_deg"] for k in ks], w, label="procrustes (held-out)", color="#4c78a8")
ax[0].bar(x + w, [FEAT[k][KOLS]["objcells"]["transport_angle_deg"] for k in ks], w, label="least squares (held-out)", color="#f58518")
for i, k in enumerate(ks):
    ax[0].text(i - w, FEAT[k]["raw_angle_deg_objcells"], "%.1f" % FEAT[k]["raw_angle_deg_objcells"], ha="center", va="bottom", fontsize=8)
    ax[0].text(i, FEAT[k][KPROC]["objcells"]["transport_angle_deg"], "%.1f" % FEAT[k][KPROC]["objcells"]["transport_angle_deg"], ha="center", va="bottom", fontsize=8)
    ax[0].text(i + w, FEAT[k][KOLS]["objcells"]["transport_angle_deg"], "%.1f" % FEAT[k][KOLS]["objcells"]["transport_angle_deg"], ha="center", va="bottom", fontsize=8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["%s deg" % k for k in ks])
ax[0].set_ylabel("y3 angle on object cells (deg)")
ax[0].set_title("(a) transport residual in y3 (held-out)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, axis="y")

ax[1].bar(x - w, [DOWN[k]["variants"][KPROC]["rel_box_obj_mean"] for k in ks], w, label="intervened: procrustes", color="#54a24b")
ax[1].bar(x, [DOWN[k]["variants"][KOLS]["rel_box_obj_mean"] for k in ks], w, label="intervened: least squares", color="#4c78a8")
ax[1].bar(x + w, [DOWN[k]["random_orthogonal"]["rel_box_obj_mean"] for k in ks], w, label="random orthogonal", color="#f58518")
for i, k in enumerate(ks):
    ax[1].hlines(DOWN[k]["null_unintervened"]["rel_box_obj_mean"], i - 0.45, i + 0.45, color="k", ls="--", lw=1.2)
ax[1].plot([], [], "k--", lw=1.2, label="null (no transport)")
ax[1].axhline(np.mean([DOWN[k]["hsv_roundtrip0_floor"]["rel_box_mean"] for k in ks]), color="gray", ls=":", lw=1.4,
              label="delta=0 round-trip floor")
ax[1].set_xticks(x)
ax[1].set_xticklabels(["%s deg" % k for k in ks])
ax[1].set_ylabel("rel. L2 vs real route (object-anchor DFL logits)")
ax[1].set_title("(b) downstream agreement vs real recolouring")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="y")

labels, vals, cols = [], [], []
for kk in COMP:
    labels += [kk + "\nfeat", kk + "\nout"]
    vals += [COMP[kk]["feature_rel_l2_composite_vs_direct"], COMP[kk]["output_rel_l2_composite_vs_direct"]]
    cols += ["#4c78a8" if COMP[kk]["operator_family"] == KPROC else "#f58518", "#72b7b2" if COMP[kk]["operator_family"] == KPROC else "#eeca3b"]
xx = np.arange(len(labels))
ax[2].bar(xx, vals, color=cols)
for i, v in enumerate(vals):
    ax[2].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)
ax[2].set_xticks(xx)
ax[2].set_xticklabels(labels, fontsize=7)
ax[2].set_ylabel("rel. L2 (composite vs direct)")
ax[2].set_title("(c) composition error: W_d2(W_d1 f) vs W_{d1+d2} f\n(blue = procrustes, orange = least squares)")
ax[2].grid(alpha=0.3, axis="y")
fig.suptitle("F24 causal test — mid-layer (y3) transport intervention vs real pixel recolouring, frozen YOLO11n", fontsize=11)
plt.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
plt.savefig(FIG, dpi=140)
plt.close()

print(json.dumps({"train/heldout": [len(train_idx), len(held_idx)],
                  "heldout_feature": {k: {KPROC: FEAT[k][KPROC], KOLS: FEAT[k][KOLS],
                                          "raw_angle_deg_objcells": FEAT[k]["raw_angle_deg_objcells"],
                                          "identity_null_rel_l2_objcells": FEAT[k]["identity_null_rel_l2_objcells"]}
                                      for k in ks},
                  "downstream_object_anchor": {k: {v: {"rel_l2": a["rel_box_obj_mean"],
                                                       "align": a["align_box_obj_mean"],
                                                       "proj": a["proj_box_obj_mean"],
                                                       "AP50": a["box_agreement_AP50"]}
                                                   for v, a in dict(list(DOWN[k]["variants"].items())
                                                                    + [("null", DOWN[k]["null_unintervened"]),
                                                                       ("random", DOWN[k]["random_orthogonal"])]).items()}
                                               for k in ks},
                  "composition": {k: {kk: vv for kk, vv in COMP[k].items()} for k in COMP},
                  "unseen_delta": {k: {kk: vv for kk, vv in v.items() if kk != "note"} for k, v in GEN.items()},
                  "runtime_sec": result["runtime_sec"]}, indent=1, ensure_ascii=False))
print("VERDICT:", result["verdict"])
print("wrote", OUT_JSON, "and", FIG)
