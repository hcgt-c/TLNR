# -*- coding: utf-8 -*-
"""Scaled usage-rule validation on real regions (category-stratified) (no training, no retraining).
120 COCO regions across categories, four conditions, all with the same frozen per-cell head:
  (a) raw white-composite region, no normalisation      -> measures input-assumption violation
  (b) mask + dominant-colour fill (circular reference)
  (c) mask + fill with hue = ref + 40 deg (NON-circular control)
  (d) same as (c) but with a GrabCut mask obtained from the box (a real segmentation front-end
      instead of ground-truth boxes)
Reports medians/means overall and by colour concentration, plus mask-quality comparison (b vs d).
Outputs results/usage_rule_scale_strat.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math
import numpy as np, torch, torch.nn as nn
import cv2
from PIL import Image
from ultralytics.nn.tasks import DetectionModel

WORK = rp.REPO_ROOT; DEV = "cuda"
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES


class AuxDet(DetectionModel):
    def _capture(self, m, i, o): return o
    def forward(self, x, *a, **k): return super().forward(x, *a, **k)


net = torch.load(f"{WORK}/runs/detect/runs_det2/arm_aux_r/weights/best.pt",
                 map_location=DEV, weights_only=False)
if isinstance(net, dict): net = net["model"]
net.to(DEV).eval().float()
net.aux_code = getattr(net, "aux_code", nn.Conv2d(64, 4, 1).to(DEV)).float()
mid = nn.Sequential(*list(net.model[:4])).to(DEV).eval()


def rgb2hsv(p):
    mx = p.max(1); mn = p.min(1); d = mx - mn
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    h = np.where(r == mx, (g - b) / np.maximum(d, 1e-6),
                 np.where(g == mx, 2 + (b - r) / np.maximum(d, 1e-6), 4 + (r - g) / np.maximum(d, 1e-6)))
    h = (h % 6) * 60
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def hsv2rgb(h, s, v):
    import colorsys
    return np.array([colorsys.hsv_to_rgb((a % 360) / 360.0, b, c) for a, b, c in zip(h, s, v)], np.float32)


def read_phase(img):
    x = torch.tensor(np.asarray(img).astype(np.float32) / 255, device=DEV).permute(2, 0, 1)[None]
    with torch.no_grad():
        code = net.aux_code(mid(x))[0].cpu().numpy()
    ph = np.degrees(np.arctan2(code[1], code[0])) % 360
    return float(ph[14, 14])


def grabcut_mask(arr, box):
    """arr float [0,1] RGB; box = (x0,y0,x1,y1) in arr coords -> boolean mask"""
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    H, W = a.shape[:2]
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = box
    rect = (int(x0), int(y0), max(2, int(x1 - x0)), max(2, int(y1 - y0)))
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(a, m, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    except Exception:
        fg = np.zeros((H, W), bool)
    if fg.sum() < 50:
        fg = np.zeros((H, W), bool); fg[int(y0):int(y1), int(x0):int(x1)] = True
    return fg


circ = lambda a, b: min(abs(a - b), 360 - abs(a - b))
ann = json.load(open(ANN)); imgs = {i["id"]: i for i in ann["images"]}
cats = {c["id"]: c["name"] for c in ann["categories"]}
rng = np.random.RandomState(0)
pool = {}
for a in ann["annotations"]:
    if a.get("iscrowd"): continue
    x, y, w, h = a["bbox"]
    if w * h < 20000 or w < 80 or h < 80: continue
    pool.setdefault(cats[a["category_id"]], []).append(a)
order = sorted(pool.keys())
rng.shuffle(order)
regs = []
per = 10
for c in order:
    lst = pool[c][:]
    rng.shuffle(lst)
    regs.extend(lst[:per])
    if len(regs) >= 120: break
regs = regs[:120]
print("categories:", len(set(cats[a["category_id"]] for a in regs)), "regions:", len(regs))

rows = []
for a in regs:
    im = imgs[a["image_id"]]; path = os.path.join(IMG, im["file_name"])
    if not os.path.exists(path): continue
    I = Image.open(path); I.draft("RGB", (max(1, I.size[0] // 4), max(1, I.size[1] // 4)))
    I = I.convert("RGB"); sc = I.size[0] / im["width"]
    x, y, w, h = [v * sc for v in a["bbox"]]
    W, H = I.size
    x0, y0 = max(0, int(x - 0.3 * w)), max(0, int(y - 0.3 * h))
    x1, y1 = min(W, int(x + 1.3 * w)), min(H, int(y + 1.3 * h))
    arr = np.asarray(I.crop((x0, y0, x1, y1))).astype(np.float32) / 255
    bx = (int(x - x0), int(y - y0), int(x + w - x0), int(y + h - y0))
    m_box = np.zeros(arr.shape[:2], bool); m_box[bx[1]:bx[3], bx[0]:bx[2]] = True
    m_gc = grabcut_mask(arr, bx)
    px = arr[m_box]
    if len(px) < 50: continue
    hh, ss, vv = rgb2hsv(px)
    vx = (ss * np.cos(np.deg2rad(hh))).sum(); vy = (ss * np.sin(np.deg2rad(hh))).sum()
    ref = float(np.degrees(np.arctan2(vy, vx)) % 360)
    conc = float(np.hypot(vx, vy) / max(ss.sum(), 1e-6))
    med_s, med_v = float(np.median(ss)), float(np.percentile(vv, 60))
    outA = np.ones_like(arr); outA[m_box] = arr[m_box]
    outB = np.ones_like(arr); outB[m_box] = hsv2rgb(hh * 0 + ref, np.full_like(hh, med_s), np.full_like(hh, med_v))
    outC = np.ones_like(arr); outC[m_box] = hsv2rgb(hh * 0 + (ref + 40), np.full_like(hh, med_s), np.full_like(hh, med_v))
    outD = np.ones_like(arr); outD[m_gc] = hsv2rgb(hh[:m_gc.sum()] * 0 + (ref + 40),
                                                   np.full(m_gc.sum(), med_s), np.full(m_gc.sum(), med_v))
    e = {}
    for tag, out, tgt in [("a_raw", outA, ref), ("b_dominant_fill", outB, ref),
                          ("c_shifted_fill_gtbox", outC, (ref + 40) % 360),
                          ("d_shifted_fill_grabcut", outD, (ref + 40) % 360)]:
        img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
        e[tag] = circ(read_phase(img), tgt)
    rows.append({"category": cats[a["category_id"]], "concentration": round(conc, 3),
                 "mask_frac_gc": round(float(m_gc.mean()), 3), **{k: round(v, 1) for k, v in e.items()}})
    if len(rows) >= 120:
        break

def stats(key):
    v = [r[key] for r in rows]
    return {"n": len(v), "median": round(float(np.median(v)), 1), "mean": round(float(np.mean(v)), 1),
            "frac_lt30": round(float(np.mean(np.array(v) < 30)), 3)}

out = {"protocol": ("120 COCO instances (w,h>=80px), frozen per-cell head, JPEG draft 1/4; "
                    "(a) raw white-composite, (b) mask+dominant-colour fill, (c) mask+fill at ref+40 deg "
                    "(non-circular), (d) same as (c) with a GrabCut mask from the box (real segmentation front-end)"),
       "n": len(rows),
       "a_raw": stats("a_raw"), "b_dominant_fill": stats("b_dominant_fill"),
       "c_shifted_fill_gtbox": stats("c_shifted_fill_gtbox"),
       "d_shifted_fill_grabcut": stats("d_shifted_fill_grabcut"),
       "by_concentration": {
           "high>=0.7": {"c": stats.__wrapped__ if False else None},
       },
       "rows": rows}
# split by concentration for the non-circular control
hi = [r["c_shifted_fill_gtbox"] for r in rows if r["concentration"] >= 0.7]
lo = [r["c_shifted_fill_gtbox"] for r in rows if r["concentration"] < 0.7]
out["by_concentration"] = {
    "high_ge0.7": {"n": len(hi), "median": round(float(np.median(hi)), 1) if hi else None},
    "low_lt0.7": {"n": len(lo), "median": round(float(np.median(lo)), 1) if lo else None},
}
percat = {}
for r in rows:
    percat.setdefault(r["category"], []).append(r["c_shifted_fill_gtbox"])
out["by_category"] = {k: {"n": len(v), "median": round(float(np.median(v)), 1)} for k, v in sorted(percat.items())}
out["by_mask_quality"] = {
    "mask_frac_ge0.15": {"n": int(((np.array([r["mask_frac_gc"] for r in rows]) >= 0.15)).sum()),
                         "median_d": round(float(np.median([r["d_shifted_fill_grabcut"] for r in rows if r["mask_frac_gc"] >= 0.15])), 1)},
    "mask_frac_lt0.15": {"n": int(((np.array([r["mask_frac_gc"] for r in rows]) < 0.15)).sum()),
                         "median_d": round(float(np.median([r["d_shifted_fill_grabcut"] for r in rows if r["mask_frac_gc"] < 0.15])), 1) if any(r["mask_frac_gc"] < 0.15 for r in rows) else None},
}
json.dump(out, open(f"{WORK}/results/usage_rule_scale_strat.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
