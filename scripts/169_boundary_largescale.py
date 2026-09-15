# -*- coding: utf-8 -*-
"""169_boundary_largescale.py — large-scale, pre-registered, out-of-sample test of the m* = 0.70 rule.

Answers the strongest anticipated objection to Paper A ("the boundary is calibrated on aligned data;
on raw real regions the statistic has AUC 0.52") with a study that is larger, image-split, and evaluated
with the threshold fixed in advance.

Protocol, fixed before the test half is examined (mirrors scripts 111/112, which used 120 regions):
  * sample COCO train2017 instance regions (non-crowd, >= 80 px per side, >= 20000 px^2), up to
    `--per_cat` per category, target `--n_target`;
  * per region: dominant share m of the saturated colour weight, GrabCut mask fraction, the region's own
    circular-mean hue, and read-out error under four presentations with the SAME frozen per-cell code head
      a_raw                 white composite, no normalisation      (input-assumption violation)
      b_dominant_fill       mask + dominant-colour fill            (circular reference)
      c_shifted_fill_gtbox  mask + fill at ref+40 deg, truth box   (non-circular control)
      d_shifted_fill_grabcut  same, GrabCut mask from the box      (real segmentation front end)
  * split regions by IMAGE 60/40 (no photograph on both sides); fit a one-parameter logistic calibration
    of success on m on the fit half; on the test half, with m* = 0.70 fixed in advance, report success
    above vs below m*, the gap with bootstrap CIs, ROC AUC with bootstrap CI, a decile calibration curve,
    per-category success, and the raw route's AUC for contrast.

Usage: python scripts/169_boundary_largescale.py [--per_cat 30 --n_target 750 --m_star 0.70]
Outputs results/boundary_largescale.{json,md}
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, math, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from ultralytics.nn.tasks import DetectionModel

WORK = rp.REPO_ROOT; DEV = "cuda"
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
circ = lambda a, b: min(abs(a - b), 360 - abs(a - b))


class AuxDet(DetectionModel):
    def _capture(self, m, i, o): return o


def build_net():
    net = torch.load(f"{WORK}/runs/detect/runs_det2/arm_aux_r/weights/best.pt",
                     map_location=DEV, weights_only=False)
    if isinstance(net, dict):
        net = net["model"]
    net.to(DEV).eval().float()
    net.aux_code = getattr(net, "aux_code", nn.Conv2d(64, 4, 1).to(DEV)).float()
    mid = nn.Sequential(*list(net.model[:4])).to(DEV).eval()
    return net, mid


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
    return np.array([colorsys.hsv_to_rgb((x % 360) / 360.0, y, z) for x, y, z in zip(h, s, v)], np.float32)


def grabcut_mask(arr, box):
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


def select_regions(per_cat, n_target):
    ann = json.load(open(ANN))
    imgs = {i["id"]: i for i in ann["images"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    rng = np.random.RandomState(0)
    pool = collections.defaultdict(list)
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w * h < 20000 or w < 80 or h < 80:
            continue
        pool[cats[a["category_id"]]].append(a)
    order = sorted(pool)
    rng.shuffle(order)
    regs = []
    for c in order:
        lst = pool[c][:]; rng.shuffle(lst)
        regs.extend(lst[:per_cat])
        if len(regs) >= n_target:
            break
    return regs[:n_target], imgs, cats


def bootstrap_ci(vals, stat=np.mean, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return [None, None]
    bs = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for k, c in enumerate(cnt):
        if c > 1:
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cat", type=int, default=30)
    ap.add_argument("--n_target", type=int, default=750)
    ap.add_argument("--m_star", type=float, default=0.70)
    ap.add_argument("--success_deg", type=float, default=30.0)
    a = ap.parse_args()

    net, mid = build_net()

    def read_phase(img):
        x = torch.tensor(np.asarray(img).astype(np.float32) / 255, device=DEV).permute(2, 0, 1)[None]
        with torch.no_grad():
            code = net.aux_code(mid(x))[0].cpu().numpy()
        return float((np.degrees(np.arctan2(code[1], code[0])) % 360)[14, 14])

    regs, imgs, cats = select_regions(a.per_cat, a.n_target)
    rows = []
    for a_ in regs:
        im = imgs[a_["image_id"]]
        path = os.path.join(IMG, im["file_name"])
        if not os.path.exists(path):
            continue
        I = Image.open(path); I.draft("RGB", (max(1, I.size[0] // 4), max(1, I.size[1] // 4)))
        I = I.convert("RGB"); sc = I.size[0] / im["width"]
        x, y, w, h = [v * sc for v in a_["bbox"]]
        W, H = I.size
        x0, y0 = max(0, int(x - 0.3 * w)), max(0, int(y - 0.3 * h))
        x1, y1 = min(W, int(x + 1.3 * w)), min(H, int(y + 1.3 * h))
        arr = np.asarray(I.crop((x0, y0, x1, y1))).astype(np.float32) / 255
        bx = (int(x - x0), int(y - y0), int(x + w - x0), int(y + h - y0))
        m_box = np.zeros(arr.shape[:2], bool); m_box[bx[1]:bx[3], bx[0]:bx[2]] = True
        if m_box.sum() < 50:
            continue
        m_gc = grabcut_mask(arr, bx)
        px = arr[m_box]
        hh, ss, vv = rgb2hsv(px)
        vx = (ss * np.cos(np.deg2rad(hh))).sum(); vy = (ss * np.sin(np.deg2rad(hh))).sum()
        ref = float(np.degrees(np.arctan2(vy, vx)) % 360)
        conc = float(np.hypot(vx, vy) / max(ss.sum(), 1e-6))
        med_s, med_v = float(np.median(ss)), float(np.percentile(vv, 60))
        outA = np.ones_like(arr); outA[m_box] = arr[m_box]
        outB = np.ones_like(arr); outB[m_box] = hsv2rgb(np.full_like(hh, ref), np.full_like(hh, med_s), np.full_like(hh, med_v))
        outC = np.ones_like(arr); outC[m_box] = hsv2rgb(np.full_like(hh, ref + 40), np.full_like(hh, med_s), np.full_like(hh, med_v))
        outD = np.ones_like(arr)
        n_gc = int(m_gc.sum())
        outD[m_gc] = hsv2rgb(np.full(n_gc, ref + 40), np.full(n_gc, med_s), np.full(n_gc, med_v))
        e = {}
        for tag, out, tgt in [("a_raw", outA, ref), ("b_dominant_fill", outB, ref),
                              ("c_shifted_fill_gtbox", outC, (ref + 40) % 360),
                              ("d_shifted_fill_grabcut", outD, (ref + 40) % 360)]:
            img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
            e[tag] = circ(read_phase(img), tgt)
        rows.append({"image_id": int(im["id"]), "category": cats[a_["category_id"]],
                     "m": round(conc, 4), "mask_frac_gc": round(float(m_gc.mean()), 4),
                     **{k: round(v, 2) for k, v in e.items()}})
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} regions", flush=True)

    m = np.array([r["m"] for r in rows])
    y_align = (np.array([r["d_shifted_fill_grabcut"] for r in rows]) <= a.success_deg).astype(int)
    y_gt = (np.array([r["c_shifted_fill_gtbox"] for r in rows]) <= a.success_deg).astype(int)
    y_raw = (np.array([r["a_raw"] for r in rows]) <= a.success_deg).astype(int)

    # image-level 60/40 split
    imgs_ids = np.array(sorted({r["image_id"] for r in rows}))
    rng = np.random.RandomState(0); rng.shuffle(imgs_ids)
    n_fit = int(0.6 * len(imgs_ids))
    fit_ids, test_ids = set(imgs_ids[:n_fit].tolist()), set(imgs_ids[n_fit:].tolist())
    is_fit = np.array([r["image_id"] in fit_ids for r in rows])
    is_test = ~is_fit

    def rates(mask, y):
        idx = np.where(mask)[0]
        hi_idx = [i for i in idx if m[i] >= a.m_star]
        lo_idx = [i for i in idx if m[i] < a.m_star]
        hi = float(np.mean([y[i] for i in hi_idx])) if hi_idx else None
        lo = float(np.mean([y[i] for i in lo_idx])) if lo_idx else None
        return {"n": len(idx), "n_hi": len(hi_idx), "n_lo": len(lo_idx),
                "success_hi": hi, "success_lo": lo,
                "gap": (hi - lo) if (hi is not None and lo is not None) else None}

    # pre-registered threshold: fixed, no fitting. Calibration: one-parameter logistic fitted on the fit half.
    def fit_logistic(mv, yv):
        # slope fixed at 1 after centring, so this is a one-parameter calibration of P(success | m)
        z = (mv - a.m_star) * 4.0
        p = 1 / (1 + np.exp(-z))
        eps = 1e-6
        w = np.clip(yv, eps, 1 - eps)
        return {"p_at_m_star": 0.5, "offset": float(np.mean(w - p))}

    cal = fit_logistic(m[is_fit], y_align[is_fit])
    pred = np.clip(1 / (1 + np.exp(-(m[is_test] - a.m_star) * 4.0)) + cal["offset"], 0, 1)

    deciles = np.quantile(m[is_test], np.linspace(0, 1, 11))
    curve = []
    for k in range(10):
        sel = (m[is_test] >= deciles[k]) & (m[is_test] <= deciles[k + 1] if k == 9 else m[is_test] < deciles[k + 1])
        if sel.sum() == 0:
            continue
        curve.append({"m_lo": round(float(deciles[k]), 3), "m_hi": round(float(deciles[k + 1]), 3),
                      "n": int(sel.sum()), "predicted": round(float(pred[sel].mean()), 3),
                      "observed": round(float(y_align[is_test][sel].mean()), 3)})

    percat = {}
    for r, yy, tt in zip(rows, y_align, is_test):
        if not tt:
            continue
        percat.setdefault(r["category"], []).append((r["m"], yy))
    by_cat = {c: {"n": len(v), "n_hi": sum(1 for mm, _ in v if mm >= a.m_star),
                  "success_hi": round(float(np.mean([y for mm, y in v if mm >= a.m_star])), 3) if any(mm >= a.m_star for mm, _ in v) else None}
              for c, v in sorted(percat.items()) if len(v) >= 5}

    auc_align = roc_auc(y_align[is_test], m[is_test])
    auc_raw = roc_auc(y_raw[is_test], m[is_test])
    auc_gt = roc_auc(y_gt[is_test], m[is_test])
    boot = []
    rngb = np.random.RandomState(1)
    for _ in range(2000):
        idx = rngb.choice(np.where(is_test)[0], int(is_test.sum()), replace=True)
        boot.append(roc_auc(y_align[idx], m[idx]))
    out = {
        "protocol": ("COCO train2017 instance regions (non-crowd, >=80 px side, >=20000 px^2); frozen "
                     "per-cell code head; four presentations a_raw / b_dominant_fill / c_shifted_fill_gtbox / "
                     "d_shifted_fill_grabcut (non-circular, +40 deg); success = error <= %.0f deg; "
                     "threshold m* = %.2f fixed before the test half; images split 60/40" % (a.success_deg, a.m_star)),
        "n_regions": len(rows), "n_images": int(len(imgs_ids)),
        "m_star": a.m_star, "success_deg": a.success_deg,
        "m_median": round(float(np.median(m)), 3), "frac_m_ge_star": round(float((m >= a.m_star).mean()), 3),
        "fit_half": rates(is_fit, y_align), "test_half": rates(is_test, y_align),
        "test_half_gtbox": rates(is_test, y_gt), "test_half_raw": rates(is_test, y_raw),
        "auc_test_align": auc_align, "auc_test_align_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "auc_test_gtbox": auc_gt, "auc_test_raw": auc_raw,
        "gap_ci95": bootstrap_ci([y_align[i] for i in np.where(is_test & (m >= a.m_star))[0]],
                                 n=1000),
        "calibration": {"model": "one-parameter logistic on (m - m*) fixed slope 4, offset fitted on the fit half",
                        "offset": cal["offset"], "deciles_test": curve},
        "by_category_test": by_cat,
        "medians": {k: round(float(np.median([r[k] for r in rows])), 1)
                    for k in ("a_raw", "b_dominant_fill", "c_shifted_fill_gtbox", "d_shifted_fill_grabcut")},
        "rows": rows,
    }
    json.dump(out, open(os.path.join(WORK, "results", "boundary_largescale.json"), "w"), indent=1)

    L = ["# Large-scale boundary validation (pre-registered m* = %.2f)" % a.m_star, "",
         "COCO train2017 instance regions, frozen per-cell code head, success = read-out error <= %.0f°." % a.success_deg,
         "Images split 60/40 before the test half was examined; the threshold was fixed in advance.", "",
         "| split | n | n(m>=m*) | n(m<m*) | success m>=m* | success m<m* | gap |", "|---|---|---|---|---|---|---|"]
    for name, r in [("fit", out["fit_half"]), ("test", out["test_half"])]:
        L.append(f"| {name} | {r['n']} | {r['n_hi']} | {r['n_lo']} | "
                 f"{r['success_hi']:.3f} | {r['success_lo']:.3f} | {r['gap']:+.3f} |")
    L += ["", f"- test-half ROC AUC of m for the aligned route: **{auc_align:.3f}** "
              f"(bootstrap 95% CI {out['auc_test_align_ci95'][0]:.3f}-{out['auc_test_align_ci95'][1]:.3f})",
          f"- test-half ROC AUC for the ground-truth-box route: {auc_gt:.3f}",
          f"- test-half ROC AUC for the raw route (input-assumption violation): {auc_raw:.3f}", "",
          "| m decile | n | predicted | observed |", "|---|---|---|---|"]
    for c in curve:
        L.append(f"| {c['m_lo']}-{c['m_hi']} | {c['n']} | {c['predicted']:.3f} | {c['observed']:.3f} |")
    open(os.path.join(WORK, "results", "boundary_largescale.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("saved results/boundary_largescale.{json,md}")
    print(f"regions {len(rows)} | test success hi {out['test_half']['success_hi']:.3f} vs lo "
          f"{out['test_half']['success_lo']:.3f} | AUC align {auc_align:.3f} raw {auc_raw:.3f}")


if __name__ == "__main__":
    main()
