# -*- coding: utf-8 -*-
"""140_gen_yolo_hue.py — colour-sensitive synthetic detection dataset (class = hue bin).

Motivation: the existing data/yolodet labels classes by SHAPE, so the detector's logits are nearly
hue-insensitive (script 133: real recolour moves object-anchor logits by only 0.04-0.065 rel L2 and
box AP50 stays 1.0 -> low-power causal test). Here the class IS the hue bin (8 bins x 45 deg) while
the shape is randomised across classes, so a detector must use colour to classify.

Writes data/yolodet_hue/{train,val}/{images,labels} + data.yaml + object_hues.json
(per-image list of (bin, cx, cy, w, h, hue_deg) for later region-restricted recolouring).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, math, json, importlib.util
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("gen", os.path.join(HERE, "01_gen_synthetic.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
import colorsys

SIZE = 224
NBINS = 8
BIN = 360.0 / NBINS
SHAPES = ["triangle", "rectangle", "circle", "star", "pentagon", "hexagon"]
NAMES = [f"hue_{i*int(BIN)}_{(i+1)*int(BIN)}" for i in range(NBINS)]
rng = np.random.RandomState(1234)


def draw_to_mask(shape, center, R, rot):
    m = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(m)
    if shape == "circle":
        d.ellipse([center[0] - R, center[1] - R, center[0] + R, center[1] + R], fill=255)
    elif shape == "rectangle":
        w, h = R * 1.7, R * 1.4
        c, s = math.cos(rot), math.sin(rot)
        pts = [(center[0] + (sx * w / 2 * c - sy * h / 2 * s), center[1] + (sx * w / 2 * s + sy * h / 2 * c))
               for sx, sy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]]
        d.polygon(pts, fill=255)
    else:
        n = {"triangle": 3, "pentagon": 5, "hexagon": 6, "star": 5}[shape]
        if shape == "star":
            outer, inner = R, R * 0.45
            pts = []
            for k in range(10):
                rad = outer if k % 2 == 0 else inner
                ang = rot + k * math.pi / 5 - math.pi / 2
                pts.append((center[0] + rad * math.cos(ang), center[1] + rad * math.sin(ang)))
            d.polygon(pts, fill=255)
        else:
            pts = [(center[0] + R * math.cos(rot + 2 * math.pi * k / n),
                    center[1] + R * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]
            d.polygon(pts, fill=255)
    a = np.asarray(m); ys, xs = np.where(a > 0)
    return (xs.min(), ys.min(), xs.max(), ys.max())


def scene(nobj):
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    boxes = []
    for _ in range(nobj):
        sh = SHAPES[rng.randint(len(SHAPES))]          # shape independent of the class
        for _try in range(30):
            R = rng.uniform(24, 46)
            cx = rng.uniform(R + 10, SIZE - R - 10); cy = rng.uniform(R + 10, SIZE - R - 10)
            rot = rng.uniform(0, 2 * math.pi)
            x0, y0, x1, y1 = draw_to_mask(sh, (cx, cy), R, rot)
            w_, h_ = x1 - x0, y1 - y0
            if w_ > 14 and h_ > 14 and x0 > 0 and y0 > 0 and x1 < SIZE - 1 and y1 < SIZE - 1:
                # keep hue away from bin edges so the class is unambiguous
                b = rng.randint(NBINS)
                hh = (b * BIN + BIN * rng.uniform(0.25, 0.75)) % 360.0
                s = rng.uniform(0.55, 1.0); v = rng.uniform(0.55, 1.0)
                rgb = colorsys.hsv_to_rgb(hh / 360.0, s, v)
                gen.draw_shape(img, sh, (cx, cy), R, rot,
                               (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
                boxes.append(dict(cls=b, cx=(x0 + x1) / 2 / SIZE, cy=(y0 + y1) / 2 / SIZE,
                                  w=w_ / SIZE, h=h_ / SIZE, hue=hh, shape=sh))
                break
    return np.asarray(img), boxes


def write_set(name, n):
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(OUT, "data", "yolodet_hue", name, sub), exist_ok=True)
    meta = {}
    cnt = 0
    while cnt < n:
        img, boxes = scene(rng.randint(1, 4))
        if not boxes:
            continue
        idx = "img%04d" % cnt
        Image.fromarray(img).save(os.path.join(OUT, "data", "yolodet_hue", name, "images", idx + ".jpg"))
        with open(os.path.join(OUT, "data", "yolodet_hue", name, "labels", idx + ".txt"), "w") as f:
            for b in boxes:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (b["cls"], b["cx"], b["cy"], b["w"], b["h"]))
        meta[idx] = [{k: b[k] for k in ("cls", "cx", "cy", "w", "h", "hue", "shape")} for b in boxes]
        cnt += 1
    return meta


if __name__ == "__main__":
    meta_tr = write_set("train", 600)
    meta_va = write_set("val", 120)
    with open(os.path.join(rp.YOLO_DET_HUE, "data.yaml"), "w") as f:
        f.write("path: %s\n" % rp.YOLO_DET_HUE)
        f.write("train: train/images\nval: val/images\nnames:\n")
        for i, s in enumerate(NAMES):
            f.write("  %d: %s\n" % (i, s))
    json.dump({"train": meta_tr, "val": meta_va, "names": NAMES, "nbins": NBINS},
              open(os.path.join(OUT, "data", "yolodet_hue", "object_hues.json"), "w"), indent=1)
    cnt = {}
    for v in meta_tr.values():
        for b in v:
            cnt[b["cls"]] = cnt.get(b["cls"], 0) + 1
    print("yolodet_hue ready:", {NAMES[k]: v for k, v in sorted(cnt.items())})
