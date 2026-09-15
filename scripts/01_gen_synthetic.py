# -*- coding: utf-8 -*-
"""
生成合成图案数据集：多形状 × HSV 可控颜色
============================================
设计：
  - 形状：triangle/rectangle/circle/star（训练） + pentagon/hexagon（留出=未见实体）
  - 颜色：色相 h ∈ {0,40,80,120,160,200,240,280,320}（训练）
          留出色相 {20, 260} 为"未见颜色"（测试色相外推）
          饱和度 s ∈ {0.5, 1.0}，明度 v ∈ {0.5, 1.0}
  - 每 (形状, 颜色) 渲染 N_VAR=12 张，随机平移/缩放/旋转扰动（骨干看到类内方差）
  - 输出：
      data/images/   （不落盘全部，直接回传数组）
      data/meta.npz  ：shape / hue_deg / sat / val / idx 元数据（供特征提取对齐）
数据规模：6 形状 × (9+2 色相 × 2 饱和度 × 2 明度=44 颜色) × 12 张 ≈ 3168 张
留出定义（防数据泄漏）：
  - 训练形状 = 4，测试形状 = 2（其全部颜色张在训练中不可见）
  - 训练色相 = 9，测试色相 = 2（全部形状的所有测试色相张在训练中不可见）
"""
import os, json, math
import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = os.path.join(OUT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SIZE = 224
N_VAR = 12          # 每(形状,颜色)变体数
RNG = np.random.RandomState(20260902)

def polygon_pts(n, cx, cy, R, rot):
    pts = []
    for k in range(n):
        ang = rot + 2*math.pi*k/n - math.pi/2
        pts.append((cx + R*math.cos(ang), cy + R*math.sin(ang)))
    return pts

def star_pts(cx, cy, R, rot):
    pts = []
    for k in range(10):
        r = R if k % 2 == 0 else R*0.45
        ang = rot + math.pi*k/5 - math.pi/2
        pts.append((cx + r*math.cos(ang), cy + r*math.sin(ang)))
    return pts

def draw_shape(img, shape, center, R, rot, color_rgb):
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([center[0]-R, center[1]-R, center[0]+R, center[1]+R], fill=color_rgb)
    elif shape == "rectangle":
        # 随机长宽比 1.0~1.6，旋转由外接旋转近似（简化：用旋转矩形顶点）
        w, h = R*1.7, R*(1.1 + 0.5*(RNG.rand()))
        ang = rot
        cos, sin = math.cos(ang), math.sin(ang)
        pts = []
        for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            x = center[0] + (sx*w/2*cos - sy*h/2*sin)
            y = center[1] + (sx*w/2*sin + sy*h/2*cos)
            pts.append((x, y))
        d.polygon(pts, fill=color_rgb)
    elif shape == "triangle":
        d.polygon(polygon_pts(3, center[0], center[1], R, rot), fill=color_rgb)
    elif shape == "pentagon":
        d.polygon(polygon_pts(5, center[0], center[1], R, rot), fill=color_rgb)
    elif shape == "hexagon":
        d.polygon(polygon_pts(6, center[0], center[1], R, rot), fill=color_rgb)
    elif shape == "star":
        d.polygon(star_pts(center[0], center[1], R, rot), fill=color_rgb)
    else:
        raise ValueError(shape)

def hsv_to_rgb(h, s, v):
    """h 度, s,v ∈[0,1] → uint8 RGB"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h % 360)/360.0, s, v)
    return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

def render(shape, h, s, v):
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    cx, cy = SIZE/2, SIZE/2
    # 扰动（中等幅度；物体尽量大、居中，保证骨干看见清晰形状+颜色）
    cx += RNG.uniform(-12, 12); cy += RNG.uniform(-12, 12)
    R = RNG.uniform(70, 92)
    rot = RNG.uniform(0, 2*math.pi)
    rgb = hsv_to_rgb(h, s, v)
    draw_shape(img, shape, (cx, cy), R, rot, rgb)
    return np.asarray(img, dtype=np.uint8)

# ---- 颜色集 ----
TRAIN_HUES = [0, 40, 80, 120, 160, 200, 240, 280, 320]   # 9 训练色相
TEST_HUES  = [20, 260]                                     # 2 留出色相（未见颜色）
SATURATIONS = [0.5, 1.0]
VALUES      = [0.5, 1.0]

TRAIN_SHAPES = ["triangle", "rectangle", "circle", "star"]
TEST_SHAPES  = ["pentagon", "hexagon"]

def build(store_images=False, max_per_cell=None):
    """返回 (arrays, meta_dict)。arrays: dict[(shape,h,s,v)] = list[img]"""
    arrays = {}
    meta_rows = []
    cells = []
    for shape in TRAIN_SHAPES + TEST_SHAPES:
        for h in TRAIN_HUES + TEST_HUES:
            for s in SATURATIONS:
                for v in VALUES:
                    n = N_VAR if max_per_cell is None else max_per_cell
                    imgs = [render(shape, h, s, v) for _ in range(n)]
                    arrays[(shape, h, s, v)] = imgs
                    cells.append((shape, h, s, v))
    # 元数据
    all_hues = TRAIN_HUES + TEST_HUES
    for (shape, h, s, v) in cells:
        for i in range(len(arrays[(shape, h, s, v)])):
            meta_rows.append({
                "shape": shape, "hue": h, "sat": s, "val": v,
                "hue_train": h in TRAIN_HUES,
                "shape_train": shape in TRAIN_SHAPES,
            })
    np.savez(os.path.join(DATA_DIR, "meta.npz"),
             shape=np.array([r["shape"] for r in meta_rows]),
             hue=np.array([r["hue"] for r in meta_rows]),
             sat=np.array([r["sat"] for r in meta_rows]),
             val=np.array([r["val"] for r in meta_rows]))
    return arrays, meta_rows

if __name__ == "__main__":
    arrays, meta = build()
    n_img = sum(len(v) for v in arrays.values())
    print(f"cells={len(arrays)}  总图像={n_img}  形状数=6  色相数={len(TRAIN_HUES)+len(TEST_HUES)}")
    # 采样检查：保存一张合成样例图确认颜色正确
    img0 = arrays[("triangle", 0, 1.0, 1.0)][0]   # 红色三角形
    Image.fromarray(img0).save(os.path.join(DATA_DIR, "sample_red_triangle.png"))
    img1 = arrays[("triangle", 80, 1.0, 1.0)][0]  # 绿色
    Image.fromarray(img1).save(os.path.join(DATA_DIR, "sample_green_triangle.png"))
    img2 = arrays[("pentagon", 120, 1.0, 1.0)][0] # 未见形状
    Image.fromarray(img2).save(os.path.join(DATA_DIR, "sample_green_pentagon.png"))
    # 平均像素颜色校验
    for name, img in [("red_tri", img0), ("green_tri", img1)]:
        px = img.reshape(-1, 3).mean(0)
        print(name, "平均RGB=", np.round(px, 1), "≈预期", "red(254,0,0)/green(0,254,0)"[:20])
    # 保存配置
    json.dump({"train_shapes": TRAIN_SHAPES, "test_shapes": TEST_SHAPES,
               "train_hues": TRAIN_HUES, "test_hues": TEST_HUES,
               "saturations": SATURATIONS, "values": VALUES,
               "n_var": N_VAR, "size": SIZE},
              open(os.path.join(DATA_DIR, "config.json"), "w"), indent=1)
    print("config saved:", DATA_DIR)
