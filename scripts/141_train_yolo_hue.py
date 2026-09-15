# -*- coding: utf-8 -*-
"""141_train_yolo_hue.py — train a COLOUR-SENSITIVE detector on the hue-bin dataset (script 140).

Why: data/yolodet labels classes by shape, so the detector is nearly hue-insensitive and the
133-style causal test is low-power. data/yolodet_hue labels classes by hue bin (8 bins) with shape
randomised, so the class head must consume colour. Training recipe mirrors script 77 (augmentation
off, so colour is not washed out): hsv_h/s/v=0, mosaic=0, mixup=0, flips off, seed 0, imgsz 224.

Usage: python scripts/141_train_yolo_hue.py --epochs 60 --name hue_cls8
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import argparse, os
from ultralytics import YOLO

WORK = rp.REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="hue_cls8")
    ap.add_argument("--weights", default=rp.YOLO_WEIGHTS)
    a = ap.parse_args()
    model = YOLO(a.weights)
    model.train(data=os.path.join(WORK, "data/yolodet_hue/data.yaml"),
                epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                fliplr=0.0, flipud=0.0, seed=0,
                project=os.path.join(WORK, "runs/detect/runs_det_hue"), name=a.name,
                exist_ok=True, verbose=False)
    m = model.val(data=os.path.join(WORK, "data/yolodet_hue/data.yaml"), imgsz=a.imgsz, verbose=False)
    print("hue detector mAP50=%.4f mAP50-95=%.4f" % (m.box.map50, m.box.map))


if __name__ == "__main__":
    main()
