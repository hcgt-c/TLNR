# -*- coding: utf-8 -*-
"""173_record_yolo_val.py — reproduce and record the hue-bin detector's validation metrics.

Provenance gap found by scripts/172_code_audit.py: `results/yolo_hue_val.json` (mAP50 0.987 / mAP50-95
0.909, cited by both manuscripts) had no writer script — it had been produced by an ad-hoc shell
command. This script makes it reproducible: it loads the trained checkpoint, evaluates it on the
hue-bin validation split with ultralytics, and writes the metrics with their provenance.

Usage: python scripts/173_record_yolo_val.py [--weights runs/detect/runs_det_hue/hue_cls8/weights/best.pt]
Outputs results/yolo_hue_val.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, json, argparse

WORK = rp.REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs_det_hue/hue_cls8/weights/best.pt")
    ap.add_argument("--data", default="data/yolodet_hue/data.yaml")
    ap.add_argument("--imgsz", type=int, default=224)
    a = ap.parse_args()
    from ultralytics import YOLO
    w = os.path.join(WORK, a.weights)
    assert os.path.exists(w), w
    model = YOLO(w)
    m = model.val(data=os.path.join(WORK, a.data), imgsz=a.imgsz, verbose=False,
                  project=os.path.join(WORK, "runs/detect"), name="hue_val_recompute", exist_ok=True)
    out = {"weights": a.weights, "data": a.data, "imgsz": a.imgsz,
           "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
           "produced_by": "scripts/173_record_yolo_val.py"}
    json.dump(out, open(os.path.join(WORK, "results", "yolo_hue_val.json"), "w"), indent=1)
    print("mAP50 %.4f  mAP50-95 %.4f" % (out["mAP50"], out["mAP50_95"]))
    print("saved results/yolo_hue_val.json")


if __name__ == "__main__":
    main()
