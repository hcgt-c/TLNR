# -*- coding: utf-8 -*-
"""Low-saturation / low-value boundary of the code3d_v3 hue read (identifiability Level 0).

Extends results/code3d_sv_domain.json (4x4 grid, s,v in {.25,.5,.75,1}) downward onto
    s in {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}  (same 8 values for v, all 64 cells)
and asks where hue stops being an identifiable property of the input at all.

Protocol (identical conventions to scripts/95_code3d_sv_map.py):
  * renderer  : scripts/01_gen_synthetic.py gen.render(shape,h,s,v)  (pose randomised per call)
  * shapes    : gen.TEST_SHAPES only = pentagon, hexagon (unseen)
  * hues      : 0..330 step 30 (12 hues)
  * poses     : 2 renders per (shape,hue,s,v)  -> 8*8*12*2*2 = 3072 images
  * features  : YOLO11n the YOLO11n checkpoint (repo_paths.YOLO_WEIGHTS), forward hook on mm.model[3],
                spatial mean (GAP) -> 64-dim, then / feat_std  (head trained scale)
  * head      : results/code3d_v3.pt  (Head6, as defined in scripts/95)
  * metric    : hue phase = degrees(arctan2(z1_imag, z1_real)) % 360, circular error vs
                nominal hue; |z1| = hypot(z1_real, z1_imag); l = softplus(o0), s_hat = sigmoid(o1)

Idempotent: deterministic seeds; cached data/features npz are reused when their metadata
matches the requested lattice exactly, otherwise regenerated (`--regen` forces a rebuild).

Outputs:
  data/dense3d_lowext.npz, features/dense3d_lowext_y3.npz
  results/code3d_sv_lowext.json
  results/figures/F19a_sv_lowext.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WORK = rp.REPO_ROOT
sys.path.insert(0, WORK + "/scripts")

import importlib.util
_spec = importlib.util.spec_from_file_location("gen", os.path.join(WORK, "scripts/01_gen_synthetic.py"))
gen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen)

GRID_S = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
GRID_V = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
HUES = list(range(0, 360, 30))
SHAPES = list(gen.TEST_SHAPES)          # pentagon, hexagon
POSES = 2
ERR_THR = 30.0
YOLO_W = rp.YOLO_WEIGHTS
DATA_NPZ = os.path.join(WORK, "data", "dense3d_lowext.npz")
FEAT_NPZ = os.path.join(WORK, "features", "dense3d_lowext_y3.npz")
JSON_OUT = os.path.join(WORK, "results", "code3d_sv_lowext.json")
FIG_OUT = os.path.join(WORK, "results", "figures", "F19a_sv_lowext.png")
REGEN = "--regen" in sys.argv


# ---------------------------------------------------------------- head (as scripts/95)
def load_pool_train_feats():
    """feat_std over exactly the pools used by scripts/95 (== scripts/79)."""
    feats = []
    d = np.load(os.path.join(WORK, "features/dense5_y3.npz"))
    F_ = d["feats"].reshape(6, 72, 4, 64).mean(2)
    for si in range(4):
        for hi in range(72):
            feats.append(F_[si, hi])
    d = np.load(os.path.join(WORK, "features/sheet_hv_y3.npz"))
    F_ = d["feats"].reshape(10, 6, 72, 64)
    for vi in range(10):
        for si in range(4):
            for hi in range(72):
                feats.append(F_[vi, si, hi])
    d = np.load(os.path.join(WORK, "features/dense3d_rings_y3.npz"))
    F_ = d["feats"].reshape(6, 5, 36, 4, 64).mean(3)
    for si in range(4):
        for svi in range(5):
            for k in range(36):
                feats.append(F_[si, svi, k])
    d = np.load(os.path.join(WORK, "features/dense3d_lattice_y3.npz"))
    F_ = d["feats"].reshape(6, 12, 4, 4, 4, 64).mean(4)
    for si in range(4):
        for i in range(12):
            for a in range(4):
                for b in range(4):
                    feats.append(F_[si, i, a, b])
    d = np.load(os.path.join(WORK, "features/sheet_sv_y3.npz"))
    F_ = d["feats"].reshape(3, 6, 10, 10, 64)
    for hi in range(3):
        for si in range(4):
            for a in range(10):
                for b in range(10):
                    feats.append(F_[hi, si, a, b])
    A = np.stack(feats)
    return float(np.std(A))


class Head6(nn.Module):
    def __init__(self, dh=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, dh), nn.SiLU(), nn.Linear(dh, dh), nn.SiLU(), nn.Linear(dh, 6))

    def forward(self, x):
        o = self.net(x)
        l = F.softplus(o[:, 0:1]); s = torch.sigmoid(o[:, 1:2])
        return torch.cat([l, s, o[:, 2:6]], 1)


def circ(a, b):
    return min(abs(a - b), 360 - abs(a - b))


# ---------------------------------------------------------------- data / features
def expected_meta():
    sh, hh, ss, vv = [], [], [], []
    for shp in SHAPES:
        for h in HUES:
            for s in GRID_S:
                for v in GRID_V:
                    for _ in range(POSES):
                        sh.append(shp); hh.append(h); ss.append(s); vv.append(v)
    return (np.array(sh), np.array(hh), np.array(ss), np.array(vv))


def meta_matches(path, keys):
    """True if the cached npz exists and its metadata arrays equal the requested lattice."""
    if not os.path.exists(path):
        return False
    d = np.load(path)
    exp = expected_meta()                      # (shape, h, s, v)
    try:
        for i, k in enumerate(["shape", "h", "s", "v"]):
            a, b = d[k], exp[i]
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        n = len(exp[0])
        for k in keys:
            if k in ("img", "feats") and k in d.files:
                if d[k].shape[0] != n:
                    return False
        if "feats" in keys and "feats" in d.files and d["feats"].shape != (n, 64):
            return False
        if "img" in keys and "img" in d.files and d["img"].shape[1:] != (gen.SIZE, gen.SIZE, 3):
            return False
    except KeyError:
        return False
    return True


def build_images():
    if not REGEN and meta_matches(DATA_NPZ, ["shape", "h", "s", "v", "img"]):
        d = np.load(DATA_NPZ)
        print("[data] reuse", DATA_NPZ, d["img"].shape, flush=True)
        return d["shape"], d["h"], d["s"], d["v"], d["img"], 0.0, True
    sh, hh, ss, vv = expected_meta()
    t0 = time.time()
    imgs = np.empty((len(sh), gen.SIZE, gen.SIZE, 3), dtype=np.uint8)
    for i in range(len(sh)):
        imgs[i] = gen.render(sh[i], int(hh[i]), float(ss[i]), float(vv[i]))
        if (i + 1) % 512 == 0:
            print("[data] rendered %d/%d  %.1fs" % (i + 1, len(sh), time.time() - t0), flush=True)
    np.savez(DATA_NPZ, shape=sh, h=hh, s=ss, v=vv, img=imgs)
    dt = time.time() - t0
    print("[data] wrote", DATA_NPZ, imgs.shape, "%.1fs" % dt, flush=True)
    return sh, hh, ss, vv, imgs, dt, False


def build_feats(sh, hh, ss, vv, imgs):
    if not REGEN and meta_matches(FEAT_NPZ, ["shape", "h", "s", "v", "feats"]):
        d = np.load(FEAT_NPZ)
        print("[feat] reuse", FEAT_NPZ, d["feats"].shape, flush=True)
        return d["feats"], 0.0, True
    from ultralytics import YOLO
    mm = YOLO(YOLO_W).model.to(DEV).eval()
    hooks = {}
    mm.model[3].register_forward_hook(lambda m, i, o: hooks.__setitem__('y3', o.detach() if o.dim() == 4 else None))
    t0 = time.time()
    feats = np.empty((len(imgs), 64), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(imgs)):
            a = imgs[i].astype(np.float32) / np.float32(255.0)
            x = torch.tensor(a, device=DEV).permute(2, 0, 1).unsqueeze(0)
            mm(x)
            feats[i] = hooks['y3'].mean(dim=(2, 3))[0].cpu().numpy()
            if (i + 1) % 512 == 0:
                print("[feat] %d/%d  %.1fs" % (i + 1, len(imgs), time.time() - t0), flush=True)
    np.savez(FEAT_NPZ, feats=feats, shape=sh, h=hh, s=ss, v=vv)
    dt = time.time() - t0
    print("[feat] wrote", FEAT_NPZ, feats.shape, "%.1fs" % dt, flush=True)
    return feats, dt, False


# ---------------------------------------------------------------- input-side sanity
def rgb_headroom(s, v, hue_step=0.25):
    """Quantisation headroom of the 8-bit HSV->RGB render at this (s,v), no network involved.

    Scans hue over [0,360) at `hue_step` and counts distinct uint8 RGB triples the renderer
    can actually produce, plus the channel spread across hue. Purely a property of the input
    encoding: at s=0 (or v=0) every hue maps to one RGB triple, so hue is not represented.
    """
    n = int(round(360.0 / hue_step))
    cols = [gen.hsv_to_rgb(h * hue_step, s, v) for h in range(n)]
    arr = np.array(cols, dtype=np.int32)
    uniq = np.unique(arr, axis=0)
    return {
        "hue_scan_step_deg": hue_step,
        "n_distinct_rgb_scan": int(len(uniq)),
        "n_distinct_rgb_at_12_hues": int(len(np.unique(
            np.array([gen.hsv_to_rgb(h, s, v) for h in HUES], dtype=np.int32), axis=0))),
        "max_channel_range_8bit": int((arr.max(0) - arr.min(0)).max()),
        "rgb_at_hue0": list(map(int, gen.hsv_to_rgb(0, s, v))),
        "rgb_at_hue180": list(map(int, gen.hsv_to_rgb(180, s, v))),
    }


def random_phase_baseline_deg():
    """Mean circular error of a uniform random phase against a fixed hue (no magic constants)."""
    p = np.arange(0, 360, 0.1)
    return float(np.mean([circ(x, 0.0) for x in p]))


def validate_against_published(head, feat_std):
    """Reproduce results/code3d_sv_domain.json from the stored 4-pose lattice features using the
    scripts/95 convention (pose-averaged GAP feature -> head). Anchors our pipeline to the
    already-published 4x4 panel."""
    fn = os.path.join(WORK, "results/code3d_sv_domain.json")
    if not os.path.exists(fn):
        return {"available": False}
    pub = json.load(open(fn))
    d = np.load(os.path.join(WORK, "features/dense3d_lattice_y3.npz"))
    G = d["feats"].reshape(6, 12, 4, 4, 4, 64).mean(4)
    SV = [0.25, 0.5, 0.75, 1.0]
    diffs = []
    for si, shp in [(4, "pentagon"), (5, "hexagon")]:
        for a, s in enumerate(SV):
            for b, v in enumerate(SV):
                with torch.no_grad():
                    c = head(torch.tensor(G[si, :, a, b] / feat_std, dtype=torch.float32, device=DEV)).cpu().numpy()
                ph = (np.degrees(np.arctan2(c[:, 3], c[:, 2])) % 360.0)
                e = float(np.mean([circ(ph[i], float(i * 30)) for i in range(12)]))
                diffs.append(abs(e - float(pub["per_cell"]["s%.2f_v%.2f" % (s, v)][shp])))
    return {"available": True,
            "reference": "results/code3d_sv_domain.json",
            "convention": "pose-averaged GAP feature -> head (as scripts/95)",
            "n_cells_compared": len(diffs),
            "max_abs_diff_deg": round(float(max(diffs)), 3),
            "mean_abs_diff_deg": round(float(np.mean(diffs)), 3)}


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    os.makedirs(os.path.join(WORK, "results", "figures"), exist_ok=True)
    sh, hh, ss, vv, imgs, data_sec, data_cached = build_images()
    feats, feat_sec, feat_cached = build_feats(sh, hh, ss, vv, imgs)
    timing = {"data_build_sec": round(data_sec, 1), "data_cached": data_cached,
              "feat_extract_sec": round(feat_sec, 1), "feat_cached": feat_cached}

    print("[head] compute feat_std...", flush=True)
    feat_std = load_pool_train_feats()
    head = Head6().to(DEV)
    head.load_state_dict(torch.load(os.path.join(WORK, "results/code3d_v3.pt"), map_location=DEV))
    head.eval()
    print("[head] feat_std=%.6f" % feat_std, flush=True)

    X = torch.tensor(feats / feat_std, dtype=torch.float32, device=DEV)
    with torch.no_grad():
        C = head(X).cpu().numpy()                      # (N,6): l, s_hat, z1_re, z1_im, z2_re, z2_im
    l_val = C[:, 0]
    s_hat = C[:, 1]
    z1 = np.hypot(C[:, 2], C[:, 3])
    phase = (np.degrees(np.arctan2(C[:, 3], C[:, 2])) % 360.0)
    if not np.isfinite(C).all() or not np.isfinite(feats).all():
        raise RuntimeError("non-finite head output or features: naN/inf present")

    return _report(sh, hh, ss, vv, feat_std, l_val, s_hat, z1, phase, t_start,
                   feats, head, C, timing)


def _report(sh, hh, ss, vv, feat_std, l_val, s_hat, z1, phase, t_start, feats, head, C, timing):
    per_cell = {}
    err_grid = np.zeros((len(GRID_V), len(GRID_S)))
    z1_grid = np.zeros((len(GRID_V), len(GRID_S)))
    l_grid = np.zeros((len(GRID_V), len(GRID_S)))
    sh_grid = np.zeros((len(GRID_V), len(GRID_S)))
    rgb_grid = np.zeros((len(GRID_V), len(GRID_S)), dtype=int)

    for bi, v in enumerate(GRID_V):
        for ai, s in enumerate(GRID_S):
            key = "s%.2f_v%.2f" % (s, v)
            m = (np.abs(ss - s) < 1e-9) & (np.abs(vv - v) < 1e-9)
            errs = np.array([circ(phase[i], float(hh[i])) for i in np.where(m)[0]])
            cell = {
                "n": int(m.sum()),
                "mean_err": round(float(errs.mean()), 2),
                "median_err": round(float(np.median(errs)), 2),
                "err_p90": round(float(np.percentile(errs, 90)), 2),
                "z1_mag": round(float(z1[m].mean()), 4),
                "l": round(float(l_val[m].mean()), 4),
                "s_hat": round(float(s_hat[m].mean()), 4),
            }
            for shp in SHAPES:
                ms = m & (sh == shp)
                cell[shp] = round(float(np.mean([circ(phase[i], float(hh[i])) for i in np.where(ms)[0]])), 2)
            # per-hue diagnostics (pooled over shapes and poses) - auditability of the collapse
            cell["phase_deg_by_hue"] = [round(float(phase[m & (hh == h)].mean() % 360.0), 1) for h in HUES]
            cell["err_by_hue"] = [round(float(np.mean([circ(phase[i], float(hh[i]))
                                                       for i in np.where(m & (hh == h))[0]])), 2) for h in HUES]
            cell["rgb_headroom"] = rgb_headroom(s, v)
            per_cell[key] = cell
            err_grid[bi, ai] = cell["mean_err"]
            z1_grid[bi, ai] = cell["z1_mag"]
            l_grid[bi, ai] = cell["l"]
            sh_grid[bi, ai] = cell["s_hat"]
            rgb_grid[bi, ai] = cell["rgb_headroom"]["n_distinct_rgb_scan"]

    bi1, ai1 = GRID_V.index(1.0), GRID_S.index(1.0)
    err_11 = float(err_grid[bi1, ai1]); z1_11 = float(z1_grid[bi1, ai1])
    l_11 = float(l_grid[bi1, ai1]); shat_11 = float(sh_grid[bi1, ai1])

    # ---- second aggregation, exactly the scripts/95 convention: average the GAP feature over
    # the poses of a (shape,hue,s,v) cell FIRST, then run the head on the pose-averaged feature.
    # (Verified below to reproduce results/code3d_sv_domain.json on the stored 4-pose lattice.)
    pa_err = np.zeros((len(GRID_V), len(GRID_S)))
    pa_z1 = np.zeros((len(GRID_V), len(GRID_S)))
    Xall = feats / feat_std
    for bi, v in enumerate(GRID_V):
        for ai, s in enumerate(GRID_S):
            errs, zs = [], []
            for shp in SHAPES:
                block = []
                for h in HUES:
                    m = (sh == shp) & (hh == h) & (np.abs(ss - s) < 1e-9) & (np.abs(vv - v) < 1e-9)
                    block.append(Xall[m].mean(0))
                with torch.no_grad():
                    c = head(torch.tensor(np.stack(block), dtype=torch.float32, device=DEV)).cpu().numpy()
                ph = (np.degrees(np.arctan2(c[:, 3], c[:, 2])) % 360.0)
                errs += [circ(ph[i], float(HUES[i])) for i in range(len(HUES))]
                zs += list(np.hypot(c[:, 2], c[:, 3]))
            pa_err[bi, ai] = float(np.mean(errs))
            pa_z1[bi, ai] = float(np.mean(zs))
            per_cell["s%.2f_v%.2f" % (s, v)]["mean_err_poseavgfeat"] = round(float(pa_err[bi, ai]), 2)
            per_cell["s%.2f_v%.2f" % (s, v)]["z1_mag_poseavgfeat"] = round(float(pa_z1[bi, ai]), 4)

    validation = validate_against_published(head, feat_std)
    print("[val] pose-avg convention vs results/code3d_sv_domain.json: max|diff| = %.3f deg (n=%d)"
          % (validation["max_abs_diff_deg"], validation["n_cells_compared"]), flush=True)

    # largest s with err<=30 for each v (over the sampled grid; None if even s=1.0 fails)
    max_s_ok = {}
    fail_by_v = {}
    min_s_ok = {}
    for bi, v in enumerate(GRID_V):
        ok = [s for ai, s in enumerate(GRID_S) if err_grid[bi, ai] <= ERR_THR]
        bad = [s for ai, s in enumerate(GRID_S) if err_grid[bi, ai] > ERR_THR]
        max_s_ok["v%.2f" % v] = (max(ok) if ok else None)
        min_s_ok["v%.2f" % v] = (min(ok) if ok else None)
        fail_by_v["v%.2f" % v] = bad
    # largest v with err<=30 for each s
    max_v_ok = {}
    min_v_ok = {}
    fail_by_s = {}
    for ai, s in enumerate(GRID_S):
        ok = [v for bi, v in enumerate(GRID_V) if err_grid[bi, ai] <= ERR_THR]
        bad = [v for bi, v in enumerate(GRID_V) if err_grid[bi, ai] > ERR_THR]
        max_v_ok["s%.2f" % s] = (max(ok) if ok else None)
        min_v_ok["s%.2f" % s] = (min(ok) if ok else None)
        fail_by_s["s%.2f" % s] = bad

    z1_ratio = z1_grid / z1_11
    low_z1 = [[GRID_S[ai], GRID_V[bi], round(float(z1_ratio[bi, ai]), 4)]
              for bi in range(len(GRID_V)) for ai in range(len(GRID_S)) if z1_ratio[bi, ai] < 0.5]
    below = [[GRID_S[ai], GRID_V[bi]] for bi in range(len(GRID_V)) for ai in range(len(GRID_S))
             if z1_ratio[bi, ai] < 0.5]
    above30 = [[GRID_S[ai], GRID_V[bi], round(float(err_grid[bi, ai]), 2)]
               for bi in range(len(GRID_V)) for ai in range(len(GRID_S)) if err_grid[bi, ai] > ERR_THR]
    clean = [[GRID_S[ai], GRID_V[bi]] for bi in range(len(GRID_V)) for ai in range(len(GRID_S))
             if err_grid[bi, ai] <= ERR_THR]

    summary = {
        "err_deg_at_s1_v1": round(err_11, 2),
        "z1_mag_at_s1_v1": round(z1_11, 4),
        "err_deg_at_s1_v1_poseavgfeat": round(float(pa_err[bi1, ai1]), 2),
        "z1_mag_at_s1_v1_poseavgfeat": round(float(pa_z1[bi1, ai1]), 4),
        "l_at_s1_v1": round(l_11, 4),
        "s_hat_at_s1_v1": round(shat_11, 4),
        "err_threshold_deg": ERR_THR,
        "max_s_with_err_le_thr_per_v": max_s_ok,
        "min_s_with_err_le_thr_per_v": min_s_ok,
        "failing_s_per_v": fail_by_v,
        "max_v_with_err_le_thr_per_s": max_v_ok,
        "min_v_with_err_le_thr_per_s": min_v_ok,
        "failing_v_per_s": fail_by_s,
        "note_on_max_s_fields": ("max_s/max_v are trivially 1.0 wherever the failure set does not "
                                 "reach s=1 or v=1; the boundary is non-monotone, so the informative "
                                 "quantities are failing_s_per_v / failing_v_per_s and "
                                 "min_s_with_err_le_thr_per_v / min_v_with_err_le_thr_per_s."),
        "cells_err_gt_thr": above30,
        "n_cells_err_le_thr": len(clean),
        "n_cells_total": len(GRID_S) * len(GRID_V),
        "region_z1_below_half_of_11": {"z1_ratio_threshold": 0.5, "cells": below},
        "z1_ratio_map": {("s%.2f_v%.2f" % (GRID_S[ai], GRID_V[bi])): round(float(z1_ratio[bi, ai]), 4)
                         for bi in range(len(GRID_V)) for ai in range(len(GRID_S))},
        "random_phase_baseline_deg": round(random_phase_baseline_deg(), 2),
        "hue_undefined_note": ("By construction HSV hue is undefined at s=0 (greyscale) and at v=0 "
                               "(black): the renderer's hsv_to_rgb maps every hue to a single RGB "
                               "triple, so no hue information reaches the network and the hue error "
                               "of any read-out is uninformative there. The measured grid starts at "
                               "s=v=0.02, where the 8-bit render still resolves a limited number of "
                               "distinct RGB triples per hue sweep (see per_cell[*].rgb_headroom)."),
        "sanity_is_input_side_only": ("rgb_headroom is computed directly from gen.hsv_to_rgb and "
                                      "involves no network; it bounds how many hues the 8-bit render "
                                      "can even distinguish at that (s,v)."),
    }

    out = {
        "protocol": ("code3d_v3 zero-shot hue read on the UNSEEN shapes (pentagon, hexagon); "
                     "grid s,v in {0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0} (64 cells) x 12 hues "
                     "(0..330 step 30) x 2 poses per shape/hue/s/v = 3072 images rendered by "
                     "scripts/01_gen_synthetic.py gen.render at 224x224 px (the size the head was "
                     "trained on; NOT 64x64 as sketched); y3 GAP feature = spatial mean of the "
                     "forward hook on mm.model[3] of YOLO11n (the YOLO11n checkpoint (repo_paths.YOLO_WEIGHTS)), "
                     "divided by feat_std from the scripts/95 training pools so the head operates "
                     "in its trained scale; head = Head6 loaded from results/code3d_v3.pt; metric = "
                     "circular error (deg) between nominal hue and degrees(arctan2(z1_imag,z1_real)) "
                     "mod 360, aggregated per (s,v) over 12 hues x 2 poses x 2 shapes; |z1| = "
                     "hypot(z1_real,z1_imag); l = softplus(o0); s_hat = sigmoid(o1)."),
        "grid_s": GRID_S,
        "grid_v": GRID_V,
        "shapes": SHAPES,
        "hues": HUES,
        "poses_per_cell": POSES,
        "n_images": int(len(ss)),
        "render_size_px": int(gen.SIZE),
        "feat_std": feat_std,
        "device": DEV,
        "seed": {"torch": 0, "numpy": 0, "renderer_RNG": 20260902},
        "per_cell": per_cell,
        "per_cell_key_format": "per_cell keys are 's%.2f_v%.2f' (same style as "
                               "results/code3d_sv_domain.json); per_cell_alias maps the shorter "
                               "'s%g_v%g' spelling to those keys.",
        "per_cell_alias": {("s%g_v%g" % (s, v)): ("s%.2f_v%.2f" % (s, v))
                           for v in GRID_V for s in GRID_S},
        "mean_err_grid_v_by_s": [[round(float(err_grid[bi, ai]), 2) for ai in range(len(GRID_S))]
                                 for bi in range(len(GRID_V))],
        "z1_mag_grid_v_by_s": [[round(float(z1_grid[bi, ai]), 4) for ai in range(len(GRID_S))]
                               for bi in range(len(GRID_V))],
        "mean_err_grid_v_by_s_poseavgfeat": [[round(float(pa_err[bi, ai]), 2) for ai in range(len(GRID_S))]
                                            for bi in range(len(GRID_V))],
        "z1_mag_grid_v_by_s_poseavgfeat": [[round(float(pa_z1[bi, ai]), 4) for ai in range(len(GRID_S))]
                                          for bi in range(len(GRID_V))],
        "pipeline_validation": validation,
        "summary": summary,
        "timing": dict(timing,
                       head_and_analysis_sec=round(time.time() - t_start
                                                   - timing["data_build_sec"]
                                                   - timing["feat_extract_sec"], 1),
                       total_sec=round(time.time() - t_start, 1)),
    }
    json.dump(out, open(JSON_OUT, "w"), indent=1)
    print("[out] wrote", JSON_OUT, flush=True)

    make_figure(err_grid, z1_grid, err_11, z1_11)
    print("[out] wrote", FIG_OUT, flush=True)

    # ---- compact summary ----
    print("\n=== grid: rows v (top=1.0), cols s -> mean circular hue error (deg) ===")
    print("        " + "".join("%8.2f" % s for s in GRID_S))
    for bi in range(len(GRID_V) - 1, -1, -1):
        print("v=%.2f  " % GRID_V[bi] + "".join("%8.2f" % err_grid[bi, ai] for ai in range(len(GRID_S))))
    print("\n=== grid: mean |z1| ===")
    print("        " + "".join("%8.3f" % s for s in GRID_S))
    for bi in range(len(GRID_V) - 1, -1, -1):
        print("v=%.2f  " % GRID_V[bi] + "".join("%8.3f" % z1_grid[bi, ai] for ai in range(len(GRID_S))))
    print("\n=== grid (scripts/95 convention: pose-averaged feature): mean hue error (deg) ===")
    print("        " + "".join("%8.2f" % s for s in GRID_S))
    for bi in range(len(GRID_V) - 1, -1, -1):
        print("v=%.2f  " % GRID_V[bi] + "".join("%8.2f" % pa_err[bi, ai] for ai in range(len(GRID_S))))
    print("\nerr(1,1)=%.2f deg  |z1|(1,1)=%.4f  |l|(1,1)=%.3f  shat(1,1)=%.3f" % (err_11, z1_11, l_11, shat_11))
    print("poseavgfeat err(1,1)=%.2f deg  |z1|(1,1)=%.4f" % (pa_err[bi1, ai1], pa_z1[bi1, ai1]))
    print("validation vs code3d_sv_domain.json: max|diff|=%.3f deg over %d cells"
          % (validation["max_abs_diff_deg"], validation.get("n_cells_compared", 0)))
    print("cells with err>30 deg: %d/%d -> %s" % (len(above30), len(GRID_S) * len(GRID_V), above30[:8]))
    print("max s with err<=30 per v:", max_s_ok)
    print("max v with err<=30 per s:", max_v_ok)
    print("failing s per v (err>30):", fail_by_v)
    print("failing v per s (err>30):", fail_by_s)
    print("min s with err<=30 per v:", min_s_ok)
    print("min v with err<=30 per s:", min_v_ok)
    print("|z1| < 0.5*|z1|(1,1) at:", below)
    print("runtime %.1fs" % (time.time() - t_start))
    return out


def make_figure(err_grid, z1_grid, err_11, z1_11):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for ax, G, title, cmap, fmt in [
        (axes[0], err_grid, "mean circular hue error (deg)", "magma", "%.0f"),
        (axes[1], z1_grid, "mean |z1| (code amplitude)", "viridis", "%.2f"),
    ]:
        im = ax.imshow(G, origin="lower", cmap=cmap, aspect="auto", vmin=0.0)
        ax.set_xticks(range(len(GRID_S))); ax.set_xticklabels([("%g" % s) for s in GRID_S])
        ax.set_yticks(range(len(GRID_V))); ax.set_yticklabels([("%g" % v) for v in GRID_V])
        ax.set_xlabel("saturation"); ax.set_ylabel("value")
        ax.set_title(title)
        lo, hi = float(G.min()), float(G.max())
        for bi in range(G.shape[0]):
            for ai in range(G.shape[1]):
                frac = 0.0 if hi <= lo else (G[bi, ai] - lo) / (hi - lo)
                ax.text(ai, bi, fmt % G[bi, ai], ha="center", va="center", fontsize=9,
                        color=("black" if frac > 0.6 else "white"))
        if G is err_grid:
            for bi in range(G.shape[0]):
                for ai in range(G.shape[1]):
                    if G[bi, ai] > ERR_THR:
                        ax.add_patch(plt.Rectangle((ai - .5, bi - .5), 1, 1, fill=False,
                                                   edgecolor="deepskyblue", lw=2.2))
            ax.set_title(title + "  (blue box: >%g deg)" % ERR_THR)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("F19a  code3d_v3 hue-read boundary on unseen shapes (pentagon/hexagon)\n"
                 "8x8 (s,v) grid x 12 hues x 2 poses x 2 shapes = 3072 imgs; "
                 "err(1,1)=%.2f deg, |z1|(1,1)=%.3f" % (err_11, z1_11))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_OUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
