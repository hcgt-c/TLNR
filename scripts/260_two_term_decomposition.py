#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""260_two_term_decomposition.py — measure the two sources of realisation error on real sites.

Theory (v16 §3.3): for a fixed linear map, the realisation error splits into
    within-cell error  (what the source rule has already made unrecoverable)  +
    sharing cost       (the price of satisfying every visited region with one operator).
On a rectifier the within-cell term is predicted to sit on the off->on cell: there the source
activation is zero for every sample while the target varies, so no map of the source can recover it.

Estimator, per site and algebra.  Treat each (channel, spatial location) as a unit and each crop as a
sample.  For a unit, u = z(unit), v = z'(unit), and the cell is (1[u>0], 1[v>0]) in {0,1}^2.
    within-cell error = sum_cells w_c * (residual of the best affine fit v ~ a u + b on that cell)
    pooled error      = residual of the single best affine fit on all samples
    sharing cost      = pooled - within-cell
Reported as the mean over sampled units, with the off->on cell's share of the within-cell term.

Usage: python scripts/260_two_term_decomposition.py [--n 120] [--locs 8] [--backbone resnet50]
Outputs results/two_term_decomposition_<backbone>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import argparse, importlib.util, json, os
import numpy as np, torch
from PIL import Image

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
SIZE = 224


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WORK, 'scripts', path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def affine_residual(u, v):
    """Residual sum of squares of the best affine fit v ~ a u + b."""
    if len(u) < 3 or np.allclose(u, u[0]):
        return float(((v - v.mean()) ** 2).sum()), len(u)
    A = np.stack([u, np.ones_like(u)], 1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(((v - A @ coef) ** 2).sum()), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='resnet50'); ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--locs', type=int, default=8); ap.add_argument('--sites', default='layer1,layer2,layer3,layer4')
    ap.add_argument('--hue_deg', type=float, default=90.0); ap.add_argument('--heat_sigma', type=float, default=2.0)
    a = ap.parse_args()
    h188 = load('h188', '188_depth_map_harness.py')

    ann = json.load(open(ANN)); imgs = {im['id']: im for im in ann['images']}
    boxes = [(an['image_id'], *an['bbox']) for an in ann['annotations']
             if not an.get('iscrowd') and an['bbox'][2] >= 80 and an['bbox'][3] >= 80
             and an['bbox'][2] * an['bbox'][3] >= 20000]
    rng = np.random.default_rng(0); rng.shuffle(boxes)
    crops = []
    for imid, x, y, w, h in boxes:
        try:
            I = Image.open(os.path.join(IMG, imgs[imid]['file_name'])).convert('RGB')
        except Exception:
            continue
        crops.append(I.crop((int(x), int(y), int(x + w), int(y + h))).resize((SIZE, SIZE), Image.BILINEAR))
        if len(crops) >= a.n:
            break
    X = torch.from_numpy(np.stack([np.asarray(c, np.float32) / 255.0 for c in crops])).permute(0, 3, 1, 2).contiguous()
    print('crops', X.shape[0], flush=True)

    from torchvision.models import resnet50, ResNet50_Weights
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)

    def site_features(x01, site):
        with torch.no_grad():
            h = stem((x01 - mean) / std)
            h = net.layer1(h)
            if site in ('layer2', 'layer3', 'layer4'):
                h = net.layer2(h)
            if site in ('layer3', 'layer4'):
                h = net.layer3(h)
            if site == 'layer4':
                h = net.layer4(h)
            return h                                  # (N, C, H, W), post-activation

    out = {'backbone': a.backbone, 'n_crops': int(X.shape[0]), 'locs_per_site': a.locs,
           'definition': 'within-cell = sum_cells w_c * SSR_cell(best affine v~au+b); '
                         'sharing cost = pooled SSR - within-cell', 'sites': {}}
    for site in a.sites.split(','):
        Z = site_features(X, site)
        C, H, W = Z.shape[1:]
        locs = [(int(rng.integers(H)), int(rng.integers(W))) for _ in range(a.locs)]
        rec = {}
        for fam, Xt in (('hue_90.0', h188.hue_shift(X, a.hue_deg)), ('heat_2.0', h188.heat(X, a.heat_sigma))):
            Zt = site_features(Xt, site)
            wc, sh, tot = [], [], 0.0
            offon_share = []
            for yy, xx in locs:
                for c in range(C):
                    u = Z[:, c, yy, xx].numpy().astype(np.float64)
                    v = Zt[:, c, yy, xx].numpy().astype(np.float64)
                    cells = {}
                    for gu in (0, 1):
                        for gv in (0, 1):
                            m = ((u > 0) == bool(gu)) & ((v > 0) == bool(gv))
                            if m.sum() >= 3:
                                cells[(gu, gv)] = (u[m], v[m])
                    if len(cells) < 2:
                        continue
                    N = len(u)
                    wcell = {k: len(vv[0]) / N for k, vv in cells.items()}
                    sse_cells = {k: affine_residual(*vv)[0] for k, vv in cells.items()}
                    within = sum(wcell[k] * sse_cells[k] for k in cells)
                    pooled, _ = affine_residual(u, v)
                    wc.append(within / N); sh.append(max(pooled - within, 0.0) / N); tot += 1
                    offon = cells.get((0, 1))
                    offon_share.append((wcell.get((0, 1), 0.0) * sse_cells.get((0, 1), 0.0)) / max(within, 1e-12))
            rec[fam] = {'n_units': tot,
                        'within_cell': float(np.mean(wc)) if wc else None,
                        'sharing_cost': float(np.mean(sh)) if sh else None,
                        'sharing_share_of_total': float(np.mean(sh) / (np.mean(sh) + np.mean(wc)))
                        if wc and (np.mean(sh) + np.mean(wc)) > 0 else None,
                        'off_on_share_of_within': float(np.mean(offon_share)) if offon_share else None}
        out['sites'][site] = rec
        print(f"  {site}: " + ' | '.join(
            f"{k.split('_')[0]} within {v['within_cell']:.4f} sharing {v['sharing_cost']:.4f} "
            f"(sharing {100*v['sharing_share_of_total']:.0f}% of total; off->on {100*v['off_on_share_of_within']:.0f}% of within)"
            for k, v in rec.items()), flush=True)
    json.dump(out, open(os.path.join(WORK, 'results', f'two_term_decomposition_{a.backbone}.json'), 'w'), indent=1)
    print('->', f'results/two_term_decomposition_{a.backbone}.json')


if __name__ == '__main__':
    main()
