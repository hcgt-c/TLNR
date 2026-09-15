#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""258_measure_lipschitz.py — measure the truth action's Lipschitz ratio at real sites.

Theorem 4's accumulation law needs one number per (site, algebra): the Lipschitz constant L of the
truth action in the consumer's seminorm.  Until now the paper *assumed* which regime each algebra
sits in (rotations isometric, diffusion contractive).  This script measures it.

Estimator.  For a metric d on features (here the relative L2 distance that the site's own scale makes
comparable), draw pairs (i, j) of held-out crops, and compute
    r_ij = d( psi(tau x_i), psi(tau x_j) ) / d( psi(x_i), psi(x_j) ).
L is summarised by the median and the 90th percentile of r over pairs; L > 1 is expansive, L = 1
isometric, L < 1 contractive for that metric.  The consumer-referenced version replaces d by the
consumer's response distance || F(.) - F(.) ||.

Usage: python scripts/258_measure_lipschitz.py [--backbone resnet50] [--n 150] [--sites layer1,layer2,layer3,layer4]
Outputs results/lipschitz_<backbone>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from scipy.fftpack import dctn, idctn

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = rp.COCO_ANNOTATIONS
IMG = rp.COCO_IMAGES
SIZE = 224


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WORK, 'scripts', path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--sites', default='layer1,layer2,layer3,layer4')
    ap.add_argument('--hue_deg', type=float, default=90.0)
    ap.add_argument('--heat_sigma', type=float, default=2.0)
    a = ap.parse_args()

    h188 = load_script('h188', '188_depth_map_harness.py')      # exact hue/heat transforms
    led = load_script('led', '154_backbone_ledger.py') if os.path.exists(
        os.path.join(WORK, 'scripts', '154_backbone_ledger.py')) else None

    # ---- data: COCO instance crops, same protocol as the depth grid
    import json as _json
    from PIL import Image
    ann = _json.load(open(ANN))
    imgs = {im['id']: im for im in ann['images']}
    boxes = []
    for an in ann['annotations']:
        if an.get('iscrowd'):
            continue
        x, y, w, h = an['bbox']
        if w < 80 or h < 80 or w * h < 20000:
            continue
        boxes.append((an['image_id'], x, y, w, h))
    rng = np.random.default_rng(0)
    rng.shuffle(boxes)
    crops = []
    for imid, x, y, w, h in boxes[:a.n]:
        try:
            I = Image.open(os.path.join(IMG, imgs[imid]['file_name'])).convert('RGB')
        except Exception:
            continue
        crops.append(I.crop((int(x), int(y), int(x + w), int(y + h))).resize((SIZE, SIZE), Image.BILINEAR))
        if len(crops) >= a.n:
            break
    X = torch.from_numpy(np.stack([np.asarray(c, dtype=np.float32) / 255.0 for c in crops])
                         ).permute(0, 3, 1, 2).contiguous()
    print(f'crops {X.shape[0]}')

    # ---- backbone
    if a.backbone == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        site_mods = {'layer1': net.layer1, 'layer2': net.layer2, 'layer3': net.layer3, 'layer4': net.layer4}
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    else:
        raise SystemExit('resnet50 only for this measurement')
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def features(x01, site):
        x = (x01 - mean) / std
        with torch.no_grad():
            h = stem(x)
            h = net.layer1(h)
            if site in ('layer2', 'layer3', 'layer4'):
                h = net.layer2(h)
            if site in ('layer3', 'layer4'):
                h = net.layer3(h)
            if site == 'layer4':
                h = net.layer4(h)
            return h.mean(dim=(2, 3))                      # global average pool

    def consumer_response(x01):
        with torch.no_grad():
            return net((x01 - mean) / std)

    out = {'backbone': a.backbone, 'n_crops': int(X.shape[0]), 'protocol':
           'pairwise Lipschitz ratio of the truth action at the site, relative-L2 metric; '
           'and the same with the consumer response metric', 'sites': {}}
    pairs = [(i, j) for i in range(min(60, X.shape[0])) for j in range(i + 1, min(60, X.shape[0]))]
    for site in a.sites.split(','):
        Z = features(X, site)
        rec = {}
        for fam, Xt in (('hue_90.0', h188.hue_shift(X, a.hue_deg)),
                        ('heat_2.0', h188.heat(X, a.heat_sigma))):
            Zt = features(Xt, site)
            ratio = []
            for i, j in pairs:
                d0 = float((Z[i] - Z[j]).norm())
                d1 = float((Zt[i] - Zt[j]).norm())
                if d0 > 1e-6:
                    ratio.append(d1 / d0)
            ratio = np.array(ratio)
            rec[fam] = {'median': float(np.median(ratio)),
                        'p90': float(np.percentile(ratio, 90)),
                        'mean': float(ratio.mean()),
                        'n_pairs': int(len(ratio))}
            # consumer-referenced version
            F0 = consumer_response(X); F1 = consumer_response(Xt)
            cr = []
            for i, j in pairs:
                d0 = float((F0[i] - F0[j]).norm())
                d1 = float((F1[i] - F1[j]).norm())
                if d0 > 1e-6:
                    cr.append(d1 / d0)
            rec[fam]['consumer_median'] = float(np.median(cr))
            rec[fam]['consumer_p90'] = float(np.percentile(cr, 90))
            rec[fam]['displacement_rel'] = float(
                (Zt - Z).norm(dim=1).mean() / Z.norm(dim=1).mean())
        out['sites'][site] = rec
        print(f"  {site}: " + ' | '.join(
            f"{k.split('_')[0]} med {v['median']:.3f} p90 {v['p90']:.3f} (cons med "
            f"{v['consumer_median']:.3f})" for k, v in rec.items()))
    json.dump(out, open(os.path.join(WORK, 'results', f'lipschitz_{a.backbone}.json'), 'w'), indent=1)
    print('->', f'results/lipschitz_{a.backbone}.json')


if __name__ == '__main__':
    main()
