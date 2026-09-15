#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""275_fig07_depth.py — Figure 7: what depth does, and what it does it to.

Data study:
  * `depth_map_summary.json` — four backbones x two algebras x four depths, one transfer each.  Four
    series per panel is the comfortable limit, so the two algebras get one panel each instead of eight
    lines in one axes; depth is an ordinal 1..4, so the x axis is categorical, not continuous.
  * `quotientization.json` — the crossed design holds four backbone curves per cell; plotting them
    all would be sixteen lines.  The quantity the claim is about is the *trend over depth*, so each
    cell is summarised by its Spearman rho per backbone and shown as a strip of four rho values with
    the median marked; the reference at 0 is "no trend".
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
BB = [('resnet50', 'ResNet-50'), ('convnext', 'ConvNeXt-T'), ('vitb16', 'ViT-B/16'),
      ('dinov2b14', 'DINOv2-B/14')]
CELLS = [('resnet50,convnext,vitb16,dinov2b14_seed0', 'hue / hue'),
         ('resnet50,convnext,vitb16,dinov2b14_seed0_hue90.0_rohf', 'hue / high-freq'),
         ('resnet50,convnext,vitb16,dinov2b14_seed0_heat2.0_rohue', 'heat / hue'),
         ('resnet50,convnext,vitb16,dinov2b14_seed0_heat2.0_rohf', 'heat / high-freq')]


def site_order(names):
    return sorted(names, key=lambda s: int(re.search(r'(\d+)', s).group(1)))


def fig06():
    d = R('depth_map_summary.json')
    q = R('quotientization.json')
    fig, axes = fs.figure('double', ncols=3, height=2.7)
    a, b, c = axes

    for ax, family, letter in ((a, 'hue_90.0', 'a'), (b, 'heat_2.0', 'b')):
        for i, (bb, label) in enumerate(BB):
            rec = d[f'{bb}|{family}']
            sites = site_order(rec['sites'])
            tf = [rec['sites'][s]['O2_transfer_mean'] for s in sites]
            ax.plot(range(1, len(tf) + 1), tf, '-o', color=fs.C[i], ms=3.2, mfc='white', mew=0.9,
                    label=label)
        vals = [r['sites'][s]['O2_transfer_mean'] for r in
                (d[f'{b2}|{family}'] for b2, _ in BB) for s in site_order(r['sites'])]
        lo, hi = min(vals), max(vals)
        pad = 0.06 * (hi - lo)
        ax.axhline(0, color=fs.REF, lw=0.7, ls=':')
        ax.set_xticks(range(1, 5))
        ax.set_xlabel('relative depth')
        ax.set_ylabel(r'transfer $T_F$')
        ax.set_ylim(lo - pad, hi + pad)
        ax.legend(loc='lower left', ncol=2)
        fs.box_axes(ax, grid='y')
        fs.panel(ax, letter)

    # ---- (c) the crossed design, summarised by its trend over depth -----------------------------
    rhos = []
    for key, _ in CELLS:
        row = []
        for bb, _ in BB:
            sites = site_order(q[key]['backbones'][bb].keys())
            vals = [q[key]['backbones'][bb][s]['matched_rank_control']['visible_over_random_effective']
                    for s in sites]
            row.append(spearmanr(range(len(vals)), vals).statistic)
        rhos.append(row)
    rhos = np.array(rhos)
    x = np.arange(len(CELLS))
    for i in range(len(BB)):
        c.plot(x, rhos[:, i], 'o', color=fs.GREY, ms=3.0, mfc='white', mew=0.8, ls='none')
    c.plot(x, np.median(rhos, axis=1), '_', color=fs.HUE, ms=11, mew=1.6)
    c.axhline(0, color=fs.REF, lw=0.7, ls=':')
    c.set_xticks(x)
    c.set_xticklabels([lab for _, lab in CELLS], rotation=20, ha='right')
    c.set_ylim(-1.05, 1.05)
    c.set_xlabel('readout / transformation cell')
    c.set_ylabel(r'Spearman $\rho$ over depth')
    fs.box_axes(c, grid='y')
    fs.panel(c, 'c')

    fs.panel_report(fig)
    fs.save(fig, 'fig07_depth', max_text=6)


if __name__ == '__main__':
    fig06()
