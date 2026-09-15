#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""277_fig11_uses.py — Figure 11: what the interface does on real images.

Data study:
  * `detector_o8_multiseed.json` — five routes x three seeds for one shift; three seeds means the
    spread is visible, so each route is a strip of seed dots rather than a bare mean bar.
  * `usage_rule_scale_strat.json` — four routes and twelve categories, both long-labelled and
    one-dimensional; horizontal bars keep the labels readable and the ordering explicit.  The
    category panel carries the 30-degree reference the paper uses to call a read-out usable.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
ROUTE = [('null', 'no intervention'), ('O1_procrustes', 'orthogonal Procrustes'),
         ('O2_ridge', 'linear (ridge)'), ('O8_conv_residual', 'residual conv.'),
         ('O6_random', 'random orthogonal')]
RNAME = [('a_raw', 'raw'), ('b_dominant_fill', 'dominant fill'),
         ('c_shifted_fill_gtbox', 'shifted fill (box)'), ('d_shifted_fill_grabcut', 'shifted fill (grabcut)')]


def fig08():
    det = R('detector_o8_multiseed.json')['deltas']['90']['routes']
    u = R('usage_rule_scale_strat.json')
    fig, axes = fs.figure('double', ncols=3, height=2.9)
    a, b, c = axes

    y = np.arange(len(ROUTE))[::-1]
    for i, (key, _) in enumerate(ROUTE):
        seeds = det[key]['f1_per_seed']
        a.plot(seeds, [y[i]] * len(seeds), 'o', ms=3.0, mfc='white', mec=fs.HUE, mew=0.9, ls='none')
        a.plot(det[key]['f1_mean'], y[i], '|', color=fs.HEAT, ms=9, mew=1.4)
    a.set_yticks(y)
    a.set_yticklabels([lab for _, lab in ROUTE])
    a.set_ylim(-0.7, len(ROUTE) - 0.3)
    a.set_xlim(-0.05, 1.05)
    a.set_xlabel('box F1 at IoU 0.5')
    a.set_ylabel('route')
    fs.box_axes(a, grid='x')
    fs.panel(a, 'a')

    vals = [u[k]['median'] for k, _ in RNAME]
    yb = np.arange(len(RNAME))[::-1]
    b.barh(yb, vals, 0.55, color=fs.HUE, edgecolor='white', linewidth=0.5)
    b.set_yticks(yb)
    b.set_yticklabels([lab for _, lab in RNAME])
    b.set_ylim(-0.6, len(RNAME) - 0.4)
    b.set_xlim(0, 78)
    b.set_xlabel('median hue error (deg)')
    b.set_ylabel('route')
    fs.box_axes(b, grid='x')
    fs.panel(b, 'b')

    cats = sorted(u['by_category'].items(), key=lambda kv: kv[1]['median'])
    yc = np.arange(len(cats))
    c.barh(yc, [v['median'] for _, v in cats], 0.7, color=fs.GREY, edgecolor='white', linewidth=0.4)
    c.axvline(30, color=fs.REF, lw=0.8, ls='--')
    c.set_yticks(yc)
    c.set_yticklabels([k for k, _ in cats], fontsize=6.5)
    c.set_ylim(-0.7, len(cats) - 0.3)
    c.set_xlim(0, 36)
    c.set_xlabel('median hue error (deg)')
    c.set_ylabel('category')
    fs.box_axes(c, grid='x')
    fs.panel(c, 'c')

    fs.panel_report(fig)
    fs.save(fig, 'fig11_uses', max_text=3, min_pt=6.5)


if __name__ == '__main__':
    fig08()
