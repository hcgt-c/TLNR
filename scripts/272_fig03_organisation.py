#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""272_fig04_organisation.py — Figure 4: the low-order, cross-shape-shared organisation.

The design follows the data study in `meta/FIGURE_SPEC.md`:

  * the harmonic content of an orbit is a *composition* (k1 + k2 + tail = 1 for every shape), so it
    is drawn as stacked bars, not as a spectrum line;
  * the plane-alignment table has only five samples per harmonic (the triangle is the reference), so
    each harmonic is drawn as a min--max bar with a median marker instead of six lines and a large
    legend;
  * the grey-axis data carries two quantities that both live in [0, 1] -- how much orbit energy
    survives at low saturation and how well the harmonic plane is preserved -- so they share one
    panel with two encodings and the empty band between them carries the legend;
  * `mean_winding` (2.0--2.8) and `intrinsic_dim` (d80 = 3) are single numbers per shape and repeat
    what the harmonic composition already says, so they are stated in the caption rather than given
    panels of their own.

Output: paper/figures_final/fig04_organisation.{pdf,png}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
SHAPES = ['triangle', 'rectangle', 'circle', 'star', 'pentagon', 'hexagon']
SHORT = ['tri', 'rect', 'circ', 'star', 'pent', 'hex']


def fig03():
    sp = R('dense_continuous_v3.json')
    d3 = R('dense3d_v4.json')
    fig, axes = fs.figure('double', ncols=3, height=2.6)
    a, b, c = axes
    x = np.arange(len(SHAPES))

    # ---- (a) harmonic composition: a stacked bar, because the parts sum to one -----------------
    k1 = np.array([sp['spectrum'][s]['k1'] for s in SHAPES])
    k2 = np.array([sp['spectrum'][s]['k2'] for s in SHAPES])
    tail = np.array([sp['spectrum'][s]['tail_k3plus'] for s in SHAPES])
    w = 0.66
    a.bar(x, k1, w, color=fs.HUE, edgecolor='white', linewidth=0.5, label=r'$k=1$')
    a.bar(x, k2, w, bottom=k1, color=fs.HEAT, edgecolor='white', linewidth=0.5, label=r'$k=2$')
    a.bar(x, tail, w, bottom=k1 + k2, color=fs.GREY, edgecolor='white', linewidth=0.5,
          label=r'$k\geq3$')
    a.set_xticks(x)
    a.set_xticklabels(SHORT)
    a.set_ylim(0, 1.25)
    a.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    a.set_ylabel('share of orbit energy')
    a.set_xlabel('shape')
    a.legend(loc='upper center', ncol=3, columnspacing=1.1, handlelength=1.1)
    fs.box_axes(a)
    fs.panel(a, 'a')

    # ---- (b) plane alignment across shapes, per harmonic ---------------------------------------
    ks = list(range(1, 11))
    vals = np.array([[sp['alignment_vs_triangle'][s][f'k{k}'] for s in SHAPES[1:]] for k in ks])
    lo, med, hi = vals.min(1), np.median(vals, 1), vals.max(1)
    b.vlines(ks, lo, hi, color=fs.HUE, lw=1.4, alpha=0.55)
    b.plot(ks, med, 'o', ms=3.4, mfc=fs.HUE, mec=fs.HUE)
    b.axhline(0.96, color=fs.REF, lw=0.7, ls=':')
    b.set_xticks(ks)
    b.set_xlim(0.5, 10.5)
    b.set_ylim(0.5, 1.03)
    b.set_xlabel(r'harmonic index $k$')
    b.set_ylabel(r'plane alignment (cos)')
    fs.box_axes(b, grid='y')
    fs.panel(b, 'b')

    # ---- (c) what happens on the grey axis: energy collapses, the plane survives ---------------
    ratio = np.array([d3['rings']['energy_gray_ratio'][s] for s in SHAPES])
    align = np.array([d3['rings']['plane_align_gray_vs_mid'][s][0] for s in SHAPES])
    c.bar(x, ratio, w, color=fs.GREEN, edgecolor='white', linewidth=0.5, label='orbit energy')
    c.plot(x, align, 'o-', color=fs.HUE, lw=1.0, ms=3.6, mfc='white', mec=fs.HUE, mew=0.9,
           label=r'$k=1$ plane')
    c.set_xticks(x)
    c.set_xticklabels(SHORT)
    c.set_ylim(0, 1.12)
    c.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    c.set_ylabel('ratio / alignment')
    c.set_xlabel('shape')
    c.legend(loc='center left', ncol=1)
    fs.box_axes(c)
    fs.panel(c, 'c')

    fs.panel_report(fig)
    fs.save(fig, 'fig04_organisation', max_text=5)


if __name__ == '__main__':
    fig03()
