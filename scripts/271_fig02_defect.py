#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""271_fig03_defect.py — Figure 3: the existence condition and the defect law.

The plot design follows the data, not habit.  Reading `results/transport_theory.json` first:

  * twelve sweep entries, but **two of them are controls** (kappa ~ 4e-16, the kernel-preserving
    case of Theorem 1: defect < 2e-6).  The sweep proper is ten points with kappa in [0.40, 9.34]
    and defect in [0.373, 0.996];
  * therefore a log axis over "sixteen decades" is set by the controls and squeezes the ten real
    points into the top 1.5% of the panel, which is what made the earlier version unreadable;
  * the deviation |measured - closed form| is a hump: 2.5e-4 at kappa = 0.40, a maximum of 9.1e-3
    near kappa = 1.2, and 1.5e-3 at kappa = 9.34.

So the sweep is drawn in its own coordinates (linear kappa, linear defect), the deviation gets its
own panel over the same range with a reference line at 0.01, and the two control entries are stated
in the caption rather than plotted.  Output: paper/figures_final/fig03_defect.{pdf,png}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
FLOOR = 1e-3          # entries below this kappa are controls (the no-mixing limit), not sweep points


def sweep(d):
    """(kappa, defect, deviation) per retained rank, controls removed."""
    out = {}
    for rank in (16, 32):
        rows = sorted(((v['kappa'], 1.0 - v['T_F_squared']) for k, v in d['theorem2_closed_form'].items()
                       if k.startswith(f'rank{rank}_')))
        rows = [(k, y, abs(y - k / np.sqrt(1 + k ** 2))) for k, y in rows if k > FLOOR]
        out[rank] = rows
    return out


def fig02():
    d = R('transport_theory.json')
    sw = sweep(d)
    fig, axes = fs.figure('double', ncols=2, nrows=2, height=4.6)
    a, b, c, e = axes

    # ---- (a) the law over the swept range ------------------------------------------------------
    kk = np.logspace(np.log10(0.3), 1.0, 400)
    kk = kk[kk / np.sqrt(1 + kk ** 2) >= 0.35]   # draw only where the truncated axis shows it
    a.plot(kk, kk / np.sqrt(1 + kk ** 2), '-', color=fs.REF, lw=1.0, label='closed form')
    for rank, mk, col, lab in ((16, 'o', fs.HUE, r'$r=16$'), (32, 's', fs.HEAT, r'$r=32$')):
        k, y, _ = zip(*sw[rank])
        a.plot(k, y, mk, ms=3.6, mfc='white', mec=col, mew=0.9, ls='none', label=lab)
    fs.log_axis(a, 'x', 0.3, 10.0, scalar=True)  # log kappa: the sweep spans a factor of 23, and on a log axis
    a.set_ylim(0.35, 1.02)          # the curve runs corner to corner instead of hugging the top
    a.set_xlabel(r'mixing ratio $\kappa$')
    a.set_ylabel(r'defect $1-T_{\mathrm{RMS}}$')
    fs.fixed_ticks(a, 'x', (0.5, 1, 2, 5, 10))
    fs.legend_auto(a)
    fs.box_axes(a, grid='both')
    fs.panel(a, 'a')

    # ---- (b) how far the measurement is from the closed form -----------------------------------
    for rank, mk, col, lab in ((16, 'o', fs.HUE, r'$r=16$'), (32, 's', fs.HEAT, r'$r=32$')):
        k, _, dv = zip(*sw[rank])
        b.plot(k, dv, mk, ms=3.6, mfc='white', mec=col, mew=0.9, ls='none', label=lab)
    b.axhline(0.01, color=fs.REF, lw=0.7, ls=':')
    fs.log_axis(b, 'x', 0.3, 10.0, scalar=True)
    b.set_ylim(0, 0.0115)
    b.set_xlabel(r'mixing ratio $\kappa$')
    b.set_ylabel('absolute deviation')
    fs.fixed_ticks(b, 'x', (0.5, 1, 2, 5, 10))
    fs.legend_auto(b)
    fs.box_axes(b, grid='both')
    fs.panel(b, 'b')

    # ---- (c) retained rank is not the variable --------------------------------------------------
    rs = d['corollary3_rank_sweep']
    ranks = sorted(int(k[4:]) for k in rs)
    tf = [rs[f'rank{r}']['T_F'] for r in ranks]
    c.plot(ranks, tf, '-', color=fs.HUE, lw=1.0)
    c.plot(ranks, tf, 'o', ms=3.6, mfc='white', mec=fs.HUE, mew=0.9)
    c.set_xlabel('retained rank')
    c.set_ylabel(r'transfer $T_F$')
    c.set_ylim(0.3, 0.72)
    fs.integer_ticks(c, 'x', n=5)
    fs.box_axes(c)
    fs.panel(c, 'c')

    # ---- (d) the score follows the geometry of the split, not the discarded energy --------------
    pv = d['proxy_bottom_variance']
    keys = sorted(pv, key=lambda k: float(k[5:]))
    xs = [float(k[5:]) for k in keys]
    e.plot(xs, [pv[k]['T_F'] for k in keys], '-', color=fs.HUE, lw=1.0)
    e.plot(xs, [pv[k]['T_F'] for k in keys], 'o', ms=3.6, mfc='white', mec=fs.HUE, mew=0.9,
           label=r'$T_F$')
    e.plot(xs, [pv[k]['displacement_share_in_bottom50'] for k in keys], '--', color=fs.HEAT, lw=1.0)
    e.plot(xs, [pv[k]['displacement_share_in_bottom50'] for k in keys], 's', ms=3.6, mfc='white',
           mec=fs.HEAT, mew=0.9, label='share in bottom half')
    e.set_xlabel('input anisotropy')
    e.set_ylabel('value')
    e.set_ylim(0.0, 0.62)
    e.set_xticks(xs)
    fs.legend_auto(e)
    fs.box_axes(e)
    fs.panel(e, 'd')

    fs.panel_report(fig)
    fs.save(fig, 'fig03_defect', max_text=5)


if __name__ == '__main__':
    fig02()
