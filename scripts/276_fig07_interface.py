#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""276_fig09_interface.py — Figure 9: the interface as built, and where it holds.

Data study:
  * `code3d_sv_lowext.json` carries two 8x8 grids over the same (saturation, value) domain: the
    readout error (3.2--94 deg, three orders of magnitude) and the code magnitude (0.04--0.41).  A
    2-D continuous field on a regular grid is a heat map; the error spans decades, so it gets a
    logarithmic colour scale, and the two panels share both axes because they describe the same
    domain.
  * `usage_rule_scale_strat.json` has 120 real regions with a concentration statistic and a per-route
    hue error: a scatter, with the pre-registered threshold drawn as a line.
  * `region_rho_demo.json` has two wall-clock numbers four orders of magnitude apart: two horizontal
    bars on a log axis.
"""
import json
import os
import sys

import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))


def fig07():
    g = R('code3d_sv_lowext.json')
    u = R('usage_rule_scale_strat.json')
    p = R('region_rho_demo.json')
    fig, axes = fs.figure('double', ncols=2, nrows=2, height=4.7)
    a, b, c, e = axes

    s = g['grid_s']
    v = g['grid_v']
    ext = (0, len(s), 0, len(v))
    err = np.array(g['mean_err_grid_v_by_s'])
    mag = np.array(g['z1_mag_grid_v_by_s'])
    ticks = [0, 2, 4, 6, 7]
    labels = [f'{x:g}' for x in [s[i] for i in ticks]]

    im = a.imshow(err, origin='lower', aspect='auto', extent=ext, cmap='cividis',
                  norm=LogNorm(vmin=3, vmax=100))
    a.set_xticks([i + 0.5 for i in ticks])
    a.set_xticklabels(labels)
    a.set_yticks([i + 0.5 for i in ticks])
    a.set_yticklabels(labels)
    a.set_xlabel('saturation')
    a.set_ylabel('value')
    fig.colorbar(im, ax=a, label='readout error (deg)', fraction=0.046, pad=0.02)
    fs.panel(a, 'a')

    im2 = b.imshow(mag, origin='lower', aspect='auto', extent=ext, cmap='cividis')
    b.set_xticks([i + 0.5 for i in ticks])
    b.set_xticklabels(labels)
    b.set_yticks([i + 0.5 for i in ticks])
    b.set_yticklabels(labels)
    b.set_xlabel('saturation')
    b.set_ylabel('value')
    fig.colorbar(im2, ax=b, label='code magnitude', fraction=0.046, pad=0.02)
    fs.panel(b, 'b')

    conc = np.array([r['concentration'] for r in u['rows']])
    err_raw = np.array([r['a_raw'] for r in u['rows']])
    c.axvline(0.70, color=fs.REF, lw=0.8, ls='--')
    c.plot(conc, err_raw, 'o', ms=2.4, mfc=fs.HUE, mec='none')
    c.set_yscale('log')
    c.set_ylim(0.4, 200)
    c.set_xlim(0, 1.02)
    c.set_xlabel('hue concentration')
    c.set_ylabel('readout error (deg)')
    fs.box_axes(c, grid='both')
    fs.panel(c, 'c')

    times = [p['region_rho_cost_s'] * 1e6, p['pixel_route_cost_s'] * 1e3]
    y = [1, 0]
    e.barh(y, times, 0.5, color=[fs.HUE, fs.HEAT], edgecolor='white', linewidth=0.5)
    e.set_xscale('log')
    e.set_yticks(y)
    e.set_yticklabels(['region shift', 'pixel re-forward'])
    e.set_xlim(5, 1e5)
    e.set_xlabel('wall clock ($\\mu$s)')
    e.set_ylabel('operation')
    e.set_ylim(-0.6, 1.6)
    fs.box_axes(e, grid='x')
    fs.panel(e, 'd')

    fs.panel_report(fig)
    fs.save(fig, 'fig09_interface', max_text=3)


if __name__ == '__main__':
    fig07()
