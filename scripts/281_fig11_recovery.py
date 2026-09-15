#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""281_fig10_recovery.py — Figure 10: reading the transformation back out of the features.

Data study, before any layout was chosen.

(a) `fig_correspondence.json:panel_B_band_limited_vs_copy` is four (shape, band) settings of a
    zero-shot operator fit, each carrying the recovered operator's relative error and the copy
    baseline's error on the same held-out shape.  Two series, four categories, values 0.17--2.90 %:
    a grouped bar on a log axis is the only encoding that shows both the level and the ratio, and the
    ratio is the quantity of interest, so each recovered bar carries its gain.
(b) `code3d_sv_lowext.json:per_cell[*].phase_deg_by_hue` holds one recovered phase per known hue for
    each of the 64 (saturation, value) cells -- this is the recovery itself, so it is drawn as a
    parity plot against the true hue rather than as an error summary.  Sweeping the cells shows two
    regimes and both are plotted: at s=v=1.00 and s=v=0.50 the cloud tracks the diagonal to a median
    of 1.6 and 1.1 degrees, while at s=v=0.02 the twelve phases are all near 240--300 degrees whatever
    the hue.  That cell is not merely hard: its `rgb_headroom.n_distinct_rgb_at_12_hues` is 1, i.e. the
    twelve hues render to the same pixel RGB, so there is no physical hue to recover.
(c) The same file carries `rgb_headroom.max_channel_range_8bit` per cell, which is exactly how much
    physical hue exists at that cell.  Median recovery error against that range over the 64 cells is a
    monotone relation (Spearman -0.93 over the 61 cells with a non-degenerate reference) and the three
    cells whose range is 0 have the largest errors, so the panel is drawn as error against reference
    range with the degenerate cells kept visible off to the left of a divider instead of being dropped.
(d) `second_attribute_chain.json:boundary_rows` is the code's report of an attribute after an
    intervention against the attribute the intervention actually realised -- a parity plot again, 160
    rows, with `clipped_frac` marking the rows where the intervention reached the gamut and could not
    change the pixel.  Markers distinguish the two attributes, colour the clipping state.

Style: log axes wherever the range is multiplicative; parity panels carry the identity line only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
HUES = np.arange(0, 360, 30)


def fig11():
    corr = R('fig_correspondence.json')['panel_B_band_limited_vs_copy']
    sv = R('code3d_sv_lowext.json')
    pc = sv['per_cell']
    sac = R('second_attribute_chain.json')['boundary_rows']

    fig, axes = fs.figure('double', ncols=2, nrows=2, height=4.9)
    a, b, c, d = axes

    # ---- (a) zero-shot operator recovery against the copy baseline -----------------------------
    lab = ['P$K$3', 'P$K$4', 'H$K$3', 'H$K$4']
    x = np.arange(len(lab))
    op = np.array(corr['relMSE_op_pct'])
    cp = np.array(corr['copy_pct'])
    w = 0.34
    a.bar(x - w / 2, cp, w, color=fs.TINT[fs.HUE], edgecolor=fs.HUE, lw=0.7, label='copy baseline')
    a.bar(x + w / 2, op, w, color=fs.HEAT, edgecolor=fs.HEAT, lw=0.7, label='recovered operator')
    fs.log_axis(a, 'y', 0.09, 9.0, scalar=True)
    fs.fixed_ticks(a, 'y', [0.1, 0.3, 1, 3])
    a.set_xticks(x)
    a.set_xticklabels(lab)
    a.set_ylim(0.09, 12)
    a.set_xlabel('shape and band (pentagon / hexagon, $K$ harmonic order)')
    a.set_ylabel('relative error (\\%)')
    a.legend(loc='upper right', ncol=1)
    fs.box_axes(a, grid='y')
    fs.panel(a, 'a')

    # ---- (b) recovered phase against the true hue ----------------------------------------------
    cells = [('s1.00_v1.00', fs.HUE, 'o', '$s,v=1.00$'),
             ('s0.50_v0.50', fs.CYAN, 's', '$s,v=0.50$'),
             ('s0.02_v0.02', fs.GREY, '^', '$s,v=0.02$')]
    for key, col, mk, name in cells:
        rec = pc[key]
        med = rec['median_err']
        b.plot(HUES, rec['phase_deg_by_hue'], mk, ms=3.4, mfc='white', mec=col, mew=0.9, ls='none',
               label=f'{name} (median ${med:.1f}^\\circ$)')
    b.plot([0, 330], [0, 330], color=fs.REF, lw=0.8, ls=':', label=None)
    b.set_xlim(-18, 348)
    b.set_ylim(-18, 378)
    fs.fixed_ticks(b, 'both', [0, 90, 180, 270, 360])
    b.set_xlabel('true hue (deg)')
    b.set_ylabel('phase read from the code (deg)')
    b.legend(loc='upper left', ncol=1)
    fs.box_axes(b, grid='both')
    fs.panel(b, 'b')

    # ---- (c) recovery error against how much physical hue exists -------------------------------
    rng = np.array([pc[k]['rgb_headroom']['max_channel_range_8bit'] for k in pc])
    err = np.array([pc[k]['median_err'] for k in pc])
    ok = rng > 0
    c.plot(rng[ok], err[ok], 'o', ms=3.0, mfc='white', mec=fs.HUE, mew=0.8, ls='none',
           label='one (saturation, value) cell')
    xd = 0.55
    c.plot([xd] * int((~ok).sum()), err[~ok], '^', ms=3.6, mfc=fs.GREY, mec=fs.INK, mew=0.5,
           ls='none', label='reference degenerate (constant RGB)')
    c.axvline(0.85, color=fs.REF, lw=0.7, ls='--')
    c.text(0.14, 0.96, 'no physical hue exists\nat these three cells', fontsize=7, color=fs.INK,
           ha='left', va='top', transform=c.transAxes)
    edges = [1, 5, 20, 60, 150, 255]
    bm = [np.median(err[ok & (rng >= lo) & (rng <= hi)]) for lo, hi in
          zip(edges[:-1], edges[1:])]
    bc = [np.sqrt(lo * hi) for lo, hi in zip(edges[:-1], edges[1:])]
    c.plot(bc, bm, '-', color=fs.HEAT, lw=1.1, marker='D', ms=2.8, mec=fs.HEAT, label='bin median')
    fs.log_axis(c, 'both', 0.4, 400, scalar=True)
    fs.fixed_ticks(c, 'x', [0.5, 1, 5, 20, 60, 150])
    fs.fixed_ticks(c, 'y', [0.5, 2, 10, 60])
    c.set_ylim(0.4, 400)
    c.set_xlabel('range of the twelve rendered RGB values (8-bit levels)')
    c.set_ylabel('median recovery error (deg)')
    c.legend(loc='lower left', ncol=1)
    fs.box_axes(c, grid='y')
    fs.panel(c, 'c')

    # ---- (d) the code's report of an intervened attribute --------------------------------------
    for attr, mk, col, name in (('v', 'o', fs.HUE, 'value'), ('s', 's', fs.HEAT, 'saturation')):
        rows = sac[attr]
        for clip, alpha in ((0.0, 1.0), (1.0, 0.0)):
            sel = [r for r in rows if r['clipped_frac'] == clip]
            d.plot([r['true_new'] for r in sel], [r['reported'] for r in sel], mk, ms=3.0,
                   mfc=col if alpha else 'white', mec=col, mew=0.8, alpha=alpha, ls='none',
                   label=(f'{name}, intervention realised' if clip == 0.0 else None))
    d.plot([], [], 'o', mfc='white', mec=fs.REF, mew=0.8, ls='none', label='intervention clipped')
    d.plot([0.2, 1.02], [0.2, 1.02], color=fs.REF, lw=0.8, ls=':')
    d.set_xlim(0.2, 1.04)
    d.set_ylim(0.2, 1.04)
    fs.fixed_ticks(d, 'both', [0.2, 0.4, 0.6, 0.8, 1.0])
    d.set_xlabel('attribute the intervention realised')
    d.set_ylabel('attribute the code reports')
    d.legend(loc='upper left', ncol=1)
    fs.box_axes(d, grid='both')
    fs.panel(d, 'd')

    fs.panel_report(fig)
    fs.save(fig, 'fig10_recovery', max_text=6)


if __name__ == '__main__':
    fig11()
