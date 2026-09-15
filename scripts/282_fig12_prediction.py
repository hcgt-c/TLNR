#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""282_fig12_prediction.py — Figure 12: what a fitted law predicts out of sample.

Data study, before any layout was chosen.

(a) `kappa_predicts_transfer.json` carries, for each of the 42 reachable (backbone, configuration,
    depth) cells, the measured downstream transfer and the closed form's prediction from that cell's
    own kappa (`T_hat.cons_r4`).  Measured values run -0.40 to 0.71 and predictions 0.37 to 0.95; 90 %
    of the points sit above the identity, so the panel is a parity plot with the identity line, the
    bias and the mean absolute error stated in the panel, and the two algebras distinguished by
    marker.
(b) `two_term_heldout_resnet50.json` evaluates the two-term decomposition on a held-out split, and
    `two_term_decomposition_resnet50.json` evaluates the same quantity inside the fit split.  The two
    numbers for each of the eight (site, family) pairs differ by ten orders of magnitude, so the panel
    pairs them per site on a logarithmic axis: an encoding that shows the collapse rather than two
    separate bar groups.
(c) The same held-out file carries the rank-k subspace operator against the diagonal restriction.  The
    ratio, not the level, is the quantity: it is below one only at the first block and only for wide
    enough k, and above one at every deeper site, so depth is the natural abscissa and the three ranks
    are three curves per algebra.
(d) `heat_semigroup_structure.json` extrapolates the code to an unfitted diffusion time from the shared
    mean rate of the fit times (`shared_rate_prediction_at_held_out_t`), with an exact synthetic
    carrier run first as a control.  Three arms on a logarithmic gain axis against the copy baseline,
    with each arm's rate dispersion in the panel: the control is 14.9x the baseline and the two real
    sites are 0.83x and 0.75x.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
SITES = ['layer1', 'layer2', 'layer3', 'layer4']
DEPTH = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
FAM = {'hue': (fs.HUE, 'hue rotation'), 'heat': (fs.HEAT, 'heat diffusion')}


def fig12():
    kp = R('kappa_predicts_transfer.json')
    hold = R('two_term_heldout_resnet50.json')['sites']
    infit = R('two_term_decomposition_resnet50.json')['sites']
    semi = R('heat_semigroup_structure.json')

    fig, axes = fs.figure('double', ncols=2, nrows=2, height=4.9)
    a, b, c, d = axes

    # ---- (a) the closed form as a predictor across real sites ----------------------------------
    for fam, (col, name) in FAM.items():
        xs, ys = [], []
        for key, cv in kp['cells'].items():
            if not key.endswith(fam) and f'|{fam}' not in key:
                continue
            for sv in cv['sites'].values():
                if sv['reachable']:
                    xs.append(sv['measured_O2_transfer'])
                    ys.append(sv['T_hat']['cons_r4'])
        a.plot(xs, ys, 'o' if fam == 'hue' else 's', ms=3.2, mfc='white', mec=col, mew=0.9,
               ls='none', label=name)
    a.plot([0.30, 0.80], [0.30, 0.80], color=fs.REF, lw=0.8, ls=':')
    a.axhline(0.312, color=fs.GREY, lw=0.8, ls='--')
    a.text(-0.43, 0.322, 'mean measured $0.31$', fontsize=7, color=fs.REF, ha='left', va='bottom')
    a.set_xlim(-0.45, 0.80)
    a.set_ylim(0.22, 1.00)
    a.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    a.set_yticks([0.4, 0.6, 0.8, 1.0])
    a.set_xlabel('measured transfer $T_F$')
    a.set_ylabel('predicted $\\hat T_F$ from $\\kappa$')
    a.legend(loc='lower right', ncol=1)
    fs.box_axes(a, grid='both')
    fs.panel(a, 'a')

    # ---- (b) the sharing term inside and outside the fit split ---------------------------------
    off = {'hue': -0.16, 'heat': 0.16}
    labelled = set()
    for i, site in enumerate(SITES):
        for fam, (col, name) in FAM.items():
            wi = infit[site][f'{fam}_90.0' if fam == 'hue' else f'{fam}_2.0']['sharing_share_of_total']
            h = hold[site][f'{fam}_90.0' if fam == 'hue' else f'{fam}_2.0']
            wo = (h['S_pool'] - h['S_cell']) / h['S_pool']
            y = i + off[fam]
            b.plot([wi, wo], [y, y], '-', color=fs.GREY, lw=0.8, zorder=1)
            b.plot(wi, y, 'o', ms=3.4, mfc=col, mec=col, ls='none', zorder=3,
                   label='evaluated in the fit split' if 'in' not in labelled else None)
            b.plot(wo, y, 'o', ms=3.4, mfc='white', mec=col, mew=0.9, ls='none', zorder=3,
                   label='evaluated on held-out data' if 'out' not in labelled else None)
            labelled |= {'in', 'out'}
    b.set_yticks(np.arange(len(SITES)))
    b.set_yticklabels(['block 1', 'block 2', 'block 3', 'block 4'])
    b.set_ylim(-0.55, len(SITES) - 0.45)
    fs.log_axis(b, 'x', 5e-10, 3.0, scalar=False)
    fs.fixed_ticks(b, 'x', [1e-8, 1e-5, 1e-2, 1.0])
    b.set_xlabel('sharing term as a share of the realisation error')
    b.set_ylabel('depth')
    b.legend(loc='lower left', ncol=1)
    fs.box_axes(b, grid='x')
    fs.panel(b, 'b')

    # ---- (c) does cross-channel coupling predict better with rank and depth? --------------------
    ks = ['8', '16', '32']
    mk = {'8': 'o', '16': 's', '32': '^'}
    for fam, (col, name) in FAM.items():
        for k in ks:
            ys = [hold[s][f'{fam}_90.0' if fam == 'hue' else f'{fam}_2.0']['R_full'][k] /
                  hold[s][f'{fam}_90.0' if fam == 'hue' else f'{fam}_2.0']['R_diag'] for s in SITES]
            c.plot([DEPTH[s] for s in SITES], ys, '-' if fam == 'heat' else '--', color=col, lw=1.0,
                   marker=mk[k], ms=3.0, mfc='white', mec=col, mew=0.8,
                   label=(f'rank ${k}$' if fam == 'hue' else None))
    c.axhline(1.0, color=fs.REF, lw=0.8, ls=':')
    c.plot([], [], '-', color=fs.HEAT, lw=1.2, label='heat diffusion')
    c.plot([], [], '--', color=fs.HUE, lw=1.2, label='hue rotation')
    fs.log_axis(c, 'y', 0.18, 7.0, scalar=True)
    fs.fixed_ticks(c, 'y', [0.2, 0.5, 1, 2, 5])
    c.set_xticks([1, 2, 3, 4])
    c.set_xticklabels(['block 1', 'block 2', 'block 3', 'block 4'])
    c.set_xlim(0.7, 4.3)
    c.set_ylim(0.18, 7.0)
    c.set_xlabel('depth')
    c.set_ylabel('subspace error / diagonal error')
    c.legend(loc='upper left', ncol=2)
    fs.box_axes(c, grid='y')
    fs.panel(c, 'c')

    # ---- (d) extrapolation to a diffusion time that was never fitted ---------------------------
    arms = [('exact carrier', semi['control']['numbers'], fs.GREEN),
            ('layer 2', semi['sites']['layer2'], fs.HUE),
            ('layer 3', semi['sites']['layer3'], fs.HUE)]
    x = np.arange(len(arms))
    gain = [t[1]['shared_rate_prediction_at_held_out_t']['gain'] for t in arms]
    disp = [t[1]['rate_dispersion']['median_relative_sd_of_per_image_rate'] for t in arms]
    d.bar(x, gain, 0.55, color=[t[2] for t in arms], edgecolor=fs.INK, lw=0.5,
          hatch=['' if i == 0 else '///' for i in range(len(arms))])
    d.axhline(1.0, color=fs.REF, lw=0.8, ls=':')
    d.text(2.42, 1.12, 'copy baseline', fontsize=7, color=fs.REF, ha='right', va='bottom')
    d.set_xticks(x)
    d.set_xticklabels([f'{t[0]}\n{s:.2f}' for t, s in zip(arms, disp)])
    fs.log_axis(d, 'y', 0.4, 40.0, scalar=True)
    fs.fixed_ticks(d, 'y', [0.5, 1, 2, 5, 10, 20])
    d.set_ylim(0.4, 40.0)
    d.set_xlim(-0.6, len(arms) - 0.4)
    d.set_xlabel('arm (second line: rate dispersion)')
    d.set_ylabel('gain over copy at unfitted $t$')
    fs.box_axes(d, grid='y')
    fs.panel(d, 'd')

    fs.panel_report(fig)
    fs.save(fig, 'fig12_prediction', max_text=6)


if __name__ == '__main__':
    fig12()
