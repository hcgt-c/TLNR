#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""279_fig08_action_comparison.py — Figure 8: the measured action against learnable ones.

Data study (`interface_action_comparison.json`): four arms x three seeds x five error metrics, all in
degrees, all small.  The arms are categories (four long names), the seeds are replicates, and the
fixed arm sits at exactly zero -- which a logarithmic axis cannot show, so everything stays linear.
Encoding: three facets sharing the arm axis, each a strip of the three seed values with the mean
marked; the composition panel carries the zero reference, and the increment panel separates the two
never-fitted increments by marker.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
ARMS = [('fixed', 'fixed $A_\\Delta$'), ('gen', 'learned generator'), ('lin', 'learned linear'),
        ('mlp', 'generic map')]


def fig10():
    d = R('interface_action_comparison.json')['per_seed']
    fig, axes = fs.figure('double', ncols=3, height=2.5, sharey=True)
    a, b, c = axes
    y = np.arange(len(ARMS))[::-1]

    for i, (arm, _) in enumerate(ARMS):
        rec = d[arm]
        a.plot([r['e0_median'] for r in rec], [y[i]] * len(rec), 'o', ms=3.2, mfc='white',
               mec=fs.HUE, mew=0.9, ls='none')
        a.plot(np.mean([r['e0_median'] for r in rec]), y[i], '|', color=fs.HEAT, ms=9, mew=1.4)
        b.plot([r['inc37_median'] for r in rec], [y[i] - 0.13] * len(rec), 'o', ms=2.8,
               mfc='white', mec=fs.HUE, mew=0.8, ls='none')
        b.plot([r['inc90_median'] for r in rec], [y[i] + 0.13] * len(rec), 's', ms=2.8,
               mfc='white', mec=fs.HEAT, mew=0.8, ls='none')
        c.plot([r['comp_defect_median'] for r in rec], [y[i]] * len(rec), 'o', ms=3.2, mfc='white',
               mec=fs.GREEN, mew=0.9, ls='none')

    a.set_yticks(y)
    a.set_yticklabels([lab for _, lab in ARMS])
    a.set_ylim(-0.7, len(ARMS) - 0.3)
    a.set_xlim(-0.3, 7.5)
    a.set_xlabel('read-out error (deg)')
    a.set_ylabel('arm')
    fs.box_axes(a, grid='x')
    fs.panel(a, 'a')

    b.set_xlim(-0.3, 12)
    b.set_xlabel('increment error (deg)')
    b.plot([], [], 'o', mfc='white', mec=fs.HUE, ls='none', label=r'$37°$')
    b.plot([], [], 's', mfc='white', mec=fs.HEAT, ls='none', label=r'$90°$')
    b.legend(loc='lower right', ncol=2)
    fs.box_axes(b, grid='x')
    fs.panel(b, 'b')

    c.axvline(0, color=fs.REF, lw=0.7, ls=':')
    c.set_xlim(-0.6, 12)
    c.set_xlabel('composition defect (deg)')
    fs.box_axes(c, grid='x')
    fs.panel(c, 'c')

    fs.panel_report(fig)
    fs.save(fig, 'fig08_action_comparison', max_text=5)


if __name__ == '__main__':
    fig10()
