#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""273_fig05_provenance.py — Figure 5: where the organisation comes from.

Data study (`results/harmonic_controls.json`, arms = the eight sites compared under one protocol):
the data is a *categorical comparison* — eight arms (two input-space, three trained, three untrained)
times three continuous metrics of different units.  Encoding follows: one horizontal dot plot per
metric, laid out as facets that share the arm axis, so the eight long names are printed once and the
panels cannot drift.  Values per metric are single seeds per arm, so no error bars are drawn and the
caption says so.  The harmonic share is shown twice per arm (raw, standardised) because the two
differ for the trained sites and that difference is the point.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
ARMS = ['pixels_hue', 'pixels_rgb', 'trained_y3', 'trained_y5', 'trained_y7',
        'random_y3', 'random_y5', 'random_y7']
# the axis carries reader-facing names, not the internal arm keys
ARM_LBL = {'pixels_hue': 'input hue', 'pixels_rgb': 'input RGB',
           'trained_y3': 'trained, stage 3', 'trained_y5': 'trained, stage 5',
           'trained_y7': 'trained, stage 7', 'random_y3': 'untrained, stage 3',
           'random_y5': 'untrained, stage 5', 'random_y7': 'untrained, stage 7'}
FACE = {'input': 'input pixels', 'trained': 'trained', 'random': 'untrained'}


def fig04():
    a = R('harmonic_controls.json')['arms']
    fig, axes = fs.figure('double', ncols=3, height=2.9, sharey=True)
    p1, p2, p3 = axes
    y = np.arange(len(ARMS))[::-1]
    raw = [a[k]['k1k2_share_mean'] for k in ARMS]
    std = [a[k]['k1k2_share_mean_standardised'] for k in ARMS]
    ph = [a[k]['cross_shape_phase_cos_mean'] for k in ARMS]
    r_hue = [a[k]['readout_median_deg'] for k in ARMS]
    r_mlp = [a[k]['mlp_readout_median_deg'] for k in ARMS]

    p1.plot(raw, y, 'o', ms=3.6, mfc=fs.HUE, mec=fs.HUE, ls='none', label='raw')
    p1.plot(std, y, 's', ms=3.4, mfc='white', mec=fs.HEAT, mew=0.9, ls='none', label='standardised')
    p1.set_xlim(0.45, 1.02)
    p1.set_xlabel(r'$k_1{+}k_2$ share')
    p1.set_ylabel('arm')
    p1.legend(loc='lower left', ncol=2)
    fs.box_axes(p1, grid='x')
    fs.panel(p1, 'a')

    p2.plot(ph, y, 'o', ms=3.6, mfc=fs.GREEN, mec=fs.GREEN, ls='none')
    p2.set_xlim(0.9, 1.005)
    p2.set_xlabel('cross-shape phase cos')
    fs.box_axes(p2, grid='x')
    fs.panel(p2, 'b')

    p3.plot(r_hue, y, 'o', ms=3.6, mfc=fs.HUE, mec=fs.HUE, ls='none', label='code head')
    p3.plot(r_mlp, y, 's', ms=3.4, mfc='white', mec=fs.HEAT, mew=0.9, ls='none', label='MLP probe')
    p3.set_xscale('log')
    p3.set_xlim(0.4, 30)
    p3.set_xlabel('zero-shot hue error (deg)')
    p3.legend(loc='lower right', ncol=1)
    fs.box_axes(p3, grid='x')
    fs.panel(p3, 'c')

    p1.set_yticks(y)
    p1.set_yticklabels([ARM_LBL.get(k, k.replace('_', ' ')) for k in ARMS])
    p1.set_ylim(-0.7, len(ARMS) - 0.3)
    fs.panel_report(fig)
    fs.save(fig, 'fig05_provenance', max_text=5)


if __name__ == '__main__':
    fig04()
