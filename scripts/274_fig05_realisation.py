#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""274_fig06_realisation.py — Figure 6: which realisations work, and what composing costs.

Data study: three different data shapes, three encodings.
  * `locality_*.json` — five sites, each with one local and one global gain over the copy baseline
    (0.08--15.9).  Paired horizontal dots on a log axis, with a reference line at 1 (no better than
    copy): the pairing is the message, so the two marks sit on one row.
  * `multipath_composition.json` — six records (two algebras x three seeds) with a transfer per path.
    Grouped bars, one panel per algebra, the direct fit drawn as a reference line, because the claim
    is "chains match the direct fit" and a line states that reference once.
  * the cyclic round trip and the single-step control are paths like any other and are kept in the bar
    order, so no separate panel is needed.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
LBL = {'resnet18_l2': 'ResNet-18 layer 2', 'resnet18_l3': 'ResNet-18 layer 3',
       'yolo11n_y3': 'YOLO11n stage 3', 'yolo11n_y5': 'YOLO11n stage 5',
       'yolo11n_y7': 'YOLO11n stage 7'}
SHORT = {'two_step_45+45': '45+45', 'two_step_30+60': '30+60',
         'two_step_60+30 (order swap)': '60+30', 'four_step_22.5x4': '4$\\times$22.5',
         'single_22.5': 'single 22.5', 'cyclic_120-120 (identity)': 'cyclic',
         'two_step_1.0+1.0': '2 steps', 'four_step_0.7071x4': '4 steps',
         'eight_step_0.5x8': '8 steps', 'single_0.5 (t*/8)': 'single $t^\\star/8$',
         'direct_fit': 'direct'}


def fig05():
    fig, axes = fs.figure('double', ncols=3, height=2.7)
    a, b, c = axes

    # ---- (a) local versus global, per site ------------------------------------------------------
    files = sorted(glob.glob(os.path.join(WORK, 'results', 'locality_*.json')))
    sites = [os.path.basename(f)[len('locality_'):-len('.json')] for f in files]
    loc = np.array([R(f'locality_{s}.json')['gain_local_over_copy'] for s in sites])
    glo = np.array([R(f'locality_{s}.json')['gain_global_over_copy'] for s in sites])
    y = np.arange(len(sites))[::-1]
    a.axvline(1.0, color=fs.REF, lw=0.7, ls=':')
    a.hlines(y, np.minimum(loc, glo), np.maximum(loc, glo), color=fs.GREY, lw=0.8, zorder=1)
    a.plot(glo, y, 's', ms=3.4, mfc='white', mec=fs.HEAT, mew=0.9, ls='none', label='global')
    a.plot(loc, y, 'o', ms=3.6, mfc=fs.HUE, mec=fs.HUE, ls='none', label='local')
    a.set_xscale('log')
    a.set_xlim(0.05, 60)
    a.set_yticks(y)
    a.set_yticklabels([LBL.get(s, s) for s in sites])
    a.set_ylim(-0.7, len(sites) - 0.3)
    a.set_xlabel('gain over the copy baseline')
    a.set_ylabel('site')
    a.legend(loc='lower right', ncol=1)
    fs.box_axes(a, grid='x')
    fs.panel(a, 'a')

    # ---- (b, c) composition at a fixed total, one panel per algebra -----------------------------
    for ax, family, letter in ((b, 'hue', 'b'), (c, 'heat', 'c')):
        recs = [v for v in R('multipath_composition.json')['records'].values() if v['family'] == family]
        keys = [k for k in recs[0]['paths'] if k.endswith('|O8')]

        def rank(key):
            stem = key[:-3]
            for i, pre in enumerate(('direct_fit', 'two_step', 'four_step', 'eight_step',
                                     'single', 'cyclic')):
                if stem.startswith(pre):
                    return i
            return 9

        order = sorted(keys, key=rank)
        means = np.array([[r['paths'][k]['transfer'] for r in recs] for k in order])
        x = np.arange(len(order))
        direct = means[0].mean()
        ax.axhline(direct, color=fs.REF, lw=0.8, ls='--')
        ax.bar(x, means.mean(1), 0.6, yerr=means.std(1),
               color=fs.HUE if family == 'hue' else fs.HEAT,
               edgecolor='white', linewidth=0.5, error_kw=dict(lw=0.7, capsize=1.6))
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(k[:-3], k[:-3].replace('_', ' ')) for k in order])
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(r'$T_F$')
        ax.set_xlabel('path (fixed total)')
        fs.box_axes(ax, grid='y')
        fs.panel(ax, letter)

    fs.panel_report(fig)
    fs.save(fig, 'fig06_realisation', max_text=4)


if __name__ == '__main__':
    fig05()
