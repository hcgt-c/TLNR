#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""278_fig02_fibre_test.py — Figure 2: the fibre condition, tested directly.

Data study (`star_existence_test.json`): for each backbone and algebra, a consumer-level preservation
rate at a tolerance quantile q, against a random-pair baseline that *is* q by construction, plus per
site rates at four depths.  The preservation rates (0.25--0.82) and the baseline (0.01) differ by two
orders of magnitude, so the panel that compares them plots the ratio to chance, not the raw rate; the
raw rates and the baseline go in the caption.  The per-site rates are four points per backbone, which
is a line per backbone, one panel per algebra.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(WORK, 'results', f)))
BB = [('resnet50', 'ResNet-50'), ('convnext', 'ConvNeXt-T'), ('vitb16', 'ViT-B/16'),
      ('dinov2b14', 'DINOv2-B/14')]
FAM = [('hue', 'hue'), ('heat', 'heat')]


def fig09():
    d = R('star_existence_test.json')['results']
    fig, axes = fs.figure('double', ncols=3, height=2.7)
    a, b, c = axes

    x = np.arange(len(BB))
    w = 0.34
    for j, (fam, lab) in enumerate(FAM):
        ratios = []
        for bb, _ in BB:
            entry = list(d[bb]['families'][fam].values())[0]
            q = entry['consumer']['q=0.01']
            ratios.append(q['preservation_rate'] / q['baseline_rate'])
        a.bar(x + (j - 0.5) * w, ratios, w, color=fs.HUE if fam == 'hue' else fs.HEAT,
              edgecolor='white', linewidth=0.5, label=lab)
    a.set_xticks(x)
    a.set_xticklabels([lab for _, lab in BB], rotation=18, ha='right')
    a.set_ylim(0, 95)
    a.set_xlabel('backbone')
    a.set_ylabel('preservation / chance')
    a.legend(loc='upper left', ncol=2)
    fs.box_axes(a, grid='y')
    fs.panel(a, 'a')

    for ax, fam, letter in ((b, 'hue', 'b'), (c, 'heat', 'c')):
        for i, (bb, label) in enumerate(BB):
            entry = list(d[bb]['families'][fam].values())[0]
            sites = sorted(entry['sites'].keys(),
                           key=lambda s: int(re.search(r'(\d+)', s).group(1)))
            vals = [entry['sites'][s]['q=0.01']['preservation_rate'] for s in sites]
            ax.plot(range(1, len(vals) + 1), vals, '-o', color=fs.C[i], ms=3.2, mfc='white',
                    mew=0.9, label=label)
        ax.axhline(0.01, color=fs.REF, lw=0.7, ls=':')
        ax.set_xticks(range(1, 5))
        ax.set_ylim(0, 0.92)
        ax.set_xlabel('relative depth')
        ax.set_ylabel('preservation rate')
        fs.box_axes(ax, grid='y')
        fs.panel(ax, letter)
    b.legend(loc='lower left', ncol=2)
    c.legend(loc='lower left', ncol=2)

    fs.panel_report(fig)
    fs.save(fig, 'fig02_fibre_test', max_text=6)


if __name__ == '__main__':
    fig09()
