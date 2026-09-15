#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""270_fig01_object.py — Figure 1: the correspondence and the four-block decomposition.

Paper-style line art, not a flow chart: a commutative square in panel (a) and the block matrix of
the induced linear map in panel (b).  Labels are mathematical symbols only; every sentence and every
number lives in the caption.  Output: paper/figures_final/fig01_object.{pdf,png}.

Data: none (both panels are schematic and carry no measurements).
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs


def arrow(ax, p, q, colour=fs.INK, lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=lw,
                                 color=colour, shrinkA=0, shrinkB=0))


def fig01():
    fig, axes = fs.figure('double', ncols=2, height=2.55)
    ax, bx = axes

    # ---------------- panel (a): the commutative square ----------------
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_axis_off()
    P = {'s': (0.26, 0.74), 'ts': (0.74, 0.74), 'z': (0.26, 0.26), 'rz': (0.74, 0.26)}
    for xy in P.values():
        ax.plot(*xy, marker='o', ms=2.4, color=fs.INK, zorder=3)
    arrow(ax, P['s'], P['ts'])                     # tau: the world
    arrow(ax, P['s'], P['z'])                      # psi: the encoder
    arrow(ax, P['ts'], P['rz'])                    # psi: the encoder
    arrow(ax, P['z'], P['rz'], colour=fs.HUE)      # rho_tau: the operator we look for
    ax.text(0.50, 0.78, r'$\tau$', ha='center', va='bottom', fontsize=8)
    ax.text(0.50, 0.22, r'$\rho_\tau$', ha='center', va='top', fontsize=8)
    ax.text(0.22, 0.50, r'$\psi$', ha='right', va='center', fontsize=8)
    ax.text(0.78, 0.50, r'$\psi$', ha='left', va='center', fontsize=8)
    ax.text(0.23, 0.80, r'$s$', ha='right', va='bottom', fontsize=8)
    ax.text(0.77, 0.80, r'$\tau s$', ha='left', va='bottom', fontsize=8)
    ax.text(0.23, 0.20, r'$z$', ha='right', va='top', fontsize=8)
    ax.text(0.77, 0.20, r'$\rho_\tau z$', ha='left', va='top', fontsize=8)
    ax.text(0.50, 0.94, r'$\rho_\tau\,\psi=\psi\,\tau$', ha='center', va='center', fontsize=8)
    fs.panel(ax, 'a', dx=0.005, dy=0.93)

    # ---------------- panel (b): the four blocks ----------------
    bx.set_xlim(-0.55, 2.15)
    bx.set_ylim(-0.50, 2.15)
    bx.set_axis_off()
    blocks = {(0, 1): 'A', (1, 1): 'C', (0, 0): 'D', (1, 0): 'B'}
    for (i, j), lab in blocks.items():
        accent = lab == 'C'
        bx.add_patch(Rectangle((i, j), 1, 1,
                               facecolor=fs.TINT[fs.HUE] if accent else 'white',
                               edgecolor=fs.HUE if accent else fs.INK,
                               lw=0.7 if accent else 0.6, zorder=1))
        bx.text(i + 0.5, j + 0.5, lab, ha='center', va='center', fontsize=9,
                color=fs.HUE if accent else fs.INK)
    bx.text(0.5, 2.06, r'$\mathcal{R}$', ha='center', va='bottom', fontsize=8)
    bx.text(1.5, 2.06, r'$\mathcal{K}$', ha='center', va='bottom', fontsize=8)
    bx.text(-0.10, 1.5, r'$\mathcal{R}$', ha='right', va='center', fontsize=8)
    bx.text(-0.10, 0.5, r'$\mathcal{K}$', ha='right', va='center', fontsize=8)
    bx.text(1.0, -0.28, r'$g$', ha='center', va='top', fontsize=8)
    fs.panel(bx, 'b', dx=0.005, dy=0.93)

    fs.save(fig, 'fig01_object', max_text=11)


if __name__ == '__main__':
    fig01()
