#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figstyle.py — one visual language for every figure in the paper, plus a layout audit.

Rules follow the usual journal guidance for line art (IEEE/Nature-style): vector output, 300-600 dpi
raster preview, fonts embedded, 7-8 pt minimum at final print size, line widths 0.5-1.5 pt, one
consistent palette across every figure, top/right spines dropped on data panels, and no text inside
a panel beyond axis labels, ticks and one legend.

Enforced by :func:`audit`, which runs before any file is written:

  * **No text inside a panel except axis labels, tick labels and at most one legend.**  Panel titles,
    sentences, bullet text and annotations carrying numbers are forbidden: the claim and every number
    belong in the caption.
  * **Nothing is placed by hand where the layout engine can place it.**  Figures use
    ``constrained_layout``; the only hand-placed artists are arrow/glyph labels in schematics.
  * **One typeface, fixed sizes**: STIX (Times-like, ships with matplotlib) for text and maths;
    axis labels 8 pt, ticks 7 pt, legend 7 pt, panel letters 8 pt bold.  Nothing below 7 pt.
  * **Vector first**: every figure is written as PDF plus a 400 dpi PNG preview.
  * **Widths follow the class**: 3.4 in single column, 5.0 in 1.5 columns, 7.0 in double column.

Colour follows Paul Tol's *bright* scheme (colour-blind safe; Tol, SRON technical note), used in a
fixed order so that a colour means the same thing in every figure of the paper:

    blue   #4477AA  hue / rotation (the group family)
    red    #EE6677  heat / diffusion (the semigroup family)
    green  #228833  measured quantities that are not a transformation
    yellow #CCBB44  controls
    cyan   #66CCEE  secondary series
    purple #AA3377  reference constructions
    grey   #BBBBBB  null / random controls

Text and axes are dark grey (``INK``) rather than pure black; grid lines are light and sit behind the
data; ``REF`` is the colour of closed-form and identity lines.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(WORK, 'paper', 'figures_final')

WIDTH = {'single': 3.40, 'onehalf': 5.00, 'double': 7.00}

# ---- Paul Tol "bright", fixed order; semantics documented in the module docstring ----------------
HUE, HEAT, GREEN, YELLOW, CYAN, PURPLE, GREY = ('#4477AA', '#EE6677', '#228833', '#CCBB44',
                                                '#66CCEE', '#AA3377', '#BBBBBB')
C = [HUE, HEAT, GREEN, YELLOW, CYAN, PURPLE, GREY]
INK = '#262626'          # text, axes, markers
REF = '#4D4D4D'          # reference curves and identity lines
GRID = '#E4E4E4'
TINT = {HUE: '#DCE9F5', HEAT: '#FBE0E3', GREEN: '#DDEDDD'}   # 15% tints for fills

rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.minor.size': 1.4, 'ytick.minor.size': 1.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.linewidth': 1.0, 'lines.markersize': 3.0, 'lines.markeredgewidth': 0.8,
    'grid.linewidth': 0.5, 'grid.color': GRID, 'grid.alpha': 1.0,
    'axes.grid': False, 'axes.axisbelow': True,
    'legend.handlelength': 1.5, 'legend.handletextpad': 0.5, 'legend.borderpad': 0.2,
    'legend.labelspacing': 0.3, 'legend.columnspacing': 0.9,
    'savefig.bbox': None, 'savefig.pad_inches': 0.01,
    'figure.dpi': 400,
    'pdf.fonttype': 42, 'ps.fonttype': 42,      # embed TrueType; text stays editable
})


def figure(width='single', ncols=2, nrows=1, height=None, sharey=False, **kw):
    """A figure with constrained layout; height defaults to a per-panel rule of thumb.

    ``sharey=True`` is used for facet rows that carry the same categorical axis (Fig. 4, Fig. 10), so
    the category labels appear once and the panels cannot drift out of alignment.
    """
    w = WIDTH[width]
    if height is None:
        height = 1.9 * nrows + 0.55
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), constrained_layout=True,
                             squeeze=False, sharey=sharey, **kw)
    if nrows * ncols == 1:
        return fig, axes[0][0]
    return fig, list(axes.ravel())


def box_axes(ax, grid='y'):
    """Standard data panel for IEEE-style figures: full box, light grid behind the data.

    IEEE/TPAMI line art normally keeps all four spines; the Nature-style open axes used earlier read
    as an unfinished frame.  ``open_axes`` is kept for the few panels where a spine would collide
    with a twin construction.
    """
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.7)
    if grid:
        ax.grid(True, axis=grid)
        ax.set_axisbelow(True)
    return ax


def open_axes(ax, grid='y'):
    """Data panel without top/right spines (used only where a full box would crowd)."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid)
        ax.set_axisbelow(True)
    return ax


def log_axis(ax, which, lo, hi, scalar=False):
    """A log axis that does not drown the reader in tick labels.

    ``scalar=True`` prints 0.1, 1, 10 instead of 10^{-1}, 10^{0}, 10^{1}; use it when the spanned
    range is narrow (a factor of ~30), where exponents are harder to read than the numbers.
    """
    if which in ('x', 'both'):
        ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if scalar:
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())
    if which in ('y', 'both'):
        ax.set_yscale('log')
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    return ax


def fixed_ticks(ax, which, values):
    """Explicit, readable tick positions (e.g. 0.5, 1, 2, 5, 10 on a narrow log axis)."""
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    if which in ('x', 'both'):
        ax.xaxis.set_major_locator(FixedLocator(list(values)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if which in ('y', 'both'):
        ax.yaxis.set_major_locator(FixedLocator(list(values)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
    return ax


def integer_ticks(ax, which='x', n=5):
    if which in ('x', 'both'):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, integer=True))
    if which in ('y', 'both'):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, integer=True))
    return ax


def _occupancy(ax, n=24):
    """Coarse occupancy grid of the drawn data, in axes coordinates."""
    import numpy as np
    grid = np.zeros((n, n))
    pts = []
    for ln in list(ax.lines) + list(ax.collections):
        if hasattr(ln, 'get_xydata'):
            xy = np.asarray(ln.get_xydata(), dtype=float)
        elif hasattr(ln, 'get_offsets'):
            xy = np.asarray(ln.get_offsets(), dtype=float)
        else:
            continue
        if xy.ndim != 2 or len(xy) == 0:
            continue
        pts.append(xy)
    if not pts:
        return grid
    xy = np.vstack(pts)
    good = np.isfinite(xy).all(axis=1)
    xy = xy[good]
    if len(xy) == 0:
        return grid
    disp = ax.transData.transform(xy)
    ax_pts = ax.transAxes.inverted().transform(disp)
    for x, y in ax_pts:
        if 0 <= x <= 1 and 0 <= y <= 1:
            grid[min(int(y * n), n - 1), min(int(x * n), n - 1)] += 1
    return grid


def _data_axes_coords(ax):
    import numpy as np
    pts = []
    for ln in list(ax.lines) + list(ax.collections):
        if hasattr(ln, 'get_xydata'):
            xy = np.asarray(ln.get_xydata(), dtype=float)
        elif hasattr(ln, 'get_offsets'):
            xy = np.asarray(ln.get_offsets(), dtype=float)
        else:
            continue
        if xy.ndim == 2 and len(xy):
            pts.append(xy)
    if not pts:
        return None
    xy = np.vstack(pts)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return None
    return ax.transAxes.inverted().transform(ax.transData.transform(xy)), xy


def panel_report(fig, verbose=True):
    """Flag axes whose range is set by a control point rather than by the data.

    A log axis spanning many decades is only justified when the *data* spans them.  If almost every
    point falls inside one decade, the axis is being stretched by an outlier or a control entry and
    the informative points are squeezed into a corner.
    """
    import numpy as np
    notes = []
    for i, ax in enumerate(fig.axes):
        got = _data_axes_coords(ax)
        if got is None or not ax.axison:
            continue
        ap, raw = got
        inside = (ap[:, 0] >= -0.02) & (ap[:, 0] <= 1.02) & (ap[:, 1] >= -0.02) & (ap[:, 1] <= 1.02)
        ap, raw = ap[inside], raw[inside]
        if len(ap) == 0:
            continue
        cells = {(min(int(x * 12), 11), min(int(y * 12), 11)) for x, y in ap}
        coverage = len(cells) / 144.0
        quad = [int(((ap[:, 1] >= .5) & (ap[:, 0] < .5)).sum()),
                int(((ap[:, 1] >= .5) & (ap[:, 0] >= .5)).sum()),
                int(((ap[:, 1] < .5) & (ap[:, 0] < .5)).sum()),
                int(((ap[:, 1] < .5) & (ap[:, 0] >= .5)).sum())]
        msg = (f'panel {i}: n={len(ap)} coverage={coverage:.0%} '
               f'quadrants(TL,TR,BL,BR)={quad}')
        for axis, scale, lim, vals in (('x', ax.get_xscale(), ax.get_xlim(), raw[:, 0]),
                                       ('y', ax.get_yscale(), ax.get_ylim(), raw[:, 1])):
            if scale == 'log':
                lo, hi = sorted(lim)
                if lo > 0:
                    span = np.log10(hi / lo)
                    dec = np.floor(np.log10(np.abs(vals[vals > 0]))).astype(int)
                    if len(dec):
                        busiest = np.bincount(dec - dec.min()).max() / len(dec)
                        if span > 4 and busiest > 0.8:
                            msg += (f'  [!] {axis} log-range {span:.1f} decades but '
                                    f'{busiest:.0%} of points in one decade')
        notes.append(msg)
        if verbose:
            print('  ' + msg)
    return notes


def legend_auto(ax, candidates=('upper left', 'upper right', 'lower right', 'lower left'), **kw):
    """Place the legend in the emptiest corner, measured from the data itself.

    Hand-placed legends collide with curves; this measures how much data sits under each candidate
    box (a dilated occupancy grid) and keeps the quietest one.
    """
    import numpy as np
    occ = _occupancy(ax)
    n = occ.shape[0]
    leg = ax.legend(loc=candidates[0], **kw)
    ax.figure.canvas.draw()
    best, best_mass = candidates[0], np.inf
    r = ax.figure.canvas.get_renderer()
    for cand in candidates:
        leg.set_loc(cand)
        ax.figure.canvas.draw()
        bb = leg.get_window_extent(renderer=r)
        p0 = ax.transAxes.inverted().transform((bb.x0, bb.y0))
        p1 = ax.transAxes.inverted().transform((bb.x1, bb.y1))
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        i0, i1 = max(int(x0 * n), 0), min(int(np.ceil(x1 * n)), n)
        j0, j1 = max(int(y0 * n), 0), min(int(np.ceil(y1 * n)), n)
        mass = float(occ[j0:j1, i0:i1].sum()) if i1 > i0 and j1 > j0 else 0.0
        if mass < best_mass:
            best, best_mass = cand, mass
    leg.set_loc(best)
    return leg


def panel(ax, letter, dx=0.02, dy=0.98):
    """Standard panel letter: bold lower-case, inside the top-left corner (never clipped)."""
    t = ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight='bold',
                va='top', ha='left', color=INK)
    t.set_in_layout(False)
    return t


def audit(fig, name='figure', max_text=6, min_pt=7.0, tol=0.6, verbose=True):
    """Fail unless the figure is typographically clean.  See the module docstring."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []                                    # (bbox, kind, label, axes-index, fontsize)

    def add(art, kind, ai):
        if art is None or not art.get_visible():
            return
        try:
            txt = art.get_text()
        except AttributeError:
            return
        if not txt:
            return
        items.append((art.get_window_extent(renderer=r), kind, txt[:28], ai, art.get_fontsize()))

    for ai, ax in enumerate(fig.axes):
        for t in ax.texts:
            add(t, 'other', ai)
        if not ax.axison:                 # schematic panel: no ticks or axis labels to audit
            continue
        add(ax.xaxis.label, 'label', ai)
        add(ax.yaxis.label, 'label', ai)
        add(ax.title, 'other', ai)
        # matplotlib keeps tick Text objects alive beyond the view limits; they are not drawn, so
        # they are excluded by a display-coordinate test against the axes box
        ab = ax.get_window_extent(renderer=r)
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        xtol, ytol = 1e-9 * max(abs(x1 - x0), 1e-12), 1e-9 * max(abs(y1 - y0), 1e-12)
        for t in ax.get_xticklabels():
            p = t.get_position()[0]
            if ax.get_xscale() == 'log':
                bb = t.get_window_extent(renderer=r)
                keep = bb.x1 >= ab.x0 - 2 and bb.x0 <= ab.x1 + 2
            else:
                keep = x0 - xtol <= p <= x1 + xtol
            if keep:
                add(t, 'tick', ai)
        for t in ax.get_yticklabels():
            p = t.get_position()[1]
            if ax.get_yscale() == 'log':
                bb = t.get_window_extent(renderer=r)
                keep = bb.y1 >= ab.y0 - 2 and bb.y0 <= ab.y1 + 2
            else:
                keep = y0 - ytol <= p <= y1 + ytol
            if keep:
                add(t, 'tick', ai)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                add(t, 'other', ai)
    for t in fig.texts:
        add(t, 'other', -1)

    problems = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            bi, ki, li, ai, _ = items[i]
            bj, kj, lj, aj, _ = items[j]
            if {ki, kj} == {'tick'} and ai == aj:
                continue
            if {ki, kj} <= {'tick', 'label'} and ai == aj:
                continue
            if not bi.overlaps(bj):
                continue
            ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ix > tol and iy > tol:
                problems.append(f'overlap [{ki} p{ai}] {li!r} x [{kj} p{aj}] {lj!r} '
                                f'({ix:.1f}x{iy:.1f} px)')
    W, H = fig.canvas.get_width_height()
    for bb, kind, lab, ai, _ in items:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            problems.append(f'outside canvas: {lab!r} [{kind} p{ai}] '
                            f'bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) canvas={W}x{H}')
    for bb, kind, lab, ai, size in items:
        if size < min_pt - 1e-6:
            problems.append(f'font {size:.1f} pt < {min_pt}: {lab!r}')
    for ai, ax in enumerate(fig.axes):
        n = len([1 for bb, kind, lab, a, _ in items if a == ai and kind == 'other'])
        if n > max_text:
            problems.append(f'panel {ai} carries {n} text items (budget {max_text})')

    if verbose:
        print(f'  audit {name}: {len(items)} text items, {len(fig.axes)} panels, '
              f'{len(problems)} problem(s)')
    if problems:
        for p in problems:
            print('    ! ' + p)
        raise SystemExit(f'{name}: layout audit failed ({len(problems)} problems)')
    return True


def figure_check(fig, name='figure', verbose=True):
    """Completeness check: axis labels, legends, and data inside the axes.

    `audit` checks typography; this checks that a reader can actually read the panel: every visible
    axis is labelled, a panel with more than one drawn series carries a legend, and no series is
    clipped by the limits (the failure that hid the deepest heat-diffusion point in Figure 6).
    Reference lines drawn with ``axhline``/``axvline`` live in axes coordinates and colourbar axes
    need no axis labels, so both are excluded.
    """
    import numpy as np
    problems = []
    labelled_siblings = {}
    for i, ax in enumerate(fig.axes):
        if ax.get_label() == '<colorbar>':
            continue
        sibs = [a for a in ax.get_shared_y_axes().get_siblings(ax) if a is not ax]
        labelled_siblings[i] = any(a.get_ylabel().strip() for a in sibs)
    for i, ax in enumerate(fig.axes):
        if not ax.axison or ax.get_label() == '<colorbar>':
            continue
        if not ax.get_xlabel().strip():
            problems.append(f'panel {i}: no x label')
        if not ax.get_ylabel().strip() and not labelled_siblings.get(i):
            problems.append(f'panel {i}: no y label')
        labelled = [ln for ln in ax.lines + ax.collections if ln.get_label() and
                    not ln.get_label().startswith('_')]
        if len(labelled) > 1 and ax.get_legend() is None:
            names = sorted({ln.get_label() for ln in labelled})
            problems.append(f'panel {i}: {len(names)} labelled series {names} but no legend')
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        xs, ys = [], []

        def take(xy):
            xy = np.asarray(xy, dtype=float)
            if xy.ndim == 2 and len(xy):
                xs.append(xy[:, 0])
                ys.append(xy[:, 1])

        for ln in ax.lines:
            if ln.get_transform() is not ax.transData:      # axhline/axvline: axes coordinates
                continue
            take(ln.get_xydata())
        for ln in ax.collections:
            if hasattr(ln, 'get_segments'):
                for seg in ln.get_segments():
                    take(seg)
            elif hasattr(ln, 'get_offsets'):
                off = ln.get_offsets()
                if off is not None and len(off) and not isinstance(off[0], (int, float)):
                    take(off)
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            for axis, vals, lo, hi, scale in (('x', x, x0, x1, ax.get_xscale()),
                                              ('y', y, y0, y1, ax.get_yscale())):
                if len(vals) == 0:
                    continue
                if scale == 'log':
                    bad = vals[(vals <= 0) | (vals > hi * 1.001)]
                else:
                    span = max(abs(hi - lo), 1e-12)
                    bad = vals[(vals < lo - 0.005 * span) | (vals > hi + 0.005 * span)]
                if len(bad):
                    problems.append(f'panel {i}: {len(bad)} point(s) outside the {axis} limits '
                                    f'[{lo:.4g},{hi:.4g}] (e.g. {bad[0]:.4g})')
    if verbose:
        print(f'  check {name}: {len(problems)} completeness problem(s)')
        for p in problems:
            print('    ! ' + p)
    return problems


def save(fig, name, audit_first=True, **akw):
    if audit_first:
        figure_check(fig, name)
        audit(fig, name, **akw)
    os.makedirs(OUTDIR, exist_ok=True)
    pdf = os.path.join(OUTDIR, name + '.pdf')
    png = os.path.join(OUTDIR, name + '.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(pdf, WORK)} and .png')
    return pdf, png
