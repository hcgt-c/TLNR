#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""283_eqwidth_probe.py — fail if a display equation would have to be shrunk to fit the column.

`267_md_to_tex.py` wraps every display equation in `\\adjustbox{max width=\\columnwidth}`, which keeps a
long formula inside the measure by scaling it down.  Scaling is a last resort: at 0.63 of full size the
type is visibly smaller than the text around it, which is not an academic typesetting standard.  The
proper fix is to break the formula at a relation or a sum into an `aligned` block (see
`meta/MANUSCRIPT_PLAN.md` §17), so the guard should never actually be needed.

This script measures every display equation at its natural width in the real two-column context:

  * it builds the LaTeX project with `--measure`, which emits `\\sbox0{...}\\typeout{EQWIDTH n wd}` in
    front of each display,
  * compiles that copy with the local TeX engine,
  * reads `COLWIDTH` and every `EQWIDTH` from the log, and

reports the ratio width/columnwidth for each.  Exit status is non-zero if any ratio exceeds 1.0, i.e. if
the document would print a shrunk equation.

Usage:  python scripts/283_eqwidth_probe.py [--limit 1.0] [--paper paper/paper_v17.md]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TECTONIC = rp.TECTONIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', default='paper/paper_v17.md')
    ap.add_argument('--limit', type=float, default=1.0,
                    help='largest allowed width as a fraction of the column (default 1.0)')
    ap.add_argument('--keep', action='store_true', help='keep the probe build')
    a = ap.parse_args()

    out = tempfile.mkdtemp(prefix='eqwidth_')
    try:
        rc = subprocess.run([sys.executable, os.path.join(WORK, 'scripts', '267_md_to_tex.py'),
                             '--paper', a.paper, '--out', out, '--measure'],
                            cwd=WORK, capture_output=True, text=True)
        if rc.returncode:
            raise SystemExit('267_md_to_tex.py --measure failed:\n' + rc.stderr[-2000:])
        env = dict(os.environ)
        rc = subprocess.run([TECTONIC, '-X', 'compile', 'main.tex', '--keep-logs'],
                            cwd=out, capture_output=True, text=True, env=env)
        log_path = os.path.join(out, 'main.log')
        if not os.path.isfile(log_path):
            raise SystemExit('no main.log in the probe build:\n' + (rc.stderr or '')[-2000:])
        log = open(log_path, errors='replace').read()

        m = re.search(r'COLWIDTH ([\d.]+)pt', log)
        if not m:
            raise SystemExit('the probe build did not report COLWIDTH')
        col = float(m.group(1))
        widths = [(int(n), float(w)) for n, w in re.findall(r'EQWIDTH (\d+) ([\d.]+)pt', log)]
        if not widths:
            raise SystemExit('the probe build reported no equation widths')

        md = open(os.path.join(WORK, a.paper), encoding='utf-8').read()
        bodies = re.findall(r'\$\$(.+?)\$\$', md, re.S)
        print(f'column width {col:.1f}pt | display equations measured {len(widths)}')
        over = []
        for n, w in sorted(widths, key=lambda t: -t[1]):
            ratio = w / col
            if ratio > a.limit:
                body = re.sub(r'\s+', ' ', bodies[n - 1]).strip() if n <= len(bodies) else '?'
                over.append((n, w, ratio, body))
        for n, w, ratio, body in over:
            print(f'  OVER  eq {n:>2}  {w:6.1f}pt  {ratio*100:5.1f}% of the column  {body[:80]}')
        widest = max(w for _, w in widths)
        print(f'widest equation {widest:.1f}pt ({100*widest/col:.0f}% of the column); '
              f'{len(over)} over the limit')
        if over:
            print('\nbreak each one at a relation or a sum (see meta/MANUSCRIPT_PLAN.md §17); '
                  'do not let \\adjustbox shrink it')
            return 1
        print('PASS: every display equation is set at full size')
        return 0
    finally:
        if a.keep:
            print('probe build kept in', out)
        else:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
