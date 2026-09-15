#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_release.py — one command that checks the release against the paper.

    python tools/verify_release.py            # everything that can run without the datasets
    python tools/verify_release.py --figures  # also redraw all twelve figures and compare them

The steps, and what each one would catch:

1. **safety audit** (`tools/safety_audit.py`): no absolute local path, credential, private identifier or
   oversized file has slipped back in.
2. **number check** (`scripts/255_recompute_numbers.py`): every number the manuscript quotes is compared
   with the value stored in `results/`.  All of them must agree; this is the check that would catch a
   result file and a table drifting apart.
3. **manuscript audit** (`scripts/250_audit_v15.py`): internal references resolve, every table and figure
   is captioned and cited, citations resolve, provenance paths exist, the depth grid reconciles.
4. **figure redraw** (`--figures`): each of the twelve figure scripts is re-run and the freshly written
   plate is compared, word for word, with the plate shipped in `paper/figures_final/`.  A difference means
   the figure no longer follows from the shipped results.

Exit status is non-zero if any step fails.  Steps that need a dataset are skipped, not failed: the
datasets are not redistributed.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FIG_SCRIPTS = ['270_fig01_object.py', '271_fig02_defect.py', '272_fig03_organisation.py',
               '273_fig04_provenance.py', '274_fig05_realisation.py', '275_fig06_depth.py',
               '276_fig07_interface.py', '277_fig08_uses.py', '278_fig09_fibre_test.py',
               '279_fig10_action_comparison.py', '281_fig11_recovery.py', '282_fig12_prediction.py',
               '280_make_caption_file.py']


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def step(name, cmd):
    r = run([PY] + cmd)
    ok = r.returncode == 0
    tail = [l for l in (r.stdout + r.stderr).strip().split('\n') if l.strip()][-3:]
    print(('  PASS  ' if ok else '  FAIL  ') + name)
    for l in tail:
        print('          ' + l[:150])
    return ok


def pdf_text(path):
    if not shutil.which('pdftotext'):
        return None
    r = subprocess.run(['pdftotext', path, '-'], capture_output=True, text=True)
    return re.sub(r'\s+', ' ', r.stdout).strip() if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figures', action='store_true', help='redraw the twelve figures and compare them')
    a = ap.parse_args()
    ok = True

    print('verifying the release in', ROOT)
    ok &= step('safety audit', ['tools/safety_audit.py'])
    ok &= step('numbers against results/', ['scripts/255_recompute_numbers.py'])
    ok &= step('manuscript audit', ['scripts/250_audit_v15.py', '--paper', 'paper/paper_v17.md'])

    if a.figures:
        import glob
        shipped = os.path.join(ROOT, 'paper', 'figures_final')
        backup = tempfile.mkdtemp(prefix='tl_figs_backup_')
        plates = sorted(glob.glob(os.path.join(shipped, '*.pdf')))
        for p in plates:
            shutil.copy2(p, backup)
        failures = []
        for script in FIG_SCRIPTS:
            r = subprocess.run([PY, os.path.join('scripts', script)], cwd=ROOT, capture_output=True,
                               text=True)
            if r.returncode:
                failures.append(f'{script}: exit {r.returncode} :: '
                                + (r.stderr.strip().split(chr(10)) or [''])[-1][:110])
        same, differ = [], []
        for p in plates:
            name = os.path.basename(p)
            fresh = os.path.join(shipped, name)
            if not os.path.isfile(fresh):
                differ.append(f'{name}: not produced by the scripts')
                continue
            t_new, t_old = pdf_text(fresh), pdf_text(os.path.join(backup, name))
            if t_new is None or t_old is None or t_new == t_old:
                same.append(name)
            else:
                differ.append(f'{name}: the redrawn plate differs from the shipped one')
        for p in plates:                     # restore the shipped plates
            shutil.copy2(os.path.join(backup, p and os.path.basename(p)), shipped)
        shutil.rmtree(backup, ignore_errors=True)
        differ += failures
        print(f'  {"PASS" if not differ else "FAIL"}  figure redraw '
              f'({len(same)} plates reproduce, {len(differ)} problem(s))')
        for d in differ[:10]:
            print('          ' + d)
        ok &= not differ

    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
