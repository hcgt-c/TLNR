#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""252_update_real_multiattr_table.py — rewrite Appendix D's real-region multi-attribute table from the
regenerated summary.

`results/realmultiattr_intervention_<backbone>_b4_s*.json` were re-run under the image-grouped split
(script 159 records `split_grouped_by_image: true`); this script reads the regenerated
`results/multiattr_summary.json` and rewrites Table D.14 in `paper/v15_appendices/D_full_grids.md`
in place, printing the old and new rows so the change can be reviewed.

Convention of the table (unchanged): `err null -> real` is the probe error before and after the real
transformation; the operator columns are the attribute-transfer fraction (mean ± sd over seeds), and the
last column is the random-orthogonal control's paired win rate.

Usage: python scripts/252_update_real_multiattr_table.py [--dry-run]
"""
import argparse
import json
import os
import re
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(WORK, 'paper', 'v15_appendices', 'D_full_grids.md')
SUMMARY = os.path.join(WORK, 'results', 'multiattr_summary.json')

ATTRS = ['hue', 'saturation', 'value']
LABEL = {'vitb16': 'ViT-B/16', 'dinov2b14': 'DINOv2-B/14'}
OPS = [('O1', 'O1_procrustes'), ('O5', 'O5_mlp'), ('O8', 'O8_conv_residual')]


def fmt(v, d=3):
    return f'{v:.{d}f}'


def fmt_err(v):
    """Hue errors are in degrees (tens); saturation and value errors are in attribute units (<1),
    so the decimal width is chosen by magnitude rather than fixed."""
    return f'{v:.1f}' if abs(v) >= 10 else f'{v:.3f}'


def row(backbone, attr, rec):
    n = rec['n_runs']
    ops = rec['attributes'][attr]['ops']
    cells = [f"{fmt(rec['attributes'][attr]['power_mean'])} ± {fmt(rec['attributes'][attr]['power_sd'])}",
             f"{fmt_err(rec['attributes'][attr]['probe_err_null_mean'])} $\\to$ "
             f"{fmt_err(rec['attributes'][attr]['probe_err_real_mean'])}"]
    for _, key in OPS:
        cells.append(f"{fmt(ops[key]['attr_transfer_mean'])} ± {fmt(ops[key]['attr_transfer_sd'])}")
    o6 = ops['O6_random_orthogonal']
    cells.append(f"{fmt(o6['attr_transfer_mean'])} ± {fmt(o6['attr_transfer_sd'])}")
    cells.append(fmt(o6['win_rate_mean']))
    return f"| {LABEL[backbone]} ({n}) | {attr} ({cells[0]}) | {cells[1]} | " + ' | '.join(cells[2:]) + ' |'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    d = json.load(open(SUMMARY))
    real = d['real']
    lines = []
    for bb in ('vitb16', 'dinov2b14'):
        if bb not in real:
            continue
        for attr in ATTRS:
            if attr in real[bb]['attributes']:
                lines.append(row(bb, attr, real[bb]))
    new_block = '\n'.join(lines)

    md = open(TABLE).read()
    m = re.search(r'(\*\*Table D\.14\.\*\*.*?\n\n)(\|.*?\n)+(\n\*Source:.*?\*\n)', md, re.S)
    if not m:
        raise SystemExit('could not locate Table D.14 block')
    old_block = m.group(0)
    header = ("| backbone | attribute (power) | err null $\\to$ real | O1 | O5 | O8 | O6 random | O6 win |\n"
              "|---|---|---|---|---|---|---|---|")
    new_full = m.group(1) + header + '\n' + new_block + '\n' + m.group(3)
    print('--- old ---')
    for ln in old_block.split('\n'):
        if ln.startswith('|') and 'backbone' not in ln and '---' not in ln:
            print(ln)
    print('--- new ---')
    print(new_block)
    if not a.dry_run:
        open(TABLE, 'w').write(md.replace(old_block, new_full))
        print(f'-> rewrote Table D.14 in {os.path.relpath(TABLE, WORK)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
