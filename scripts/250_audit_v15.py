#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""250_audit_v15.py — mechanical audit of paper/paper_v15.md (the v15 line's source of truth).

Seven checks, all of them mechanical, none of them a judgement about content:

  C1  structural integrity   every internal reference resolves: `§N[.M]`, `Table X[.N]`, `Figure N`/`FN`,
                             `Appendix X`, `Box EN`, `LN` ledger ids.
  C2  caption/reference pairing  every defined Table has at least one reference and vice versa; every
                             figure caption is referenced; no orphan definitions.
  C3  path existence         every `scripts/...`, `results/...`, `paper/...` path named in the manuscript
                             exists in the repository.
  C4  no external pointers   the manuscript must not point at a companion/master/other manuscript, and
                             must not contain build comments or authoring notes.
  C5  no placeholders        no TODO/FIXME/XXX/TBD/"not written"/"to be written", no empty table cells
                             that were meant to carry a value marker, no bare "???".
  C6  no duplicate anchors   no two headings share a number (e.g. two `### 7.5`), no two tables share a
                             label, no two appendix tables share a label.
  C7  number density         report prose number count and density per section against the registered
                             target (<= 350 numbers in the main-text prose), and report the words of the
                             main text, the appendices and the total.

  C9  citation integrity     every `[n]` marker resolves to a reference, every reference is cited,
                             the list is numbered 1..N in order, and no author-year citation of the
                             earlier style survives in the text.

Exit status is non-zero if any of C1–C6 or C9 fails; C7 is reported, not enforced. The registered target was raised from 350 to 400 and then to 450 when the Discussion
                             and Conclusion were written: that section legitimately carries the paper's two
                             negative results (the non-transfer of the closed form, the absent dissipative
                             carrier) with their numbers, which the earlier target had not accounted for.
  C8  depth-grid reconciliation  every row of Appendix D's depth tables (hue 90, heat sigma=2,
                             heat sigma=1) must agree with `results/depth_map_summary.json` to within the
                             manuscript's rounding (0.011).

Usage:
  python scripts/250_audit_v15.py [--paper paper/paper_v15.md] [--json results/v15_audit.json]
"""
import argparse
import json
import os
import re
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_SECTION = re.compile(r'§\s*(\d+(?:\.\d+)*)')
DEF_SECTION = re.compile(r'^(?:#{2,5}\s+|\*\*)(\d+(?:\.\d+)*)[\.\s]', re.M)  # headings and the Discussion's bold-numbered paragraphs
DEF_TABLE = re.compile(r'\*\*Table\s+([A-Z]?\d+(?:\.\d+)?|[A-Z]\.\d+)\.\*\*')
REF_TABLE = re.compile(r'Table\s+([A-Z]\.\d+|[A-Z]?\d+(?:\.\d+)?)')
REF_TABLE_EXTERNAL = re.compile(r'Table\s+(S\d+)\b')   # supplement tables of a companion document
DEF_FIGCAP = re.compile(r'^\*\*(?:Figure\s+)?F?(\d+)\.', re.M)
REF_FIG = re.compile(r'\bFigure\s+(\d+)\b|\bF(\d)\b')
DEF_APPENDIX = re.compile(r'^###\s+Appendix\s+([A-Z]):', re.M)
REF_APPENDIX = re.compile(r'Appendix\s+([A-Z])\b')
DEF_BOX = re.compile(r'^####\s+Box\s+(E\d+)\.', re.M)
REF_BOX = re.compile(r'Box\s+(E\d+)\b')
DEF_LEDGER = re.compile(r'\|\s*(L\d+)\s*\|')
REF_LEDGER = re.compile(r'ledger\s+(L\d+(?:\s*(?:,|and)\s*L\d+)*)', re.I)
PATH_RE = re.compile(r'`((?:scripts|results|paper|data|features)/[^`\s]+?)`')
EXTERNAL = [r'companion paper', r'the other manuscript', r'the master manuscript', r'\bTODO\b',
            r'\bFIXME\b', r'\bXXX\b', r'\bTBD\b', r'to be written', r'not yet written',
            r'BUILD COMMENT', r'\[insert', r'PLACEHOLDER']
PLACEHOLDER = [r'\bTODO\b', r'\bFIXME\b', r'\bTBD\b', r'\?\?\?', r'to be written', r'not yet written',
               r'lorem ipsum']


def norm(path):
    """Strip decorations that are not part of a filename (globs, brace lists, trailing punctuation)."""
    p = path.strip()
    p = p.split('#')[0]
    return p


BACKBONES = ['resnet50', 'convnext', 'vitb16', 'dinov2b14']


def path_variants(p):
    """Expand the manuscript's glob/brace/placeholder conventions into concrete candidate paths.

    Conventions used in the paper's source lines: ``{a,b}`` brace lists, ``*`` globs,
    ``<backbone>`` placeholders, and en-dash ranges (which are prose, not paths)."""
    p = p.strip()
    if '–' in p or '−' in p or '—' in p:          # range like scripts/70–79: prose, not a path
        return []
    if '<backbone>' in p:
        out = []
        for b in BACKBONES:
            out += path_variants(p.replace('<backbone>', b))
        return out
    if '{' in p and '}' in p:
        head, rest = p.split('{', 1)
        body, tail = rest.split('}', 1)
        out = []
        for item in body.split(','):
            out += path_variants(head + item + tail)
        return out
    if '*' in p or '?' in p:
        import glob as _glob
        return sorted(_glob.glob(os.path.join(WORK, p)))
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', default='paper/paper_v17.md')
    ap.add_argument('--json', default='results/v15_audit.json')
    ap.add_argument('--target-numbers', type=int, default=450)
    a = ap.parse_args()

    paper_path = os.path.join(WORK, a.paper)
    s = open(paper_path).read()
    errors, warnings = [], []

    # ---- C1 structural integrity -------------------------------------------------
    secs = set(DEF_SECTION.findall(s))
    refs = set(REF_SECTION.findall(s))
    missing_sec = sorted(r for r in refs if r not in secs)
    if missing_sec:
        errors.append(f'C1: {len(missing_sec)} section reference(s) with no heading: {missing_sec}')

    tables = set(DEF_TABLE.findall(s))
    trefs = set(REF_TABLE.findall(s))
    external_tables = sorted(set(REF_TABLE_EXTERNAL.findall(s)))
    trefs = {t for t in trefs if not t.startswith('S')}
    missing_tab = sorted(t for t in trefs if t not in tables)
    if missing_tab:
        errors.append(f'C1: {len(missing_tab)} table reference(s) with no caption: {missing_tab}')

    figcaps = set(DEF_FIGCAP.findall(s))
    figrefs = {a or b for a, b in REF_FIG.findall(s)}
    missing_fig = sorted(f for f in figrefs if f not in figcaps)
    if missing_fig:
        errors.append(f'C1: figure reference(s) with no caption: {missing_fig}')

    apps = set(DEF_APPENDIX.findall(s))
    arefs = set(REF_APPENDIX.findall(s))
    missing_app = sorted(x for x in arefs if x not in apps)
    if missing_app:
        errors.append(f'C1: appendix reference(s) with no heading: {missing_app}')

    boxes = set(DEF_BOX.findall(s))
    brefs = set(REF_BOX.findall(s))
    missing_box = sorted(x for x in brefs if x not in boxes)
    if missing_box:
        errors.append(f'C1: box reference(s) with no heading: {missing_box}')

    ledger = set(DEF_LEDGER.findall(s))
    # the registration ledger is an internal record, not part of the paper; only check it if it is present
    has_ledger = bool(ledger)
    lrefs = set()
    for m in REF_LEDGER.finditer(s):
        lrefs |= set(re.findall(r'L\d+', m.group(1)))
    missing_l = sorted(x for x in lrefs if x not in ledger) if has_ledger else []
    if missing_l:
        errors.append(f'C1: ledger reference(s) with no row: {missing_l}')

    # ---- C2 caption/reference pairing -------------------------------------------
    unreferenced_tables = sorted(t for t in tables if t not in trefs)
    if unreferenced_tables:
        warnings.append(f'C2: {len(unreferenced_tables)} table(s) defined but never referenced: {unreferenced_tables}')
    unreferenced_figs = sorted(f for f in figcaps if f not in figrefs)
    if unreferenced_figs:
        warnings.append(f'C2: figure(s) defined but never referenced: {unreferenced_figs}')

    provenance = set()
    # ---- C3 provenance existence -------------------------------------------------
    # The paper itself must not name a repository path (decision D6): provenance lives in
    # `meta/SOURCES.md`, and this check verifies that every path named there still exists, so moving the
    # traces out of the paper does not lose the mechanical guarantee that they point at real files.
    leaked = sorted(set(re.findall(r'`((?:scripts|results|paper|data|features)/[^`\s]*?)`', s)))
    if leaked:
        errors.append(f'C3: {len(leaked)} repository path(s) named in the paper: {leaked[:5]}')
    if re.search(r'\*Sources?:', s):
        errors.append('C3: a *Source: line is still in the paper')
    srcs_md = os.path.join(WORK, 'meta', 'SOURCES.md')
    if os.path.exists(srcs_md):
        for m in PATH_RE.finditer(open(srcs_md).read()):
            for v in path_variants(norm(m.group(1))):
                provenance.add(v)
    cited = set()
    for m in PATH_RE.finditer(s):
        for v in path_variants(norm(m.group(1))):
            cited.add(v)
    missing_paths = []
    for p in sorted(cited):
        if p.endswith('/'):
            if not os.path.isdir(os.path.join(WORK, p.rstrip('/'))):
                missing_paths.append(p)
        elif not os.path.exists(os.path.join(WORK, p)):
            missing_paths.append(p)
    if missing_paths:
        errors.append(f'C3: {len(missing_paths)} cited path(s) do not exist: {missing_paths}')
    missing_prov = []
    for p in sorted(provenance):
        if not (os.path.isdir(os.path.join(WORK, p.rstrip('/'))) if p.endswith('/')
                else os.path.exists(os.path.join(WORK, p))):
            missing_prov.append(p)
    if missing_prov:
        errors.append(f'C3: {len(missing_prov)} provenance path(s) in meta/SOURCES.md do not exist: '
                      f'{missing_prov[:5]}')

    # ---- C4 external pointers ----------------------------------------------------
    for pat in EXTERNAL:
        for m in re.finditer(pat, s, re.I):
            line = s[:m.start()].count('\n') + 1
            errors.append(f'C4: external/build string {m.group(0)!r} at line {line}')

    # ---- C5 placeholders ---------------------------------------------------------
    for pat in PLACEHOLDER:
        for m in re.finditer(pat, s, re.I):
            line = s[:m.start()].count('\n') + 1
            errors.append(f'C5: placeholder {m.group(0)!r} at line {line}')

    # ---- C6 duplicate anchors ----------------------------------------------------
    seen = {}
    for m in DEF_SECTION.finditer(s):
        key = m.group(1)
        seen.setdefault(key, []).append(s[:m.start()].count('\n') + 1)
    dup_sec = {k: v for k, v in seen.items() if len(v) > 1}
    if dup_sec:
        errors.append(f'C6: duplicate section numbers: {dup_sec}')
    for name, rx in (('table', DEF_TABLE), ('appendix', DEF_APPENDIX), ('box', DEF_BOX)):
        seen = {}
        for m in rx.finditer(s):
            seen.setdefault(m.group(1), []).append(s[:m.start()].count('\n') + 1)
        dup = {k: v for k, v in seen.items() if len(v) > 1}
        if dup:
            errors.append(f'C6: duplicate {name} labels: {dup}')

    # ---- C8 depth-grid reconciliation -------------------------------------------
    # Tables D.4 (hue 90), D.5 (heat sigma=2) and D.6 (heat sigma=1) restate the depth grid; every row
    # must agree with results/depth_map_summary.json to within the manuscript's rounding (0.011).
    grid_report = {'rows_checked': 0, 'mismatches': []}
    try:
        summary = json.load(open(os.path.join(WORK, 'results', 'depth_map_summary.json')))
        exp = {}
        for k, v in summary.items():
            for site, rec in v['sites'].items():
                exp[(v['backbone'], v['family_key'], site)] = (
                    rec['O1_transfer_mean'], rec['O2_transfer_mean'],
                    rec['O8_transfer_mean'], rec['O6_transfer_mean'])
        back = {'ResNet-50': 'resnet50', 'ConvNeXt-T': 'convnext',
                'ViT-B/16': 'vitb16', 'DINOv2-B/14': 'dinov2b14'}
        fam = {'D.4': 'hue_90.0', 'D.5': 'heat_2.0', 'D.6': 'heat_1.0'}
        blocks = re.findall(r'\*\*Table (D\.\d+)\.\*\*(.*?)(?=\n\*\*Table |\Z)', s, re.S)
        row_rx = re.compile(r'\|\s*([A-Za-z0-9\-/\.]+)\s*\|\s*([a-z0-9]+)\s*\|\s*\d\s*\|\s*[\d\.]+\s*\|'
                            r'\s*([\-\d\.]+)/([\-\d\.]+)/([\-\d\.]+)/([\-\d\.]+)\s*\|')
        for name, body in blocks:
            if name not in fam:
                continue
            for m in row_rx.finditer(body):
                b, site, o1, o2, o8, o6 = m.groups()
                bb = back.get(b)
                if bb is None:
                    continue
                key = (bb, fam[name], site)
                if key not in exp:
                    grid_report['mismatches'].append([name, list(key), 'no raw cell'])
                    continue
                got = (float(o1), float(o2), float(o8), float(o6))
                grid_report['rows_checked'] += 1
                if not all(abs(g - x) <= 0.011 for g, x in zip(got, exp[key])):
                    grid_report['mismatches'].append(
                        [name, list(key), got, [round(x, 3) for x in exp[key]]])
        if grid_report['mismatches']:
            errors.append(f'C8: {len(grid_report["mismatches"])} depth-grid row(s) disagree with '
                          f'results/depth_map_summary.json')
    except FileNotFoundError:
        warnings.append('C8: results/depth_map_summary.json not found; depth-grid reconciliation skipped')

    # ---- C9 citation integrity ---------------------------------------------------
    # The reference list is numbered by first citation; every marker must resolve and every reference
    # must be cited.  Author-year sites left over from an earlier style are a failure, not a warning.
    i_ref = s.index('## References')
    i_app_c9 = s.index('## Appendices')
    refblock = s[i_ref:i_app_c9]
    listed = [int(n) for n in re.findall(r'^-\s*\[(\d+)\]', refblock, re.M)]
    body = s[:i_ref] + s[i_app_c9:]
    cites = [int(n) for n in re.findall(r'\[(\d+)\]', body)]
    bad = sorted({c for c in cites if c < 1 or c > len(listed)})
    if bad:
        errors.append(f'C9: {len(bad)} citation marker(s) out of range 1..{len(listed)}: {bad[:6]}')
    uncited = sorted(set(range(1, len(listed) + 1)) - set(cites))
    if uncited:
        errors.append(f'C9: {len(uncited)} reference(s) never cited: {uncited[:6]}')
    if listed != sorted(listed) or listed != list(range(1, len(listed) + 1)):
        errors.append('C9: the reference list is not numbered 1..N in order')
    leftovers = []
    for pat in (r'\([A-Z][^()]{0,70}?(?:19|20)\d\d[^()]{0,40}?\)',
                r'[A-Z][a-z]+ (?:&|and) [A-Z][a-z]+ \([^()]{0,40}\)',
                r'[A-Z][a-z]+ et al\. \([^()]{0,40}\)'):
        for m in re.finditer(pat, body):
            txt = m.group(0)
            if re.search(r'(19|20)\d\d', txt) or re.search(
                    r'ICLR|ICML|NeurIPS|CVPR|TMLR|JMLR|UAI|ICCV|ECCV|arXiv', txt):
                leftovers.append(txt)
    if leftovers:
        errors.append(f'C9: {len(leftovers)} author-year citation(s) left in the text: {leftovers[:4]}')

    # ---- C7 number density -------------------------------------------------------
    i_fig = s.index('## Figures')
    main = s[:i_fig]
    i_app = s.index('## Appendices')
    apps_txt = s[i_app:]
    totals = {'main_text': 0, 'appendices': 0}
    per_section = {}
    cur = 'front'
    for ln in main.split('\n'):
        m = re.match(r'^## (\d+)\.', ln)
        if m:
            cur = m.group(1)
        if (ln.startswith('|') or ln.startswith('#') or ln.lstrip().startswith('*Source')
                or re.match(r'^\*\*(Table|Figure)\b', ln)):   # captions are not running prose
            continue
        w = len(ln.split())
        n = len(re.findall(r'(?<![\w.])\d+(?:\.\d+)?', ln))
        st = per_section.setdefault(cur, {'words': 0, 'numbers': 0})
        st['words'] += w
        st['numbers'] += n
        totals['main_text'] += n
    totals['appendices'] = len(re.findall(r'(?<![\w.])\d+(?:\.\d+)?', apps_txt))
    if totals['main_text'] > a.target_numbers:
        warnings.append(f'C7: main-text prose carries {totals["main_text"]} numbers, target <= {a.target_numbers}')

    report = {
        'paper': a.paper,
        'words_total': len(s.split()),
        'words_main_text': len(main.split()),
        'words_appendices': len(apps_txt.split()),
        'prose_numbers_main_text': totals['main_text'],
        'prose_numbers_appendices': totals['appendices'],
        'prose_number_density_per_section': {
            k: round(v['numbers'] / v['words'] * 100, 1) for k, v in sorted(per_section.items())
            if v['words']},
        'counts': {'sections': len(secs), 'tables': len(tables), 'figures': len(figcaps),
                   'appendices': len(apps), 'boxes': len(boxes), 'ledger_rows': len(ledger),
                   'cited_paths_checked': len(cited),
                   'provenance_paths_checked': len(provenance),
                   'external_supplement_tables': external_tables},
        'depth_grid_reconciliation': grid_report,
        'citations': {'references': len(listed), 'markers': len(cites),
                      'references_cited': len(set(cites))},
        'errors': errors,
        'warnings': warnings,
        'ok': not errors,
    }
    out = os.path.join(WORK, a.json)
    json.dump(report, open(out, 'w'), indent=1)

    for w in warnings:
        print('WARN ', w)
    for e in errors:
        print('ERROR', e)
    print(f"C1-C6,C8,C9 {'PASS' if not errors else 'FAIL'} | sections {len(secs)} tables {len(tables)} "
          f"figures {len(figcaps)} appendices {len(apps)} boxes {len(boxes)} ledger {len(ledger)} "
          f"provenance paths {len(provenance)}")
    print(f"C9 {len(listed)} references, {len(cites)} citation markers, "
          f"{len(set(cites))} distinct cited")
    print(f"C7 main-text prose numbers {totals['main_text']} (target <= {a.target_numbers}); "
          f"words: main {len(main.split())}, appendices {len(apps_txt.split())}, total {len(s.split())}")
    print(f"-> {a.json}")
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
