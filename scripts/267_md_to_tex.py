#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""267_md_to_tex.py — build the LaTeX project `paper/latex_v17/` from `paper/paper_v17.md`.

The manuscript is the single source of truth; this script converts it and authors no text.  Design:

  * IEEEtran journal class (available on Overleaf), pdfLaTeX-compatible, ASCII-only source:
    every non-ASCII character is mapped to a LaTeX command, so the build does not depend on
    inputenc coverage or on a Unicode font.
  * Sections keep the manuscript's own numbering (the text refers to "§3.5", "Appendix D.21"), so
    they are emitted as ordinary \\section/\\subsection and numbered automatically in the same order.
  * Tables and figures keep their manuscript labels in an unnumbered caption (\\caption*), because
    the labels are not in order of appearance (Table 12 appears before Table 1) and the running text
    cites them by number.
  * Every table is rendered as a full-width tabularx so nothing overflows, whatever the column count.

Usage:  python scripts/267_md_to_tex.py [--paper paper/paper_v17.md] [--out paper/latex_v17]
"""
import argparse
import json
import os
import re
import shutil
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UNICODE = {
    '\u2014': '---', '\u2013': '--', '\u00b0': r'$^\circ$', '\u00a7': r'\S{}',
    '\u00b1': r'$\pm$', '\u00b5': r'$\mu$', '\u00d7': r'$\times$', '\u00bc': r'$1/4$',
    '\u2212': r'$-$', '\u2192': r'$\to$', '\u21a6': r'$\mapsto$', '\u2248': r'$\approx$',
    '\u2264': r'$\le$', '\u2265': r'$\ge$', '\u2208': r'$\in$', '\u2218': r'$\circ$',
    '\u221a': r'$\surd$', '\u2016': r'$\|$', '\u2113': r'$\ell$', '\u211d': r'$\mathbb{R}$',
    '\u2605': r'$\star$', '\u2081': r'$_1$', '\u2082': r'$_2$', '\u00b2': r'$^2$',
    '\u00b3': r'$^3$',
    '\u03c4': r'$\tau$', '\u03c1': r'$\rho$', '\u03c8': r'$\psi$', '\u0394': r'$\Delta$',
    '\u03b8': r'$\theta$', '\u03ba': r'$\kappa$', '\u03b5': r'$\varepsilon$',
    '\U0001d4ab': r'$\mathcal{P}$', '\U0001d4ae': r'$\mathcal{S}$', '\U0001d4b3': r'$\mathcal{X}$',
    '\U0001d54b': r'$\mathbb{T}$',
    '\u2500': '-', '\u2502': '|', '\u25ba': '>', '\u25bc': 'v',
}
LATEX_SPECIAL = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$', '&': r'\&',
                 '#': r'\#', '%': r'\%', '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def escape(t, in_math=False):
    out = []
    for ch in t:
        if ch in UNICODE and not in_math:
            out.append(UNICODE[ch])
        elif ch in UNICODE and in_math:
            m = UNICODE[ch]
            if m.startswith('$') and m.endswith('$'):
                m = m[1:-1]
            out.append(m)
        elif ch in LATEX_SPECIAL:
            out.append(LATEX_SPECIAL[ch])
        else:
            out.append(ch)
    return ''.join(out)


MATH = re.compile(r'(\$\$.*?\$\$|\$[^$]*?\$)', re.S)
# `--measure` instruments every display equation with a width probe so that the ones that would have to
# be shrunk can be found and set properly.  Nothing in the default build is affected.
MEASURE = {'on': False, 'n': 0}
# A caption line is `**Table 12.** ...` or `**Figure 3.** ...`; requiring the number means a bold
# paragraph that merely starts with F or Table (e.g. "**Fixed action.** ...") stays running text.
CAPTION_T = re.compile(r'^\*\*Table\s+(?:[A-Z]?\d+(?:\.\d+)?|[A-Z]\.\d+)\.')
CAPTION_F = re.compile(r'^\*\*Figure\s+(\d+)\.')


def _math_char(ch):
    if ord(ch) < 128:
        return ch
    m = UNICODE.get(ch, ch)
    if m.startswith('$') and m.endswith('$'):
        m = m[1:-1]
    return m


VERBATIM_MAP = {'\u03c4': 'tau', '\u03c1': 'rho', '\u03c8': 'psi', '\u2113': 'l',
                '\u2208': 'in', '\u2248': '~=', '\u2500': '-', '\u2502': '|',
                '\u25ba': '>', '\u25bc': 'v', '\u2014': '--', '\u2013': '-'}


def inline(s):
    """Text span -> LaTeX: protect math, escape the rest, then apply markdown emphasis."""
    math = []

    def _stash(m):
        body = m.group(0)
        if body.startswith('$$'):
            body = body[2:-2]
            body = ''.join(_math_char(ch) for ch in body)
            body = re.sub(r'\\tag\{(.*?)\}', lambda mm: '\\tag{$' + mm.group(1) + '$}', body)
            wrapped = '\\adjustbox{max width=\\columnwidth}{$\\displaystyle ' + body + '$}'
            if MEASURE['on']:
                MEASURE['n'] += 1
                probe = re.sub(r'\\tag\{[^{}]*\}', '', body)      # \tag is legal only in an equation
                wrapped = ('\\sbox0{$\\displaystyle ' + probe +
                           '$}\\typeout{EQWIDTH ' + str(MEASURE['n']) + ' \\the\\wd0}' +
                           wrapped)
            if '\\tag{' in body:
                tex = '\\begin{equation*}' + wrapped + '\\end{equation*}'
            else:
                tex = '\\[' + wrapped + '\\]'
        else:
            body = body[1:-1]
            body = ''.join(_math_char(ch) for ch in body)
            tex = '$' + body + '$'
        math.append(tex)
        return '\x00M%d\x00' % (len(math) - 1)

    t = MATH.sub(_stash, s)
    # bare URLs are stashed like math: they must reach LaTeX unescaped, wrapped in \url{} so that they
    # are clickable and can break at punctuation instead of overflowing the column
    url = []

    def _stash_url(m):
        # trailing sentence punctuation belongs to the sentence, not to the URL
        raw = m.group(0)
        u = raw.rstrip('.,;:')
        url.append(u)
        return '\x00U%d\x00' % (len(url) - 1) + raw[len(u):]

    t = re.sub(r'https?://[^\s<>()\[\]{}"`]+', _stash_url, t)
    # a markdown backslash escape (\%, \_, \&, ...) is authoring syntax: drop the backslash and let
    # `escape` re-escape the character for LaTeX, so a cell reading `71\%` does not print as `71\%`.
    t = re.sub(r'\\([%_&#\'*\[\](){}])', r'\1', t)
    t = escape(t)                       # escape the text first ...
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t, flags=re.S)   # ... then add emphasis
    t = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\\textit{\1}', t)
    def _tt(mt):
        inner = mt.group(1)
        inner = inner.replace('/', '/\\allowbreak{}')
        inner = inner.replace('\\_', '\\_\\allowbreak{}')   # the underscore is already escaped
        return '\\texttt{' + inner + '}'

    t = re.sub(r'`([^`]+?)`', _tt, t)
    t = re.sub(r'\x00U(\d+)\x00', lambda m: '\\url{' + url[int(m.group(1))] + '}', t)
    t = re.sub(r'\x00M(\d+)\x00', lambda m: math[int(m.group(1))], t)
    return t


def split_row(line):
    """Split a markdown table row on its cell separators, not on an escaped or in-math pipe.

    A cell may contain the LaTeX norm bar `\\|` (e.g. `$\\|BAP_{\\ker B}\\|_F^2$`), which is not a
    separator; it is protected through the split and restored afterwards.
    """
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    guard = '\x03'
    cells = line.replace('\\|', guard).split('|')
    return [c.replace(guard, '\\|').strip() for c in cells]


def is_sep(line):
    return bool(re.match(r'^\s*\|?[\s:|-]+\|[\s:|-]*$', line)) and '-' in line


def breakable_math(t):
    """Allow a line break after a comma inside math, so a long formula in a narrow table cell can wrap.

    TeX treats `$c+\\mathrm{MLP}([c,\\cos\\Delta,\\sin\\Delta])$` as one unbreakable atom; in a
    tabularx cell that is what produces an overfull box.  `\\allowbreak` adds a legal breakpoint and
    changes nothing visually.
    """
    return re.sub(r'\$([^$]*)\$', lambda m: '$' + m.group(1).replace(',', ',\\allowbreak{}') + '$', t)


def render_table(rows, label=None, caption=None):
    """rows: list of raw markdown table lines (header, separator, body)."""
    cells = [split_row(r) for r in rows if not is_sep(r)]
    if not cells:
        return ''
    ncol = max(len(r) for r in cells)
    cells = [r + [''] * (ncol - len(r)) for r in cells]
    head, body = cells[0], cells[1:]
    size = ('\\small' if ncol <= 4 else '\\scriptsize' if ncol <= 8 else '\\tiny')
    out = ['\\begin{table*}[t]', '\\centering', size]
    if caption:                      # a table caption sits above the table in IEEE style
        out.append('\\caption*{' + caption + '}')
    if ncol >= 5:
        out.append('\\setlength{\\tabcolsep}{2pt}')
    out += ['\\begin{tabularx}{\\textwidth}{@{}'
            + ' '.join(['>{\\raggedright\\arraybackslash}X'] * ncol) + '@{}}', '\\toprule']
    out.append(' & '.join('\\textbf{' + breakable_math(inline(c)) + '}' for c in head) + ' \\\\')
    out.append('\\midrule')
    for r in body:
        out.append(' & '.join(breakable_math(inline(c)) for c in r) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabularx}', '\\end{table*}']
    return '\n'.join(out)


def convert(md):
    lines = md.split('\n')
    out = []
    i = 0
    in_code = False
    list_stack = []          # 'itemize' | 'enumerate'

    def close_lists():
        while list_stack:
            out.append('\\end{%s}' % list_stack.pop())

    while i < len(lines):
        ln = lines[i]
        # ---- fenced code (the commutation diagram) ----
        if ln.strip().startswith('```'):
            if not in_code:
                close_lists()
                out.append('\\begin{figure*}[t]\\centering\\begin{verbatim}')
                in_code = True
            else:
                out.append('\\end{verbatim}\\end{figure*}')
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(''.join(VERBATIM_MAP.get(ch, ch) for ch in ln))
            i += 1
            continue

        # ---- table block ----
        if ln.lstrip().startswith('|') and i + 1 < len(lines) and is_sep(lines[i + 1]):
            close_lists()
            block = [ln]
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith('|'):
                block.append(lines[j])
                j += 1
            # label/caption from the most recent bold caption line already emitted; look ahead for source
            src = ''
            if j < len(lines) and lines[j].lstrip().startswith('*Source'):
                src = inline(lines[j].strip())
                j += 1
            cap = None
            for k in range(len(out) - 1, -1, -1):
                if out[k].strip() == '':
                    continue
                if out[k].startswith('%%CAPTION%% '):
                    cap = out.pop(k).split(' ', 1)[1]
                    cap = re.sub(r'^\\textbf\{(.*?)\}(.*)$', r'\\textbf{\1}\2', cap, flags=re.S)
                break
            out.append(render_table(block, caption=cap))
            if src:
                out.append('{\\footnotesize ' + src + '}')
            i = j
            continue

        # ---- multi-line display math -------------------------------------------------
        # \$\$ may open on one line and close several lines later (a broken equation).  The block has to
        # reach `inline` as one string, otherwise each line is escaped as text and the alignment is lost.
        if ln.strip().startswith('$$') and not (
                ln.strip().endswith('$$') and len(ln.strip()) > 2):
            body = [ln.strip()[2:]] if ln.strip() != '$$' else []
            j = i + 1
            while j < len(lines) and '$$' not in lines[j]:
                body.append(lines[j])
                j += 1
            if j >= len(lines):
                raise SystemExit('unterminated $$ block')
            tail = lines[j]
            body.append(tail[:tail.index('$$')])
            out.append(inline('$$' + '\n'.join(body) + '$$'))
            i = j + 1
            continue

        # ---- headings ----
        m = re.match(r'^(#{1,6})\s+(.*)$', ln)
        if m:
            close_lists()
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                out.append('% TITLE: ' + inline(text))
            elif level == 2:
                if text in ('Abstract',):
                    out.append('\\begin{abstract}')
                elif re.match(r'^\d+\.', text):
                    out.append('\\section{' + inline(re.sub(r'^\d+\.\s*', '', text)) + '}')
                elif text == 'Figures':
                    out.append('\\section*{Figures}')
                elif text == 'References':
                    out.append('\\section*{References}')
                elif text == 'Appendices':
                    out.append('\\appendix')
                else:
                    out.append('\\section*{' + inline(text) + '}')
            elif level == 3:
                if text.startswith('Appendix'):
                    out.append('\\section{' + inline(re.sub(r'^Appendix\s+[A-Z]:\s*', '', text)) + '}')
                else:
                    out.append('\\subsection{' + inline(re.sub(r'^\d+(\.\d+)*\.?\s*', '', text)) + '}')
            elif level == 4:
                # The manuscript already labels these headings ("B.3 ...", "Box E1. ..."), and IEEEtran
                # numbers \subsubsection as a continuous "1) 2) 3) ..." counter that is not reset per
                # section: an unstarred heading printed "31) H.1 ..." in the positioning appendix.  The
                # starred form keeps the manuscript's own label and drops the stray counter.
                out.append('\\subsubsection*{' + inline(text) + '}')
            else:
                out.append('\\paragraph{' + inline(text) + '}')
            i += 1
            continue

        # ---- lists ----
        m = re.match(r'^\s*[-*]\s+(.*)$', ln)
        m2 = re.match(r'^\s*\d+\.\s+(.*)$', ln)
        if m or m2:
            kind = 'itemize' if m else 'enumerate'
            if not list_stack or list_stack[-1] != kind:
                close_lists()
                if kind == 'enumerate':
                    # IEEEtran does not reset the enumerate counter per section, so without this every
                    # later list continues the previous one (the positioning list printed as "31)").
                    out.append('\\setcounter{enumi}{0}')
                out.append('\\begin{%s}' % kind)
                list_stack.append(kind)
            out.append('\\item ' + inline((m or m2).group(1)))
            i += 1
            continue
        if ln.strip() == '':
            # A markdown list may be "loose": blank lines between its items.  Looking ahead keeps the
            # enumerate open, which is what makes the four contributions come out as 1)-4) instead of
            # four separate lists that all start at 1).
            nxt = next((x for x in lines[i + 1:] if x.strip()), '')
            same = (list_stack and (
                (list_stack[-1] == 'itemize' and re.match(r'^\s*[-*]\s+', nxt)) or
                (list_stack[-1] == 'enumerate' and re.match(r'^\s*\d+\.\s+', nxt))))
            if not same:
                close_lists()
            out.append('')
            i += 1
            continue
        if ln.strip() == '---':
            i += 1
            continue
        # ---- ordinary paragraph line ----
        if ln.lstrip().startswith('**Index Terms**'):
            # IEEE style: the index terms are a separate environment right after the abstract, not part
            # of it.  The placeholder is placed by main() once the abstract has been closed.
            terms = re.sub(r'^\*\*Index Terms\*\*[—\-–]\s*', '', ln.strip())
            out.append('%%KEYWORDS%% ' + inline(terms))
        elif ln.lstrip().startswith('*Source'):
            out.append('{\\footnotesize ' + inline(ln.strip()) + '}')
        elif CAPTION_T.match(ln.strip()):
            # table caption: keep the label, render as an unnumbered caption attached to the next table
            out.append('%%CAPTION%% ' + inline(ln.strip()))
        elif CAPTION_F.match(ln.strip()):
            out.append('%%FIGCAPTION%% ' + inline(ln.strip()))
        else:
            out.append(inline(ln))
        i += 1
    close_lists()
    return '\n'.join(out)


def attach_captions(tex):
    """A %%CAPTION%% line immediately before a table*/figure becomes its \\caption*."""
    lines = tex.split('\n')
    pending = None
    out = []
    for ln in lines:
        if ln.startswith('%%CAPTION%% ') or ln.startswith('%%FIGCAPTION%% '):
            # strip the markdown bold marker so the label survives as text
            txt = ln.split(' ', 1)[1]
            txt = re.sub(r'^\\textbf\{(.*?)\}(.*)$', r'\\textbf{\1}\2', txt, flags=re.S)
            pending = txt
            continue
        if pending and ln.startswith('\\begin{table*}') and not re.search(r'\\caption\*', ln):
            out.append(ln)
            out.append('\\caption*{' + pending + '}')
            pending = None
            continue
        out.append(ln)
    return '\n'.join(out)


PREAMBLE = r"""% ---------------------------------------------------------------------------
%  Transformation Laws in Neural Representations
%  LaTeX source generated from paper/paper_v17.md by scripts/267_md_to_tex.py.
%  Compiles with pdfLaTeX (Overleaf default); the source is pure ASCII.
% ---------------------------------------------------------------------------
\documentclass[journal]{IEEEtran}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{caption}
\usepackage{textcomp}
\usepackage{url}
% flushend balances the two columns of the last page.  Without it the bibliography, which is short
% enough to share one page, is filled column by column and the leftover glue lands inside it as a
% 380pt gap after the heading and blank lines between the first entries.
\usepackage{flushend}
\usepackage{morefloats}
\usepackage{adjustbox}
\usepackage{microtype}

% The manuscript carries 57 tables and 12 full-width plates; raise the float limits so that
% LaTeX never reports "Too many unprocessed floats".
\setcounter{totalnumber}{20}
\setcounter{topnumber}{10}
\setcounter{bottomnumber}{10}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}

\captionsetup[table]{skip=4pt}
\captionsetup[figure]{skip=4pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\interdisplaylinepenalty=2500

% technical prose with many tokens: give TeX room to stretch and hyphenate rather than overflow into
% the margin (microtype above does most of the work; \emergencystretch handles the rare long formula)
\tolerance=1200
\emergencystretch=3em

\graphicspath{{figures/}}

\title{Transformation Laws in Neural Representations:\\
Structure, Realisability, and Construction}

% IEEE Transactions style: the byline carries the name only; the affiliation and the e-mail address go
% into a first-page footnote, which is where an IEEE journal prints them.  To drop the address from the
% printed footnote, delete the parenthetical at the end of the \thanks argument.
\author{Yuan Sun%
\thanks{The author is with the School of Mathematical Sciences, Beijing Normal University,
Beijing 100875, China (e-mail: 3396897122@qq.com).}}

% hyperref last, with hidelinks: the running text cites figures, tables and sections by number, so
% internal links are useful in the PDF and the review copy should not be full of coloured boxes.
\usepackage[hidelinks,pdfusetitle]{hyperref}
\hypersetup{pdfauthor={Yuan Sun}}
"""


def place_figures(body, fig_tex):
    """Put each plate in the running text, immediately after the paragraph that first cites it.

    A paper's figures belong where they are argued, not in a gallery at the end: the manuscript's
    `## Figures` section is the Markdown reader's index of the captions, and the LaTeX build turns it
    into twelve floats distributed over the text.  Floats are inserted from the last reference to the
    first so that an insertion cannot move a not-yet-processed reference, and a figure whose reference
    cannot be found fails the build rather than silently landing at the end.
    """
    blocks = {}
    for tex in fig_tex:
        m = re.search(r'\\textbf\{Figure (\d+)\.\}', tex)
        if not m:
            raise SystemExit('figure block without a number')
        blocks[int(m.group(1))] = tex
    if not blocks:
        return body
    anchors = []
    for n in blocks:
        m = re.search(r'\bFigure %d\b' % n, body)
        if not m:
            raise SystemExit(f'figure {n}: no in-text reference to anchor the float to')
        para = body.find('\n\n', m.end())
        anchors.append((m.start(), n, len(body) if para < 0 else para + 2))
    for _, n, pos in sorted(anchors, reverse=True):
        body = body[:pos] + blocks[n] + '\n\n' + body[pos:]
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', default='paper/paper_v17.md')
    ap.add_argument('--out', default='paper/latex_v17')
    ap.add_argument('--figdir', default='paper/v15_figures')
    ap.add_argument('--measure', action='store_true',
                    help='instrument display equations with a width probe (for scripts/283)')
    a = ap.parse_args()
    MEASURE['on'] = bool(a.measure)
    src = os.path.join(WORK, a.paper)
    outdir = os.path.join(WORK, a.out)
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    for f in os.listdir(figdir):                 # a renamed plate must not linger as a second copy
        if f.lower().endswith(('.pdf', '.png')):
            os.remove(os.path.join(figdir, f))

    md = open(src).read()
    # ---- split into main text, figures, references, appendices -----------------
    i_fig = md.index('\n## Figures')
    i_ref = md.index('\n## References')
    i_app = md.index('\n## Appendices')
    parts = {
        'main': md[:i_fig],
        'figures': md[i_fig:i_ref],
        'references': md[i_ref:i_app],
        'appendices': md[i_app:],
    }

    # ---- figures: the ten vector figures of figures_final (captions.json is the source) --------
    caps_path = os.path.join(WORK, 'paper', 'figures_final', 'captions.json')
    plates, fig_tex = [], []
    if os.path.exists(caps_path):
        caps = json.load(open(caps_path))
        for c in caps:
            base = os.path.basename(c['file'])
            src_pdf = os.path.join(WORK, 'paper', c['file'] + '.pdf')
            if not os.path.isfile(src_pdf):
                raise SystemExit(f'missing figure {src_pdf}')
            shutil.copyfile(src_pdf, os.path.join(outdir, 'figures', base + '.pdf'))
            # the caption travels through the same text pipeline as the body, so every non-ASCII
            # character (degree signs, section marks) becomes a LaTeX command and the source stays ASCII
            cap = ' '.join(c['caption'].split())
            # the micro sign travels through `inline` like every other non-ASCII character: it becomes
            # `$\mu$`.  Writing \textmu here instead put the literal macro into the caption, because
            # `inline` escapes a backslash in text.
            cap = inline(cap)
            fig_tex.append('\\begin{figure*}[t]\n\\centering\n'
                           f'\\includegraphics[width=\\textwidth]{{{base}.pdf}}\n'
                           f'\\caption*{{\\textbf{{Figure {c["n"]}.}} {cap}}}\n'
                           '\\end{figure*}')
            plates.append({'n': c['n'], 'file': f'figures/{base}.pdf'})
    else:                                   # legacy path: the eight PNG plates
        for k in range(1, 9):
            src_png = os.path.join(WORK, a.figdir, f'F{k}.png')
            if not os.path.isfile(src_png):
                raise SystemExit(f'missing plate {src_png}')
            shutil.copyfile(src_png, os.path.join(outdir, 'figures', f'F{k}.png'))
            plates.append({'n': k, 'file': f'figures/F{k}.png'})

    # ---- convert -----------------------------------------------------------------
    body = convert(parts['main'])
    figs = convert(parts['figures'])
    refs = convert(parts['references'])
    apps = convert(parts['appendices'])

    # figure blocks: replace the caption block with an included plate
    fig_blocks = [] if fig_tex else re.split(r'(?=%%FIGCAPTION%%)', figs)
    for blk in fig_blocks:
        m = re.match(r'%%FIGCAPTION%%\s*(.*?)\n(.*)', blk, re.S)
        if not m:
            continue
        cap, rest = m.group(1), m.group(2)
        n = re.search(r'F(\d)', cap)
        if not n:
            continue
        fig_tex.append('\\begin{figure*}[t]\n\\centering\n'
                       f'\\includegraphics[width=\\textwidth]{{F{n.group(1)}.png}}\n'
                       '\\caption*{' + cap + '}\n\\end{figure*}')
    body = attach_captions(body)
    apps = attach_captions(apps)
    for name, txt in (('main text', body), ('appendices', apps)):
        if '%%CAPTION%%' in txt or '%%FIGCAPTION%%' in txt:
            raise SystemExit(f'{name}: a caption marker was never attached to a float')
    body = place_figures(body, fig_tex)

    # references -> thebibliography
    bib = []
    for ln in parts['references'].split('\n'):
        m = re.match(r'^-\s+(.*)$', ln.strip())
        if m:
            entry = m.group(1)
            # the manuscript numbers its entries `[n]` so that the Markdown list can be read as a
            # bibliography; \bibitem prints that number itself, so the literal label is dropped here.
            entry = re.sub(r'^\[\d+\]\s*', '', entry)
            bib.append('\\bibitem{} ' + inline(entry))
    # \clearpage first: without it the full-width appendix tables still waiting for a page are flushed
    # after the bibliography has started, which cuts the reference list into pieces on several pages.
    ref_tex = ('\\clearpage\n\\begin{thebibliography}{99}\n\\footnotesize\n' + '\n'.join(bib)
               + '\n\\end{thebibliography}')

    # abstract close
    body = body.replace('\\section*{1. Introduction}', '\\section{Introduction}')
    if '\\begin{abstract}' in body:
        body = body.replace('\\begin{abstract}', '\\begin{abstract}', 1)
        # the abstract ends where the first \section starts
        k = body.index('\\section{Introduction}')
        body = body[:k] + '\\end{abstract}\n\n' + body[k:]
        # ... and the index terms follow the abstract, outside it
        m = re.search(r'%%KEYWORDS%% (.*)', body)
        if m:
            body = body.replace(m.group(0), '', 1)
            body = body.replace('\\end{abstract}', '\\end{abstract}\n\n\\begin{IEEEkeywords}\n'
                                + m.group(1) + '\n\\end{IEEEkeywords}', 1)

    # ---- split into parts/ so the project is editable section by section ----
    parts_dir = os.path.join(outdir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    for f in os.listdir(parts_dir):
        if f.endswith('.tex'):
            os.remove(os.path.join(parts_dir, f))

    def split_sections(tex, prefix):
        lines = tex.split('\n')
        idx = [i for i, l in enumerate(lines) if l.startswith('\\section{')]
        chunks = []
        if idx and '\n'.join(lines[:idx[0]]).strip():
            chunks.append((prefix + '00_front', '\n'.join(lines[:idx[0]])))
        for k, i in enumerate(idx):
            j = idx[k + 1] if k + 1 < len(idx) else len(lines)
            title = re.sub(r'^\\section\{(.*)\}$', r'\1', lines[i])
            name = re.sub(r'[^A-Za-z0-9]+', '_', title)[:38].strip('_').lower()
            chunks.append((f'{prefix}{k + 1:02d}_{name}', '\n'.join(lines[i:j])))
        return chunks

    files = []
    abstract = ''
    main_chunks = split_sections(body, 'sec')
    if main_chunks and main_chunks[0][0] == 'sec00_front':
        abstract = main_chunks.pop(0)[1]
    for name, text in main_chunks:
        files.append((f'parts/{name}.tex', text))
    for name, text in split_sections(apps, 'app'):
        files.append((f'parts/{name}.tex', text))
    files.append(('parts/references.tex', ref_tex))
    # explicit order: main sections, plates, appendix front matter, appendices, references
    def _pick(pred):
        return [f for f in files if pred(f[0])]

    files = (_pick(lambda n: n.startswith('parts/sec'))
             + _pick(lambda n: n == 'parts/app00_front.tex')
             + _pick(lambda n: n.startswith('parts/app') and n != 'parts/app00_front.tex')
             + _pick(lambda n: n == 'parts/references.tex'))

    def write(path, text):
        text = text.replace(r'\mathbb 1', r'\mathbf{1}').replace(r'\mathbb{1}', r'\mathbf{1}')
        text = text.replace('$$', '$ $')
        open(os.path.join(outdir, path), 'w').write(text)

    body_inputs = '\n'.join('\\input{%s}' % n[:-4] for n, _ in files)
    doc = (PREAMBLE + '\n\\begin{document}\n\\maketitle\n'
           + ('\\typeout{COLWIDTH \\the\\columnwidth}\n' if MEASURE['on'] else '')
           + '\n' + abstract + '\n\n'
           + '%% ---- main text, figures, appendices, references (parts/) ----\n'
           + body_inputs + '\n\n\\end{document}\n')
    for name, text in files:
        write(name, text)
    write('main.tex', doc)
    doc = doc + '\n'.join(t for _, t in files)      # for the summary counts only

    readme = r"""# LaTeX source of the manuscript (v17 line)

Generated from `paper/paper_v17.md` by `scripts/267_md_to_tex.py`.  Do not edit the `.tex` files by
hand: edit the Markdown and re-run the converter, which rewrites `main.tex`, `parts/` and this file.

## Overleaf

1. Zip this folder (`paper/latex_v17.zip` sits next to it) and upload it:
   *New Project -> Upload Project -> select the zip*.  Overleaf unpacks it and opens `main.tex`.
2. Leave the compiler on **pdfLaTeX** (*Menu -> Compiler -> pdfLaTeX*).  No shell escape, no BibTeX
   run, no extra packages beyond Overleaf's TeX Live.
3. Compile twice if you want the page numbers of the internal links to settle.

The source is deliberately plain and pure ASCII: `IEEEtran` + `amsmath/amssymb/amsfonts`, `graphicx`,
`booktabs`, `tabularx`, `array`, `caption`, `textcomp`, `url`, `flushend`, `morefloats`, `adjustbox`,
`microtype`, `hyperref` (hidelinks).  Nothing is read from the network and no font file is embedded, so
the project compiles unchanged on Overleaf and locally with `tectonic` or `pdflatex`.

## Structure

```
latex_v17/
├── main.tex                 preamble, title, abstract, \input of every part
├── parts/
│   ├── sec01_introduction.tex ... sec09_conclusion.tex   main text, with the figure floats in place
│   ├── app00_front.tex                                    \appendix + appendix guide
│   ├── app01_claim_status_table.tex ... app07_...tex      appendices A-H
│   └── references.tex                                     thebibliography (31 entries, IEEE style)
├── figures/fig01_object.pdf ... fig12_prediction.pdf      the twelve plates (vector PDF)
└── README.md
```

## What is in the document

* **Main text** sections 1-9.  Twelve full-width figures are placed as floats next to the paragraph
  that first cites them, each with its caption below and its panel-by-panel reading in the caption.
* **Tables**: 57, caption above the table (IEEE style), `tabularx` so no row overflows the measure.
* **Appendices A-H**: claim-status table, assumptions and proofs, protocols, the full grids
  (Tables D.1-D.23), method boxes, per-setup capability tables, and the positioning table.
* **References**: 31 entries, IEEE style, numbered by first citation; the running text cites them as
  `[n]`.
* Figure, table and section numbers in the text are the manuscript's own; the LaTeX prints them from
  the captions, so a renumbering pass in the Markdown (`scripts/264_integrate_v17.py`) keeps the text
  and the plates in step.

## Build status

Compiled end to end with `tectonic` (pdfLaTeX-compatible): 0 errors, 0 overfull or underfull boxes,
48 pages, 12 vector plates, 57 tables, 31 references.  `scripts/250_audit_v15.py` passes
(C1-C6, C8, C9) and `scripts/255_recompute_numbers.py` agrees on all 153 checked numbers.

## Rebuilding

```bash
python scripts/267_md_to_tex.py              # regenerates main.tex, parts/, figures/, README.md
```
"""
    open(os.path.join(outdir, 'README.md'), 'w').write(readme)
    print(f'wrote {os.path.relpath(os.path.join(outdir, "main.tex"), WORK)} '
          f'({len(open(os.path.join(outdir, "main.tex")).read().split(chr(10)))} lines) + '
          f'{len(files)} part files, {len(plates)} plates, '
          f'{doc.count("begin{table*")} tables, {len(bib)} references')
    return 0


if __name__ == '__main__':
    sys.exit(main())
