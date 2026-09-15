# LaTeX source of the manuscript (v17 line)

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
