# Transformation Laws in Neural Representations — code and results

Code, stored results and figures for the paper

**Repository:** <https://github.com/hcgt-c/TLNR>  ·  **Paper:** [arXiv:2609.18190](https://arxiv.org/abs/2609.18190)

> **Transformation Laws in Neural Representations: Structure, Realisability, and Construction**
> Yuan Sun, School of Mathematical Sciences, Beijing Normal University — arXiv:2609.18190

The paper asks when a physical transformation of an input — a hue rotation, a heat diffusion, a clipped
scaling — survives the trip through a frozen network as an *operator on the features*, and what that
operator costs.  This repository contains everything behind its numbers: the experiment scripts, the
stored result files the manuscript quotes, the twelve figures and the scripts that draw them, and the
checks that keep the two in step.

## Quick start (no datasets needed)

```bash
python -m pip install -r requirements.txt
python tools/verify_release.py            # safety audit + every quoted number + manuscript audit
python tools/verify_release.py --figures  # also redraw the twelve figures and compare them
```

`tools/verify_release.py` re-derives all 153 quoted numbers from the stored results, audits the
manuscript's internal references, and (with `--figures`) rebuilds every plate from `results/` and checks
it against the shipped PDF word for word.  It needs no dataset: the stored results *are* the evidence.

To sanity-check your path configuration:

```bash
python scripts/repo_paths.py
```

## Full re-runs (datasets required)

The datasets are not redistributed — the COCO and ImageNet licences do not allow it.  Download them
yourself and point the release at them:

```bash
export COCO_ROOT=/path/to/coco              # needs train2017/ and annotations/instances_train2017.json
export CIFAR10_ROOT=/path/to/cifar10        # a directory containing cifar-10-batches-py/
export YOLO_WEIGHTS=/path/to/yolo11n.pt     # ultralytics YOLO11n checkpoint
export TL_DATA=/scratch/tl                  # optional: reruns write here instead of the checkout
```

Then run the experiment whose table you want, for example

```bash
python scripts/188_depth_map_harness.py --backbone resnet50 --depths 1,2,3,4   # the depth grid
python scripts/263_two_term_heldout.py --n 300                                 # the held-out decomposition
python scripts/229_star_existence_test.py --threads 4                          # the fibre test
```

Each script writes its JSON into `results/`; `docs/MAPPING.md` says which script and which result file
stand behind every table and figure of the paper.  `docs/REPRODUCING.md` has the details, the hardware
the measurements were made on, and what each script needs.

## Layout

```
scripts/         the experiment scripts the paper cites, the figure scripts, the checks, and
                 everything they import (see docs/MAPPING.md for the count and the mapping)
  repo_paths.py  every filesystem root in one place (environment-overridable)
results/         the stored results the manuscript quotes
paper/           the manuscript source, the twelve vector figures, the LaTeX project
meta/            provenance: SOURCES.md (table -> results -> scripts), ARTIFACTS.md, FIGURE_SPEC.md
tools/           safety_audit.py, verify_release.py
docs/            REPRODUCING.md, MAPPING.md, FIGURES.md, SAFETY.md
```

## What the paper measures, in one table

| | |
|---|---|
| Transformation algebras | hue rotation (group), heat diffusion (semigroup), clipped scaling (monoid) |
| Sites | ResNet-50/18, ConvNeXt-T, ViT-B/16, DINOv2-B/14, YOLO11n — four depths each |
| Data | synthetic orbit suites (generated), COCO train2017 crops and instance regions, CIFAR-10 |
| Figures / tables | 12 vector figures, 63 tables, 31 references |
| Hardware | one NVIDIA RTX 2060 (6 GB); the CPU-only controls ran on four threads |

## Safety and privacy

The scripts were written on one machine over several months and originally carried absolute paths.  They
are gone: every script imports `scripts/repo_paths.py`, which resolves each root from an environment
variable with a checkout-relative default.  `tools/safety_audit.py` fails on any absolute path,
credential, private identifier, oversized file or dangling symlink, and `tools/verify_release.py` runs it
first.  No dataset, no raw image, no model checkpoint and no personal data are redistributed.  See
`docs/SAFETY.md`.

## Publishing this release

The checkout is already a git repository with one commit, so publishing is two commands:

```bash
git remote add origin https://github.com/hcgt-c/TLNR.git
git push -u origin main
```

That is all this release needs: the documentation lives in `README.md`, `docs/` and the release notes, and
no external archive is required.  GitHub shows the licence in the repository sidebar from the `LICENSE`
file (MIT for the code, CC BY 4.0 for the docs and figures — `LICENSE` and `LICENSING.md` state the split and
the third-party terms that are *not* passed on to you).

If you later want the code to be citable as its own object, that is what an archival DOI (Zenodo,
Software Heritage) is for; it changes nothing in this repository or in the paper, and it is not needed to
publish the code.

## Licence and citation

**Code: MIT** (`LICENSE`) **— documentation and figures: CC BY 4.0** (`LICENSING.md`).  The manuscript
Markdown is the author's preprint, and the third-party terms (COCO, CIFAR-10, DINOv2, YOLO11n, torchvision
weights) are listed in `LICENSING.md` because none of that material is distributed here.  GitHub reads the
root `LICENSE`, so the repository sidebar shows *MIT license*.

If you use this material, please cite the paper (`CITATION.cff` carries the same entry):

```bibtex
@misc{sun2026transformation,
  title  = {Transformation Laws in Neural Representations: Structure, Realisability, and Construction},
  author = {Sun, Yuan},
  year   = {2026},
  eprint = {2609.18190},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2609.18190}
}
```
