# Licensing

This repository is not under a single licence: the pieces are under different terms on purpose.

| part | licence | what it means |
|---|---|---|
| `scripts/`, `tools/` (the code) | **MIT** — see `LICENSE` | use, modify and redistribute freely, keep the copyright notice |
| `docs/`, `paper/figures_final/`, `meta/` | **CC BY 4.0** | share and adapt for any purpose including commercial, with attribution |
| `paper/paper_v17.md` (the manuscript) | author's preprint, no blanket licence | if the paper is published by IEEE, the published version is copyright IEEE; cite that for the paper |
| `CITATION.cff` | metadata, not a licence | how to cite the code and the paper |

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

## Third-party material that is *not* distributed here

No dataset, no pretrained checkpoint and no third-party image is included, so none of their licences is
passed on to you.  To reproduce the measurements you obtain them yourself, under their own terms:

* **COCO** (images and annotations): https://cocodataset.org/#termsofuse
* **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
* **torchvision ImageNet weights** (ResNet, ConvNeXt, ViT): BSD-3-Clause, downloaded by torchvision
* **DINOv2**: the code is Apache-2.0, the published weights are **CC BY-NC-4.0 (non-commercial)** —
  https://github.com/facebookresearch/dinov2
* **YOLO11n (Ultralytics)**: used as an installed dependency; the `ultralytics` package is **AGPL-3.0** and
  its weights are not redistributed here

If you add any of these to a fork, check their terms first: they are stricter than this repository's MIT
licence for the code.

## How this is detected by GitHub

GitHub reads the root `LICENSE` file, which contains the MIT text, and shows *MIT license* in the
repository sidebar.  The rest of the terms live here, in `LICENSING.md`, so that the sidebar stays
accurate rather than reading "Other".
