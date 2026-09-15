# Safety and privacy of this release

The code was written over several months on one workstation, and it started out full of absolute paths.
This document says what was done about that, what the release deliberately does **not** contain, and how
to check both claims yourself.

## What was changed before publishing

Every script now imports `scripts/repo_paths.py` and takes its filesystem roots from there.  The original
literals are gone; a release blocker if any comes back.

| was | is now |
|---|---|
| the author's checkout root (57 scripts) | `rp.REPO_ROOT`, computed from the script's own location |
| the private dataset root, COCO images and annotations | `rp.COCO_IMAGES`, `rp.COCO_ANNOTATIONS` (`$COCO_ROOT`) |
| the YOLO11n checkpoint under that root | `rp.YOLO_WEIGHTS` (`$YOLO_WEIGHTS`) |
| the CIFAR-10 raw batches under that root | `rp.CIFAR10_ROOT` (`$CIFAR10_ROOT`) |
| `/home/<user>/.cache/torch/hub/...` | `rp.TORCH_HUB`, `rp.DINOV2_HUB`, `rp.DINOV2_WEIGHTS` (`$TORCH_HOME`) |
| `/home/<user>/miniconda3/envs/yolo11/bin/python` | `sys.executable` |
| `/tmp/mplcfg` | `rp.MPLCONFIGDIR`, a temporary directory chosen at run time |

Points in the *other* direction — a path invented by the changes — is impossible: `repo_paths` only reads
the environment and the checkout, never the network, and stores no credential.

## What is not in the release

| excluded | why |
|---|---|
| COCO, CIFAR-10, ImageNet images and annotations | their licences forbid redistribution; the download table is in `docs/REPRODUCING.md` |
| pretrained checkpoints (`yolo11n.pt`, `dinov2_vitb14_pretrain.pth`, torchvision weights) | obtained from their public sources, not vendored |
| `data/`, `features/`, `runs/` (2.4 GB of intermediate arrays and training runs) | derived artefacts; the scripts that create them are included |
| 206 exploratory scripts from the project's history | nothing in the paper depends on them; the 95 shipped scripts are the paper's pipeline and its imports |
| the manuscript's internal provenance notes | they name result files and scripts; the reader-facing form is `docs/MAPPING.md` and `meta/SOURCES.md` |
| personal data of any kind | none is used by the experiments; the only name and address in the release are the author's, in `CITATION.cff`, on purpose |

## Checking it

```bash
python tools/safety_audit.py           # fails on any absolute path, credential, private identifier, big file, dangling link
python tools/verify_release.py         # runs the audit first, then the scientific checks
```

The audit also enforces a size limit (20 MB per file by default) so that a future commit cannot quietly
add a dataset dump, and it refuses files that look like credentials or private identifiers
(access-key and token patterns, and a short list of publication markers used internally).

## If you find something that should not be here

Please open an issue, or e-mail the address in `CITATION.cff`.  A path that leaks a machine layout is a bug
in this release, not a curiosity.
