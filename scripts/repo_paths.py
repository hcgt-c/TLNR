"""repo_paths.py — every filesystem root this release uses, in one place.

The scripts in this repository were written over several months on one machine, and the paths they
started with were absolute (`/data/...`, `/home/...`).  Those are gone: each script imports this module
and reads its roots from here, so the code can be checked out anywhere and run from anywhere.

Resolution order for every root is the same:

1. the environment variable named below, if it is set;
2. otherwise a default inside the checkout (`<repo>/data/...`, `<repo>/results/...`).

Datasets are **not** redistributed (the COCO and ImageNet licences do not allow it).  Point the
environment variables at your own copies; `check()` prints what is present and what is still missing.

    export COCO_ROOT=/path/to/coco            # expects train2017/ and annotations/instances_train2017.json
    export CIFAR10_ROOT=/path/to/cifar10_raw  # the torchvision CIFAR-10 python batches
    export YOLO_WEIGHTS=/path/to/yolo11n.pt   # ultralytics YOLO11n checkpoint
    export TL_DATA=/path/to/scratch           # optional: reruns write here instead of inside the repo

Nothing here reads the network, and no credential is stored: the only external artefacts are the public
datasets and the public pretrained checkpoints named in docs/REPRODUCING.md.
"""
import os
import shutil
import sys
import tempfile

__all__ = [
    'REPO_ROOT', 'SCRIPTS_DIR', 'RESULTS_DIR', 'FIGURES_DIR', 'DATA_ROOT', 'FEATURES_DIR', 'RUNS_DIR',
    'PAPER_DIR', 'META_DIR', 'COCO_ROOT', 'COCO_IMAGES', 'COCO_ANNOTATIONS', 'CIFAR10_ROOT',
    'YOLO_WEIGHTS', 'YOLO_DET_HUE', 'CLIP_WEIGHTS', 'TORCH_HUB', 'DINOV2_HUB', 'DINOV2_WEIGHTS',
    'MPLCONFIGDIR',
    'TECTONIC', 'require', 'check',
]

#: the checkout root: the directory that contains `scripts/`
REPO_ROOT = os.environ.get('TL_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
RESULTS_DIR = os.environ.get('TL_RESULTS', os.path.join(REPO_ROOT, 'results'))
FIGURES_DIR = os.environ.get('TL_FIGURES', os.path.join(REPO_ROOT, 'figures'))
META_DIR = os.path.join(REPO_ROOT, 'meta')
PAPER_DIR = os.path.join(REPO_ROOT, 'paper')

#: scratch and derived data.  The repository ships none of it; reruns create it.
DATA_ROOT = os.environ.get('TL_DATA', os.path.join(REPO_ROOT, 'data'))
FEATURES_DIR = os.environ.get('TL_FEATURES', os.path.join(REPO_ROOT, 'features'))
RUNS_DIR = os.environ.get('TL_RUNS', os.path.join(REPO_ROOT, 'runs'))

# --- the three external datasets / checkpoints -------------------------------------------------
COCO_ROOT = os.environ.get('COCO_ROOT', os.path.join(DATA_ROOT, 'coco'))
COCO_IMAGES = os.environ.get('COCO_IMAGES', os.path.join(COCO_ROOT, 'train2017'))
COCO_ANNOTATIONS = os.environ.get(
    'COCO_ANNOTATIONS', os.path.join(COCO_ROOT, 'annotations', 'instances_train2017.json'))
CIFAR10_ROOT = os.environ.get('CIFAR10_ROOT', os.path.join(DATA_ROOT, 'cifar10_raw'))
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', os.path.join(DATA_ROOT, 'weights', 'yolo11n.pt'))
YOLO_DET_HUE = os.environ.get('YOLO_DET_HUE', os.path.join(DATA_ROOT, 'yolodet_hue'))
CLIP_WEIGHTS = os.environ.get('CLIP_WEIGHTS', os.path.join(DATA_ROOT, 'clip', 'ViT-B-32.pt'))

# --- pretrained backbones come from the standard caches, never from the repository --------------
TORCH_HUB = os.environ.get('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
DINOV2_HUB = os.environ.get('DINOV2_HUB',
                            os.path.join(TORCH_HUB, 'hub', 'facebookresearch_dinov2_main'))
DINOV2_WEIGHTS = os.environ.get(
    'DINOV2_WEIGHTS', os.path.join(TORCH_HUB, 'hub', 'checkpoints', 'dinov2_vitb14_pretrain.pth'))

# --- tooling -----------------------------------------------------------------------------------
#: matplotlib writes its font cache here; a writable directory is enough
MPLCONFIGDIR = os.environ.get('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'tl_mplcfg'))
#: the TeX engine used by the equation-width probe (optional dependency)
TECTONIC = os.environ.get('TECTONIC', shutil.which('tectonic') or 'tectonic')


def require(path, what, how):
    """Return `path`, or exit with an actionable message naming what is missing and how to get it."""
    if not os.path.exists(path):
        sys.stderr.write(
            f'\n{what} not found at:\n    {path}\n'
            f'This release does not redistribute it. {how}\n'
            f'Set the matching environment variable (see docs/REPRODUCING.md) and try again.\n\n')
        raise SystemExit(2)
    return path


def check(verbose=True):
    """Report which roots are present.  Returns the list of missing ones (empty when all are there)."""
    items = [
        ('repository results', RESULTS_DIR, 'shipped with the release'),
        ('COCO images', COCO_IMAGES, 'export COCO_ROOT=/path/to/coco'),
        ('COCO annotations', COCO_ANNOTATIONS, 'export COCO_ROOT=/path/to/coco'),
        ('CIFAR-10 raw', CIFAR10_ROOT, 'export CIFAR10_ROOT=/path/to/cifar10_raw'),
        ('YOLO11n weights', YOLO_WEIGHTS, 'export YOLO_WEIGHTS=/path/to/yolo11n.pt'),
        ('DINOv2 hub', DINOV2_HUB, 'export DINOV2_HUB=/path/to/facebookresearch_dinov2_main'),
        ('CLIP ViT-B/32', CLIP_WEIGHTS, 'export CLIP_WEIGHTS=/path/to/ViT-B-32.pt'),
    ]
    missing = []
    if verbose:
        print('path configuration')
        print('  repo root      ', REPO_ROOT)
        for name, path, how in items:
            ok = os.path.exists(path)
            print('  %-15s %s  %s' % (name, 'OK  ' if ok else 'MISS', path if ok else f'{path}   ({how})'))
            if not ok:
                missing.append(name)
    return missing


if __name__ == '__main__':
    check()
