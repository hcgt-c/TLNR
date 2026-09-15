# Reproducing the paper

Two tiers, in increasing cost.  Tier 1 needs nothing but this repository and runs in minutes; tier 2 needs
the public datasets and a GPU and reproduces the measurements themselves.

```bash
python scripts/repo_paths.py        # which roots resolve, which datasets are still missing
```

## Tier 1 — check the paper against the stored results (no datasets)

```bash
python -m pip install -r requirements.txt
python tools/verify_release.py --figures
```

What each step does:

| step | script | what it proves |
|---|---|---|
| safety audit | `tools/safety_audit.py` | nothing private is in the release |
| number check | `scripts/255_recompute_numbers.py` | every number quoted in the manuscript equals the value in `results/` (153 of them) |
| manuscript audit | `scripts/250_audit_v15.py` | internal references resolve; every table and figure is captioned and cited; citations resolve; provenance paths exist; the depth grid reconciles with its result file |
| figure redraw | the twelve `scripts/27x/28x_fig*.py` | each plate follows from `results/`: the redrawn PDF is compared word for word with the shipped one |

The figures are drawn from stored JSON only, so `--figures` is a genuine reproduction of the paper's
pictures and a strong check on the result files.  `scripts/266_artifact_ledger.py` regenerates
`meta/ARTIFACTS.md` (the result-file to script index) if you want to see the whole map.

## Tier 2 — reproduce the measurements

### Environment

Python 3.10 with the versions in `requirements.txt` (tested: torch 2.10, torchvision 0.25, numpy 2.2,
scipy 1.15, matplotlib 3.10, ultralytics 8.4, scikit-learn 1.7).  A single NVIDIA RTX 2060 with 6 GB was
enough; the CPU-only controls were run on four threads (`OMP_NUM_THREADS=4`).  `environment.yml` builds a
matching conda environment.

### Datasets and checkpoints

None of these is redistributed.  All are public:

| what | how to get it | where the release expects it |
|---|---|---|
| COCO train2017 images | `http://images.cocodataset.org/zips/train2017.zip` | `$COCO_ROOT/train2017` |
| COCO train2017 annotations | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | `$COCO_ROOT/annotations/instances_train2017.json` |
| CIFAR-10 (python batches) | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | `$CIFAR10_ROOT/cifar-10-batches-py` |
| ImageNet-pretrained backbones | downloaded automatically by `torchvision` (`IMAGENET1K_V1`/`V2` weights) | torch hub cache |
| DINOv2 ViT-B/14 | `torch.hub` from `facebookresearch/dinov2`, checkpoint `dinov2_vitb14_pretrain.pth` | `$TORCH_HOME/hub` |
| YOLO11n | ultralytics release, `yolo11n.pt` | `$YOLO_WEIGHTS` |

Set `COCO_ROOT`, `CIFAR10_ROOT`, `YOLO_WEIGHTS` (and `TL_DATA` if you want reruns to write outside the
checkout).  The synthetic orbit suites and the cached features are **generated**, not shipped:

```bash
python scripts/41_gen_dense3d.py        # the 3-D lattice, rings and low-extension suites
python scripts/42_extract_dense3d.py    # their features at the probed sites
python scripts/79_code3d_v3.py          # the six-dimensional code head used by the read-out experiments
python scripts/115_sv_lowext.py         # the (saturation, value) boundary grid
python scripts/176_second_attribute_chain.py   # the clipped-scaling chain
```

### The measurement runs

Every script is standalone: run it from the checkout root, it reads `results/` (and the generated features
where it needs them), and it writes its JSON back into `results/`.  Scripts that take no arguments use the
settings the paper reports.  The main ones, with their real flags:

```bash
# the defect law in the linear layer (Theorems 1-2, Tables 3 and B.1)
python scripts/206_transport_theory.py
python scripts/207_deep_linear_transport.py

# orbits, organisation, inheritance
python scripts/31_dense_features.py                    # features of the dense orbit suite
python scripts/43_preview_3d_sparse.py

# the depth grid: four backbones, four depths, three seeds, both algebras
python scripts/188_depth_map_harness.py --backbone resnet50 --family hue --params 90.0 --depths 1,2,3,4
python scripts/188_depth_map_harness.py --backbone resnet50 --family heat --params 1.0 --depths 1,2,3,4

# the fibre condition, tested directly (Table 2 of the appendix; CPU-only)
python scripts/229_star_existence_test.py --backbones resnet50,convnext,vitb16,dinov2b14 --n 300 --threads 4

# real images: detector, per-sample studies, boundary grid
python scripts/156_detector_o8_multiseed.py
python scripts/159_realmultiattr_intervention.py
python scripts/115_sv_lowext.py                        # add --regen to recompute the code head's grid

# the two-source decomposition, in-split and held out (Table 13 and Appendix D.20)
python scripts/260_two_term_decomposition.py --backbone resnet50 --n 100
python scripts/263_two_term_heldout.py --n 300

# metric distortion of the truth action (Table D.22)
python scripts/258_measure_lipschitz.py --backbone resnet50 --n 150 --hue_deg 90.0

# the interface and its uses
python scripts/265_interface_action_comparison.py --arms fixed,gen,lin,mlp --seeds 0,1,2
python scripts/109_region_rho_demo.py
python scripts/112_usage_rule_strat.py
```

`python scripts/<name>.py --help` prints the full flag set of the harness-style scripts (`188`, `229`,
`258`, `260`, `263`, `265`).  The remaining scripts are single-purpose and take no arguments.

### Cost

The measurements are wide rather than deep: four backbones × four depths × two algebras × three seeds for
the depth grid, plus per-sample studies over COCO crops.  On one 6 GB GPU the depth grid is the dominant
cost; the CPU-only controls (`229`, and the existence test in general) run in minutes to hours.  No
measurement in the paper needed more than a single GPU or more than a few hundred images per arm — the
per-arm sample sizes are stated with every table (Appendix C of the manuscript).

## What is not reproducible from this checkout

Stated here rather than left for the reader to discover:

* **The 206 exploratory scripts** of the project's history are not included: nothing in the paper depends
  on them, and many were superseded by the scripts that are here.
* **Cached features, checkpoints and runs** (`features/`, `runs/`) are derived artefacts; the scripts that
  create them are included.
* **The datasets themselves**, for licensing reasons — see the table above.
* **Wall-clock numbers**: they belong to the RTX 2060 named in the paper and will differ elsewhere; the
  paper reports ratios against a copy baseline for exactly that reason.

## Reporting a problem

If a script here and the manuscript disagree, the manuscript is the claim and the script is the evidence:
please open an issue with the table or figure number and the result file involved.
