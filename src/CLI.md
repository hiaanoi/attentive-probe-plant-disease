# Core GPU reproduction commands

The CLI is new packaging glue around the preserved experimental functions.
It is statically checked, not execution-validated. It covers the PlantWild
518-pixel core pipeline, both official and deduplicated training roles.
PlantSeg three-fold search and the supplementary protocols remain separate
function interfaces; this CLI does not implement their full orchestration.

Use a Python environment with the dependencies listed in the root README.
Set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`). Run commands from
the repository root. Example paths below are relative and user supplied.
Output directories must be new. Acquire images and licensed backbone weights
separately; no images, weights, or extracted arrays are included here.

Prepare `inputs/manifest.csv` with columns:

```csv
image_id,label,split,relative_path,sha256
example_id,0,train,class/example.jpg,REPLACE_WITH_IMAGE_SHA256
```

Join the released split identifiers and checksums to your local image paths.
Labels must be the published contiguous class IDs, starting at zero. Use
`train`, `val`, `test`, or `exclude` for split. Every feature array follows
exactly this manifest's row order. For deduplicated reproduction preserve
the same array rows, mark all removed training and validation records `exclude`
using the released `keep=1` assignments, and retain the same test assignments.
The deduplicated validation set has 1787 rows rather than 1820. Do not replace the published split with a new
random split. The example row is a schema illustration, not an actual image.
For deduplicated attentive or MLP fitting, pass `--fixed-config` and the
official run's `selection.json` (or the corresponding published selected
configuration). Do not run the eight-candidate neural search again on the
deduplicated rows. The class-token probe C selection remains separate.
Deduplicated feature-fusion and uniform-control protocol orchestration is not
claimed by these core CLI instructions; use their source-specific recipes.

```sh
python -m attentive_probe.cli extract --manifest inputs/manifest.csv --images inputs/images --model inputs/dinov2-large --backbone dinov2 --output outputs/dinov2
```

Repeat with `--backbone dinov3` and `--backbone clip`, the appropriate local
model directories, and separate output directories. DINOv3 requires its
documented Transformers environment. CLIP loads the official processor from
the model directory. Do not use PlantSeg float32 DINOv3 patch storage for
PlantWild; the default is PlantWild float16. Extraction writes `global.npy`,
`patch.npy`, optional `mean_patch.npy`, and an index without local image paths.

```sh
python -m attentive_probe.cli train --manifest inputs/manifest.csv --features outputs/dinov2/global.npy --head probe --output outputs/dinov2_probe
python -m attentive_probe.cli train --manifest inputs/manifest.csv --features outputs/dinov2/global.npy --patches outputs/dinov2/patch.npy --head attentive --config configs/plantwild_head_candidates.json --output outputs/dinov2_attention
```

The train command chooses C on validation accuracy then macro-F1; fits each
neural candidate with seed 101; chooses validation accuracy, then macro-F1,
then candidate order; reuses the selected seed-101 model; fits seeds 202, 303,
404 and 505; then produces test predictions and probabilities. Patch data are
transferred from the local cache in batches, preserving stored values and
the source training order, optimizer and schedule. This transfer adaptation
has not been numerically exercised. Disk space for full patch caches can be
large; extraction and fitting require CUDA.

For the official matched uniform control use `--head uniform --fixed-config` with
`configs/plantwild_dinov3_selected.json` (the same selected numerical head
recipe). For the residual MLP use `--head mlp` with
`configs/plantwild_head_candidates.json`, omitting `--patches`.
For DINOv3 and CLIP probes/attentive heads use their own global and patch
arrays and `--c-grid 0.001,0.01,0.1,1`; use the corresponding candidate
configurations. DINOv3 shares the eight PlantWild head candidates.

```sh
python -m attentive_probe.cli train --manifest inputs/manifest.csv --features outputs/dinov2/global.npy outputs/dinov3/global.npy outputs/clip/global.npy --head feature-fusion --config configs/feature_fusion_candidates.json --output outputs/feature_fusion
python -m attentive_probe.cli fuse --probabilities outputs/dinov2_attention/probabilities_101.npy outputs/dinov3_probe/probabilities_deterministic.npy outputs/clip_probe/probabilities_deterministic.npy --output outputs/swap_101.npy
```

The three fusion inputs must have identical test row order and class order.
Repeat swap for all five attention seeds. Replace the first input with the
DINOv2 class-token probe for the three-probe baseline. Feature-fusion search
keeps each candidate's best validation macro-F1 epoch, then selects the
candidate by accuracy and macro-F1. Every prediction CSV contains the image
identifier, true label and predicted class; probability arrays support the
saved-prediction statistics functions.

See `README.md` in this directory for individual APIs and unported
supplementary-protocol orchestration. No claim of successful full GPU
reproduction is made by the release's CPU/static checks.
