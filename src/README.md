# Scientific source interfaces

The core PlantWild extraction/training/fusion CLI is documented in [CLI.md](CLI.md).

Add `src` to `PYTHONPATH` to import `attentive_probe`. These are extracted and
adapted scientific functions from the experimental implementations. They do
not automatically discover datasets or run experiments on import. The local
release checks parse the source; they do not train, extract features or run a
model forward pass. Full GPU reproduction has not been exercised from this
packaged source.

| Module | Entry points and inputs |
| --- | --- |
| `geometry` | `geometry`, `transform`, `patch_weights`: original resize/crop and lesion occupancy |
| `features` | `extract_batch(images, model, backbone, processor=None, dinov3_patch_dtype="float16")`: caller-supplied PIL images and frozen CUDA model; returns global, patch, and optional mean-patch arrays |
| `attention` | `ClassConditionedMIL`, `lesion_region_mass_loss`, `combined_loss`: frozen patch tensors `[B,N,D]`, frozen probe logits `[B,C]` |
| `global_heads` | `make_head`, `PrototypeHead`, `HybridHead`, `focal_loss`, `learning_rate`: global-feature heads and exact one-based cosine schedule |
| `residual_mlp` | `CLSResidual`, `optimizer`, `update_lr`, `train_residual`: original PlantWild 1024–244–89 residual MLP and fixed-epoch training loop |
| `training` | `lr_fit`, `lr_logits`, `train_global`, `train_mil`, `predict_mil`, `mil_forward`: caller-supplied aligned feature arrays, labels and row indices |
| `plantseg_training` | `Fit(c, seed, x, g, y, rows, epochs)`: original PlantSeg fitting loop; `epoch`, `pred`, `state`, `close`; CUDA feature tensor and NumPy logits/labels |
| `fusion` | `feature_blocks`, `combine`, `choose_weights`, `scalar_fit`: block L2 normalization, three-probability arithmetic/geometric/weighted fusion, validation grid and stacking fit |
| `statistics` | `metrics`, `analyze`: probabilities `[N,C]`, class-to-host mapping, paired seed dictionaries, shared bootstrap stream |
| `errors` | `transitions`: aligned truth and hard predictions; C/W/X transition matrix and net components |

`training` runners require a callback object with `check(force=False)` and
`status(stage, **values)` methods. `io.Progress()` supplies an optional empty
callback. Pass arrays and output directories explicitly; output directories for
fits must not already exist. `train_mil` expects CUDA patch and frozen-logit
tensors, NumPy labels and row indices. `train_global` expects NumPy global
features. `lr_fit` standardizes training rows only. Models saved by these
functions should only be loaded from trusted local runs.

The PlantWild attentive head uses `variant="FULL"`; uniform pooling uses
`variant="UNIFORM"`; the no-global-logit control uses `variant="NO_G1"`.
For the fixed-update data-size protocol, pass `steps_override=8160` and
`schedule_override=40800`. Do not substitute a ten-epoch cosine horizon for
the fifty-epoch horizon. The deduplicated uniform control instead follows its
saved ten-pass training rule with its own training-set-size horizon.

`plantseg_training` preserves arm names P2 (residual MLP), P3 (uniform),
P5 (DINOv3 attentive), P7 (CLIP attentive), and D1 (PlantWild deduplicated
uniform) to avoid changing branching behavior. Its `Fit` default uses Torch
shuffle order; D1 uses `numpy_order=True`. PlantSeg MLP clips gradients at 1.0;
the PlantWild residual MLP does not. The new PlantSeg fits preserve the
candidate maximum-epoch schedule while stopping at the selected epoch.
The earlier PlantSeg final fits also used the candidate maximum-epoch horizon; consult both
`plantseg_initial_selected.json` and `plantseg_selected.json`.

Feature extraction preserves the shortest-edge 592 / center 518 view.
DINOv3 subsequently resizes to 512 and excludes four registers plus CLS;
CLIP uses its official 224 processor on the 518 view. Use float32 DINOv3
patch storage for PlantSeg and float16 for PlantWild. DINOv2 stores CLS and
patches as float16, then computes the patch mean from those stored patches.
Acquire licensed images and weights separately. Set deterministic numerical
settings with `training.configure()` before full reproduction; source runs
used the package versions documented in the main README.

Additional function-level protocols are documented in [SUPPLEMENTARY.md](SUPPLEMENTARY.md).
`out_of_fold` preserves the whole-group five-fold allocator and complementary
fold probe fitting. `overlap` exposes the directional correction overlap and
retained-record filter used by the released CPU checker; `errors.transitions`
exposes the C/W/X decomposition. Original full-model package versions are in
`configs/full_environments.json`, separate from the quick-check environment.

Remaining manual orchestration: PlantSeg three-fold candidate/epoch selection
and final fits; five-fold stacking selection; the probe-recipe multi-arm
experiments and attention refits on OOF logits; 224/336 extraction variants;
and image-hash recomputation if repeating the published overlap filtering.
Core and supplementary GPU numerical execution has not been validated in this
release preparation. All new units have static checks only.

## Implementation identifiers

| Identifier | Paper terminology |
|---|---|
| `FULL` | Attentive probe |
| `UNIFORM` | Uniform pooling |
| `NO_G1` | Attention branch without the class-token residual |
| PlantSeg `P2` | Residual MLP on the class token |
| PlantSeg `P3` | Uniform pooling |
| PlantSeg `P5` | DINOv3-L attentive probe |
| PlantSeg `P7` | CLIP-L attentive probe |

These identifiers connect the selected configuration records to implementation branches.
