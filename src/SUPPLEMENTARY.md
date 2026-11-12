# Supplementary scientific interfaces

These functions preserve the saved numerical procedures, with data paths and
process management removed. Only static checks were run during packaging.

## Out-of-fold residual-logit intervention

`out_of_fold.allocate_folds(rows)` implements the original official PlantWild
TRAIN allocation. Supply all 13,045 training records, including duplicate
members. Each row requires `feature_row`, `class_id` and `group`. Map the
released split file's `group_id` column to `group`; attach `feature_row` from
the matching feature-index order, rather than sorting by image filename.
Rows must cover feature indices 0–13044 exactly. Do not use the deduplicated
12,833-row training set for this intervention.

The allocator sorts string group IDs, applies one `default_rng(0)` permutation,
then stably sorts groups by descending size. It chooses each fold using the
original exact rational class-proportion cost, then fewer assigned images,
then lower fold ID. It retains true class histograms for mixed-label groups.
The returned int8 array is in feature-row order.

`out_of_fold.fit_oof(x, y, folds, C)` fits five StandardScaler/logistic-probe
pairs on the other four folds and stitches held-out logits in original row
order. It requires original float32 features `[13045,1024]` and integer class
IDs `[13045]`. It returns the float32 `[13045,89]` logits and fold fit records.
`fit_probe` and `predict` expose the corresponding full-training probe.
All class IDs must be present in every training complement. There is no
calibration or cross-fold logit alignment.

Use `configs/residual_mechanism.json`: the full-probe C sweep is
0.001/0.003/0.01/0.03/0.1/0.3/1; the OOF intervention is at C=1 and C=0.01.
Reuse the original full C=0.01 probe when matching the saved experiment.
For each C and each attention seed, pass either the full-probe in-sample
TRAIN logits or OOF TRAIN logits to `training.train_mil`, with
`steps_override=8160`, `schedule_override=40800`, and the recorded head config.
The head uses OOF logits only during training; **every test residual receives
the same corresponding full-training probe logits**, not an ensemble of fold
probes. No validation selection occurs in this intervention. The caller still
orchestrates all seeds, trained-head persistence, and test prediction after
all fits finish. `out_of_fold.diagnostics` reproduces the descriptive TRAIN
accuracy, cross-entropy, mean maximum probability and uncentered logit norm.

## Probe-recipe subsamples

`probe_recipe.group_subsamples(rows)` takes the deduplicated retained TRAIN
records with `group`, `class_id`, `feature_row`. It returns the 30 original
80% group subsamples indexed `rep0` through `rep29`. Majority-label ownership
uses lower class ID to break ties; classes and group identifiers are sorted;
one generator seeded by repetition controls group order. Entire groups are
taken until the ceiling of 80% of each owner's image count is reached.

`fit_balanced_probe` provides the original C=1 balanced float32 CUDA L-BFGS
arm, and `objective` exposes its weighted mean cross-entropy plus L2 penalty
on weights only. Full multi-arm and attention-refit orchestration remains
manual; this helper must not be presented as covering every arm by itself.

## Error decomposition and correction overlap

`errors.transitions(y, before, after, host_mapping)` returns the C/W/X matrix
and net within-host and wrong-host components for one seed. Average per-seed
components to match the paper. C means correct, W means wrong class within
the correct host, and X means wrong host.

`overlap.correction_overlap(y, baseline, attentive, fusion)` accepts hard
class predictions: baseline/fusion `[N]`, attention `[seeds,N]`. It matches
the released CPU checker's directional overlap: attention corrections also
corrected by fusion, divided by attention corrections; all ordered distinct
attention-seed pairs for the between-seed control; fusion corrections divided
by all baseline errors for the independence expectation. This is not a
Jaccard coefficient. `byte_overlap_mask` applies the published retained-record
list in test order. It does not recompute image hashes or perceptual hashes.

## Environments and remaining orchestration

`configs/full_environments.json` records the original Python 3.11.16,
PyTorch 2.5.1+cu121, NumPy 2.4.6 and scikit-learn 1.7.2 environment, with
Transformers 4.57.6 for DINOv2/CLIP and 5.13.0 for DINOv3. The saved CPU
statistical run used SciPy 1.17.1. These full-model environments were not
installed or tested in repository preparation. The root requirements serve
the separate saved-prediction CPU quick check.

PlantSeg's `plantseg_training.Fit` provides the inherited fitting loop and
configs provide all candidates, chosen epochs and combination C values.
The caller must still orchestrate three-fold candidate/epoch evaluation,
tie-breaking, final fits, and held-out test evaluation. Five-fold stacking
selection and 224/336-pixel extraction variants also require manual source
integration. These limitations are separate from the CPU result checks.
