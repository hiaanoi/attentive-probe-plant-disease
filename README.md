# In-the-Wild Plant Disease Recognition: Patch Tokens from One Frozen Backbone Recover Most of the Gain from Adding Backbones

The study compares reading patch tokens from an existing backbone with adding
backbones. On official PlantWild, the DINOv2-L attentive probe recovered 73% of
the accuracy gain from adding DINOv3-L and CLIP-L, with 0.72 instead of 779 extra
GFLOPs per image. Replacing its class-token probe inside the fixed three-probe
fusion raised accuracy from 80.96% to 81.93%. The analysis also examines the
errors corrected, training-data size and the choice of base-probe regularisation.

## Quick check (CPU)

Recreate the tables and data figures from saved predictions and numerical records.
Use Python 3.11 or newer and install the dependencies in `requirements.txt`:

```sh
python -m venv .venv
# Activate .venv using your platform's command.
python -m pip install -r requirements.txt
python scripts/quick_check.py
```

The check aligns image IDs, verifies checksums and recomputes per-seed accuracy,
macro-F1, means/SDs, paired test-image intervals, host/error contributions and
correction overlap. Training and profiling statistics are read from saved
measurements. Tables S1–S15 are exported from the saved numerical records;
prediction-based metrics are recomputed by the quick check.

Paired image intervals use 10,000 resamples after averaging correctness across
seeds per image. Stream hashes and generation rules are in
`results/bootstrap/streams.json`. The 1263-image PlantSeg comparison uses its
saved membership and seed-0 stream. Outputs and check summaries appear in
`generated/`; implementation details are in `docs/source_notes.md`.

After the quick check, export a particular table or figure:

```sh
python scripts/make_tables.py --table 2b
python scripts/make_tables.py --table S15
python scripts/plot_figures.py --figure 3
```

## Data

Acquire images from the dataset authors. Results use PlantWild v1.

| Dataset | Classes | Train / validation / test | License |
|---|---:|---|---|
| [PlantWild](https://tqwei05.github.io/PlantWild/) | 89 | 13,045 / 1820 / 3677 | CC BY-NC-ND 4.0 |
| [PlantSeg](https://github.com/tqwei05/PlantSeg) | 115 | 4611 / internal CV / 1436 | CC BY-NC 4.0 |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | 28; 27 in test | 2342 raw train / internal validation / 236 | CC BY 4.0 |

PlantDoc uses commit `5467f6012d78d1c446145d5f582da6096f852ae8`.
Match image IDs and SHA-256 values to `data/splits/`.
`data/splits/README.md` describes deduplication, CV folds, nested per-class
training caps and overlap-excluded images. Host and symptom mappings are in
`data/metadata/`.

`results/predictions/inventory.csv` identifies each dataset, protocol, model,
seed and prediction checksum. Five-seed runs use 101, 202, 303, 404 and 505;
deterministic probes and training-subsample repeats retain their identifiers.
`cleaned` contains the earlier fixed-epoch uniform-pooling control; Table 3
and Figure 3 use `cleaned-uniform-tuned17`.
C-scan, out-of-fold and data-budget predictions have separate protocols;
validation-selection records are in `results/diagnostics/`.

## Manuscript-to-file map

Use `python scripts/make_tables.py --table <number-or-panel>` to select a table.
The script numbers main-text tables in an earlier order; the first column below
gives the manuscript numbering.
Complete supplementary cells are in `results/supplementary/`, with block
definitions in `results/summaries/supplementary_tables.json`.

| Manuscript item | Entry point | Inputs |
|---|---|---|
| Table 1(a,b) | `make_tables.py --table 5` | dataset/backbone descriptions; split identities |
| Table 2(a–c) | `make_tables.py --table 1` | predictions; main references; profiling |
| Table 3(a,b) | `make_tables.py --table 2` | fusion/control predictions; overlap-excluded IDs |
| Table 4(a–c) | `make_tables.py --table 3` | predictions; host and health-state mappings |
| Table 5(a–c) | `make_tables.py --table 4` | subsample/OOF predictions; TRAIN statistics; configurations |
| Figure 1 | `figures/Figure_1.pdf` / `.svg`; scripts below | diagram; PlantDoc photograph; mean attention |
| Figure 2 | `plot_figures.py --figure 2` | `results/figures/Figure_2_source.csv` |
| Figure 3 | `plot_figures.py --figure 3` | `results/figures/Figure_3_source.csv` |
| Figure 4 | `plot_figures.py --figure 4` | `results/figures/Figure_4_source.csv` |
| Figure 5 | `plot_figures.py --figure 5` | `results/figures/Figure_5_source.csv` |
| S1 | `make_tables.py --table S1` | classifier/fusion results |
| S2 | `make_tables.py --table S2` | configurations, resolutions, benchmarks, seeds |
| S3 | `make_tables.py --table S3` | fusion-rule comparisons |
| S4 | `make_tables.py --table S4` | host, overlap and health-state results |
| S5 | `make_tables.py --table S5` | symptom/pathogen groups |
| S6 | `make_tables.py --table S6` | PlantSeg selection and lesion attention |
| S7 | `make_tables.py --table S7` | paired training-subsample recipes |
| S8 | `make_tables.py --table S8` | regularisation, OOF and TRAIN diagnostics |
| S9 | `make_tables.py --table S9` | linear CKA |
| S10 | `make_tables.py --table S10` | region, mask/recipe and margin analyses |
| S11 | `make_tables.py --table S11` | overlap-excluded PlantSeg comparison |
| S12 | `make_tables.py --table S12` | data cleaning and flow counts |
| S13 | `make_tables.py --table S13` | feature-fusion search |
| S14 | `make_tables.py --table S14` | host mappings and merged-host sensitivity |
| S15 | `make_tables.py --table S15` | uniform-pooling tuning and five-seed results |

## Figure 1

Photograph from PlantDoc (CC BY 4.0), centre-cropped as model input.
`docs/figure_1_attribution.md` identifies `Shoemaker_7068.JPG.jpg` and the
transformations. `results/attention/candidate_02_mean5.npy` contains the 37×37
attention output averaged over five seeds; identity and checksum are in
`results/attention/README.json`.

To recreate the overlay, acquire the photograph and prepare `input518.png`:
decode RGB preserving the stored orientation, resize the short side to 592
pixels with bicubic interpolation and integer truncation, then take the
centred 518×518 crop using floor offsets. Save it under `inputs/figure1/`.

```sh
python scripts/make_fig1_attention.py --image inputs/figure1/input518.png --attn results/attention/candidate_02_mean5.npy --out inputs/figure1/attn_spot.png
python -m pip install -r requirements-figure1.txt
python scripts/make_fig1.py --assets inputs/figure1 --out generated/Figure_1.pptx
```

## Full reproduction

Run feature extraction, training-only scaling/probe fitting, neural-head
selection and fitting, probability fusion, then evaluation. `src/CLI.md`
documents the PlantWild commands; `src/README.md` covers the scientific
functions and PlantSeg CV/OOF orchestration.

The environment uses Python 3.11, PyTorch 2.5.1/CUDA 12.1, scikit-learn 1.7.2
and NumPy 2.4. DINOv2-L/CLIP-L extraction uses Transformers 4.57.6;
DINOv3-L uses Transformers 5.13.0 in a separate environment. Geometry,
precision and checkpoint checksums are in `configs/backbones.json`.
Acquire licensed images and model files separately, placing them in `inputs/`.

Recorded RTX 4090 extraction times were about 885/419/236 seconds for the
three backbones on 18,542 PlantWild images, with about 19 seconds for a cached
DINOv2-L attentive head and 6 seconds for a class-token probe.

## License and citation

Code is licensed under MIT (`LICENSE`); numeric data products under CC BY 4.0
(`DATA_LICENSE.md`). Dataset images and backbone weights retain their original
licenses. Citation information is in `CITATION.cff`.
