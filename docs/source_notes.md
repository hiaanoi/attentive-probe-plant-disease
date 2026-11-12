# What the quick check recomputes

## Prediction-based statistics

Saved per-image predictions provide accuracy, macro-F1, seed summaries,
paired test-image intervals, host/health-state contributions and correction
overlap. Validation predictions reproduce the C-scan selection summaries.
Training-cap sizes come from the supplied image-ID lists. GFLOPs coordinates
are checked against the saved profiling records.

Image bootstrap intervals describe test-image sampling uncertainty conditional
on the fitted seeds. The recipe analyses use matched training subsamples as
their resampling unit.

## Saved numerical measurements

Training accuracy, training loss, profiling/timing records, CKA and attention
measurements are read from saved records using the relevant section, recipe,
column or configuration. Tables S1–S15 preserve the complete numerical records.
The table exporter checks the cells against their saved sources.

## Figures

Figure 1 is supplied as PDF/SVG. Its optional Python diagram script uses
relative input paths. The attention-overlay script applies the fixed formula
to the 518-pixel input crop and five-seed mean softmax attention array.

Figures 2–5 read their coordinates from CSVs. Plotting runs in separate
processes so each design keeps its own Matplotlib settings. Figures 3 and 4
prefer TeX Gyre Termes, then Times New Roman, then DejaVu Serif. The supplied
Figure 3/4 PDF/SVG files are the versions used in the article.

Figure 4 retains the sum of the author's rounded class-token accuracy and
rounded gain for the attentive coordinate. The numerical check accounts for
this display rounding when comparing with unrounded prediction-derived accuracy.
