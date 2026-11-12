# Dataset identities and splits

All CSV image identities are relative to the original dataset image root. SHA-256 values identify original image bytes. CSV row order is preserved from the saved source lists; no duplicate detection was rerun.

- `plantwild/splits.csv`: 18,542 images. `split` records the official train/validation/test role; `keep=1` selects the deduplicated setting. Counts are 13,045/1,820/3,677 before removal and 12,833/1,787/3,677 after removal.
- `plantdoc/splits.csv`: 2,578 images, with train/validation/test counts 2,107/235/236 before removal and 2,094/233/236 after removal. `version.json` identifies the original dataset commit. The 28 raw directory classes are retained in the class mapping.
- `duplicate_groups.csv`: only members of saved groups containing more than one image. Group identifiers are representative relative image IDs. Singleton groups remain in `splits.csv`.
- `plantwild/subset_{25,50,100}_per_class.csv`: nested subsets of the official training partition, shared across model seeds. Each class contributes min(requested count, available training images), giving 2,225/4,417/8,110 images. These subsets use the official training partition, including images excluded from the separate deduplicated setting.
- `plantseg/train.csv` and `test.csv`: 4,611 cleaned training and 1,436 cleaned test images.
- `plantseg/internal_split.csv`: the 3,385 training and 534 validation images used for internal selection. `cv3.csv` assigns the 3,385 internal training images to validation folds 1–3.
- `plantseg/known_duplicate_groups.csv`: saved multi-image groups among the internal and test lists.
- `plantseg/historical_train_groups.csv`: saved perceptual-hash group membership for all 4,611 training images. Its 16-character group namespace differs from the later 20-character group namespace in internal/CV/test lists; do not combine these namespaces. Later assignments for the 692 training images outside the internal partition are unavailable.

The image files themselves are not distributed. Class labels, hosts, health status, and symptom/pathogen groups are in `../metadata/`.
