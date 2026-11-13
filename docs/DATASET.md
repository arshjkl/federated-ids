# CIC-IDS2017 Dataset

## Source

- CIC-IDS2017, 8 CSV files under `data/raw/MachineLearningCVE/`
- Raw rows: 2,830,743
- After cleaning: 2,017,889 train / 504,473 test rows × 70 features

## Label mapping

- BENIGN → 0
- ATTACK → 1
- Binary classification task

## Split

- Stratified split, `test_size=0.2`, `random_state=42`
- Training set: 2,017,889 rows
- Test set: 504,473 rows

## Preprocessing

- `StandardScaler` fit on training data only, applied to both train and test
- Processed arrays saved as `.npy` files in `data/processed/`
- Manifest recorded in `reports/experiment_manifest.json`

## Dataset integrity

Validated by `src/data/preprocessing_validation.py`, which checks:
- CSV discovery and readability
- Label column presence and consistent binary mapping
- Both classes present
- Processed arrays present and shape-consistent
- Stratified split integrity (no train/test overlap)
- Array cleanliness (no NaN/inf)
- Scaler fit on train only

See `tests/test_preprocessing.py` for the 11 preprocessing unit tests (all passing).
