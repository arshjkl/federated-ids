# Preprocessing Pipeline

## Entry point

`src/data/preprocessing_validation.py` — dataset integrity validation + manifest generation.

## Steps

1. **Discover CSVs** — locate the 8 CIC-IDS2017 CSV files under `data/raw/MachineLearningCVE/`.
2. **Read & validate** — confirm each file is readable, contains the label column, and uses consistent binary mapping (BENIGN→0, ATTACK→1).
3. **Class balance check** — confirm both classes are present.
4. **Stratified split** — `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
5. **Feature scaling** — `StandardScaler` fit on training data only; applied to train and test.
6. **Persist arrays** — `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` in `data/processed/`.
7. **Write manifest** — `reports/experiment_manifest.json` records input_dim, row counts, split parameters, and file checksums.

## Integrity checks

- Processed arrays present and shape-consistent (train/test × 70 features)
- No train/test overlap
- Arrays contain no NaN or infinite values
- Scaler was fit on training data only (verified by re-fitting on train and comparing)

## Tests

11 preprocessing tests in `tests/test_preprocessing.py`, all passing:
`test_discover_csv_files`, `test_count_rows_positive`, `test_files_readable`,
`test_label_column_found`, `test_binary_mapping_consistent`, `test_classes_present`,
`test_processed_arrays_present`, `test_train_test_shapes_consistent`,
`test_stratified_split`, `test_no_overlap`, `test_processed_arrays_clean`,
`test_scaler_fit_on_train_only`.
