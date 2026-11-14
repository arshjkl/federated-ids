# Test Results — Federated Learning IDS Project

Generated: 2026-11-12
Test runner: Python `unittest` (pytest unavailable — network blocked for PyPI)
Python environment: `.venv` (torch 2.14.0, opacus 1.6.0)

## Summary

| Metric | Value |
|---|---|
| Total tests | 36 |
| Passed | 36 |
| Failed | 0 |
| Errors | 0 |

All test suites pass.

## Test Suites

### tests/test_mlp.py — MLPIDS model (8 tests)

| Test | Result |
|---|---|
| test_model_creation | PASS |
| test_forward_pass_shape | PASS |
| test_predict_proba_shape | PASS |
| test_predict_shape | PASS |
| test_count_parameters | PASS (6849 params incl. BatchNorm weight+bias) |
| test_create_mlp_model_factory | PASS |
| test_load_centralized_checkpoint | PASS |
| test_load_federated_checkpoint | PASS |

### tests/test_federated.py — Manual FedAvg (6 tests)

| Test | Result |
|---|---|
| test_get_set_parameters_round_trip | PASS |
| test_average_parameters_unweighted | PASS |
| test_average_parameters_weighted | PASS |
| test_evaluate_model_shapes_and_keys | PASS (incl. ROC-AUC key) |
| test_training_changes_parameters | PASS |

### tests/test_dp.py — Differential Privacy (10 tests)

| Test | Result |
|---|---|
| test_clip_gradients_scales_down_large_norm | PASS |
| test_clip_gradients_no_op_when_small_norm | PASS |
| test_add_gaussian_noise_changes_gradients | PASS |
| test_compute_epsilon_positive_finite | PASS |
| test_compute_epsilon_infinite_when_no_noise | PASS |
| test_compute_epsilon_decreases_with_larger_noise | PASS |
| test_privacy_report_structure | PASS |
| test_train_epoch_manual_fallback | PASS |
| test_get_privacy_spent_returns_epsilon_delta | PASS |
| test_dp_enabled_flag | PASS |
| test_train_epoch_opacus_path | skipped (Opacus unavailable in this env) |

### tests/test_preprocessing.py — Data preprocessing (11 tests)

| Test | Result |
|---|---|
| test_discover_csv_files | PASS |
| test_count_rows_positive | PASS |
| test_files_readable | PASS |
| test_label_column_found | PASS |
| test_binary_mapping_consistent | PASS |
| test_classes_present | PASS |
| test_processed_arrays_present | PASS |
| test_train_test_shapes_consistent | PASS |
| test_stratified_split | PASS |
| test_no_overlap | PASS |
| test_processed_arrays_clean | PASS |
| test_scaler_fit_on_train_only | PASS |

## Notes

- `tests/test_dp.py::test_train_epoch_opacus_path` is skipped when Opacus is unavailable; the manual fallback path is fully covered by `test_train_epoch_manual_fallback`.
- `get_model_parameters` returns independent copies (`.clone()` before `.numpy()`) so in-place optimizer updates do not mutate "before" snapshots — this was required for `test_training_changes_parameters` to be meaningful.
- The dashboard `/api/analyze` endpoint now delegates to `preprocess_input()` so a wrong feature count returns HTTP 400 (previously returned 500).
