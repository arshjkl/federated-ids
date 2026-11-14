# Reproducibility

## Environment

- Python 3 in `.venv`
- torch 2.14.0, opacus 1.6.0
- numpy, scikit-learn
- fastapi, uvicorn, pydantic (dashboard backend)
- Network blocked for PyPI — dependencies must already be present in `.venv`
- `pytest` unavailable; `unittest` is used instead

## Random seeds

- Stratified split: `random_state=42`
- Federated partitioning: `random_state=42`
- All training uses the configured seeds above for determinism within a single run

## Reproducing results

### 1. Preprocess the dataset

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/data/preprocessing_validation.py
```

Writes `data/processed/*.npy` and `reports/experiment_manifest.json`.

### 2. Validate the centralized model

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python validate_centralized.py
```

Writes `reports/centralized_validation.json`. All metrics match `models/centralized_mlp_results.json` exactly.

### 3. Run manual FedAvg

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_manual_federated.py
```

Writes `models/federated_mlp_manual.pth` and `models/manual_federated_improved_results.json`.

### 4. Run DP-FedAvg

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_dp_federated.py
```

Writes `models/federated_mlp_dp.pth` and `models/dp_federated_results.json`.

### 5. Run the test suite

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -m unittest discover -s tests
```

All 50 tests pass (see `reports/test_results.md`).

## Preservation rules

## Key artifacts

| Artifact | Description |
|---|---|
| `data/processed/X_train.npy` / `X_test.npy` | scaled feature arrays |
| `data/processed/y_train.npy` / `y_test.npy` | label arrays |
| `reports/experiment_manifest.json` | dataset manifest |
| `models/centralized_mlp.pth` | centralized checkpoint |
| `models/centralized_mlp_results.json` | centralized metrics |
| `reports/centralized_validation.json` | centralized validation report |
| `models/federated_mlp_manual.pth` | manual FedAvg checkpoint |
| `models/manual_federated_improved_results.json` | manual FedAvg metrics |
| `models/federated_mlp_dp.pth` | DP-FedAvg checkpoint + privacy report |
| `models/dp_federated_results.json` | DP-FedAvg metrics + privacy report |
| `reports/test_results.md` | full test suite results |
| `reports/final_project_audit.md` | final audit |
| `reports/final_project_status.json` | final status JSON |
| `reports/model_comparison.json` | model comparison |
| `reports/privacy_report.json` | DP privacy report |
