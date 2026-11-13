# System Architecture

## High-level design

The Federated Learning IDS pipeline consists of four layers:

1. **Data layer** — CIC-IDS2017 CSV files are cleaned, split, scaled, and partitioned into client datasets with a manifest recorded in `reports/experiment_manifest.json`.
2. **Model layer** — MLPIDS (128→64→32 hidden, BatchNorm, dropout 0.3, single logit, BCEWithLogitsLoss).
3. **Training layer** — centralized training, manual FedAvg, and DP-FedAvg.
4. **Serving layer** — FastAPI backend + React dashboard for real-time inference.

```
data/                 CIC-IDS2017 raw CSVs + processed npy arrays
    └── preprocessing_validation.py   integrity validation + manifest

src/models/mlp.py                 MLPIDS model + predict / predict_proba
src/privacy/dp_sgd.py             DPSGDTrainer (Opacus + manual fallback), privacy accounting
src/federated/run_manual_federated.py   manual FedAvg helpers + training loop
src/federated/run_dp_federated.py   DP-FedAvg end-to-end training

dashboard/backend/main.py         FastAPI inference API (MLP-only fallback)
dashboard/frontend/src/           React frontend

tests/                  50 unit tests (unittest)
models/                 checkpoints + results JSON
reports/                validation, audit, status, experiment reports
```

## Data flow

1. `preprocessing_validation.py` reads the 8 CIC-IDS2017 CSVs, maps labels (BENIGN→0, ATTACK→1), stratifies into train/test (80/20, random_state=42), fits `StandardScaler` on train only, and writes `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` plus `reports/experiment_manifest.json`.
2. Centralized training loads the processed arrays, trains MLPIDS, and saves `models/centralized_mlp.pth` + `models/centralized_mlp_results.json`.
3. Federated training partitions the training data across clients (Dirichlet α=0.5 non-IID), runs FedAvg rounds (sample-weighted aggregation), and saves the federated checkpoint + results.
4. DP-FedAvg runs the same loop with per-client DP-SGD (Opacus), tracks epsilon/delta per client and per round, and saves `models/federated_mlp_dp.pth` + `models/dp_federated_results.json`.
5. The dashboard loads the centralized MLP checkpoint and serves `/api/analyze` inference; if Laya (`src/ids_laya/`) is unavailable it falls back to MLP-only predictions.

## Component responsibilities

| Component | Responsibility |
|---|---|
| `src/data/preprocessing_validation.py` | dataset integrity validation + manifest generation |
| `src/models/mlp.py` | MLPIDS model definition, forward, predict, predict_proba |
| `src/privacy/dp_sgd.py` | DPSGDTrainer (Opacus + manual fallback), privacy accounting |
| `src/federated/run_manual_federated.py` | manual FedAvg: partition, train, aggregate, evaluate |
| `src/federated/run_dp_federated.py` | DP-FedAvg end-to-end training + privacy report |
| `dashboard/backend/main.py` | FastAPI inference API with MLP-only fallback |
| `dashboard/frontend/src/` | React UI (Header, SystemStatus, etc.) |
| `tests/` | 50 unit tests across 5 files |

## Preservation rules

- `src/ids_laya/` is **read-only / off-limits** — no source changes permitted.
- Dashboard MLP-only fallback must not modify Laya source.
- Network is blocked for PyPI — dependencies must already be present in `.venv` (torch 2.14.0, opacus 1.6.0).
