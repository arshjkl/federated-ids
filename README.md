# Federated Learning IDS

Privacy-Preserving Federated Intrusion Detection System (IDS) built on the CIC-IDS2017 dataset.

## Overview

This project implements an end-to-end privacy-preserving federated intrusion detection pipeline. It detects malicious network traffic (binary classification: BENIGN vs ATTACK) using:

- **Centralized MLP baseline** — single-node training on the full training set (97.71% accuracy)
- **Manual FedAvg** — 50 rounds, 5 clients, Dirichlet α=0.5 non-IID partitioning, warm-started from the centralized checkpoint (95.87% accuracy)
- **DP-FedAvg** — differentially-private federated averaging with DP-SGD via Opacus 1.6.0 (per-example gradient clipping + Gaussian noise, moments-accountant privacy accounting)
- **FastAPI + React dashboard** — real-time inference with an MLP-only fallback when the optional Laya component is unavailable

`src/ids_laya/` is read-only / off-limits; the dashboard degrades gracefully to MLP-only inference when Laya is unavailable.

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Centralized MLP | 97.71% | 96.16% | 90.01% | 92.98% | 99.81% |
| Manual FedAvg (50 rounds) | 95.87% | 88.18% | 87.21% | 87.70% | — |

Full metrics and confusion matrices: `models/centralized_mlp_results.json`, `models/manual_federated_improved_results.json`.

## Repository structure

```
federated-ids/
├── PROJECT_SYNOPSIS.md      # Project overview and key results
├── README.md                # This file
├── docs/                    # Detailed documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── PREPROCESSING.md
│   ├── CENTRALIZED_TRAINING.md
│   ├── FEDERATED_LEARNING.md
│   ├── DIFFERENTIAL_PRIVACY.md
│   ├── EVALUATION.md
│   └── REPRODUCIBILITY.md
├── data/                    # Dataset (raw + processed)
├── models/                  # Trained checkpoints and results JSON
├── reports/                 # Validation, audit, status, and experiment reports
├── src/
│   ├── data/preprocessing_validation.py
│   ├── models/mlp.py                 # MLPIDS model definition
│   ├── federated/
│   │   ├── run_manual_federated.py   # Manual FedAvg
│   │   └── run_dp_federated.py       # DP-FedAvg
│   ├── privacy/dp_sgd.py             # DP-SGD trainer + privacy accounting
│   └── ids_laya/                     # read-only / off-limits
├── dashboard/
│   ├── backend/main.py        # FastAPI inference API (MLP-only fallback)
│   └── frontend/src/          # React frontend
├── tests/                   # 50 unit tests (unittest), all passing
├── validate_centralized.py  # Phase 3 centralized validation script
└── .venv/                   # Python env (torch 2.14.0, opacus 1.6.0)
```

## Requirements

- Python 3 (`.venv` recommended)
- torch 2.14.0, opacus 1.6.0
- numpy, scikit-learn
- fastapi, uvicorn, pydantic (dashboard backend)
- flwr (federated learning utilities)
- react frontend (dashboard frontend)

Network is blocked for PyPI — dependencies must already be present in `.venv`.

## Run the pipeline

Centralized training validation:
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python validate_centralized.py
```

Manual FedAvg (50 rounds):
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_manual_federated.py
```

DP-FedAvg (30 rounds, noise_multiplier=1.0):
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_dp_federated.py
```

## Tests

Run the full suite with the project's unittest runner (pytest unavailable — network blocked for PyPI):

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -m unittest discover -s tests
```

All 50 tests pass (documented in `reports/test_results.md`).

## Preservation rules

- `src/ids_laya/` is **read-only / off-limits** — no source changes permitted.
- Dashboard MLP-only fallback must not modify Laya source.

## Documentation

See `docs/` for per-component details:
- `ARCHITECTURE.md` — system architecture
- `DATASET.md` — CIC-IDS2017 dataset
- `PREPROCESSING.md` — preprocessing pipeline
- `CENTRALIZED_TRAINING.md` — centralized MLP training
- `FEDERATED_LEARNING.md` — manual FedAvg
- `DIFFERENTIAL_PRIVACY.md` — DP-SGD / DP-FedAvg
- `EVALUATION.md` — evaluation methodology and results
- `REPRODUCIBILITY.md` — how to reproduce results
