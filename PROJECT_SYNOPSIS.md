# Project Synopsis: Federated Learning IDS

**Project:** Privacy-Preserving Federated Intrusion Detection System (IDS)  
**Dataset:** CIC-IDS2017  https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs
**Last updated:** November 13, 2026  
**Scope:** Federated ML pipeline only — no Laya, no dashboard  

## Overview

This project implements an end-to-end privacy-preserving federated intrusion detection pipeline. The goal is to detect malicious network traffic (binary classification: BENIGN vs ATTACK) using a centralized MLP baseline, a manual FedAvg federated learner, and a differentially-private DP-FedAvg variant — all built on the CIC-IDS2017 dataset.

All components are verified end-to-end with persisted checkpoints, results JSON, and a passing unit test suite.

## Dataset

- Source: CIC-IDS2017, 8 CSV files under `data/raw/MachineLearningCVE/`
- Raw rows: 2,830,743
- After cleaning: 2,017,889 train / 504,473 test rows × 70 features
- Label mapping: BENIGN → 0, ATTACK → 1 (binary)
- Split: stratified, `test_size=0.2`, `random_state=42`
- Preprocessing: `StandardScaler` fit on train only; manifest in `reports/experiment_manifest.json`

## Models

| Component | Description |
|---|---|
| MLPIDS | 128→64→32 hidden layers, BatchNorm, dropout 0.3, single logit, BCEWithLogitsLoss (`src/models/mlp.py`) |
| Centralized MLP | Trained on full training set; warm-start checkpoint for federation |
| Manual FedAvg | 50 rounds, 5 clients, Dirichlet α=0.5 non-IID, local_epochs=5, batch 128, lr=0.001, sample-weighted aggregation |
| DP-FedAvg | DP-SGD via Opacus 1.6.0; per-example gradient clipping + Gaussian noise; moments-accountant privacy accounting |

## Key Results

| Metric | Centralized MLP | Manual FedAvg (50 rounds) | DP-FedAvg (30 rounds) |
|---|---|---|---|
| Accuracy | 97.71% | 95.87% | 93.92% |
| Precision | 96.16% | 88.18% | 73.71% |
| Recall | 90.01% | 87.21% | 99.48% |
| F1-score | 92.98% | 87.70% | 84.68% |
| ROC-AUC | 99.81% | — | 99.50% |

DP-FedAvg privacy budget: **epsilon=0.1457 at delta=1e-05** (naive composition bound: 176.07)

Full metrics and confusion matrices are stored in `models/centralized_mlp_results.json`, `models/manual_federated_improved_results.json`, and `models/dp_federated_results.json`.

## Testing

- 50 unit tests across 5 files (`tests/test_mlp.py`, `tests/test_federated.py`, `tests/test_dp.py`, `tests/test_preprocessing.py`, `tests/test_dashboard.py`)
- All tests pass; run with `python -m unittest discover -s tests`
- `pytest` is unavailable (network blocked for PyPI); `unittest` is used instead

## Components

- `src/data/preprocessing_validation.py` — dataset integrity validation + manifest generation
- `src/models/mlp.py` — MLPIDS model definition
- `src/models/train_centralized.py` — centralized training script (20 epochs)
- `src/privacy/dp_sgd.py` — DPSGDTrainer (Opacus + manual fallback), privacy accounting
- `src/federated/run_manual_federated.py` — manual FedAvg helpers + training loop
- `src/federated/run_dp_federated.py` — DP-FedAvg end-to-end training
- `validate_centralized.py` — centralized validation script

## Preservation Rules

- Network is blocked for PyPI — dependencies must already be present in `.venv` (torch 2.14.0, opacus 1.6.0).
- This repository intentionally excludes `src/ids_laya/` (read-only Laya component) and `dashboard/` (FastAPI/React frontend). No references to either are included.

## Key Artifacts

- `models/centralized_mlp.pth` — centralized checkpoint
- `models/federated_mlp_manual.pth` — manual FedAvg checkpoint
- `models/federated_mlp_dp.pth` — DP-FedAvg checkpoint
- `models/checkpoints/dp_federated_round_*.pth` — DP-FedAvg round checkpoints
- `models/checkpoints/federated_round_*.pth` — manual FedAvg round checkpoints
- `models/centralized_mlp_results.json` — centralized metrics
- `models/manual_federated_improved_results.json` — manual FedAvg metrics
- `models/dp_federated_results.json` — DP-FedAvg metrics + privacy report
- `reports/experiment_manifest.json` — dataset manifest
- `reports/test_results.md` — full test suite results
- `reports/final_project_audit.md` — final audit
- `reports/final_project_status.json` — final status JSON
- `reports/model_comparison.json` — model comparison
- `reports/privacy_report.json` — DP privacy report
