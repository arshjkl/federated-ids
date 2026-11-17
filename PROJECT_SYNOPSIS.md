# Project Synopsis: Federated Learning IDS

**Project:** Privacy-Preserving Federated Intrusion Detection System (IDS)  
**Dataset:** CIC-IDS2017  
**Last updated:** November 17, 2026  
**Scope:** Federated ML pipeline only  

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
| ImprovedMLPIDS | Configurable hidden dims (default [256,128,64]), residual blocks, BatchNorm toggle, loss_type (bce/weighted_bce/focal) (`src/models/mlp.py`) |
| Centralized MLP | Trained on full training set; warm-start checkpoint for federation |
| Centralized MLP (improved) | ImprovedMLPIDS + weighted BCE/focal loss + ReduceLROnPlateau + early stopping + threshold optimization |
| Manual FedAvg | 50 rounds, 5 clients, Dirichlet α=0.5 non-IID, local_epochs=5, batch 128, lr=0.001, sample-weighted aggregation |
| Improved FedAvg | FedAvg + ImprovedMLPIDS + weighted BCE + per-client LR scheduling + threshold optimization |
| FedProx | FedAvg with proximal term mu * \|\|w - w_global\|\|^2 to reduce client drift |
| DP-FedAvg | DP-SGD via Opacus 1.6.0; per-example gradient clipping + Gaussian noise; moments-accountant privacy accounting |
| Improved DP-FedAvg | DP-FedAvg + correct batch-level privacy accounting + ImprovedMLPIDS + weighted BCE + threshold optimization |
| DP-FedProx | DP-FedAvg with proximal term (mu) + DP-SGD |

## Key Results

### Baselines

| Metric | Centralized MLP | Manual FedAvg (50 rounds) | DP-FedAvg (30 rounds) |
|---|---|---|---|
| Accuracy | 97.71% | 95.87% | 93.92% |
| Precision | 96.16% | 88.18% | 73.71% |
| Recall | 90.01% | 87.21% | 99.48% |
| F1-score | 92.98% | 87.70% | 84.68% |
| ROC-AUC | 99.81% | — | 99.50% |
| FPR | 0.73% | 2.37% | 7.21% |
| FNR | 9.99% | 12.79% | 0.52% |

### Confusion Matrices (Test Set: 504,473 samples)

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---|---|---|---|
| Centralized MLP | 416,238 | 3,059 | 8,512 | 76,664 |
| Manual FedAvg | 409,343 | 9,954 | 10,891 | 74,285 |
| DP-FedAvg | 389,079 | 30,218 | 445 | 84,731 |

### Improved Variants (ImprovedMLPIDS: 256→128→64→32, residual blocks, LayerNorm, FocalLoss, LR scheduling, threshold optimization)

| Metric | Centralized Improved | Federated v2 (round 20) |
|---|---|---|
| Accuracy | 99.72% | 99.44% |
| Precision | 99.54% | 97.43% |
| Recall | 98.79% | 99.30% |
| F1-score | 99.16% | 98.36% |
| ROC-AUC | 99.99% | 99.98% |
| FPR | 0.09% | 0.53% |
| FNR | 1.21% | 0.70% |
| Optimal Threshold | 0.58 | 0.50 |

DP-FedAvg privacy budget: **epsilon=0.1457 at delta=1e-05** (naive composition bound: 176.07)

Improved variant results are written to `models/model_comparison_improved.json` by `src/federated/run_comparison.py`. Baseline results remain in `models/centralized_mlp_results.json`, `models/manual_federated_improved_results.json`, and `models/dp_federated_results.json`. Improved checkpoints: `models/centralized_mlp_improved.pth`, `models/checkpoints/federated_v2_round_20.pth`.

## Testing

- 66 unit tests across 5 files (`tests/test_mlp.py`, `tests/test_federated.py`, `tests/test_dp.py`, `tests/test_improved.py`, `tests/test_preprocessing.py`)
- All tests pass; run with `python -m unittest discover -s tests`
- `pytest` is unavailable (network blocked for PyPI); `unittest` is used instead

## Components

- `src/data/preprocessing_validation.py` — dataset integrity validation + manifest generation
- `src/models/mlp.py` — MLPIDS + ImprovedMLPIDS model definitions, FocalLoss
- `src/models/train_centralized.py` — centralized training (20 epochs baseline + improved with weighted/focal loss, LR scheduling, threshold optimization)
- `src/privacy/dp_sgd.py` — DPSGDTrainer (Opacus + manual fallback), privacy accounting, batches_per_epoch helper
- `src/federated/run_manual_federated.py` — manual FedAvg + improved + FedProx
- `src/federated/run_dp_federated.py` — DP-FedAvg + improved + DP-FedProx
- `src/federated/run_comparison.py` — all-variants comparison runner
- `validate_centralized.py` — centralized validation script

## Preservation Rules

- Network is blocked for PyPI — dependencies must already be present in `.venv` (torch 2.14.0, opacus 1.6.0).

## Key Artifacts

- `models/centralized_mlp.pth` — centralized checkpoint
- `models/federated_mlp_manual.pth` — manual FedAvg checkpoint
- `models/federated_mlp_dp.pth` — DP-FedAvg checkpoint
- `models/federated_mlp_improved.pth` — improved FedAvg checkpoint
- `models/federated_mlp_dp_improved.pth` — improved DP-FedAvg checkpoint
- `models/federated_mlp_dp_prox.pth` — DP-FedProx checkpoint
- `models/checkpoints/dp_federated_round_*.pth` — DP-FedAvg round checkpoints
- `models/checkpoints/federated_round_*.pth` — manual FedAvg round checkpoints
- `models/centralized_mlp_results.json` — centralized metrics
- `models/manual_federated_improved_results.json` — manual FedAvg metrics
- `models/dp_federated_results.json` — DP-FedAvg metrics + privacy report
- `models/model_comparison_improved.json` — all-variants comparison (from run_comparison.py)
- `reports/experiment_manifest.json` — dataset manifest
- `reports/test_results.md` — full test suite results
- `reports/final_project_audit.md` — final audit
- `reports/final_project_status.json` — final status JSON
- `reports/model_comparison.json` — model comparison
- `reports/privacy_report.json` — DP privacy report