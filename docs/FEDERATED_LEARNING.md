# Federated Learning (Manual FedAvg + Improved Variants)

## Entry Points

- `src/federated/run_manual_federated.py` — manual FedAvg (baseline + improved + FedProx)
- `src/federated/run_federated_v2.py` — Federated v2 (ImprovedMLPIDS, Flower-based)

## Configuration (Baseline Manual FedAvg)

| Parameter | Value |
|---|---|
| Rounds | 50 |
| Clients | 5 |
| Partitioning | non-IID Dirichlet, α=0.5 |
| Local epochs | 5 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Warm start | centralized checkpoint (MLPIDS) |
| Aggregation | sample-weighted FedAvg |

## Configuration (Federated v2 — Improved)

| Parameter | Value |
|---|---|
| Rounds | 30 |
| Clients | 5 |
| Partitioning | non-IID Dirichlet, α=0.5 |
| Local epochs | 5 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Warm start | centralized improved checkpoint (ImprovedMLPIDS) |
| Loss | FocalLoss (alpha=0.25, gamma=2.0) |
| Architecture | ImprovedMLPIDS (256→128→64→32, residual, LayerNorm, dropout 0.2) |

## How it works (Baseline)

1. Load and partition training data across 5 clients using Dirichlet α=0.5 non-IID partitioning.
2. Initialize client models from the centralized checkpoint (warm start).
3. For each round:
   - Each client trains locally for 5 epochs on its partition.
   - Server aggregates client updates with sample-weighted FedAvg (`average_parameters`).
   - Evaluate the global model on the test set.
4. Save final checkpoint to `models/federated_mlp_manual.pth` and results to `models/manual_federated_improved_results.json`.

## How it works (Federated v2 — Improved)

1. Load and partition training data across 5 clients using Dirichlet α=0.5 non-IID partitioning (val_ratio=0.1).
2. Initialize global model from `models/centralized_mlp_improved.pth` (warm start).
3. For each round:
   - Each client trains locally for 5 epochs on its partition using FocalLoss.
   - Server aggregates client updates with sample-weighted FedAvg.
   - Evaluate the global model on the test set.
4. Save round checkpoints to `models/checkpoints/federated_v2_round_*.pth`.

## Results (Baseline — round 50)

| Metric | Value |
|---|---|
| Accuracy | 0.958680 (95.87%) |
| Precision | 0.881836 (88.18%) |
| Recall | 0.872135 (87.21%) |
| F1-score | 0.876959 (87.70%) |
| FPR | 0.023740 |
| FNR | 0.127865 |

Confusion matrix: TP=74285, TN=409343, FP=9954, FN=10891.

## Results (Federated v2 — ImprovedMLPIDS)

| Round | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---|---|---|---|---|---|
| 0 (warm start) | 0.997138 | 0.994578 | 0.988436 | 0.991497 | 0.999906 | 0.001095 | 0.011564 |
| 10 | 0.993262 | 0.966173 | 0.994928 | 0.980340 | 0.999826 | 0.007076 | 0.005072 |
| 20 | 0.994392 | 0.974311 | 0.992968 | 0.983551 | 0.999843 | 0.005318 | 0.007032 |

Confusion matrix (round 20): TP=84577, TN=417067, FP=2230, FN=599.

Checkpoint: `models/checkpoints/federated_v2_round_20.pth`.

## Helpers

- `get_model_parameters` / `set_model_parameters` — extract/inject model weights (`.clone()` before `.numpy()` to avoid aliasing).
- `average_parameters` — sample-weighted average of client parameter lists.
- `evaluate_model` — compute loss, accuracy, precision, recall, f1, roc_auc on a dataset.
- `load_and_partition_data` — partition training data across clients.
- `load_centralized_model` — load the centralized checkpoint for warm start.

## Tests

6 tests in `tests/test_federated.py`, all passing:
`test_get_set_parameters_round_trip`, `test_average_parameters_unweighted`,
`test_average_parameters_weighted`, `test_evaluate_model_shapes_and_keys`,
`test_training_changes_parameters`.

31 tests in `tests/test_improved.py` cover ImprovedMLPIDS, FocalLoss, WeightedBCELoss, get_optimal_threshold, FedProx, and privacy accounting.
