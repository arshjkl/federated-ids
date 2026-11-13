# Federated Learning (Manual FedAvg)

## Entry point

`src/federated/run_manual_federated.py` — manual FedAvg helpers + training loop.

## Configuration

| Parameter | Value |
|---|---|
| Rounds | 50 |
| Clients | 5 |
| Partitioning | non-IID Dirichlet, α=0.5 |
| Local epochs | 5 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Warm start | centralized checkpoint |
| Aggregation | sample-weighted FedAvg |

## How it works

1. Load and partition training data across 5 clients using Dirichlet α=0.5 non-IID partitioning.
2. Initialize client models from the centralized checkpoint (warm start).
3. For each round:
   - Each client trains locally for 5 epochs on its partition.
   - Server aggregates client updates with sample-weighted FedAvg (`average_parameters`).
   - Evaluate the global model on the test set.
4. Save final checkpoint to `models/federated_mlp_manual.pth` and results to `models/manual_federated_improved_results.json`.

## Results (round 50)

| Metric | Value |
|---|---|
| Accuracy | 0.958680 (95.87%) |
| Precision | 0.881836 (88.18%) |
| Recall | 0.872135 (87.21%) |
| F1-score | 0.876959 (87.70%) |
| FPR | 0.023740 |
| FNR | 0.127865 |

Confusion matrix: TP=74285, TN=409343, FP=9954, FN=10891.

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
