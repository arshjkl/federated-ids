# Centralized Training

## Model

MLPIDS defined in `src/models/mlp.py`:
- Input → 128 → 64 → 32 hidden layers
- BatchNorm, dropout 0.3
- Single logit output
- BCEWithLogitsLoss

## Training

- Trained on the full preprocessed training set (2,017,889 rows × 70 features)
- Validated on the test set (504,473 rows)
- Checkpoint saved to `models/centralized_mlp.pth`
- Metrics saved to `models/centralized_mlp_results.json`

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.977063 (97.71%) |
| Precision | 0.961630 (96.16%) |
| Recall | 0.900066 (90.01%) |
| F1-score | 0.929830 (92.98%) |
| ROC-AUC | 0.998076 (99.81%) |

Confusion matrix: TN=409343, FP=9954, FN=10891, TP=74285 (on the federated test set; centralized validation uses its own test split).

## Validation script

`validate_centralized.py` (Phase 3) loads `models/centralized_mlp.pth`, runs a forward pass on `X_test.npy` (batched, 4096), recomputes metrics, compares against `models/centralized_mlp_results.json`, and writes `reports/centralized_validation.json`.

All metrics match exactly (abs diff < 1e-6). See `reports/centralized_validation.json`.

## Warm-start role

The centralized checkpoint is the warm-start for federated training: client models are initialized from centralized weights before FedAvg rounds begin.
