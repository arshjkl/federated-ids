# Centralized Training

## Models

### MLPIDS (Baseline)

Defined in `src/models/mlp.py`:
- Input → 128 → 64 → 32 hidden layers
- BatchNorm, dropout 0.3
- Single logit output
- BCEWithLogitsLoss

### ImprovedMLPIDS (Improved)

Defined in `src/models/improved_mlp.py`:
- Input → 256 → 128 → 64 → 32 hidden layers
- Residual connections (every 2 layers), LayerNorm, dropout 0.2
- Single logit output
- FocalLoss (alpha=0.25, gamma=2.0) or WeightedBCELoss
- Cosine warmup LR schedule (5 warmup epochs)
- Early stopping on val_f1 (patience=10)
- Gradient clipping (1.0)
- Optimal threshold selection on validation set

## Training (Baseline)

- Trained on the full preprocessed training set (2,017,889 rows × 70 features)
- Validated on the test set (504,473 rows)
- 20 epochs, batch_size=256, Adam lr=0.001
- Checkpoint saved to `models/centralized_mlp.pth`
- Metrics saved to `models/centralized_mlp_results.json`

## Training (Improved)

- Trained on the full preprocessed training set (2,017,889 rows × 70 features)
- Validated on the test set (504,473 rows)
- 50 epochs, batch_size=512, AdamW lr=0.001, weight_decay=1e-4
- Checkpoint saved to `models/centralized_mlp_improved.pth`
- Metrics saved within checkpoint (`results` key)

## Results (Baseline)

| Metric | Value |
|---|---|
| Accuracy | 0.977063 (97.71%) |
| Precision | 0.961630 (96.16%) |
| Recall | 0.900066 (90.01%) |
| F1-score | 0.929830 (92.98%) |
| ROC-AUC | 0.998076 (99.81%) |
| FPR | 0.007296 |
| FNR | 0.099934 |

Confusion matrix: TN=416238, FP=3059, FN=8512, TP=76664.

## Results (Improved)

| Metric | Value |
|---|---|
| Accuracy | 0.997187 (99.72%) |
| Precision | 0.995398 (99.54%) |
| Recall | 0.987907 (98.79%) |
| F1-score | 0.991639 (99.16%) |
| ROC-AUC | 0.999906 (99.99%) |
| FPR | 0.000928 (0.09%) |
| FNR | 0.012093 (1.21%) |
| Optimal Threshold | 0.58 |
| Training Time | 589.7s |

Confusion matrix: TN=418908, FP=389, FN=1030, TP=84146.

## Validation script

`validate_centralized.py` loads `models/centralized_mlp.pth`, runs a forward pass on `X_test.npy` (batched, 4096), recomputes metrics, compares against `models/centralized_mlp_results.json`, and writes `reports/centralized_validation.json`.

All metrics match exactly (abs diff < 1e-6). See `reports/centralized_validation.json`.

## Warm-start role

The centralized checkpoint is the warm-start for federated training: client models are initialized from centralized weights before FedAvg rounds begin. The improved centralized checkpoint (`centralized_mlp_improved.pth`) is used as warm-start for Improved FedAvg and Federated v2.
