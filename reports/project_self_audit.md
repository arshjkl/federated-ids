# Project Self-Audit: Federated IDS (federated-ids)

**Audit date:** 2026-11-12
**Scope:** `src/`, `models/`, `data/`, `dashboard/backend/`, project documentation
**Out of scope:** `dashboard/` frontend — not touched per instruction

## Executive summary

**Overall weighted completion: 54% / 100%**

The federated IDS pipeline is implemented end-to-end and the central model-training claims are substantiated by persisted artifacts: a centralized MLP (97.7% accuracy, 0.998 ROC-AUC) and a completed 50-round manual FedAvg federated run (95.9% accuracy), each with a metrics JSON and full training history. However, two entire audit domains — differential privacy and evaluation/testing — are unimplemented (empty directories), and `PROJECT_SYNOPSIS.md` contains false claims (Random Forest/XGBoost and differential privacy were never implemented; the Ray/Flower simulation was never run). The code works where executed, but reproducibility and test coverage are weak.

## Methodology

Seven components, weighted per the audit charter:

| Component | Weight |
|---|---:|
| Dataset / preprocessing | 15% |
| Centralized ML | 15% |
| Federated learning | 25% |
| Differential privacy | 15% |
| Evaluation / testing | 15% |
| Documentation / reproducibility | 10% |
| Dashboard readiness | 5% |

Each component is scored 0–100 against verified evidence (source read and/or persisted artifacts inspected), then multiplied by its weight. **Verified** = confirmed by reading source and/or persisted output files. **Assumption** = not independently validated in this session.

## 1. Dataset / preprocessing — 90/100 (13.5 weighted pts)

Verified:
- `src/data/preprocessing.py`: `CICIDS2017Preprocessor` implements the full pipeline — load 8 CIC-IDS2017 CSVs → clean labels → binary conversion (BENIGN=0, ATTACK=1) → clean (inf→NaN, drop duplicates, median imputation, non-numeric coercion, constant-feature removal) → stratified train/test split (`test_size=0.2`, `random_state=42`) → `StandardScaler` fit **only on training data** (no data leakage) → transform test.
- Artifacts saved to `data/processed/`: `scaler.pkl`, `feature_names.json`, `label_mapping.json`, `constant_features.json`.
- Processed arrays saved: `X_train.npy` (2,017,889 × 70 ≈ 1.08 GB), `X_test.npy` (504,473 × 70), `y_train.npy`, `y_test.npy`.
- Raw source present: `data/raw/MachineLearningCVE/` contains the 8 CSV files.

Assumptions / gaps:
- Class balance and per-class distributions not independently recomputed (relied on printed split logs).
- No data-integrity/validation scripts, no dataset manifest, no documentation of feature semantics.
- Raw data is an untracked directory (`?? data/` in git status) — not version controlled.

## 2. Centralized ML — 90/100 (13.5 weighted pts)

Verified:
- `src/models/mlp.py`: `MLPIDS` now inherits `nn.Module` (fix applied at line 38). Architecture: input → [Linear→BatchNorm→ReLU→Dropout]×3 with `hidden_dims=[128,64,32]`, `dropout=0.3` → Linear output (1 logit). `predict_proba`/`predict`/`count_parameters` implemented.
- `src/models/train_centralized.py`: 20 epochs, `batch_size=256`, Adam `lr=0.001`, `BCEWithLogitsLoss`; saves `models/centralized_mlp.pth` and `models/centralized_mlp_results.json`.
- Results **verified from JSON**: accuracy 0.9771, precision 0.9616, recall 0.9001, F1 0.9298, ROC-AUC 0.9981, FPR 0.0073, FNR 0.0999. Training history (loss/acc/prec/rec/f1/roc_auc over 20 epochs) persisted.

Gaps:
- CPU-only training (no CUDA/GPU support).
- Single architecture; no hyperparameter tuning, no learning-rate scheduling, no cross-validation, no early stopping.
- No model card / architecture documentation beyond docstrings.

## 3. Federated learning — 80/100 (20.0 weighted pts)

Verified:
- `src/federated/run_manual_federated.py`: manual FedAvg, no Ray/Flower dependency. 50 rounds, 5 clients, non-IID Dirichlet partitioning (`alpha=0.5`), `local_epochs=5`, `batch_size=128`, `lr=0.001`, warm start from `centralized_mlp.pth`, `max_samples_per_client=100000`, weighted aggregation by sample count, checkpoints every 10 rounds (rounds 0,10,20,30,40,50 saved in `models/checkpoints/`).
- Final model saved: `models/federated_mlp_manual.pth`; results: `models/manual_federated_improved_results.json`.
- Results **verified from JSON**: final accuracy 0.9587, precision 0.8818, recall 0.8721, F1 0.8770, FPR 0.0237, FNR 0.1279 (TP 74,285 / TN 409,343 / FP 9,954 / FN 10,891). Full per-round history and per-client metrics persisted for all 50 rounds.

Not verified / gaps:
- Flower `server.py` / `client.py` code exists but was **not executed** (sandbox blocks port binding).
- `run_federated_improved.py` (Ray+Flower) not run for the same reason.
- `manual_simulation.py` and `test_manual_simulation.py` exist; the latter produced a small 5-round test result (accuracy 0.80) — cited in the synopsis but not representative of the full 50-round run.
- Final federated metrics do **not** include ROC-AUC (evaluation function omits it).
- No secure aggregation, no client sampling fraction, no straggler/dropout handling.

## 4. Differential privacy — 0/100 (0.0 weighted pts)

**Not implemented.**
- `src/privacy/` directory is empty — no DP-SGD, no gradient clipping/noise injection, no epsilon/delta accounting, no secure aggregation.
- The claim in `PROJECT_SYNOPSIS.md` ("Implementation of privacy-preserving techniques (differential privacy, secure aggregation)") is **false**.
- This is the largest single gap in the project relative to its stated goals ("Privacy-Preserving Federated IDS").

## 5. Evaluation / testing — 10/100 (1.5 weighted pts)

**Not implemented.**
- `src/evaluation/` directory is empty — no evaluation scripts, no unit tests, no integration tests, no test harness.
- Model quality evidence exists only as persisted results JSONs (output artifacts), not as executable tests.
- No independent validation harness, no holdout re-scoring, no threshold-sensitivity analysis, no statistical significance testing.

## 6. Documentation / reproducibility — 35/100 (3.5 weighted pts)

Verified:
- Code files carry docstrings.
- `PROJECT_SYNOPSIS.md` exists and results JSONs/checkpoints are persisted.

Gaps (significant):
- `PROJECT_SYNOPSIS.md` contains **false claims**: "Training of Random Forest and XGBoost models" (never implemented — no RF/XGB code anywhere), "differential privacy, secure aggregation" (not implemented), and it reports federated results from the 5-round test (80% accuracy) rather than the completed 50-round run (95.9%).
- No README for training or federated scripts; no experiment-config documentation; no instructions to reproduce a run.
- `dashboard/backend/requirements.txt` omits `torch` and `flwr` though `main.py` imports both.
- No `README.md` exists at the repository root.

## 7. Dashboard readiness — 40/100 (2.0 weighted pts)

Not touched per instruction (reviewed only).
- `dashboard/backend/main.py`: FastAPI app with `/api/health`, `/api/system-status`, `/api/model-metrics`, `/api/analyze`, `/api/demo-events`; loads preprocessor, MLP model, and evidence builder at startup; prefers federated then centralized metrics.
- Gaps: `requirements.txt` missing `torch` and `flwr`; not verified running (sandbox blocks port 8000); component-loading can fail silently if weights absent.

## Model performance summary (verified from persisted JSONs)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Centralized MLP (20 epochs) | 0.9771 | 0.9616 | 0.9001 | 0.9298 | 0.9981 | 0.0073 | 0.0999 |
| Federated MLP manual (50 rounds) | 0.9587 | 0.8818 | 0.8721 | 0.8770 | — | 0.0237 | 0.1279 |
| Federated 5-round test (small subset) | 0.8000 | 0.7500 | 0.8000 | 0.7700 | — | 0.1000 | 0.1500 |

## False claims in PROJECT_SYNOPSIS.md

1. "Training of Random Forest and XGBoost models across federated nodes" — **False.** No Random Forest or XGBoost code exists in the repository. Only the MLP is implemented.
2. "Implementation of privacy-preserving techniques (differential privacy, secure aggregation)" — **False.** `src/privacy/` is empty.
3. Implies the FL simulation ran and produced results — the Ray/Flower path was never executed (sandbox port restrictions); only the manual FedAvg path ran.

## Weighted completion summary

| Component | Weight | Score | Weighted pts |
|---|---:|---:|---:|
| Dataset / preprocessing | 15% | 90 | 13.5 |
| Centralized ML | 15% | 90 | 13.5 |
| Federated learning | 25% | 80 | 20.0 |
| Differential privacy | 15% | 0 | 0.0 |
| Evaluation / testing | 15% | 10 | 1.5 |
| Documentation / reproducibility | 10% | 35 | 3.5 |
| Dashboard readiness | 5% | 40 | 2.0 |
| **Overall** | **100%** | — | **54.0** |

**Overall weighted completion: 54%**

## Recommendations (priority order)

1. Implement differential privacy (`src/privacy/`): DP-SGD with gradient clipping + Gaussian noise, epsilon/delta accounting — required given the project's stated privacy goal.
2. Build an evaluation/test suite (`src/evaluation/`): unit tests for preprocessing, model forward pass, FedAvg aggregation; end-to-end test that re-runs a short federated round and asserts metrics.
3. Correct `PROJECT_SYNOPSIS.md`: remove RF/XGBoost and DP claims; report the actual 50-round manual federated result.
4. Add `torch` and `flwr` to `dashboard/backend/requirements.txt`; add a root `README.md` with run/reproduce instructions.
5. Add ROC-AUC to federated evaluation; consider GPU/CUDA support and hyperparameter tuning.

