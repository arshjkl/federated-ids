# Federated Learning IDS

Privacy-Preserving Federated Intrusion Detection System (IDS) built on the CIC-IDS2017 dataset.

## Overview

This project implements an end-to-end privacy-preserving federated intrusion detection pipeline. It detects malicious network traffic (binary classification: BENIGN vs ATTACK) using:

- **Centralized MLP baseline** — single-node training on the full training set (97.71% accuracy)
- **Centralized MLP (improved)** — ImprovedMLPIDS with weighted BCE / focal loss, LR scheduling, early stopping, threshold optimization
- **Manual FedAvg** — 50 rounds, 5 clients, Dirichlet α=0.5 non-IID partitioning, warm-started from the centralized checkpoint (95.87% accuracy)
- **Improved FedAvg** — FedAvg with ImprovedMLPIDS, weighted BCE, per-client LR scheduling, threshold optimization
- **FedProx** — FedAvg with proximal term (mu) to reduce client drift
- **DP-FedAvg** — differentially-private federated averaging with DP-SGD via Opacus 1.6.0 (per-example gradient clipping + Gaussian noise, moments-accountant privacy accounting, epsilon=0.1457 at delta=1e-05)
- **Improved DP-FedAvg** — DP-FedAvg with correct batch-level privacy accounting, ImprovedMLPIDS, weighted BCE, threshold optimization
- **DP-FedProx** — DP-FedAvg with proximal term (mu) + DP-SGD

## Key Results

### Baselines

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Centralized MLP (20 epochs) | 97.71% | 96.16% | 90.01% | 92.98% | 99.81% |
| Manual FedAvg (50 rounds) | 95.87% | 88.18% | 87.21% | 87.70% | — |
| DP-FedAvg (30 rounds) | 93.92% | 73.71% | 99.48% | 84.68% | 99.50% |

### Improved Variants (ImprovedMLPIDS: 256→128→64→32, residual, LayerNorm, FocalLoss, threshold optimization)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Optimal Threshold |
|---|---|---|---|---|---|---|---|---|
| Centralized Improved | 99.72% | 99.54% | 98.79% | 99.16% | 99.99% | 0.09% | 1.21% | 0.58 |
| Federated v2 (round 20, 30 rounds total) | 99.44% | 97.43% | 99.30% | 98.36% | 99.98% | 0.53% | 0.70% | 0.50 |

Baseline results in `models/centralized_mlp_results.json`, `models/manual_federated_improved_results.json`, `models/dp_federated_results.json`. Improved checkpoints: `models/centralized_mlp_improved.pth`, `models/checkpoints/federated_v2_round_20.pth`.

## Repository structure

```
federated-ids/
├── PROJECT_SYNOPSIS.md   # Project overview and key results
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── docs/                 # Detailed documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── PREPROCESSING.md
│   ├── CENTRALIZED_TRAINING.md
│   ├── FEDERATED_LEARNING.md
│   ├── DIFFERENTIAL_PRIVACY.md
│   ├── EVALUATION.md
│   └── REPRODUCIBILITY.md
├── data/                 # Dataset (raw + processed — generated, not version controlled)
├── models/               # Trained checkpoints and results JSON
├── reports/              # Validation, audit, status, and experiment reports
├── src/
│   ├── data/preprocessing_validation.py
│   ├── models/mlp.py                 # MLPIDS + ImprovedMLPIDS, FocalLoss
│   ├── models/train_centralized.py   # Centralized training (20 epochs baseline + improved)
│   ├── federated/
│   │   ├── run_manual_federated.py   # Manual FedAvg (50 rounds) + improved + FedProx
│   │   ├── run_dp_federated.py       # DP-FedAvg (30 rounds) + improved + DP-FedProx
│   │   └── run_comparison.py         # NEW: all-variants comparison runner
│   ├── privacy/dp_sgd.py             # DP-SGD trainer + privacy accounting (+ batches_per_epoch)
│   └── utils/explore_dataset.py
├── tests/
│   ├── test_mlp.py
│   ├── test_federated.py
│   ├── test_dp.py
│   ├── test_improved.py              # NEW: improved MLP, losses, FedProx, threshold, privacy
│   └── test_preprocessing.py
├── validate_centralized.py # Centralized validation script
└── .venv/                # Python env (torch 2.14.0, opacus 1.6.0)
```

## Requirements

Install dependencies (network is blocked for PyPI — packages must already be present in `.venv`):

```bash
pip install -r requirements.txt
```

Or create the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run the pipeline (full commands)

### 1. Preprocess the CIC-IDS2017 dataset

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/data/preprocessing_validation.py
```

This loads the raw CSVs, cleans labels, converts to binary (BENIGN=0, ATTACK=1), applies stratified train/test split, fits `StandardScaler` on training data only, and saves processed arrays + scaler to `data/processed/`.

### 2. Train centralized MLP baseline

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/models/train_centralized.py
```

Trains for 20 epochs, saves `models/centralized_mlp.pth` and `models/centralized_mlp_results.json`.

### 3. Run centralized validation

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python validate_centralized.py
```

### 4. Run Manual FedAvg (50 rounds)

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_manual_federated.py
```

Trains 50 rounds with 5 clients, warm-started from the centralized checkpoint. Saves `models/federated_mlp_manual.pth`, `models/manual_federated_improved_results.json`, and per-round checkpoints.

### 5. Run DP-FedAvg (30 rounds)

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_dp_federated.py
```

Trains 30 rounds with DP-SGD (noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-05). Saves `models/federated_mlp_dp.pth`, `models/dp_federated_results.json`, and `reports/privacy_report.json`.

### 6. Run improved variants (all models)

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python src/federated/run_comparison.py
```

Runs Centralized (baseline + improved), FedAvg (baseline + improved), FedProx, DP-FedAvg (baseline + improved), DP-FedProx under identical data/seed conditions. Writes `models/model_comparison_improved.json` and prints a side-by-side table.

### 7. Run individual improved variants

Centralized improved:
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -c "from src.models.train_centralized import train_centralized_improved; train_centralized_improved()"
```

Improved FedAvg / FedProx:
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -c "from src.federated.run_manual_federated import run_manual_federated_improved, run_manual_federated_prox; run_manual_federated_improved(); run_manual_federated_prox()"
```

Improved DP-FedAvg / DP-FedProx:
```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -c "from src.federated.run_dp_federated import run_dp_federated_improved, run_dp_federated_prox; run_dp_federated_improved(); run_dp_federated_prox()"
```

### Expected order

Run steps 1 → 2 → 4 → 5 in order (step 3 is optional validation). Steps 4, 5, and 6 all warm-start from the centralized checkpoint produced in step 2. Step 6 runs all improved variants in one go.

## Tests

Run the full suite with the project's unittest runner (pytest unavailable — network blocked for PyPI):

```bash
PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python -m unittest discover -s tests
```

All tests pass (66 tests across 6 files; documented in `reports/test_results.md`).

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

## Notes

- All model files are under GitHub's 100MB per-file limit.