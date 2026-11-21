# Federated Learning IDS — Universal Network Dataset Intelligence & Adaptive Intrusion Detection System

Privacy-Preserving Federated Intrusion Detection System (IDS) combining high-accuracy supervised CIC-IDS2017 models (`MLPIDS` & `ImprovedMLPIDS`, centralized, FedAvg, FedProx, and DP-FedAvg) with a **Universal Network Dataset Intelligence and Adaptive Intrusion Detection Engine** capable of ingesting, profiling, semantically mapping, deriving, and scoring diverse network traffic datasets (`CIC-IDS2017`, `CIC-IDS2018/2019`, `UNSW-NB15`, `NSL-KDD`, `TON_IoT`, `Edge-IIoTset`, and custom flow tables).

## Overview

This project implements an end-to-end privacy-preserving federated intrusion detection and universal adaptive inference pipeline:

- **Centralized MLP baseline** — single-node training on the full training set (97.71% accuracy)
- **Centralized MLP (improved)** — `ImprovedMLPIDS` with residual connections, LayerNorm, FocalLoss, and threshold optimization (99.72% accuracy)
- **Manual FedAvg** — 50 rounds, 5 clients, Dirichlet $\alpha=0.5$ non-IID partitioning, warm-started from the centralized checkpoint (95.87% accuracy)
- **Improved FedAvg (Federated v2)** — FedAvg with `ImprovedMLPIDS`, weighted BCE, per-client LR scheduling, and threshold optimization (99.44% accuracy)
- **FedProx** — FedAvg with proximal term ($\mu$) to reduce client drift
- **DP-FedAvg** — differentially-private federated averaging with DP-SGD via Opacus 1.6.0 (per-example gradient clipping + Gaussian noise, moments-accountant privacy accounting, $\varepsilon=0.1457$ at $\delta=10^{-5}$)
- **Universal Dataset Intelligence & 18-Stage Adaptive Inference Engine (`src/inference/`)**:
  - **Multi-Format Ingestion (`UniversalIngestionEngine`)**: CSV, TSV, JSON, JSONL/NDJSON, and Parquet with automatic delimiter/encoding detection and malformed line recovery.
  - **Dataset Profiling & Schema Family Detection (`DatasetProfiler`)**: Computes a 0–100 Data Quality Score, feature distributions, 3-IQR outlier telemetry, and identifies dataset families (`CIC-IDS2017`, `CIC-IDS2018/2019`, `UNSW-NB15`, `NSL-KDD`, `TON_IoT`, `Edge-IIoTset`, or custom flow logs).
  - **Semantic Feature Mapping (`SemanticFeatureMapper`)**: Maps arbitrary column names to 77 canonical network features across 8 groups with exact, alias, normalized, and token-similarity matching, automatic unit conversion (`seconds_to_microseconds`), statistical plausibility validation, and user manual overrides.
  - **Safe Mathematical Feature Derivation (`FeatureDerivationEngine`)**: Safely derives 20 mathematical network flow properties (including `Min Packet Length`, `Max Packet Length`, packet rates, byte rates, directional means, and subflow aggregates) with explicit provenance tracking (`original`, `derived`, `approximated`, `unavailable`) without fabricating non-derivable measurements (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, TCP flags, IATs).
  - **Adaptive Model Routing (`ModelCompatibilityRouter` & `AdaptiveHybridScorer`)**: Routes datasets across 4 transparent detection modes (`Mode 1: Supervised CICIDS2017_SUPERVISED_MODEL`, `Mode 2: Adaptive Hybrid Ensemble`, `Mode 3: Unsupervised Anomaly + Rule Engine`, `Mode 4: Network Intrusion Rule Engine`).

## Key Results

### Baselines

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---|---|---|---|---|---|
| Centralized MLP (20 epochs) | 97.71% | 96.16% | 90.01% | 92.98% | 99.81% | 0.73% | 9.99% |
| Manual FedAvg (50 rounds) | 95.87% | 88.18% | 87.21% | 87.70% | — | 2.37% | 12.79% |
| DP-FedAvg (30 rounds) | 93.92% | 73.71% | 99.48% | 84.68% | 99.50% | 7.21% | 0.52% |

### Confusion Matrices (Test Set: 504,473 samples)

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---|---|---|---|
| Centralized MLP | 416,238 | 3,059 | 8,512 | 76,664 |
| Manual FedAvg | 409,343 | 9,954 | 10,891 | 74,285 |
| DP-FedAvg | 389,079 | 30,218 | 445 | 84,731 |

### Improved Variants (`ImprovedMLPIDS`: 256→128→64→32, residual, LayerNorm, FocalLoss, threshold optimization)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Optimal Threshold |
|---|---|---|---|---|---|---|---|---|
| Centralized Improved | 99.72% | 99.54% | 98.79% | 99.16% | 99.99% | 0.09% | 1.21% | 0.58 |
| Federated v2 (round 20, 30 rounds total) | 99.44% | 97.43% | 99.30% | 98.36% | 99.98% | 0.53% | 0.70% | 0.50 |

Baseline results in `models/centralized_mlp_results.json`, `models/manual_federated_improved_results.json`, `models/dp_federated_results.json`. Improved checkpoints: `models/centralized_mlp_improved.pth`, `models/checkpoints/federated_v2_round_20.pth`.

## Repository Structure

```
federated-ids/
├── PROJECT_SYNOPSIS.md
├── README.md
├── requirements.txt
├── start_backend.sh                  # Launches FastAPI backend on port 18003
├── start_frontend.sh                 # Launches React/Vite frontend on port 3000+
├── docs/                             # Comprehensive technical documentation
│   ├── UNIVERSAL_DATASET_AUDIT.md
│   ├── UNIVERSAL_DATASET_ARCHITECTURE.md
│   ├── FEATURE_MAPPING_REFERENCE.md
│   ├── MODEL_ROUTING.md
│   ├── INFERENCE_PIPELINE.md
│   ├── DATASET_COMPATIBILITY.md
│   ├── TESTING_REPORT.md
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── PREPROCESSING.md
│   ├── CENTRALIZED_TRAINING.md
│   ├── FEDERATED_LEARNING.md
│   ├── DIFFERENTIAL_PRIVACY.md
│   ├── EVALUATION.md
│   └── REPRODUCIBILITY.md
├── data/                             # Dataset samples and processed artifacts
├── models/                           # Trained checkpoints and results JSON
├── reports/                          # Validation, audit, status, and experiment reports
├── frontend/                         # React + TypeScript + Vite dashboard
│   └── src/
│       ├── pages/Inference.tsx       # 18-stage Universal Dataset Intelligence UI
│       ├── pages/Detections.tsx      # Multi-mode Detection Explorer & Rule Inspector
│       └── services/api.ts           # Typed API client
├── src/
│   ├── data/preprocessing.py         # CIC-IDS2017 training preprocessing pipeline
│   ├── models/
│   │   ├── mlp.py                    # MLPIDS
│   │   └── improved_mlp.py           # ImprovedMLPIDS & FocalLoss
│   ├── inference/                    # Universal Dataset Intelligence & Adaptive Inference
│   │   ├── canonical_schema.py       # 77 canonical features across 8 network groups
│   │   ├── ingestion.py              # Multi-format UniversalIngestionEngine
│   │   ├── profiler.py               # DatasetProfiler & schema family detection
│   │   ├── feature_mapper.py         # SemanticFeatureMapper & unit conversion
│   │   ├── feature_derivation.py     # Safe mathematical FeatureDerivationEngine
│   │   ├── model_router.py           # ModelCompatibilityRouter (4 detection modes)
│   │   ├── detectors.py              # Supervised, IsolationForest+ZScore, 8-Rule Engine, Hybrid Scorer
│   │   └── pipeline.py               # 18-stage traceable AdaptiveInferencePipeline
│   ├── federated/
│   │   ├── run_manual_federated.py   # Manual FedAvg (50 rounds)
│   │   ├── run_dp_federated.py       # DP-FedAvg (30 rounds)
│   │   └── schema_validator.py       # FederatedCompatibilityValidator for heterogeneous clients
│   ├── dashboard/
│   │   ├── main.py                   # FastAPI application & Universal Intelligence endpoints
│   │   ├── schemas/__init__.py       # Pydantic schemas
│   │   └── services/
│   │       ├── model_loader.py       # PyTorch checkpoint loader (MLPIDS & ImprovedMLPIDS)
│   │       └── inference_runner.py   # Background inference orchestrator & artifact storage
│   └── privacy/dp_sgd.py             # DP-SGD trainer + privacy accounting
└── tests/
    ├── test_mlp.py
    ├── test_federated.py
    ├── test_dp.py
    └── test_universal_pipeline.py    # 14 test suites covering all 16 Universal IDS phases
```

## Quick Start — Running the Full-Stack Dashboard & Backend

### 1. Start the FastAPI Backend (`http://localhost:18003`)

```bash
./start_backend.sh
```

### 2. Start the React + Vite Frontend (`http://localhost:3000`)

```bash
./start_frontend.sh
```

The frontend proxies `/api`, `/health`, and `/ws` directly to the FastAPI backend (`http://localhost:18003`), avoiding cross-origin issues regardless of which local port Vite selects.

## Run the ML Training & Validation Pipeline

### 1. Validate Centralized & Improved Checkpoints

```bash
PYTHONPATH=. .venv/bin/python validate_centralized.py
```

### 2. Train Centralized MLP Baseline

```bash
PYTHONPATH=. .venv/bin/python src/models/train_centralized.py
```

### 3. Run Manual FedAvg (50 Rounds)

```bash
PYTHONPATH=. .venv/bin/python src/federated/run_manual_federated.py
```

### 4. Run DP-FedAvg (30 Rounds)

```bash
PYTHONPATH=. .venv/bin/python src/federated/run_dp_federated.py
```

## Tests

Run all 50 unit and integration tests across the repository:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

## Documentation

See `docs/` for detailed technical documentation:

- `docs/DASHBOARD_INTERFACE.md` — complete guide to the React + FastAPI Dashboard UI, 6-step Inference Wizard, Model Selector, and Detections Explorer
- `docs/UNIVERSAL_DATASET_AUDIT.md` — Phase 1 audit of repository, models, and legacy assumptions
- `docs/UNIVERSAL_DATASET_ARCHITECTURE.md` — System architecture of the Universal Dataset Intelligence Engine
- `docs/FEATURE_MAPPING_REFERENCE.md` — 77-feature canonical schema, alias mapping rules, and mathematical derivations
- `docs/MODEL_ROUTING.md` — 4-mode routing decision matrix, neutral mean baseline strategy, and severity tiers
- `docs/INFERENCE_PIPELINE.md` — Complete 18-stage traceable pipeline specification
- `docs/DATASET_COMPATIBILITY.md` — Supported file formats, dataset families, and federated schema compatibility
- `docs/TESTING_REPORT.md` — 50-test verification report, 10-scenario evaluation matrix, and 100k-row scaling benchmarks
- `docs/ARCHITECTURE.md` — Base federated system architecture
- `docs/DATASET.md` — CIC-IDS2017 dataset details
- `docs/PREPROCESSING.md` — Training preprocessing pipeline
- `docs/CENTRALIZED_TRAINING.md` — Centralized MLP training
- `docs/FEDERATED_LEARNING.md` — Manual FedAvg & FedProx
- `docs/DIFFERENTIAL_PRIVACY.md` — DP-SGD / DP-FedAvg
- `docs/EVALUATION.md` — Evaluation methodology and results
- `docs/REPRODUCIBILITY.md` — How to reproduce results
