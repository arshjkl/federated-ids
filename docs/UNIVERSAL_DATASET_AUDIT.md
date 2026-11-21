# Universal Dataset Intelligence & Adaptive Detection — Complete Repository Audit

**Date:** 2026-11-18  
**Project:** Federated Learning Intrusion Detection System (`federated-ids`)

---

## 1. Existing Architecture

The repository consists of two major layers:

### A. Core Research & Federated ML Layer (Protected / Read-Only Artifacts)
- **Preprocessing (`src/data/preprocessing.py`, `src/data/preprocessing_validation.py`)**:
  - `CICIDS2017Preprocessor` loads 8 raw CIC-IDS2017 CSV files (2,830,743 total rows, 79 raw columns), strips column whitespace, maps 15 attack classes to binary (`BENIGN=0`, `ATTACK=1`), drops duplicates and 8 zero-variance constant columns (`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`), performs a stratified 80/20 split (`2,017,889` train / `504,473` test × `70` features), and fits a `StandardScaler` strictly on the training set.
  - Persisted read-only artifacts in `data/processed/`: `scaler.pkl`, `feature_names.json` (70 ordered features), `label_mapping.json`, `constant_features.json`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.
- **Supervised Neural Architectures (`src/models/mlp.py`, `src/models/improved_mlp.py`)**:
  - `MLPIDS`: 70 → 128 → 64 → 32 → 1 with optional `BatchNorm1d`, `ReLU`, and `Dropout(0.3)`.
  - `ImprovedMLPIDS`: 70 → 256 → 128 → 64 → 32 → 1 with residual connections (`ResidualBlock`), `LayerNorm` / `BatchNorm1d`, `GELU`, `FocalLoss`, and threshold optimization (`get_optimal_threshold`).
- **Federated Learning & Differential Privacy (`src/federated/`, `src/privacy/dp_sgd.py`)**:
  - `run_manual_federated.py` / `run_manual_federated_v2.py`: Manual FedAvg and FedProx across 5 clients with Dirichlet ($\alpha=0.5$) non-IID partitioning.
  - `run_dp_federated.py`: DP-FedAvg using Opacus `DPSGDTrainer` (`use_batchnorm=False`, per-example gradient clipping `max_grad_norm=1.0`, Gaussian noise multiplier `1.0`, $\varepsilon=0.1457$ at $\delta=10^{-5}$).
- **Trained Checkpoints (`models/`)**:
  - `centralized_mlp.pth` (Acc: 97.71%, ROC-AUC: 99.81%)
  - `centralized_mlp_improved.pth` (Acc: 99.72%, F1: 99.16%, ROC-AUC: 99.99%, optimal threshold: 0.58)
  - `federated_mlp_manual.pth` / `manual_federated_mlp.pth` (Acc: 95.87%, F1: 87.70%)
  - `checkpoints/federated_v2_round_20.pth` (Acc: 99.44%, F1: 98.36%, ROC-AUC: 99.98%)
  - `federated_mlp_dp.pth` (Acc: 93.92%, Recall: 99.48%, ROC-AUC: 99.50%)

### B. Full-Stack Platform & Dashboard Layer
- **Backend (`src/dashboard/main.py`, `src/dashboard/db/`, `src/dashboard/schemas/`, `src/dashboard/services/`)**:
  - FastAPI application on port `18003` backed by async SQLAlchemy + SQLite (`federated_ids.db`).
  - Core ORM models: `Organisation`, `Dataset`, `FederatedExperiment`, `TrainingRound`, `ClientParticipation`, `ClientRoundMetrics`, `GlobalModel`, `InferenceJob`, `DetectionResult`, `DetectionHistory`, `TrainingHistory`, `PrivacyRecord`, `SystemEvent`, `Incident`.
  - Services: `PreprocessingLoader` (`src/dashboard/services/preprocessing.py`), `ModelLoader` (`src/dashboard/services/model_loader.py`), `InferenceRunner` (`src/dashboard/services/inference_runner.py`).
- **Frontend (`frontend/src/`)**:
  - React 18 + TypeScript + Vite + TailwindCSS + TanStack Query on port `3000`.
  - Pages: `Dashboard.tsx`, `Organisations.tsx`, `Experiments.tsx`, `Models.tsx`, `Inference.tsx`, `Detections.tsx`, `Privacy.tsx`, `Settings.tsx`.

---

## 2. Trace of Current Dataset-to-Detection Pipeline

1. **Dataset Upload / Sample Selection**:
   - Frontend `Inference.tsx` Step 1 (`UploadStep`) accepts `.csv` files or selects a sample CSV from `data/mock/inference/`.
   - Calls `POST /api/inference/analyze-csv` or `POST /api/inference/analyze-sample`.
2. **File Parsing & Analysis**:
   - `main.py` calls `pd.read_csv(temp_path, low_memory=False)` directly with default comma delimiter and UTF-8 encoding.
   - Strips column whitespace, appends `.1`, `.2` to duplicate column names, checks for exact `"Label"` column name, computes basic min/max/mean/std/null_count, and saves a `Dataset` record in SQLite.
3. **Preview**:
   - `GET /api/inference/preview/{dataset_id}` re-reads the entire CSV file into memory on every page request (`pd.read_csv(dataset.file_path)`) and slices `df.iloc[offset:offset + page_size]`.
4. **Validation**:
   - `GET /api/inference/validate/{dataset_id}` compares `df.columns` against a hardcoded 70-element `EXPECTED_FEATURES` list (`CIC-IDS2017` exact column names).
   - If even 1 of the 70 CIC-IDS2017 column names is missing or renamed (e.g., `Dst Port` instead of `Destination Port`, or missing `Min Packet Length`, `Max Packet Length`, `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`), `missing_features` is non-empty, `feature_count` check returns `status="error"`, and `is_valid=False`.
5. **Feature Detection & Matching**:
   - **None exists.** Only exact string equality after `.strip()` and duplicate `.1` suffixing is performed.
6. **Preprocessing**:
   - `POST /api/inference/preprocess/{dataset_id}` calls `PreprocessingLoader.transform(df)`.
   - `PreprocessingLoader.transform` checks `missing = set(self.feature_names) - set(df.columns)`. If any of the 70 exact CIC-IDS2017 names are missing, it raises `ValueError(f"Missing features: {missing}")`.
   - Replaces `NaN`, `+inf`, `-inf` with `0.0` via `np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)` and applies `self.scaler.transform(X)`.
7. **Model Loading & Routing**:
   - **No routing exists.** `ModelLoader` only loads the 70-input PyTorch MLP checkpoints (`MLPIDS` / `ImprovedMLPIDS`).
   - If a dataset has fewer than 70 features or different feature semantics, the supervised MLP cannot run without either crashing or feeding fabricated zeros into `StandardScaler` (where `0.0` in raw space maps to $-\mu_i / \sigma_i$ in standardized space, distorting neural activations).
8. **Inference**:
   - `POST /api/inference/run-on-dataset`, `/api/inference/upload-csv`, `/api/inference/upload-sample` run `InferenceRunner.run_inference_job()`.
   - `InferenceRunner` loads the CSV via `PreprocessingLoader.transform(df)`, runs `model(X_tensor)` in batches, applies `sigmoid(logits) >= threshold`, and writes `DetectionResult` rows (`prediction` 0 or 1, `probability`, `confidence`).
9. **Detection Storage & Visualization**:
   - `DetectionResult` only stores `record_index`, `prediction` (0/1), `probability`, `confidence`, `true_label`, and `features` (which was previously left `None` during batch insertion).
   - No fields or logic exist for `detection_mode`, `anomaly_score`, `risk_score`, `severity`, `predicted_category`, `triggered_rules`, `explanation`, or `compatibility_info`.

---

## 3. Model Feature Requirements

All existing supervised models (`centralized_mlp.pth`, `centralized_mlp_improved.pth`, `federated_mlp_manual.pth`, `manual_federated_mlp.pth`, `federated_mlp_dp.pth`, `federated_v2_round_*.pth`) require:
- **Exact Input Dimension**: `input_dim = 70` float32 features.
- **Exact Feature Ordering**: As defined in `data/processed/feature_names.json` (indices 0 through 69, from `"Destination Port"` to `"Idle Min"`).
- **Exact Normalization**: `data/processed/scaler.pkl` (`StandardScaler` fitted on 2,017,889 CIC-IDS2017 training samples).
- **Implication**: Because `StandardScaler` transforms each feature $x_i$ as $(x_i - \mu_i) / \sigma_i$, substituting `0.0` for an unavailable feature does **not** produce a neutral zero activation after scaling—it injects $-\mu_i / \sigma_i$. Furthermore, if a majority of the 70 CIC-IDS2017 features are missing (e.g., UNSW-NB15, NSL-KDD, or partial NetFlow exports), running the 70-input CIC-IDS2017 MLP produces scientifically invalid predictions.

---

## 4. Hardcoded CIC-IDS2017 Assumptions & Places Filling Arbitrary Zeros

1. `src/dashboard/main.py` lines 1468–1539: Hardcoded 70-item `EXPECTED_FEATURES` list inside `validate_dataset_for_inference`; rejects any dataset missing even one exact CIC-IDS2017 column name.
2. `src/dashboard/main.py` lines 1231, 1331, 1585, 1647: Hardcoded `"Label"` column check (case-sensitive `"Label"` only; ignores `label`, `attack_cat`, `class`, `target`, `type`, `malicious`).
3. `src/dashboard/main.py` lines 336–345: `POST /api/datasets/{dataset_id}/validate` returns a hardcoded mock `DatasetValidationResponse(rows=100000, columns=70, feature_names=["feature_0"...])`.
4. `src/dashboard/services/preprocessing.py` lines 123–135:
   - Requires `set(self.feature_names) - set(df.columns)` to be empty (`ValueError("Missing features: ...")`).
   - Blindly fills all `NaN`/`Inf` with `0.0` (`np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)`).
5. `test_inference.csv` in root directory: Uses column names `feature_1` .. `feature_70`, which immediately fails `validate_dataset_for_inference` and `PreprocessingLoader.transform`.

---

## 5. Bugs, Failure Points, and Risks

1. **Rigid Schema Rejection**: Datasets with aliased column names (e.g. CICFlowMeter v3/v4 `Dst Port`, `Tot Fwd Pkts`, `Flow Byts/s`, `Fwd Pkt Len Max`, `Init Fwd Win Byts`, `Fwd Act Data Pkts`, `Fwd Seg Size Min`) or missing TCP/packet-length features (`Min Packet Length`, `Max Packet Length`, `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`) are rejected outright.
2. **No Derivations for Derivable Features**: When `Min Packet Length`, `Max Packet Length`, `Average Packet Size`, `Flow Bytes/s`, `Flow Packets/s`, `Down/Up Ratio`, `Subflow Fwd Packets`, or `Fwd Header Length.1` are omitted from a CSV, they are not derived even when their exact mathematical constituents (`Fwd Packet Length Min`, `Bwd Packet Length Min`, `Total Length of Fwd Packets`, `Flow Duration`, etc.) are present in the dataset.
3. **No Fallback Detection Mode for Non-CICIDS Schemas**: When a user uploads UNSW-NB15, NSL-KDD, TON_IoT, or a custom flow table with 10–40 columns, the system has no unsupervised anomaly detector, rule-based engine, or hybrid detector to fall back to.
4. **Conflation of Anomaly vs. Confirmed Attack**: The system only outputs binary `0` (Benign) or `1` (Malicious) with no distinction between a high-confidence supervised classification, a statistical outlier/anomaly, and a heuristic rule alert.
5. **Memory & Performance Bottlenecks**: `preview_dataset`, `validate_dataset_for_inference`, `preprocess_dataset`, and `InferenceRunner` each call `pd.read_csv(file_path)` on the full file repeatedly without chunked reading or caching profile metadata.
6. **Placeholder Modules**: `src/inference/__init__.py` and `src/model_registry/__init__.py` are 4-line empty docstrings with no implementation.

---

## 6. Proposed Migration Architecture

We will build a modular **Universal Dataset Intelligence & Adaptive Intrusion Detection Engine** under `src/inference/` and integrate it cleanly into `src/dashboard/` and `frontend/src/` while leaving all read-only research artifacts (`models/*.pth`, `data/processed/*`, `src/models/mlp.py`, `src/models/improved_mlp.py`, `src/data/preprocessing.py`, `src/federated/*`, `src/privacy/*`) untouched:

1. **Phase 2 — Universal Data Ingestion Engine (`src/inference/ingestion.py`)**:
   - Supports CSV, TSV, JSON, JSONL/NDJSON, and Parquet (when engine available).
   - Auto-detects encoding (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`), delimiter (`,`, `\t`, `;`, `|`), malformed lines, duplicate columns, empty columns, constant columns, missing/infinite values, column types (numeric, categorical, datetime, boolean, text, identifier), and candidate label columns (`Label`, `label`, `class`, `attack_cat`, `attack_type`, `target`, `malicious`, `is_attack`, `category`, `type`).
   - Supports chunked streaming (`iter_chunks`) and bounded-memory sampling for large files.
2. **Phase 3 — Dataset Profiler & Schema Family Intelligence (`src/inference/profiler.py`)**:
   - Computes row/column counts, memory footprint, data types, missing/infinite/duplicate counts, numeric distributions (min, max, mean, std, median, IQR, outlier counts via 3-IQR/Z-score), categorical distributions, quality score (0–100), and network-feature coverage.
   - Identifies dataset schema family from feature signatures: `CICIDS2017`, `CICIDS2018_2019` (CICFlowMeter shorthand), `UNSW_NB15`, `NSL_KDD`, `TON_IOT`, `EDGE_IIOT`, `GENERIC_FLOW`, or `CUSTOM_TABULAR`.
3. **Phase 4 & 5 — Canonical Feature Schema & Semantic Feature Mapping Engine (`src/inference/canonical_schema.py`, `src/inference/feature_mapper.py`)**:
   - Defines the 8-group Canonical Network Traffic Feature Schema (`flow_properties`, `directional_properties`, `packet_statistics`, `timing`, `protocol_properties`, `tcp_properties`, `rate_properties`, `behavioral_features`) and maps each of the 70 CIC-IDS2017 model features to its canonical definition, unit/scale constraints, and category criticality.
   - Maps uploaded columns to canonical features and CIC-IDS2017 features using:
     1. Exact match
     2. Known alias / schema profile match (CICFlowMeter, UNSW-NB15, NSL-KDD, TON_IoT, Zeek/NetFlow aliases)
     3. Normalized token & semantic similarity with unit/type/scale validation
     4. Positional/indexed fallback (only when explicitly matching 70-column standardized exports like `feature_1..feature_70` with statistical distribution verification, flagged as `approximate_match` with explicit warning)
   - Classifies every canonical/model feature into: `exact_match`, `strong_semantic_match`, `derived_feature`, `approximate_match`, `ambiguous_match`, `unavailable_feature`, or `incompatible_feature`, with a `[0.0, 1.0]` confidence score and human-readable explanation.Supports manual user overrides.
4. **Phase 6 — Safe Feature Derivation Engine (`src/inference/feature_derivation.py`)**:
   - Safely derives mathematically determinable features (e.g., `Min Packet Length` = $\min(\text{Fwd Min}, \text{Bwd Min})$, `Max Packet Length` = $\max(\text{Fwd Max}, \text{Bwd Max})$, `Packet Length Mean`, `Packet Length Variance` = $\text{Std}^2$, `Packet Length Std` = $\sqrt{\text{Var}}$, `Average Packet Size`, `Avg Fwd Segment Size`, `Avg Bwd Segment Size`, `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`, `Down/Up Ratio`, `Fwd Header Length.1` = `Fwd Header Length`, `Subflow Fwd/Bwd Packets/Bytes` from flow totals when 1 subflow per flow).
   - Never fabricates non-derivable measurements (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, TCP flags, IATs when no timestamps exist). When non-critical features are unavailable in a high-coverage CICIDS dataset, records explicit provenance (`original`, `derived`, `approximated_training_median`, `unavailable`) so `StandardScaler` receives the training-set mean ($\mu_i$, which scales to `0.0` in standardized space rather than raw `0.0` which scales to $-\mu_i/\sigma_i$!) and lowers confidence accordingly.
5. **Phase 7 — Model Compatibility & Multi-Mode Detection Router (`src/inference/model_router.py`, `src/inference/detectors.py`)**:
   - Preserves existing trained checkpoints under `CICIDS2017_SUPERVISED_MODEL`.
   - Evaluates feature coverage, core-feature availability, mapping confidence, and distribution plausibility to route among 4 explicit modes:
     - **Mode 1: `supervised`** (`CICIDS2017_SUPERVISED_MODEL`) — used when direct + safely derived CIC-IDS2017 feature coverage is high ($\ge 85\%$ overall and $\ge 80\%$ core flow/packet/rate features).
     - **Mode 2: `hybrid`** (`CICIDS2017_SUPERVISED_MODEL` + Unsupervised Anomaly Detector + Evidence-Based Rule Engine) — used when CICIDS coverage is moderate ($65\%–85\%$, e.g., missing TCP window features like `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`) OR to cross-verify supervised predictions with anomaly & rule signals using documented, configurable weights.
     - **Mode 3: `anomaly`** (`IsolationForest` + Robust Statistical Mahalanobis/IQR Z-Score Detector on available canonical numeric features) — used when dataset is incompatible with the 70-feature CIC-IDS2017 supervised schema (e.g., UNSW-NB15, NSL-KDD, partial custom flow logs). Clearly labels detections as `Statistical Anomaly` / `Suspicious Anomaly` rather than confirmed attacks.
     - **Mode 4: `rule_based`** (Evidence-based configurable network heuristics: SYN flood / SYN-heavy ratio, port-scan destination/port burst, excessive packet/byte rate, asymmetric flood ratio, short-duration high-packet burst, repeated failed/RST connection patterns) — used standalone when only basic flow headers exist, and combined in `hybrid` / `anomaly` modes.
6. **Phase 8 & 9 — 18-Stage Traceable Adaptive Inference Pipeline & Rich Detection Results (`src/inference/pipeline.py`)**:
   - Executes and logs all 18 stages with real execution times, input/output counts, status (`completed`, `warning`, `failed`), warnings, errors, and metadata.
   - Produces per-record structured results: `record_index`, `predicted_category` (`Normal`, `Confirmed Attack (Supervised)`, `Probable Suspicious Traffic (Hybrid)`, `Statistical Anomaly`, `Rule Alert`, `Unsupported`), `prediction`, `supervised_prob`, `anomaly_score`, `rule_score`, `risk_score`, `confidence`, `severity` (`informational`, `low`, `medium`, `high`, `critical`), `detection_mode`, `triggered_rules`, `explanation`, `model_used`, and `compatibility_score`.
7. **Phase 11 — Federated Learning Schema Compatibility Validator (`src/federated/schema_validator.py`)**:
   - Validates client datasets and model updates before federated training/aggregation to prevent aggregating incompatible feature representations or architectures, with canonical feature schema versioning (`v1.0-canonical` / `v1.0-cicids70`).
8. **Phase 10, 12, 13 — Backend APIs & Transparent Frontend Experience**:
   - New & enhanced endpoints for universal ingestion, profile, preview (with search/sort/filter), validation, feature mapping (with manual override support), compatibility & routing report, 18-stage pipeline execution, rich detection results, and record investigation.
   - Redesigned `Inference.tsx` and `Detections.tsx` displaying Dataset Overview, searchable/sortable Data Preview, Validation Report, Visual Feature Compatibility Mapping Table (with manual override), Model Routing & Detection Mode explanation, Real 18-Stage Pipeline Progress, Rich Detection Analytics (severity, mode, risk/confidence distributions), and Record Investigation drawer.
