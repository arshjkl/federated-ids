# Universal Dataset Intelligence & Adaptive Detection Architecture

## 1. Architectural Overview

The Federated IDS platform is architected to process heterogeneous network traffic datasets without enforcing rigid single-dataset column constraints or fabricating missing measurements.

```
Uploaded File (CSV / TSV / JSON / JSONL / Parquet)
        │
        ▼
[1. Universal Data Ingestion Engine] (src/inference/ingestion.py)
  • Format, encoding (UTF-8/BOM/Latin-1/CP1252) & delimiter auto-detection
  • Malformed line recovery & duplicate column disambiguation (.1, .2)
  • Column role classification (numeric, categorical, boolean, datetime, text, identifier, label)
        │
        ▼
[2. Dataset Profiler & Schema Family Intelligence] (src/inference/profiler.py)
  • Row/column counts, memory footprint, missing/infinite/duplicate rates
  • Numeric (min, max, mean, std, median, IQR, 3-IQR outliers) & categorical distributions
  • Schema Family Detection (CICIDS2017, CICIDS2018_2019, UNSW_NB15, NSL_KDD, TON_IOT, EDGE_IIOT, GENERIC_NETWORK_FLOW, ANONYMIZED_FEATURE_MATRIX)
  • Transparent 0–100 Data Quality Score
        │
        ▼
[3. Canonical Feature Schema & Semantic Feature Mapper] (src/inference/canonical_schema.py, src/inference/feature_mapper.py)
  • 8 Canonical Groups (77 canonical features bridging the 70 ordered CIC-IDS2017 features)
  • Exact match → Known schema alias → Normalized token similarity → Statistical/unit validation
  • Classifies each feature: exact_match, strong_semantic_match, derived_feature, approximate_match, ambiguous_match, unavailable_feature, incompatible_feature
  • Supports user-defined manual mapping overrides
        │
        ▼
[4. Safe Feature Derivation Engine] (src/inference/feature_derivation.py)
  • Derives mathematically valid features (min/max/mean packet size, variance/std, byte/packet rates, down/up ratio, subflow totals)
  • Protects against division by zero, NaN, Inf, negative counts, and unit scale differences (seconds → microseconds)
  • Never fabricates non-derivable fields (TCP initial window sizes, TCP flags, packet IAT std)
  • Tracks full per-feature provenance
        │
        ▼
[5. Model Compatibility & Multi-Mode Detection Router] (src/inference/model_router.py, src/inference/detectors.py)
  ├── Mode 1: supervised  → CICIDS2017_SUPERVISED_MODEL (when effective coverage >= 90%, core >= 85%)
  ├── Mode 2: hybrid      → CICIDS2017_SUPERVISED_MODEL + IsolationForest/Modified-Z + Network Rules (coverage 55%–90%)
  ├── Mode 3: anomaly     → Unsupervised IsolationForest + Robust Modified Z-Score + Network Rules (non-CICIDS schemas)
  └── Mode 4: rule_based  → Configurable Evidence-Based Network Heuristic Rule Engine
        │
        ▼
[6. 18-Stage Traceable Pipeline & Rich Detection Synthesis] (src/inference/pipeline.py)
  • Per-stage execution telemetry (status, execution_time_ms, input/output counts, warnings, metadata)
  • Per-record outcome: predicted_category, risk_score, supervised_probability, anomaly_score, rule_score, confidence, severity, triggered_rules, explanation
```

---

## 2. Preservation of Existing Research Artifacts

All original trained checkpoints and preprocessing artifacts remain untouched and read-only:
- `models/centralized_mlp.pth` (`MLPIDS`, 97.71% accuracy)
- `models/centralized_mlp_improved.pth` (`ImprovedMLPIDS`, 99.72% accuracy, 99.16% F1)
- `models/federated_mlp_manual.pth` / `models/manual_federated_mlp.pth` (Manual FedAvg, 95.87% accuracy)
- `models/checkpoints/federated_v2_round_20.pth` (Improved Federated v2, 99.44% accuracy)
- `models/federated_mlp_dp.pth` (DP-FedAvg, $\varepsilon=0.1457, \delta=10^{-5}$, 93.92% accuracy)
- `data/processed/scaler.pkl`, `feature_names.json`, `label_mapping.json`, `constant_features.json`

These models are registered under the `CICIDS2017_SUPERVISED_MODEL` family and are invoked only when the uploaded dataset satisfies their schema and preprocessing requirements.
