# 18-Stage Traceable Adaptive Inference Pipeline (`src/inference/pipeline.py`)

Every dataset processed by the Universal Network Dataset Intelligence & Adaptive Intrusion Detection System flows through an explicit **18-stage traceable pipeline** (`AdaptiveInferencePipeline`). Every stage records its execution status, duration in milliseconds, summary text, and structured telemetry dictionary (`details`), which are returned by the API and rendered in the 18-stage stepper on the frontend Inference dashboard.

---

## Stage Reference

### Stage 1: Upload / Source Selection (`upload`)
- **Input**: File path (`csv`, `tsv`, `json`, `jsonl`, `parquet`) or selected Dataset ID.
- **Action**: Validates file existence, byte size, and readability.
- **Telemetry**: `file_name`, `file_size_bytes`, `dataset_id`.

### Stage 2: Universal Ingestion (`ingestion`)
- **Component**: `UniversalIngestionEngine.ingest_file()`
- **Action**: Auto-detects format, character encoding (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`), and delimiter (`,`, `\t`, `;`, `|`); recovers from malformed lines without crashing; coerces string-encoded numeric columns.
- **Telemetry**: `file_format`, `encoding`, `delimiter`, `total_rows`, `total_columns`, `malformed_rows_skipped`, `warnings`.

### Stage 3: Structural & Content Validation (`validation`)
- **Component**: `UniversalIngestionEngine.classify_columns()`
- **Action**: Identifies duplicate column names (including pandas `.1`/`.2` suffixes), all-null empty columns, single-value constant columns, `NaN` counts, `±Inf` counts, identifier/metadata columns (`Flow ID`, `Src IP`, `Timestamp`), and label columns (`Label`, `Attack`, `class`).
- **Telemetry**: `is_labeled`, `label_columns`, `identifier_columns`, `numeric_columns`, `categorical_columns`, `duplicate_columns`, `empty_columns`, `constant_columns`, `missing_cells`, `infinite_cells`.

### Stage 4: Dataset Profiling & Quality Intelligence (`profiling`)
- **Component**: `DatasetProfiler.profile()`
- **Action**: Computes memory footprint, duplicate row percentage, per-column distribution statistics (`min`, `max`, `mean`, `std`, `median`, `q25`, `q75`, `iqr`, 3-IQR outlier counts), and an overall **0–100 Data Quality Score**.
- **Telemetry**: `quality_score`, `memory_mb`, `duplicate_row_percentage`, `network_feature_ratio`.

### Stage 5: Schema & Dataset Family Detection (`schema_detection`)
- **Component**: `DatasetProfiler._detect_schema_family()`
- **Action**: Matches column patterns against signature markers for `CIC-IDS2017`, `CIC-IDS2018 / 2019`, `UNSW-NB15`, `NSL-KDD`, `TON_IoT`, `Edge-IIoTset`, or `Custom / Generic Network Traffic`.
- **Telemetry**: `detected_family`, `family_confidence`, `family_scores`, `evidence`.

### Stage 6: Semantic Feature Mapping (`feature_mapping`)
- **Component**: `SemanticFeatureMapper.map_columns()`
- **Action**: Maps uploaded columns to the 77 canonical network features across 8 groups using exact match, alias match, normalized identifier match, token-expanded Jaccard similarity, unit-scale detection (`seconds_to_microseconds`), statistical plausibility checks, and optional user manual overrides.
- **Telemetry**: `mapped_canonical_count`, `cicids_direct_coverage`, `average_confidence`, `mapping_type_counts`.

### Stage 7: Canonical Feature Construction (`canonical_construction`)
- **Component**: `SemanticFeatureMapper.build_canonical_dataframe()`
- **Action**: Projects mapped source columns into a canonical DataFrame, applying validated unit conversions (e.g. multiplying flow duration in seconds by $10^6$ to convert to microseconds) and recording per-feature provenance (`original`).
- **Telemetry**: `canonical_features_populated`, `unit_conversions_applied`.

### Stage 8: Safe Mathematical Feature Derivation (`feature_derivation`)
- **Component**: `FeatureDerivationEngine.derive()`
- **Action**: Safely derives mathematically determinable features (`min_packet_length`, `max_packet_length`, `packet_length_mean`, `average_packet_size`, `packet_length_variance`, `packet_length_std`, `fwd_packet_length_mean`, `bwd_packet_length_mean`, `avg_fwd_segment_size`, `avg_bwd_segment_size`, `flow_bytes_per_second`, `flow_packets_per_second`, `fwd_packets_per_second`, `bwd_packets_per_second`, `down_up_ratio`, `fwd_header_length_dup`, `subflow_*`, `flow_iat_mean`, `flow_packet_count`, `flow_byte_count`) while handling division-by-zero, `NaN`, and `±Inf`. Never fabricates TCP window sizes, flags, or IATs.
- **Telemetry**: `derived_count`, `derived_features`, `total_available_canonical`, `cicids_effective_coverage`.

### Stage 9: Model Compatibility & Detection Mode Routing (`model_routing`)
- **Component**: `ModelCompatibilityRouter.evaluate()`
- **Action**: Evaluates effective CIC-IDS2017 feature coverage, core 22-feature coverage, mapping confidence, data quality, and distribution alignment (`z <= 3.0`), then selects `supervised`, `hybrid`, `anomaly`, or `rule_based` mode.
- **Telemetry**: `selected_mode`, `selected_mode_label`, `supervised_eligible`, `feature_coverage`, `core_feature_coverage`, `distribution_alignment_score`, `weights`, `routing_reasons`, `limitations`.

### Stage 10: Adaptive Preprocessing & Matrix Alignment (`preprocessing`)
- **Component**: `FeatureDerivationEngine.prepare_cicids70_matrix()`
- **Action**: Sanitizes `±Inf` and `NaN` values. If `supervised_eligible` is true, aligns the 70 ordered CIC-IDS2017 features, neutralizes unavailable non-core features with `scaler.mean_[idx]` (so `StandardScaler` maps them to `0.0` z-score instead of $-\mu/\sigma$), and applies `StandardScaler.transform`.
- **Telemetry**: `canonical_shape`, `cicids70_prepared`, `neutral_baseline_features_count`, `scaler_applied`.

### Stage 11: Primary Detection Execution (`primary_detection`)
- **Component**: `SupervisedCICIDSDetector` (when eligible) or primary active detector (`UnsupervisedAnomalyDetector` / `NetworkRuleEngine`).
- **Action**: Runs PyTorch batch inference (`MLPIDS` or `ImprovedMLPIDS`) or primary anomaly/rule scoring.
- **Telemetry**: `executed`, `model_type`, `mean_probability`, `flagged_above_threshold`.

### Stage 12: Auxiliary Anomaly & Rule Evaluation (`fallback_detection`)
- **Component**: `UnsupervisedAnomalyDetector.score_dataset()` + `NetworkRuleEngine.evaluate_dataset()`
- **Action**: Runs multi-feature `IsolationForest` + Modified Z-Score anomaly detection on available canonical features and evaluates the 8 network intrusion signatures (`RULE_SYN_FLOOD`, `RULE_RST_SCAN`, `RULE_HIGH_PACKET_RATE`, `RULE_HIGH_BYTE_RATE`, `RULE_SENSITIVE_PORT_PROBE`, `RULE_JUMBO_PAYLOAD`, `RULE_ASYMMETRIC_EXFIL`, `RULE_DURATION_EXTREME`).
- **Telemetry**: `anomaly_mean_score`, `anomaly_high_count`, `rules_triggered_records`, `unique_rules_triggered`.

### Stage 13: Multi-Mode Risk & Confidence Scoring (`risk_scoring`)
- **Component**: `AdaptiveHybridScorer.combine()`
- **Action**: Combines active signals using the routed mode weights into a unified `risk_score` $\in [0, 1]$, assigns severity tiers (`informational`, `low`, `medium`, `high`, `critical`), and generates per-record evidence explanations.
- **Telemetry**: `mode`, `mean_risk_score`, `max_risk_score`, `flagged_records`, `severity_counts`.

### Stage 14: Ground-Truth Evaluation (`ground_truth_eval`)
- **Action**: If a ground-truth label column (`Label`, `Attack`, `class`) was detected during ingestion, computes post-inference accuracy, precision, recall, F1 score, and confusion matrix (`TP`, `FP`, `TN`, `FN`) without leaking labels into features. If unlabeled, marks stage as skipped/unlabeled.
- **Telemetry**: `has_ground_truth`, `label_column`, `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`.

### Stage 15: Post-Processing & Record Packaging (`post_processing`)
- **Action**: Packages per-record detection metadata, top canonical feature snapshots, triggered rule codes, and category distributions.
- **Telemetry**: `total_records_packaged`, `category_distribution`.

### Stage 16: Result Persistence (`storage`)
- **Component**: `InferenceRunner` + SQLite (`InferenceJob`, `DetectionResult`) + JSON artifact (`artifacts/inference_results/job_{id}.json`)
- **Action**: Persists job telemetry, stage reports, and detection records to SQLite and disk.
- **Telemetry**: `persisted_records`, `artifact_path`.

### Stage 17: Visualization Payload Assembly (`visualization`)
- **Action**: Builds ready-to-render histograms (`risk_score_histogram`), severity breakdowns, category distributions, and provenance charts for the React frontend.
- **Telemetry**: `charts_prepared`.

### Stage 18: Export & Inspection Readiness (`export_ready`)
- **Action**: Finalizes JSON and CSV export endpoints (`/api/inference/{job_id}/export?format=csv|json`) and record-level drill-down endpoints (`/api/inference/{job_id}/records/{record_index}`).
- **Telemetry**: `export_formats`, `total_pipeline_ms`.
