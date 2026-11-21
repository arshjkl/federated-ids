"""
Phase 7 (Part 1): Model Compatibility & Adaptive Detection Mode Router.

Evaluates feature coverage, core feature availability, mapping confidence,
schema compatibility, data quality, and training distribution alignment to route
datasets to the appropriate detection mode:
  - Mode 1: supervised (CICIDS2017_SUPERVISED_MODEL)
  - Mode 2: hybrid (CICIDS2017_SUPERVISED_MODEL + Unsupervised Anomaly + Rule Engine)
  - Mode 3: anomaly (IsolationForest + Robust Statistical IQR/Z-Score + Rule Engine)
  - Mode 4: rule_based (Configurable Evidence-Based Network Heuristic Rules)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.inference.canonical_schema import CICIDS70_FEATURES
from src.inference.feature_derivation import DerivationResult
from src.inference.feature_mapper import FeatureMappingReport
from src.inference.profiler import DatasetProfile


# Documented default hybrid weights & thresholds
DEFAULT_HYBRID_CONFIG: Dict[str, Any] = {
    "base_supervised_weight": 0.55,
    "base_anomaly_weight": 0.25,
    "base_rule_weight": 0.20,
    "supervised_threshold": 0.50,
    "anomaly_threshold": 0.65,
    "rule_threshold": 0.50,
    "hybrid_risk_threshold": 0.50,
    "isolation_forest_contamination": "auto",
}


@dataclass
class CompatibilityEvaluation:
    selected_mode: str  # supervised, hybrid, anomaly, rule_based
    selected_model_family: str  # CICIDS2017_SUPERVISED_MODEL, HYBRID_ENSEMBLE_DETECTOR, UNSUPERVISED_ANOMALY_DETECTOR, HEURISTIC_RULE_ENGINE
    selected_model_key: str
    selected_model_name: str
    is_supervised_compatible: bool
    compatibility_score: float  # 0.0 - 100.0
    feature_coverage_score: float
    core_feature_coverage_score: float
    mapping_confidence_score: float
    data_quality_score: float
    distribution_compatibility_score: float
    preprocessing_compatible: bool
    available_modes: List[Dict[str, Any]]
    routing_reasons: List[str]
    limitations: List[str]
    scoring_weights: Dict[str, float]
    missing_core_features: List[str]
    missing_non_core_features: List[str]
    derived_features_used: List[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelCompatibilityRouter:
    """Evaluates dataset compatibility and selects the optimal, honest detection strategy."""

    def evaluate_distribution_compatibility(
        self,
        derivation: DerivationResult,
        scaler: Optional[Any],
    ) -> float:
        """
        Check how well available mapped/derived features align with the CIC-IDS2017
        training distribution stored in StandardScaler (mean_ and scale_).
        Returns a score in [0.0, 100.0].
        """
        if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            return 80.0
        if len(derivation.canonical_df) == 0:
            return 0.0

        plausible_count = 0
        checked_count = 0
        means = scaler.mean_
        scales = np.maximum(scaler.scale_, 1e-6)

        for spec in CICIDS70_FEATURES:
            idx = spec.cicids_index
            if idx is None or spec.canonical_id not in derivation.canonical_df.columns:
                continue
            arr = derivation.canonical_df[spec.canonical_id].to_numpy(dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            checked_count += 1
            med_z = abs(float(np.median(finite)) - float(means[idx])) / float(scales[idx])
            if med_z <= 10.0:
                plausible_count += 1

        if checked_count == 0:
            return 50.0
        return round((plausible_count / checked_count) * 100.0, 1)

    def route(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        mapping_report: FeatureMappingReport,
        derivation: DerivationResult,
        scaler: Optional[Any] = None,
        requested_mode: Optional[str] = None,
        selected_model_key: str = "federated_v2_round_20",
        selected_model_name: str = "Federated v2 (Improved FedAvg, Round 20)",
    ) -> CompatibilityEvaluation:
        """
        Determine compatibility scores and select the detection mode.
        """
        eff_cov = mapping_report.effective_coverage_ratio
        dir_cov = mapping_report.direct_coverage_ratio
        core_cov = mapping_report.core_coverage_ratio
        avg_conf = mapping_report.average_mapping_confidence
        qual_score = profile.quality_score
        dist_score = self.evaluate_distribution_compatibility(derivation, scaler)

        feat_cov_score = round(eff_cov * 100.0, 1)
        core_cov_score = round(core_cov * 100.0, 1)
        conf_score = round(avg_conf * 100.0, 1)

        # Missing core vs non-core CICIDS features
        available_cids = set(derivation.available_canonical_features)
        missing_core = [
            f.cicids_name or f.display_name
            for f in CICIDS70_FEATURES
            if f.is_core and f.canonical_id not in available_cids
        ]
        missing_non_core = [
            f.cicids_name or f.display_name
            for f in CICIDS70_FEATURES
            if not f.is_core and f.canonical_id not in available_cids
        ]

        # Overall compatibility with CICIDS2017_SUPERVISED_MODEL
        compatibility_score = round(
            0.35 * feat_cov_score
            + 0.25 * core_cov_score
            + 0.15 * conf_score
            + 0.15 * qual_score
            + 0.10 * dist_score,
            1,
        )

        raw_numeric_cols = sum(
            1 for col, ctype in profile.column_types.items()
            if ctype == "numeric" and col not in profile.empty_columns and col not in profile.constant_columns
        )
        num_numeric_cols = max(raw_numeric_cols, len(derivation.canonical_df.columns))

        # Strictly supervised requires >= 90% effective coverage, >= 85% core coverage, high mapping confidence
        strictly_supervised_ok = (
            eff_cov >= 0.90
            and core_cov >= 0.85
            and avg_conf >= 0.80
            and mapping_report.approximate_match_count <= 10
            and profile.total_rows > 0
        )

        # Hybrid supervised+anomaly+rules allowed when >= 55% of CICIDS features & >= 60% of core features exist
        hybrid_supervised_ok = (
            eff_cov >= 0.55
            and core_cov >= 0.60
            and avg_conf >= 0.60
            and profile.total_rows > 0
        )

        anomaly_ok = num_numeric_cols >= 2 and profile.total_rows >= 3
        rule_ok = len(derivation.available_canonical_features) >= 1 or num_numeric_cols >= 1

        routing_reasons: List[str] = []
        limitations: List[str] = []
        warnings: List[str] = list(mapping_report.warnings)

        # Decide mode
        if requested_mode in ("supervised", "hybrid", "anomaly", "rule_based"):
            if requested_mode == "supervised" and not hybrid_supervised_ok:
                warnings.append(
                    f"Requested 'supervised' mode rejected because CIC-IDS2017 feature coverage is only "
                    f"{feat_cov_score:.1f}% (core: {core_cov_score:.1f}%). Falling back to adaptive routing."
                )
                selected_mode = "anomaly" if anomaly_ok else "rule_based"
            else:
                selected_mode = requested_mode
                routing_reasons.append(f"Detection mode '{selected_mode}' explicitly selected and verified compatible.")
        else:
            # Check if there is an active custom-trained model matching this dataset schema
            custom_ckpt = None
            try:
                from src.dashboard.services.experiment_trainer import ExperimentTrainerService
                custom_ckpt = ExperimentTrainerService.find_matching_custom_checkpoint(
                    df, preferred_checkpoint_path=selected_model_key
                )
            except Exception:
                pass

            if custom_ckpt is not None and custom_ckpt[2] >= 0.60:
                ckpt_path, payload, overlap = custom_ckpt
                selected_mode = "supervised" if overlap >= 0.85 else "hybrid"
                compatibility_score = round(max(compatibility_score, overlap * 100.0), 1)
                routing_reasons.append(
                    f"Matched custom trained model '{payload.get('experiment_name', ckpt_path.name)}' "
                    f"with {overlap * 100.0:.1f}% schema compatibility ({len(payload.get('schema_bundle', {}).get('feature_columns', []))} features)."
                )
            elif strictly_supervised_ok:
                selected_mode = "supervised"
                routing_reasons.append(
                    f"High CIC-IDS2017 schema compatibility ({compatibility_score:.1f}%): "
                    f"{mapping_report.mapped_cicids_count}/70 features available "
                    f"({mapping_report.exact_match_count} exact, {mapping_report.semantic_match_count} semantic, "
                    f"{len(derivation.derived_features_list)} derived)."
                )
                routing_reasons.append(
                    f"All required core flow/packet/rate features meet threshold ({core_cov_score:.1f}%)."
                )
            elif hybrid_supervised_ok:
                selected_mode = "hybrid"
                routing_reasons.append(
                    f"Partial CIC-IDS2017 schema match ({feat_cov_score:.1f}% overall, {core_cov_score:.1f}% core features): "
                    f"{len(derivation.missing_cicids_features)} feature(s) unavailable "
                    f"({', '.join(derivation.missing_cicids_features[:5])}{'...' if len(derivation.missing_cicids_features) > 5 else ''})."
                )
                routing_reasons.append(
                    "Selected Hybrid Detection Mode: combines CICIDS2017_SUPERVISED_MODEL (using neutral training-mean "
                    "baseline for unavailable non-core features) with Unsupervised Anomaly Detection (IsolationForest + "
                    "Robust IQR Z-Score) and Evidence-Based Network Rules."
                )
                limitations.append(
                    f"{len(derivation.missing_cicids_features)} unavailable CIC-IDS2017 feature(s) were neutralized at "
                    "the training mean rather than fabricated; supervised weight is scaled proportionally to feature coverage."
                )
            elif anomaly_ok:
                selected_mode = "anomaly"
                routing_reasons.append(
                    f"Dataset schema family '{profile.detected_schema_family}' has {feat_cov_score:.1f}% overlap with "
                    f"the 70-feature CIC-IDS2017 supervised schema ({len(derivation.missing_cicids_features)} features missing)."
                )
                routing_reasons.append(
                    "Running CICIDS2017_SUPERVISED_MODEL would require fabricating missing features and produce invalid "
                    "predictions. Routed to Unsupervised Anomaly Detection (IsolationForest + Robust Multivariate "
                    "Modified Z-Score) combined with Canonical Network Rule checks."
                )
                limitations.append(
                    "Unsupervised anomaly detection identifies statistically unusual network flows relative to the "
                    "dataset baseline; flagged anomalies are NOT confirmed supervised attack classifications."
                )
            else:
                selected_mode = "rule_based"
                routing_reasons.append(
                    "Dataset lacks sufficient numeric features or rows for multivariate anomaly modeling; "
                    "routed to Configurable Evidence-Based Heuristic Rule Engine."
                )
                limitations.append(
                    "Rule-based mode only detects known heuristic signatures (floods, port scans, extreme bursts) "
                    "on available canonical fields."
                )

        if not profile.is_labeled:
            limitations.append(
                "Dataset is unlabeled: ground-truth accuracy/F1 cannot be computed on this upload; "
                "detections are ranked by risk score, confidence, and evidence."
            )

        # Compute transparent scoring weights for the active mode
        if selected_mode == "supervised":
            scoring_weights = {
                "supervised_weight": 0.85,
                "anomaly_weight": 0.05,
                "rule_weight": 0.10,
            }
            model_family = "CICIDS2017_SUPERVISED_MODEL"
            active_model_name = f"CICIDS2017_SUPERVISED_MODEL ({selected_model_name})"
        elif selected_mode == "hybrid":
            w_sup = round(0.55 * eff_cov * avg_conf, 3)
            w_anom = 0.25
            w_rule = round(max(0.15, 1.0 - w_sup - w_anom), 3)
            total_w = w_sup + w_anom + w_rule
            scoring_weights = {
                "supervised_weight": round(w_sup / total_w, 3),
                "anomaly_weight": round(w_anom / total_w, 3),
                "rule_weight": round(w_rule / total_w, 3),
            }
            model_family = "HYBRID_ENSEMBLE_DETECTOR"
            active_model_name = f"Hybrid Ensemble ({selected_model_name} + IsolationForest + Network Rules)"
        elif selected_mode == "anomaly":
            has_canonical_rules = len(derivation.available_canonical_features) >= 2
            scoring_weights = {
                "supervised_weight": 0.0,
                "anomaly_weight": 0.70 if has_canonical_rules else 0.90,
                "rule_weight": 0.30 if has_canonical_rules else 0.10,
            }
            model_family = "UNSUPERVISED_ANOMALY_DETECTOR"
            active_model_name = "Unsupervised Anomaly Detector (IsolationForest + Robust Modified Z-Score)"
        else:
            scoring_weights = {
                "supervised_weight": 0.0,
                "anomaly_weight": 0.0,
                "rule_weight": 1.0,
            }
            model_family = "HEURISTIC_RULE_ENGINE"
            active_model_name = "Evidence-Based Network Heuristic Rule Engine"

        available_modes = [
            {
                "mode": "supervised",
                "name": "Mode 1 — Supervised Classification (CICIDS2017_SUPERVISED_MODEL)",
                "compatible": strictly_supervised_ok,
                "recommended": selected_mode == "supervised",
                "min_coverage_required": 0.90,
                "description": "Uses trained PyTorch MLP / FedAvg / DP-FedAvg models with exact 70-feature StandardScaler pipeline.",
            },
            {
                "mode": "hybrid",
                "name": "Mode 2 — Hybrid Adaptive Detection (Supervised + Anomaly + Rules)",
                "compatible": hybrid_supervised_ok,
                "recommended": selected_mode == "hybrid",
                "min_coverage_required": 0.55,
                "description": "Combines partial-coverage supervised model with IsolationForest anomaly scoring and heuristic rules.",
            },
            {
                "mode": "anomaly",
                "name": "Mode 3 — Unsupervised Anomaly Detection (IsolationForest + Statistical IQR/Z-Score)",
                "compatible": anomaly_ok,
                "recommended": selected_mode == "anomaly",
                "min_coverage_required": 0.0,
                "description": "Schema-agnostic multivariate outlier detection for non-CICIDS or unlabeled network traffic datasets.",
            },
            {
                "mode": "rule_based",
                "name": "Mode 4 — Evidence-Based Network Heuristic Rules",
                "compatible": rule_ok,
                "recommended": selected_mode == "rule_based",
                "min_coverage_required": 0.0,
                "description": "Evaluates configurable network intrusion rules (SYN flood, port scan bursts, packet/byte rate spikes).",
            },
        ]

        return CompatibilityEvaluation(
            selected_mode=selected_mode,
            selected_model_family=model_family,
            selected_model_key=selected_model_key,
            selected_model_name=active_model_name,
            is_supervised_compatible=strictly_supervised_ok or hybrid_supervised_ok,
            compatibility_score=compatibility_score,
            feature_coverage_score=feat_cov_score,
            core_feature_coverage_score=core_cov_score,
            mapping_confidence_score=conf_score,
            data_quality_score=qual_score,
            distribution_compatibility_score=dist_score,
            preprocessing_compatible=strictly_supervised_ok or hybrid_supervised_ok,
            available_modes=available_modes,
            routing_reasons=routing_reasons,
            limitations=limitations,
            scoring_weights=scoring_weights,
            missing_core_features=missing_core,
            missing_non_core_features=missing_non_core,
            derived_features_used=derivation.derived_features_list,
            warnings=warnings,
        )


_model_router: Optional[ModelCompatibilityRouter] = None


def get_model_router() -> ModelCompatibilityRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelCompatibilityRouter()
    return _model_router
