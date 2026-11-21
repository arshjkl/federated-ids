"""
Phase 11: Federated Learning Schema & Model Compatibility Validator.

Ensures federated clients cannot aggregate incompatible model architectures or
heterogeneous feature representations without explicit canonical alignment.
Preserves all existing FedAvg, FedProx, and DP-FedAvg implementations untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.inference.canonical_schema import CICIDS70_NAMES, CORE_CICIDS_FEATURES
from src.inference.feature_derivation import get_derivation_engine
from src.inference.feature_mapper import get_feature_mapper
from src.inference.ingestion import get_ingestion_engine
from src.inference.profiler import get_dataset_profiler


FEATURE_REPRESENTATION_VERSIONS = {
    "v1.0-cicids70": {
        "input_dim": 70,
        "features": CICIDS70_NAMES,
        "description": "Full 70-feature CIC-IDS2017 standardized feature representation",
        "min_effective_coverage": 0.85,
        "min_core_coverage": 0.85,
    },
    "v1.0-canonical-core22": {
        "input_dim": len(CORE_CICIDS_FEATURES),
        "features": sorted(CORE_CICIDS_FEATURES),
        "description": "Canonical 22 core flow, directional, packet-length, rate, and TCP-flag features",
        "min_effective_coverage": 0.80,
        "min_core_coverage": 0.80,
    },
}


@dataclass
class ClientCompatibilityReport:
    client_id: str
    is_compatible: bool
    feature_representation_version: str
    expected_input_dim: int
    actual_mapped_dim: int
    effective_coverage_ratio: float
    core_coverage_ratio: float
    detected_schema_family: str
    architecture_match: bool
    parameter_count_match: bool
    missing_features: List[str]
    derived_features: List[str]
    reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FederatedCompatibilityValidator:
    """
    Validates client dataset schemas and model parameter tensors before federated
    round participation and server aggregation.
    """

    def __init__(self, target_version: str = "v1.0-cicids70"):
        if target_version not in FEATURE_REPRESENTATION_VERSIONS:
            raise ValueError(f"Unsupported feature representation version: {target_version}")
        self.target_version = target_version
        self.spec = FEATURE_REPRESENTATION_VERSIONS[target_version]
        self.ingestion = get_ingestion_engine()
        self.profiler = get_dataset_profiler()
        self.mapper = get_feature_mapper()
        self.derivation = get_derivation_engine()

    def validate_client_dataset(
        self,
        client_id: str,
        df: pd.DataFrame,
        manual_overrides: Optional[Dict[str, str]] = None,
    ) -> ClientCompatibilityReport:
        """
        Check whether a client's local dataset can be mapped/derived into the
        federated experiment's required canonical feature representation.
        """
        errors: List[str] = []
        reasons: List[str] = []

        if len(df) == 0:
            return ClientCompatibilityReport(
                client_id=client_id,
                is_compatible=False,
                feature_representation_version=self.target_version,
                expected_input_dim=self.spec["input_dim"],
                actual_mapped_dim=0,
                effective_coverage_ratio=0.0,
                core_coverage_ratio=0.0,
                detected_schema_family="EMPTY",
                architecture_match=False,
                parameter_count_match=False,
                missing_features=list(self.spec["features"]),
                derived_features=[],
                reasons=[],
                errors=["Client dataset has 0 rows."],
            )

        # Infer column types
        col_types = {
            c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text")
            for c in df.columns
        }
        family, _, _, _ = self.profiler.detect_schema_family(list(df.columns), col_types)
        mapping = self.mapper.map_features(
            df, col_types, schema_family=family, manual_overrides=manual_overrides
        )
        deriv = self.derivation.derive_features(df, mapping)

        eff_cov = mapping.effective_coverage_ratio
        core_cov = mapping.core_coverage_ratio
        min_eff = self.spec["min_effective_coverage"]
        min_core = self.spec["min_core_coverage"]

        is_compat = eff_cov >= min_eff and core_cov >= min_core
        if is_compat:
            reasons.append(
                f"Client '{client_id}' schema ({family}) meets '{self.target_version}' requirements "
                f"(effective coverage {eff_cov*100:.1f}% >= {min_eff*100:.0f}%, "
                f"core coverage {core_cov*100:.1f}% >= {min_core*100:.0f}%)."
            )
        else:
            errors.append(
                f"Client '{client_id}' schema ({family}) is incompatible with '{self.target_version}': "
                f"effective coverage {eff_cov*100:.1f}% (required >= {min_eff*100:.0f}%), "
                f"core coverage {core_cov*100:.1f}% (required >= {min_core*100:.0f}%). "
                f"Missing {len(deriv.missing_cicids_features)} feature(s)."
            )

        return ClientCompatibilityReport(
            client_id=client_id,
            is_compatible=is_compat,
            feature_representation_version=self.target_version,
            expected_input_dim=self.spec["input_dim"],
            actual_mapped_dim=mapping.mapped_cicids_count,
            effective_coverage_ratio=eff_cov,
            core_coverage_ratio=core_cov,
            detected_schema_family=family,
            architecture_match=is_compat,
            parameter_count_match=is_compat,
            missing_features=deriv.missing_cicids_features,
            derived_features=deriv.derived_features_list,
            reasons=reasons,
            errors=errors,
        )

    @staticmethod
    def validate_parameter_aggregation(
        reference_parameters: List[np.ndarray],
        client_parameters_list: List[Tuple[str, List[np.ndarray]]],
    ) -> Tuple[bool, List[str]]:
        """
        Verify that all participating clients have identical tensor counts, shapes,
        and finite weights before executing FedAvg / FedProx aggregation.
        """
        errors: List[str] = []
        ref_shapes = [tuple(p.shape) for p in reference_parameters]

        if not client_parameters_list:
            return False, ["No client parameters provided for aggregation."]

        for client_id, params in client_parameters_list:
            if len(params) != len(ref_shapes):
                errors.append(
                    f"Client '{client_id}' layer count mismatch: expected {len(ref_shapes)} tensors, got {len(params)}."
                )
                continue
            for idx, (arr, expected_shape) in enumerate(zip(params, ref_shapes)):
                if tuple(arr.shape) != expected_shape:
                    errors.append(
                        f"Client '{client_id}' tensor #{idx} shape mismatch: expected {expected_shape}, got {tuple(arr.shape)}."
                    )
                if not np.all(np.isfinite(arr)):
                    errors.append(
                        f"Client '{client_id}' tensor #{idx} contains non-finite (NaN/Inf) weights."
                    )

        return len(errors) == 0, errors
