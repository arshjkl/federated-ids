"""
Phase 4: Semantic Feature Mapping Engine.

Maps uploaded dataset columns to the Canonical Network-Traffic Feature Schema
and the 70-feature CIC-IDS2017 model schema using:
  1. Exact canonical / CIC-IDS2017 name matches
  2. Normalized column name matches
  3. Schema-specific mapping profiles (CICFlowMeter, UNSW-NB15, NSL-KDD, TON_IoT)
  4. Token-expanded semantic similarity
  5. Data type compatibility checks
  6. Unit and scale plausibility checks (e.g., seconds vs microseconds for duration)
  7. Statistical distribution plausibility (e.g., port range 0..65535)
  8. Derived-feature dependency detection
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.inference.canonical_schema import (
    CANONICAL_BY_ID,
    CANONICAL_FEATURES,
    CICIDS70_FEATURES,
    CanonicalFeatureSpec,
)


TOKEN_SYNONYMS: Dict[str, str] = {
    "pkt": "packet",
    "pkts": "packets",
    "packets": "packet",
    "fwd": "forward",
    "bwd": "backward",
    "backward": "backward",
    "len": "length",
    "tot": "total",
    "totlen": "total_length",
    "cnt": "count",
    "avg": "mean",
    "byts": "bytes",
    "byt": "byte",
    "bytes": "byte",
    "seg": "segment",
    "win": "window",
    "dur": "duration",
    "dst": "destination",
    "dest": "destination",
    "src": "source",
    "hdr": "header",
    "var": "variance",
    "s": "per_second",
    "sec": "per_second",
}


# Derivability rules: canonical_id -> (human formula, list of required canonical_ids)
DERIVATION_DEPENDENCIES: Dict[str, Tuple[str, List[str]]] = {
    "min_packet_length": (
        "min(fwd_packet_length_min, bwd_packet_length_min)",
        ["fwd_packet_length_min", "bwd_packet_length_min"],
    ),
    "max_packet_length": (
        "max(fwd_packet_length_max, bwd_packet_length_max)",
        ["fwd_packet_length_max", "bwd_packet_length_max"],
    ),
    "packet_length_mean": (
        "(total_fwd_bytes + total_bwd_bytes) / max(total_fwd_packets + total_bwd_packets, 1)",
        ["total_fwd_bytes", "total_bwd_bytes", "total_fwd_packets", "total_bwd_packets"],
    ),
    "average_packet_size": (
        "(total_fwd_bytes + total_bwd_bytes) / max(total_fwd_packets + total_bwd_packets, 1)",
        ["total_fwd_bytes", "total_bwd_bytes", "total_fwd_packets", "total_bwd_packets"],
    ),
    "packet_length_variance": (
        "packet_length_std ** 2",
        ["packet_length_std"],
    ),
    "packet_length_std": (
        "sqrt(packet_length_variance)",
        ["packet_length_variance"],
    ),
    "fwd_packet_length_mean": (
        "total_fwd_bytes / max(total_fwd_packets, 1)",
        ["total_fwd_bytes", "total_fwd_packets"],
    ),
    "bwd_packet_length_mean": (
        "total_bwd_bytes / max(total_bwd_packets, 1)",
        ["total_bwd_bytes", "total_bwd_packets"],
    ),
    "avg_fwd_segment_size": (
        "fwd_packet_length_mean (or total_fwd_bytes / max(total_fwd_packets, 1))",
        ["total_fwd_bytes", "total_fwd_packets"],
    ),
    "avg_bwd_segment_size": (
        "bwd_packet_length_mean (or total_bwd_bytes / max(total_bwd_packets, 1))",
        ["total_bwd_bytes", "total_bwd_packets"],
    ),
    "flow_bytes_per_second": (
        "(total_fwd_bytes + total_bwd_bytes) / (flow_duration_seconds)",
        ["total_fwd_bytes", "total_bwd_bytes", "flow_duration"],
    ),
    "flow_packets_per_second": (
        "(total_fwd_packets + total_bwd_packets) / (flow_duration_seconds)",
        ["total_fwd_packets", "total_bwd_packets", "flow_duration"],
    ),
    "fwd_packets_per_second": (
        "total_fwd_packets / (flow_duration_seconds)",
        ["total_fwd_packets", "flow_duration"],
    ),
    "bwd_packets_per_second": (
        "total_bwd_packets / (flow_duration_seconds)",
        ["total_bwd_packets", "flow_duration"],
    ),
    "down_up_ratio": (
        "total_bwd_packets / max(total_fwd_packets, 1)",
        ["total_bwd_packets", "total_fwd_packets"],
    ),
    "fwd_header_length_dup": (
        "copy(fwd_header_length)",
        ["fwd_header_length"],
    ),
    "subflow_fwd_packets": (
        "copy(total_fwd_packets) [single-subflow approximation]",
        ["total_fwd_packets"],
    ),
    "subflow_fwd_bytes": (
        "copy(total_fwd_bytes) [single-subflow approximation]",
        ["total_fwd_bytes"],
    ),
    "subflow_bwd_packets": (
        "copy(total_bwd_packets) [single-subflow approximation]",
        ["total_bwd_packets"],
    ),
    "subflow_bwd_bytes": (
        "copy(total_bwd_bytes) [single-subflow approximation]",
        ["total_bwd_bytes"],
    ),
    "flow_iat_mean": (
        "flow_duration / max(total_fwd_packets + total_bwd_packets - 1, 1)",
        ["flow_duration", "total_fwd_packets", "total_bwd_packets"],
    ),
    "flow_packet_count": (
        "total_fwd_packets + total_bwd_packets",
        ["total_fwd_packets", "total_bwd_packets"],
    ),
    "flow_byte_count": (
        "total_fwd_bytes + total_bwd_bytes",
        ["total_fwd_bytes", "total_bwd_bytes"],
    ),
}


@dataclass
class FeatureMappingEntry:
    canonical_id: str
    canonical_name: str
    cicids_name: Optional[str]
    cicids_index: Optional[int]
    group: str
    is_core: bool
    source_column: Optional[str]
    mapping_type: str  # exact_match, strong_semantic_match, derived_feature, approximate_match, ambiguous_match, unavailable_feature, incompatible_feature
    confidence: float
    status: str  # mapped, derived, approximated, unmapped, incompatible
    unit_conversion: Optional[str] = None
    derivation_formula: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureMappingReport:
    schema_family: str
    total_uploaded_columns: int
    total_canonical_features: int
    total_cicids_features: int
    mapped_cicids_count: int
    exact_match_count: int
    semantic_match_count: int
    derived_feature_count: int
    approximate_match_count: int
    ambiguous_match_count: int
    unavailable_feature_count: int
    incompatible_feature_count: int
    direct_coverage_ratio: float
    effective_coverage_ratio: float  # (exact + semantic + derived + approximate) / 70
    core_coverage_ratio: float
    average_mapping_confidence: float
    mappings: List[FeatureMappingEntry]
    unmapped_uploaded_columns: List[str]
    group_coverage: Dict[str, Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticFeatureMapper:
    """Intelligent feature mapping engine with type, unit, and statistical validation."""

    @staticmethod
    def _normalize_identifier(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"[/\-\.\s]+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    @classmethod
    def _tokenize_and_expand(cls, name: str) -> List[str]:
        norm = cls._normalize_identifier(name)
        raw_tokens = [t for t in norm.split("_") if t]
        expanded: List[str] = []
        for tok in raw_tokens:
            mapped = TOKEN_SYNONYMS.get(tok, tok)
            expanded.extend(mapped.split("_"))
        return expanded

    @staticmethod
    def _validate_statistical_plausibility(
        spec: CanonicalFeatureSpec,
        series: pd.Series,
        col_type: str,
    ) -> Tuple[bool, Optional[str], float, str]:
        """
        Check type compatibility, unit scale, and statistical plausibility.
        Returns (is_compatible, unit_conversion, penalty_factor, note).
        """
        if spec.canonical_id in ("protocol", "encryption_strength", "client_agent_anomaly", "unusual_time_access"):
            return True, None, 1.0, f"Categorical/encoded security field '{spec.display_name}' compatible."

        if col_type in ("text", "identifier", "datetime", "label", "empty") or not pd.api.types.is_numeric_dtype(series):
            return False, None, 0.0, f"Column '{series.name}' has non-numeric type '{col_type}' incompatible with '{spec.display_name}'."

        vals = series.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return False, None, 0.0, f"Column '{series.name}' contains no finite numeric values."

        v_min = float(np.min(finite))
        v_max = float(np.max(finite))
        v_median = float(np.median(finite))

        # Port range check
        if spec.unit == "port":
            if v_min < -1 or v_max > 65535:
                return False, None, 0.0, f"Values [{v_min:.1f}, {v_max:.1f}] exceed valid TCP/UDP port range [0, 65535]."
            return True, None, 1.0, f"Port range [{v_min:.0f}, {v_max:.0f}] verified."

        # Non-negative check (allow -1 sentinel used by CICFlowMeter)
        if spec.non_negative and v_min < -1.0:
            neg_ratio = float(np.mean(finite < -1.0))
            if neg_ratio > 0.10:
                return False, None, 0.0, f"{neg_ratio*100:.1f}% of values are negative; '{spec.display_name}' must be non-negative."
            return True, None, 0.90, "Minor negative sentinel values present; clamped during preprocessing."

        # Unit check for flow_duration / IAT (seconds vs microseconds)
        if spec.canonical_id == "flow_duration":
            has_decimals = bool(np.any(np.mod(finite, 1.0) != 0))
            if has_decimals and v_max <= 7200.0 and v_median < 60.0:
                return True, "seconds_to_microseconds (* 1e6)", 0.95, "Detected duration in seconds; will convert to microseconds (* 1e6)."

        return True, None, 1.0, "Type and statistical range verified."

    def map_features(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        schema_family: str = "GENERIC_NETWORK_FLOW",
        manual_overrides: Optional[Dict[str, str]] = None,
    ) -> FeatureMappingReport:
        """
        Map uploaded DataFrame columns to Canonical and CIC-IDS2017 features.
        `manual_overrides` can map `canonical_id -> source_column` or `cicids_name -> source_column`.
        """
        manual_overrides = manual_overrides or {}
        uploaded_cols = list(df.columns)
        used_uploaded_cols: Set[str] = set()
        warnings: List[str] = []

        # Build lookup structures for uploaded columns
        exact_lookup: Dict[str, str] = {c.strip(): c for c in uploaded_cols}
        lower_lookup: Dict[str, str] = {c.strip().lower(): c for c in uploaded_cols}
        norm_lookup: Dict[str, str] = {self._normalize_identifier(c): c for c in uploaded_cols}
        token_lookup: Dict[str, Set[str]] = {
            c: set(self._tokenize_and_expand(c)) for c in uploaded_cols
        }

        mappings_by_canonical: Dict[str, FeatureMappingEntry] = {}

        # Check if dataset is an anonymized 70-column matrix (e.g., feature_1..feature_70)
        non_label_cols = [c for c in uploaded_cols if column_types.get(c) not in ("label", "identifier", "datetime")]
        is_anon_70 = (
            len(non_label_cols) == 70
            and sum(1 for c in non_label_cols if re.match(r"^(feature|feat|f|col|v|x)_?\d+$", c.strip().lower())) >= 60
        )
        if is_anon_70:
            warnings.append(
                "Detected 70 positional anonymized columns (feature_1..feature_70). "
                "Mapped positionally as approximate matches; verify whether data is pre-scaled."
            )

        # Pass 1: Direct matching (Manual overrides -> Exact -> Alias -> Token similarity)
        for spec in CANONICAL_FEATURES:
            override_col = manual_overrides.get(spec.canonical_id) or (
                manual_overrides.get(spec.cicids_name) if spec.cicids_name else None
            )
            matched_col: Optional[str] = None
            raw_match_type = "unavailable_feature"
            base_conf = 0.0
            match_reason = ""

            if override_col and override_col in df.columns:
                matched_col = override_col
                raw_match_type = "exact_match"
                base_conf = 1.0
                match_reason = f"Manually mapped to '{override_col}' by user override."
            elif spec.cicids_name and spec.cicids_name in exact_lookup:
                matched_col = exact_lookup[spec.cicids_name]
                raw_match_type = "exact_match"
                base_conf = 1.0
                match_reason = f"Exact CIC-IDS2017 column name match ('{matched_col}')."
            elif spec.canonical_id in lower_lookup:
                matched_col = lower_lookup[spec.canonical_id]
                raw_match_type = "exact_match"
                base_conf = 0.99
                match_reason = f"Exact canonical feature identifier match ('{matched_col}')."
            elif spec.cicids_name and spec.cicids_name.lower() in lower_lookup:
                matched_col = lower_lookup[spec.cicids_name.lower()]
                raw_match_type = "exact_match"
                base_conf = 0.98
                match_reason = f"Case-insensitive match to '{spec.cicids_name}' ('{matched_col}')."
            elif spec.cicids_name and self._normalize_identifier(spec.cicids_name) in norm_lookup:
                matched_col = norm_lookup[self._normalize_identifier(spec.cicids_name)]
                raw_match_type = "exact_match"
                base_conf = 0.96
                match_reason = f"Normalized identifier match to '{spec.cicids_name}' ('{matched_col}')."
            else:
                # Check aliases
                for alias in spec.aliases:
                    alias_low = alias.lower().strip()
                    alias_norm = self._normalize_identifier(alias)
                    if alias_low in lower_lookup and lower_lookup[alias_low] not in used_uploaded_cols:
                        matched_col = lower_lookup[alias_low]
                        raw_match_type = "strong_semantic_match"
                        base_conf = 0.92
                        match_reason = f"Known schema alias '{alias}' matched uploaded column '{matched_col}'."
                        break
                    if alias_norm in norm_lookup and norm_lookup[alias_norm] not in used_uploaded_cols:
                        matched_col = norm_lookup[alias_norm]
                        raw_match_type = "strong_semantic_match"
                        base_conf = 0.90
                        match_reason = f"Normalized schema alias '{alias}' matched uploaded column '{matched_col}'."
                        break

            # Token-similarity matching if not yet matched
            if matched_col is None and not is_anon_70:
                target_tokens = set(self._tokenize_and_expand(spec.cicids_name or spec.canonical_id))
                best_cand: Optional[str] = None
                best_jaccard = 0.0
                runner_up_jaccard = 0.0

                for cand_col in uploaded_cols:
                    if cand_col in used_uploaded_cols:
                        continue
                    if column_types.get(cand_col) in ("label", "identifier", "datetime", "empty"):
                        continue
                    cand_tokens = token_lookup.get(cand_col, set())
                    if not cand_tokens or not target_tokens:
                        continue
                    inter = len(target_tokens & cand_tokens)
                    union = len(target_tokens | cand_tokens)
                    jacc = inter / union if union > 0 else 0.0
                    # Directionality safeguard: don't map forward to backward!
                    if ("forward" in target_tokens and "backward" in cand_tokens) or (
                        "backward" in target_tokens and "forward" in cand_tokens
                    ):
                        continue
                    # Statistic safeguard: don't cross-map min/max/mean/std
                    stat_words = {"min", "max", "mean", "std", "variance", "total"}
                    t_stats = target_tokens & stat_words
                    c_stats = cand_tokens & stat_words
                    if t_stats and c_stats and t_stats != c_stats:
                        continue

                    if jacc > best_jaccard:
                        runner_up_jaccard = best_jaccard
                        best_jaccard = jacc
                        best_cand = cand_col
                    elif jacc > runner_up_jaccard:
                        runner_up_jaccard = jacc

                if best_cand and best_jaccard >= 0.72:
                    matched_col = best_cand
                    if best_jaccard >= 0.85 and (best_jaccard - runner_up_jaccard) >= 0.15:
                        raw_match_type = "strong_semantic_match"
                        base_conf = round(0.78 + 0.15 * best_jaccard, 2)
                        match_reason = f"High token/semantic similarity ({best_jaccard:.2f}) with '{best_cand}'."
                    else:
                        raw_match_type = "ambiguous_match"
                        base_conf = round(0.55 + 0.20 * best_jaccard, 2)
                        match_reason = f"Partial or ambiguous token overlap ({best_jaccard:.2f}) with '{best_cand}'."

            # Positional fallback for 70-column anonymized matrices (e.g. feature_1..feature_70)
            if matched_col is None and is_anon_70 and spec.cicids_index is not None:
                pos_col = non_label_cols[spec.cicids_index]
                matched_col = pos_col
                raw_match_type = "approximate_match"
                base_conf = 0.65
                match_reason = (
                    f"Positional mapping of '{pos_col}' (column #{spec.cicids_index + 1} of 70) "
                    f"to '{spec.cicids_name}'."
                )

            if matched_col is not None:
                c_type = column_types.get(matched_col, "numeric")
                is_compat, unit_conv, penalty, stat_note = self._validate_statistical_plausibility(
                    spec, df[matched_col], c_type
                )
                if not is_compat:
                    mappings_by_canonical[spec.canonical_id] = FeatureMappingEntry(
                        canonical_id=spec.canonical_id,
                        canonical_name=spec.display_name,
                        cicids_name=spec.cicids_name,
                        cicids_index=spec.cicids_index,
                        group=spec.group,
                        is_core=spec.is_core,
                        source_column=matched_col,
                        mapping_type="incompatible_feature",
                        confidence=0.0,
                        status="incompatible",
                        unit_conversion=None,
                        derivation_formula=None,
                        explanation=f"{match_reason} Rejected: {stat_note}",
                    )
                else:
                    used_uploaded_cols.add(matched_col)
                    final_conf = round(base_conf * penalty, 3)
                    status = "approximated" if raw_match_type in ("approximate_match", "ambiguous_match") else "mapped"
                    mappings_by_canonical[spec.canonical_id] = FeatureMappingEntry(
                        canonical_id=spec.canonical_id,
                        canonical_name=spec.display_name,
                        cicids_name=spec.cicids_name,
                        cicids_index=spec.cicids_index,
                        group=spec.group,
                        is_core=spec.is_core,
                        source_column=matched_col,
                        mapping_type=raw_match_type,
                        confidence=final_conf,
                        status=status,
                        unit_conversion=unit_conv,
                        derivation_formula=None,
                        explanation=f"{match_reason} {stat_note}".strip(),
                    )
            else:
                mappings_by_canonical[spec.canonical_id] = FeatureMappingEntry(
                    canonical_id=spec.canonical_id,
                    canonical_name=spec.display_name,
                    cicids_name=spec.cicids_name,
                    cicids_index=spec.cicids_index,
                    group=spec.group,
                    is_core=spec.is_core,
                    source_column=None,
                    mapping_type="unavailable_feature",
                    confidence=0.0,
                    status="unmapped",
                    unit_conversion=None,
                    derivation_formula=None,
                    explanation=f"No compatible column found for '{spec.display_name}'.",
                )

        # Pass 2: Check which unmapped features can be safely derived (two iterations for chained derivations)
        for _ in range(2):
            available_ids = {
                cid
                for cid, entry in mappings_by_canonical.items()
                if entry.status in ("mapped", "derived", "approximated")
            }
            for cid, entry in list(mappings_by_canonical.items()):
                if entry.status != "unmapped":
                    continue
                if cid in DERIVATION_DEPENDENCIES:
                    formula, req_ids = DERIVATION_DEPENDENCIES[cid]
                    can_derive = all(r in available_ids for r in req_ids)
                    if not can_derive and cid == "packet_length_mean" and "average_packet_size" in available_ids:
                        can_derive = True
                        formula = "copy(average_packet_size)"
                        req_ids = ["average_packet_size"]
                    elif not can_derive and cid == "average_packet_size" and "packet_length_mean" in available_ids:
                        can_derive = True
                        formula = "copy(packet_length_mean)"
                        req_ids = ["packet_length_mean"]
                    if can_derive:
                        parent_confs = [mappings_by_canonical[r].confidence for r in req_ids]
                        derived_conf = round(min(parent_confs) * 0.90, 3) if parent_confs else 0.80
                        spec = CANONICAL_BY_ID[cid]
                        mappings_by_canonical[cid] = FeatureMappingEntry(
                            canonical_id=cid,
                            canonical_name=spec.display_name,
                            cicids_name=spec.cicids_name,
                            cicids_index=spec.cicids_index,
                            group=spec.group,
                            is_core=spec.is_core,
                            source_column=f"DERIVED({', '.join(req_ids)})",
                            mapping_type="derived_feature",
                            confidence=derived_conf,
                            status="derived",
                            unit_conversion=None,
                            derivation_formula=formula,
                            explanation=f"Safely derived via formula: {formula}.",
                        )

        # Summarize over the 70 CIC-IDS2017 model features + additional canonical features
        all_entries = [mappings_by_canonical[f.canonical_id] for f in CANONICAL_FEATURES]
        cicids_entries = [e for e in all_entries if e.cicids_index is not None]

        exact_cnt = sum(1 for e in cicids_entries if e.mapping_type == "exact_match")
        sem_cnt = sum(1 for e in cicids_entries if e.mapping_type == "strong_semantic_match")
        der_cnt = sum(1 for e in cicids_entries if e.mapping_type == "derived_feature")
        approx_cnt = sum(1 for e in cicids_entries if e.mapping_type == "approximate_match")
        ambig_cnt = sum(1 for e in cicids_entries if e.mapping_type == "ambiguous_match")
        unavail_cnt = sum(1 for e in cicids_entries if e.mapping_type == "unavailable_feature")
        incompat_cnt = sum(1 for e in cicids_entries if e.mapping_type == "incompatible_feature")

        direct_mapped = exact_cnt + sem_cnt
        effective_mapped = exact_cnt + sem_cnt + der_cnt + approx_cnt + ambig_cnt

        direct_cov = round(direct_mapped / 70.0, 4)
        effective_cov = round(effective_mapped / 70.0, 4)

        core_entries = [e for e in cicids_entries if e.is_core]
        core_available = sum(1 for e in core_entries if e.status in ("mapped", "derived", "approximated"))
        core_cov = round(core_available / max(len(core_entries), 1), 4)

        active_confs = [e.confidence for e in all_entries if e.status in ("mapped", "derived", "approximated")]
        avg_conf = round(float(np.mean(active_confs)), 4) if active_confs else 0.0

        # Group coverage summary
        group_cov: Dict[str, Dict[str, Any]] = {}
        for spec in CANONICAL_FEATURES:
            g = spec.group
            if g not in group_cov:
                group_cov[g] = {"total": 0, "available": 0, "derived": 0, "missing": 0}
            group_cov[g]["total"] += 1
            entry = mappings_by_canonical[spec.canonical_id]
            if entry.status == "derived":
                group_cov[g]["available"] += 1
                group_cov[g]["derived"] += 1
            elif entry.status in ("mapped", "approximated"):
                group_cov[g]["available"] += 1
            else:
                group_cov[g]["missing"] += 1

        unmapped_uploaded = [
            c for c in uploaded_cols
            if c not in used_uploaded_cols and column_types.get(c) not in ("label",)
        ]

        if unavail_cnt > 0:
            missing_names = [e.cicids_name or e.canonical_name for e in cicids_entries if e.status == "unmapped"]
            warnings.append(
                f"{unavail_cnt} of 70 CIC-IDS2017 feature(s) unavailable after semantic mapping and derivation: "
                f"{', '.join(missing_names[:8])}{'...' if len(missing_names) > 8 else ''}"
            )

        return FeatureMappingReport(
            schema_family=schema_family,
            total_uploaded_columns=len(uploaded_cols),
            total_canonical_features=len(CANONICAL_FEATURES),
            total_cicids_features=70,
            mapped_cicids_count=effective_mapped,
            exact_match_count=exact_cnt,
            semantic_match_count=sem_cnt,
            derived_feature_count=der_cnt,
            approximate_match_count=approx_cnt,
            ambiguous_match_count=ambig_cnt,
            unavailable_feature_count=unavail_cnt,
            incompatible_feature_count=incompat_cnt,
            direct_coverage_ratio=direct_cov,
            effective_coverage_ratio=effective_cov,
            core_coverage_ratio=core_cov,
            average_mapping_confidence=avg_conf,
            mappings=all_entries,
            unmapped_uploaded_columns=unmapped_uploaded,
            group_coverage=group_cov,
            warnings=warnings,
        )


_feature_mapper: Optional[SemanticFeatureMapper] = None


def get_feature_mapper() -> SemanticFeatureMapper:
    global _feature_mapper
    if _feature_mapper is None:
        _feature_mapper = SemanticFeatureMapper()
    return _feature_mapper
