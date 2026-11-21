"""
Phase 3: Dataset Profiling and Schema Family Intelligence.

Analyzes any uploaded dataset before inference to produce comprehensive
statistical distributions, outlier counts, missing/infinite/duplicate rates,
network-flow family detection, and an explainable dataset quality score (0-100).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.inference.canonical_schema import DATASET_FAMILY_SIGNATURES
from src.inference.ingestion import IngestionMetadata, UniversalIngestionEngine, get_ingestion_engine


NETWORK_KEYWORDS = {
    "port", "pkt", "packet", "byte", "flow", "iat", "dur", "duration",
    "syn", "ack", "fin", "rst", "psh", "urg", "ece", "cwe", "win",
    "window", "seg", "hdr", "header", "fwd", "bwd", "subflow", "proto",
    "protocol", "ttl", "sbytes", "dbytes", "spkts", "dpkts", "rate",
    "load", "jitter", "sjit", "djit", "idle", "active", "src", "dst",
}


@dataclass
class DatasetProfile:
    filename: str
    file_size_bytes: int
    detected_format: str
    detected_encoding: str
    total_rows: int
    total_columns: int
    memory_footprint_bytes: int
    memory_footprint_mb: float
    is_labeled: bool
    primary_label_column: Optional[str]
    candidate_label_columns: List[str]
    label_distribution: Optional[Dict[str, int]]
    identifier_columns: List[str]
    datetime_columns: List[str]
    empty_columns: List[str]
    constant_columns: List[str]
    duplicate_columns: List[str]
    duplicate_rows: int
    duplicate_row_percentage: float
    total_missing_cells: int
    missing_value_percentage: float
    total_infinite_cells: int
    infinite_value_percentage: float
    valid_row_count: int
    invalid_row_count: int
    type_counts: Dict[str, int]
    column_types: Dict[str, str]
    column_stats: Dict[str, Dict[str, Any]]
    network_related_columns: List[str]
    detected_schema_family: str
    schema_family_confidence: float
    schema_family_scores: Dict[str, float]
    quality_score: float
    quality_breakdown: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetProfiler:
    """Profiles uploaded datasets and detects network traffic schema families."""

    def __init__(self, ingestion_engine: Optional[UniversalIngestionEngine] = None):
        self.ingestion = ingestion_engine or get_ingestion_engine()

    @staticmethod
    def _clean_col_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    def detect_schema_family(
        self,
        columns: List[str],
        column_types: Dict[str, str],
    ) -> Tuple[str, float, Dict[str, float], List[str]]:
        """Detect dataset family based on feature patterns and schema similarity."""
        cleaned_cols = {self._clean_col_key(c) for c in columns}
        token_cols = {re.sub(r"[^a-z0-9_/\.]+", " ", c).strip() for c in cleaned_cols}
        all_normalized = cleaned_cols | token_cols

        family_scores: Dict[str, float] = {}
        for family, sig_set in DATASET_FAMILY_SIGNATURES.items():
            matched = sum(1 for sig in sig_set if sig in all_normalized)
            score = matched / max(len(sig_set), 1)
            family_scores[family] = round(float(score), 4)

        # Detect network-related columns
        network_cols: List[str] = []
        for col in columns:
            c_low = col.lower()
            if any(kw in c_low for kw in NETWORK_KEYWORDS):
                network_cols.append(col)

        best_family, best_score = max(family_scores.items(), key=lambda kv: kv[1])

        # Check for generic anonymized feature_1..feature_N pattern
        anon_matches = sum(
            1 for c in columns if re.match(r"^(feature|feat|f|col|v|x)_?\d+$", c.strip().lower())
        )

        if best_score >= 0.45:
            detected_family = best_family
            confidence = min(1.0, round(best_score / 0.85, 3))
        elif len(network_cols) >= 5:
            detected_family = "GENERIC_NETWORK_FLOW"
            confidence = round(min(0.85, len(network_cols) / max(len(columns), 1) + 0.25), 3)
        elif anon_matches >= max(5, int(len(columns) * 0.6)):
            detected_family = "ANONYMIZED_FEATURE_MATRIX"
            confidence = round(anon_matches / max(len(columns), 1), 3)
        else:
            num_cols = sum(1 for t in column_types.values() if t == "numeric")
            detected_family = "CUSTOM_TABULAR" if num_cols > 0 else "UNSTRUCTURED_TABULAR"
            confidence = 0.5 if num_cols > 0 else 0.2

        return detected_family, confidence, family_scores, network_cols

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        meta: IngestionMetadata,
    ) -> DatasetProfile:
        """Compute full statistical and structural profile for an ingested DataFrame."""
        n_rows = len(df)
        n_cols = len(df.columns)
        total_cells = max(n_rows * n_cols, 1)

        mem_bytes = int(df.memory_usage(deep=True).sum()) if n_rows > 0 else 0
        mem_mb = round(mem_bytes / (1024 * 1024), 3)

        # Duplicate rows (cap check on huge dataframes to avoid excessive memory copy)
        if n_rows > 0 and n_rows <= 150000:
            dup_rows = int(df.duplicated().sum())
        elif n_rows > 150000:
            sample_dup = int(df.head(50000).duplicated().sum())
            dup_rows = int(sample_dup * (n_rows / 50000))
        else:
            dup_rows = 0
        dup_row_pct = round((dup_rows / max(n_rows, 1)) * 100.0, 2)

        # Label distribution
        label_dist: Optional[Dict[str, int]] = None
        if meta.primary_label_column and meta.primary_label_column in df.columns:
            vc = df[meta.primary_label_column].value_counts(dropna=False).head(30).to_dict()
            label_dist = {str(k): int(v) for k, v in vc.items()}

        column_stats: Dict[str, Dict[str, Any]] = {}
        total_missing = 0
        total_infinite = 0
        row_has_issue = np.zeros(n_rows, dtype=bool) if n_rows > 0 else np.array([], dtype=bool)

        for col in df.columns:
            series = df[col]
            c_type = meta.column_types.get(col, "text")
            null_mask = series.isna()
            null_cnt = int(null_mask.sum())
            null_pct = round((null_cnt / max(n_rows, 1)) * 100.0, 2)
            total_missing += null_cnt

            if n_rows > 0 and c_type not in ("label", "identifier", "datetime"):
                row_has_issue |= null_mask.to_numpy(dtype=bool)

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                vals = series.to_numpy(dtype=np.float64, na_value=np.nan)
                inf_mask = np.isinf(vals)
                inf_cnt = int(inf_mask.sum())
                inf_pct = round((inf_cnt / max(n_rows, 1)) * 100.0, 2)
                total_infinite += inf_cnt
                if n_rows > 0:
                    row_has_issue |= inf_mask

                finite_vals = vals[np.isfinite(vals)]
                if len(finite_vals) > 0:
                    v_min = float(np.min(finite_vals))
                    v_max = float(np.max(finite_vals))
                    v_mean = float(np.mean(finite_vals))
                    v_std = float(np.std(finite_vals))
                    q25, median, q75 = [float(x) for x in np.percentile(finite_vals, [25, 50, 75])]
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_fence = q25 - 3.0 * iqr
                        upper_fence = q75 + 3.0 * iqr
                        outliers = int(np.sum((finite_vals < lower_fence) | (finite_vals > upper_fence)))
                    elif v_std > 0:
                        outliers = int(np.sum(np.abs((finite_vals - v_mean) / v_std) > 3.5))
                    else:
                        outliers = 0
                    outlier_pct = round((outliers / len(finite_vals)) * 100.0, 2)
                    neg_cnt = int(np.sum(finite_vals < 0))
                    zero_cnt = int(np.sum(finite_vals == 0))
                    nunique = int(pd.Series(finite_vals).nunique())
                else:
                    v_min = v_max = v_mean = v_std = q25 = median = q75 = iqr = None  # type: ignore[assignment]
                    outliers = 0
                    outlier_pct = 0.0
                    neg_cnt = 0
                    zero_cnt = 0
                    nunique = 0

                column_stats[col] = {
                    "type": str(series.dtype),
                    "semantic_type": c_type,
                    "min": v_min,
                    "max": v_max,
                    "mean": v_mean,
                    "std": v_std,
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "null_count": null_cnt,
                    "null_percentage": null_pct,
                    "inf_count": inf_cnt,
                    "inf_percentage": inf_pct,
                    "unique_count": nunique,
                    "outlier_count": outliers,
                    "outlier_percentage": outlier_pct,
                    "negative_count": neg_cnt,
                    "zero_count": zero_cnt,
                }
            else:
                non_null = series.dropna()
                nunique = int(non_null.nunique())
                top_vals = {str(k): int(v) for k, v in non_null.value_counts().head(5).to_dict().items()}
                column_stats[col] = {
                    "type": str(series.dtype),
                    "semantic_type": c_type,
                    "null_count": null_cnt,
                    "null_percentage": null_pct,
                    "inf_count": 0,
                    "inf_percentage": 0.0,
                    "unique_count": nunique,
                    "top_values": top_vals,
                }

        missing_pct = round((total_missing / total_cells) * 100.0, 3)
        inf_pct_total = round((total_infinite / total_cells) * 100.0, 3)

        invalid_rows = int(row_has_issue.sum()) if n_rows > 0 else 0
        valid_rows = max(0, n_rows - invalid_rows)

        type_counts: Dict[str, int] = {}
        for t in meta.column_types.values():
            type_counts[t] = type_counts.get(t, 0) + 1

        family, family_conf, family_scores, network_cols = self.detect_schema_family(
            list(df.columns), meta.column_types
        )

        # Compute transparent Dataset Quality Score (0-100)
        if n_rows == 0 or n_cols == 0:
            quality_score = 0.0
            quality_breakdown = {
                "completeness": 0.0,
                "validity": 0.0,
                "uniqueness": 0.0,
                "feature_variance": 0.0,
                "network_relevance": 0.0,
            }
        else:
            completeness = max(0.0, 100.0 - missing_pct * 2.0)
            validity = max(0.0, 100.0 - inf_pct_total * 5.0 - (meta.malformed_rows_skipped / max(n_rows, 1)) * 100.0)
            uniqueness = max(0.0, 100.0 - dup_row_pct * 0.5)
            non_const_ratio = 1.0 - (len(meta.constant_columns) + len(meta.empty_columns)) / max(n_cols, 1)
            feature_variance = round(max(0.0, min(100.0, non_const_ratio * 100.0)), 1)
            num_feature_cols = sum(1 for t in meta.column_types.values() if t == "numeric")
            net_ratio = len(network_cols) / max(num_feature_cols, 1) if num_feature_cols > 0 else 0.0
            network_relevance = round(min(100.0, max(family_conf * 100.0, net_ratio * 100.0, 40.0 if num_feature_cols >= 4 else 10.0)), 1)

            quality_breakdown = {
                "completeness": round(completeness, 1),
                "validity": round(validity, 1),
                "uniqueness": round(uniqueness, 1),
                "feature_variance": feature_variance,
                "network_relevance": network_relevance,
            }
            quality_score = round(
                0.30 * quality_breakdown["completeness"]
                + 0.25 * quality_breakdown["validity"]
                + 0.15 * quality_breakdown["uniqueness"]
                + 0.15 * quality_breakdown["feature_variance"]
                + 0.15 * quality_breakdown["network_relevance"],
                1,
            )

        warnings = list(meta.warnings)
        errors = list(meta.errors)
        if total_missing > 0:
            warnings.append(f"Dataset contains {total_missing} missing value(s) ({missing_pct:.2f}% of cells)")
        if total_infinite > 0:
            warnings.append(f"Dataset contains {total_infinite} infinite value(s) ({inf_pct_total:.2f}% of cells)")
        if len(meta.constant_columns) > 0:
            warnings.append(f"{len(meta.constant_columns)} constant feature(s) with zero variance detected")

        return DatasetProfile(
            filename=meta.filename,
            file_size_bytes=meta.file_size_bytes,
            detected_format=meta.detected_format,
            detected_encoding=meta.detected_encoding,
            total_rows=n_rows,
            total_columns=n_cols,
            memory_footprint_bytes=mem_bytes,
            memory_footprint_mb=mem_mb,
            is_labeled=meta.is_labeled,
            primary_label_column=meta.primary_label_column,
            candidate_label_columns=meta.candidate_label_columns,
            label_distribution=label_dist,
            identifier_columns=meta.identifier_columns,
            datetime_columns=meta.datetime_columns,
            empty_columns=meta.empty_columns,
            constant_columns=meta.constant_columns,
            duplicate_columns=meta.duplicate_columns,
            duplicate_rows=dup_rows,
            duplicate_row_percentage=dup_row_pct,
            total_missing_cells=total_missing,
            missing_value_percentage=missing_pct,
            total_infinite_cells=total_infinite,
            infinite_value_percentage=inf_pct_total,
            valid_row_count=valid_rows,
            invalid_row_count=invalid_rows,
            type_counts=type_counts,
            column_types=meta.column_types,
            column_stats=column_stats,
            network_related_columns=network_cols,
            detected_schema_family=family,
            schema_family_confidence=family_conf,
            schema_family_scores=family_scores,
            quality_score=quality_score,
            quality_breakdown=quality_breakdown,
            warnings=warnings,
            errors=errors,
        )

    def profile_file(
        self,
        file_path: str,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, IngestionMetadata, DatasetProfile]:
        df, meta = self.ingestion.load_dataframe(file_path, max_rows=max_rows)
        profile = self.profile_dataframe(df, meta)
        return df, meta, profile


_profiler: Optional[DatasetProfiler] = None


def get_dataset_profiler() -> DatasetProfiler:
    global _profiler
    if _profiler is None:
        _profiler = DatasetProfiler()
    return _profiler
