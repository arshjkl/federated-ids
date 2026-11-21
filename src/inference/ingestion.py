"""
Phase 2: Universal Data Ingestion Engine.

Handles CSV, TSV, JSON, JSONL/NDJSON, and Parquet files with automatic encoding
and delimiter detection, malformed row recovery, duplicate column disambiguation,
column type classification, identifier/metadata detection, and label detection.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


KNOWN_LABEL_NAMES = {
    "label", "labels", "class", "attack_cat", "attack_category",
    "attack_type", "attack_label", "attack", "target", "malicious",
    "is_attack", "is_malicious", "threat_type", "category", "binary_label",
    "attack_detected", "intrusion", "is_intrusion",
}

KNOWN_IDENTIFIER_PATTERNS = {
    "flow id", "flow_id", "flowid", "id", "uid", "record_id", "row_id",
    "src ip", "src_ip", "source ip", "source_ip", "srcip", "id.orig_h", "ip.src_host",
    "dst ip", "dst_ip", "destination ip", "destination_ip", "dstip", "id.resp_h", "ip.dst_host",
    "src mac", "dst mac", "mac", "eth.src", "eth.dst", "session_id",
}

KNOWN_DATETIME_PATTERNS = {
    "timestamp", "time", "date", "datetime", "start_time", "end_time",
    "stime", "ltime", "frame.time", "ts",
}


@dataclass
class IngestionMetadata:
    file_path: str
    filename: str
    file_size_bytes: int
    detected_format: str
    detected_encoding: str
    detected_delimiter: Optional[str]
    total_rows: int
    total_columns: int
    malformed_rows_skipped: int
    original_columns: List[str]
    normalized_columns: List[str]
    duplicate_columns: List[str]
    empty_columns: List[str]
    constant_columns: List[str]
    column_types: Dict[str, str]  # numeric, categorical, boolean, datetime, text, identifier, label
    identifier_columns: List[str]
    datetime_columns: List[str]
    candidate_label_columns: List[str]
    primary_label_column: Optional[str]
    is_labeled: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class UniversalIngestionEngine:
    """Robust universal dataset ingestion engine."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"}

    def detect_format_and_encoding(self, file_path: str | Path) -> Tuple[str, str, Optional[str]]:
        """Detect file format, character encoding, and delimiter."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return "parquet", "binary", None

        # Read first 16 KB to detect encoding and structure
        raw_head = path.read_bytes()[:16384]
        if not raw_head.strip():
            return "empty", "utf-8", ","

        encoding = "utf-8"
        if raw_head.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        else:
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    raw_head.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue

        text_head = raw_head.decode(encoding, errors="replace").lstrip()

        if suffix == ".json" or text_head.startswith("["):
            return "json", encoding, None
        if suffix in (".jsonl", ".ndjson") or (text_head.startswith("{") and "\n" in text_head):
            first_line = text_head.splitlines()[0].strip()
            if first_line.startswith("{") and first_line.endswith("}"):
                return "jsonl", encoding, None

        # Delimited text (CSV / TSV)
        delimiter = ","
        if suffix == ".tsv":
            delimiter = "\t"
        else:
            sample_lines = [line for line in text_head.splitlines()[:10] if line.strip()]
            if sample_lines:
                header_line = sample_lines[0]
                counts = {
                    ",": header_line.count(","),
                    "\t": header_line.count("\t"),
                    ";": header_line.count(";"),
                    "|": header_line.count("|"),
                }
                best_delim, best_count = max(counts.items(), key=lambda kv: kv[1])
                if best_count > 0:
                    delimiter = best_delim

        fmt = "tsv" if delimiter == "\t" else "csv"
        return fmt, encoding, delimiter

    @staticmethod
    def _disambiguate_columns(raw_columns: List[Any]) -> Tuple[List[str], List[str], List[str]]:
        """Strip whitespace/BOM and append .1, .2 to duplicate column names."""
        import re
        orig = [str(c) for c in raw_columns]
        cleaned = [str(c).strip().replace("\ufeff", "") for c in raw_columns]
        seen: Dict[str, int] = {}
        normalized: List[str] = []
        duplicates: List[str] = []
        cleaned_set = set(cleaned)

        for idx, col in enumerate(cleaned):
            if not col or col.startswith("Unnamed:"):
                col_name = f"unnamed_col_{idx}"
            else:
                col_name = col
            if col_name in seen:
                seen[col_name] += 1
                dedup_name = f"{col_name}.{seen[col_name]}"
                normalized.append(dedup_name)
                duplicates.append(dedup_name)
            else:
                seen[col_name] = 0
                normalized.append(col_name)
                # Detect columns already mangled by pandas read_csv (e.g. 'Fwd Header Length.1')
                m = re.match(r"^(.+)\.(\d+)$", col_name)
                if m and m.group(1) in cleaned_set:
                    duplicates.append(col_name)

        return orig, normalized, duplicates

    def load_dataframe(
        self,
        file_path: str | Path,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, IngestionMetadata]:
        """Load dataset into a DataFrame with full ingestion diagnostics."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        fmt, encoding, delimiter = self.detect_format_and_encoding(path)
        warnings: List[str] = []
        errors: List[str] = []
        malformed_skipped = 0

        if fmt == "empty" or file_size == 0:
            meta = IngestionMetadata(
                file_path=str(path),
                filename=path.name,
                file_size_bytes=file_size,
                detected_format="empty",
                detected_encoding=encoding,
                detected_delimiter=delimiter,
                total_rows=0,
                total_columns=0,
                malformed_rows_skipped=0,
                original_columns=[],
                normalized_columns=[],
                duplicate_columns=[],
                empty_columns=[],
                constant_columns=[],
                column_types={},
                identifier_columns=[],
                datetime_columns=[],
                candidate_label_columns=[],
                primary_label_column=None,
                is_labeled=False,
                warnings=["Dataset file is empty"],
                errors=["Dataset contains 0 rows"],
            )
            return pd.DataFrame(), meta

        try:
            if fmt == "parquet":
                df = pd.read_parquet(path)
                if max_rows is not None:
                    df = df.head(max_rows)
            elif fmt == "json":
                df = pd.read_json(path, encoding=encoding)
                if max_rows is not None:
                    df = df.head(max_rows)
            elif fmt == "jsonl":
                records = []
                with open(path, "r", encoding=encoding, errors="replace") as f:
                    for idx, line in enumerate(f):
                        if max_rows is not None and len(records) >= max_rows:
                            break
                        line_s = line.strip()
                        if not line_s:
                            continue
                        try:
                            records.append(json.loads(line_s))
                        except json.JSONDecodeError:
                            malformed_skipped += 1
                df = pd.DataFrame(records)
            else:
                # CSV or TSV: try fast C parser first; if malformed lines exist, recover with python engine & count skipped lines
                try:
                    df = pd.read_csv(
                        path,
                        sep=delimiter or ",",
                        encoding=encoding,
                        low_memory=False,
                        nrows=max_rows,
                        on_bad_lines="error",
                        engine="c",
                    )
                except Exception:
                    bad_lines: List[int] = []

                    def _bad_line_handler(bad_line: List[str]) -> None:
                        bad_lines.append(1)
                        return None

                    df = pd.read_csv(
                        path,
                        sep=delimiter or ",",
                        encoding=encoding,
                        nrows=max_rows,
                        on_bad_lines=_bad_line_handler,
                        engine="python",
                    )
                    malformed_skipped = len(bad_lines)
        except Exception as exc:
            # Fallback to python engine with permissive error handling
            try:
                df = pd.read_csv(
                    path,
                    sep=delimiter or ",",
                    encoding="latin-1",
                    on_bad_lines="skip",
                    engine="python",
                    nrows=max_rows,
                )
                warnings.append(f"Used fallback parser due to format issue: {exc}")
            except Exception as inner_exc:
                raise ValueError(f"Failed to parse dataset {path.name}: {inner_exc}") from inner_exc

        if malformed_skipped > 0:
            warnings.append(f"Skipped {malformed_skipped} malformed row(s) during parsing")

        orig_cols, norm_cols, dup_cols = self._disambiguate_columns(list(df.columns))
        df.columns = norm_cols

        if dup_cols:
            warnings.append(f"Disambiguated {len(dup_cols)} duplicate column name(s): {', '.join(dup_cols[:5])}")

        # Coerce numeric-looking string/object columns when >= 60% of non-null values parse as numbers
        inf_map = {
            "Infinity": np.inf, "+Infinity": np.inf, "-Infinity": -np.inf,
            "inf": np.inf, "+inf": np.inf, "-inf": -np.inf,
            "INF": np.inf, "-INF": -np.inf,
        }
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                col_lower = col.lower().strip()
                if col_lower in KNOWN_LABEL_NAMES or col_lower in KNOWN_IDENTIFIER_PATTERNS:
                    continue
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    sample_s = non_null.head(500).astype(str).str.strip()
                    coerced = pd.to_numeric(sample_s.replace(inf_map), errors="coerce")
                    valid_ratio = float(coerced.notna().mean())
                    if valid_ratio >= 0.60:
                        cleaned_s = df[col].where(df[col].notna(), np.nan)
                        str_s = cleaned_s.astype(str).str.strip().replace(
                            {**inf_map, "nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan}
                        )
                        full_coerced = pd.to_numeric(str_s, errors="coerce")
                        invalid_cnt = int(df[col].notna().sum() - full_coerced.notna().sum())
                        if invalid_cnt > 0:
                            warnings.append(f"Column '{col}' contained {invalid_cnt} non-numeric string value(s) coerced to NaN")
                        df[col] = full_coerced.astype(np.float64)

        empty_cols: List[str] = []
        constant_cols: List[str] = []
        column_types: Dict[str, str] = {}
        identifier_cols: List[str] = []
        datetime_cols: List[str] = []
        candidate_labels: List[str] = []

        for col in df.columns:
            col_lower = col.lower().strip()
            series = df[col]
            non_null_count = int(series.notna().sum())

            if non_null_count == 0:
                empty_cols.append(col)
                column_types[col] = "empty"
                continue

            nunique = int(series.dropna().nunique())
            if nunique <= 1 and len(df) > 1:
                constant_cols.append(col)

            # Check label candidacy (exclude measurement/feature suffixes and threat intel tags)
            is_feature_measurement = any(
                col_lower.endswith(sfx)
                for sfx in ("_votes", "_score", "_count", "_rate", "_ids", "_id", "_bytes", "_packets", "_pkts", "_duration", "_size", "_attempts", "_logins")
            ) or col_lower in ("threat_label", "threat_category", "threat_severity")
            if col_lower in KNOWN_LABEL_NAMES or (
                not is_feature_measurement
                and any(k in col_lower for k in ("label", "attack", "malicious", "class"))
                and nunique <= max(30, int(len(df) * 0.05))
            ):
                candidate_labels.append(col)

            # Check identifier candidacy
            if col_lower in KNOWN_IDENTIFIER_PATTERNS or (
                col_lower.endswith("_id") or col_lower.endswith("_ip")
            ):
                identifier_cols.append(col)
                column_types[col] = "identifier"
                continue

            # Check datetime candidacy
            if pd.api.types.is_datetime64_any_dtype(series) or col_lower in KNOWN_DATETIME_PATTERNS:
                datetime_cols.append(col)
                column_types[col] = "datetime"
                continue

            if pd.api.types.is_bool_dtype(series):
                column_types[col] = "boolean"
            elif pd.api.types.is_numeric_dtype(series):
                column_types[col] = "numeric"
            else:
                if nunique <= min(50, max(10, int(len(df) * 0.1))):
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "text"

        # Select single primary label (prefer exact KNOWN_LABEL_NAMES match) and only mark that column as 'label'
        primary_label: Optional[str] = None
        if candidate_labels:
            exact_matches = [c for c in candidate_labels if c.lower().strip() in KNOWN_LABEL_NAMES]
            primary_label = exact_matches[0] if exact_matches else candidate_labels[0]
            column_types[primary_label] = "label"

        if len(df) == 0:
            errors.append("Dataset has headers but 0 data rows")

        meta = IngestionMetadata(
            file_path=str(path),
            filename=path.name,
            file_size_bytes=file_size,
            detected_format=fmt,
            detected_encoding=encoding,
            detected_delimiter=delimiter,
            total_rows=len(df),
            total_columns=len(df.columns),
            malformed_rows_skipped=malformed_skipped,
            original_columns=orig_cols,
            normalized_columns=norm_cols,
            duplicate_columns=dup_cols,
            empty_columns=empty_cols,
            constant_columns=constant_cols,
            column_types=column_types,
            identifier_columns=identifier_cols,
            datetime_columns=datetime_cols,
            candidate_label_columns=candidate_labels,
            primary_label_column=primary_label,
            is_labeled=primary_label is not None,
            warnings=warnings,
            errors=errors,
        )
        return df, meta

    def iter_chunks(
        self,
        file_path: str | Path,
        chunk_size: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        """Stream a large CSV/TSV dataset in memory-bounded chunks."""
        path = Path(file_path)
        fmt, encoding, delimiter = self.detect_format_and_encoding(path)
        if fmt in ("csv", "tsv"):
            reader = pd.read_csv(
                path,
                sep=delimiter or ",",
                encoding=encoding,
                low_memory=False,
                chunksize=chunk_size,
                on_bad_lines="skip",
            )
            norm_cols: Optional[List[str]] = None
            for chunk in reader:
                if norm_cols is None:
                    _, norm_cols, _ = self._disambiguate_columns(list(chunk.columns))
                chunk.columns = norm_cols
                yield chunk
        else:
            df, _ = self.load_dataframe(path)
            for start in range(0, len(df), chunk_size):
                yield df.iloc[start : start + chunk_size]


_ingestion_engine: Optional[UniversalIngestionEngine] = None


def get_ingestion_engine() -> UniversalIngestionEngine:
    global _ingestion_engine
    if _ingestion_engine is None:
        _ingestion_engine = UniversalIngestionEngine()
    return _ingestion_engine
