"""
Phase 6: Safe Feature Derivation Engine.

Derives mathematically determinable canonical network features while protecting
against division by zero, NaN, infinity, negative counts, and unit mismatches.
Never fabricates non-derivable measurements (such as TCP initial windows or flags)
and maintains complete per-feature provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.inference.canonical_schema import (
    CANONICAL_BY_ID,
    CICIDS70_FEATURES,
)
from src.inference.feature_mapper import FeatureMappingEntry, FeatureMappingReport


@dataclass
class FeatureProvenance:
    canonical_id: str
    cicids_name: Optional[str]
    provenance_type: str  # original, derived, approximated, neutral_training_mean_baseline, missing
    source_detail: str
    confidence: float
    finite_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivationResult:
    canonical_df: pd.DataFrame
    provenance: Dict[str, FeatureProvenance]
    derived_features_list: List[str]
    missing_cicids_features: List[str]
    available_canonical_features: List[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "derived_features_count": len(self.derived_features_list),
            "derived_features_list": self.derived_features_list,
            "available_canonical_count": len(self.available_canonical_features),
            "available_canonical_features": self.available_canonical_features,
            "missing_cicids_count": len(self.missing_cicids_features),
            "missing_cicids_features": self.missing_cicids_features,
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "warnings": self.warnings,
        }


class FeatureDerivationEngine:
    """Safely extracts mapped features and computes mathematically valid derived features."""

    @staticmethod
    def _clean_numeric_array(
        arr: np.ndarray,
        non_negative: bool = True,
        max_clip: float = 1e15,
    ) -> np.ndarray:
        """Clean NaN, Inf, and out-of-bounds values using column median for finite imputation."""
        out = arr.astype(np.float64, copy=True)
        finite_mask = np.isfinite(out)
        if np.any(finite_mask):
            med = float(np.median(out[finite_mask]))
        else:
            med = 0.0
        out[~finite_mask] = med
        if non_negative:
            # Preserve -1 sentinel only where not clamping, else clamp < 0 to 0
            out = np.where(out < 0.0, 0.0, out)
        out = np.clip(out, -max_clip, max_clip)
        return out

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, min_denom: float = 1e-6) -> np.ndarray:
        """Vectorized division where denominator <= 0 returns 0.0 safely."""
        valid = denominator > 0
        safe_denom = np.where(valid, np.maximum(denominator, min_denom), 1.0)
        result = np.where(valid, numerator / safe_denom, 0.0)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def derive_features(
        self,
        df: pd.DataFrame,
        mapping_report: FeatureMappingReport,
    ) -> DerivationResult:
        """
        Extract directly mapped features into canonical space and compute derived features.
        """
        n_rows = len(df)
        canonical_data: Dict[str, np.ndarray] = {}
        provenance: Dict[str, FeatureProvenance] = {}
        derived_list: List[str] = []
        warnings: List[str] = []

        entry_by_id: Dict[str, FeatureMappingEntry] = {
            m.canonical_id: m for m in mapping_report.mappings
        }

        # Step 1: Extract directly mapped & approximated features
        for cid, entry in entry_by_id.items():
            if entry.status in ("mapped", "approximated") and entry.source_column and entry.source_column in df.columns:
                spec = CANONICAL_BY_ID[cid]
                series = df[entry.source_column]
                if pd.api.types.is_numeric_dtype(series):
                    raw_arr = series.to_numpy(dtype=np.float64, na_value=np.nan)
                    finite_ratio = float(np.mean(np.isfinite(raw_arr))) if n_rows > 0 else 1.0
                    cleaned = self._clean_numeric_array(raw_arr, non_negative=spec.non_negative)
                    if entry.unit_conversion and "seconds_to_microseconds" in entry.unit_conversion:
                        cleaned = cleaned * 1e6
                    canonical_data[cid] = cleaned
                    prov_type = "approximated" if entry.status == "approximated" else "original"
                    provenance[cid] = FeatureProvenance(
                        canonical_id=cid,
                        cicids_name=spec.cicids_name,
                        provenance_type=prov_type,
                        source_detail=f"Source column '{entry.source_column}'"
                        + (f" ({entry.unit_conversion})" if entry.unit_conversion else ""),
                        confidence=entry.confidence,
                        finite_ratio=round(finite_ratio, 4),
                    )
                elif cid == "protocol":
                    # Encode protocol names (tcp->6, udp->17, icmp->1)
                    proto_map = {"tcp": 6.0, "udp": 17.0, "icmp": 1.0}
                    encoded = (
                        series.astype(str)
                        .str.strip()
                        .str.lower()
                        .map(lambda x: proto_map.get(x, float(pd.to_numeric(x, errors="coerce") or 0.0)))
                        .to_numpy(dtype=np.float64)
                    )
                    canonical_data[cid] = encoded
                    provenance[cid] = FeatureProvenance(
                        canonical_id=cid,
                        cicids_name=spec.cicids_name,
                        provenance_type="original",
                        source_detail=f"Encoded protocol column '{entry.source_column}'",
                        confidence=entry.confidence,
                        finite_ratio=1.0,
                    )
                elif cid == "encryption_strength":
                    # Encode cipher strength: None/NaN -> 0.0 (unencrypted), DES/RC4/WEP -> 1.0 (weak), AES/TLS/SSH -> 2.0 (strong)
                    def _encode_enc(val: Any) -> float:
                        if pd.isna(val):
                            return 0.0
                        s = str(val).strip().lower()
                        if s in ("", "none", "null", "nan", "unencrypted", "0", "false", "no"):
                            return 0.0
                        if s in ("des", "3des", "rc4", "wep", "md5", "weak", "1"):
                            return 1.0
                        return 2.0

                    encoded = series.map(_encode_enc).to_numpy(dtype=np.float64)
                    canonical_data[cid] = encoded
                    provenance[cid] = FeatureProvenance(
                        canonical_id=cid,
                        cicids_name=spec.cicids_name,
                        provenance_type="original",
                        source_detail=f"Encoded encryption strength from '{entry.source_column}' (0=None, 1=Weak/DES, 2=Strong/AES)",
                        confidence=entry.confidence,
                        finite_ratio=1.0,
                    )
                elif cid == "client_agent_anomaly":
                    # Encode browser/user-agent: Unknown/Bot/Scanner -> 1.0 (anomalous), Chrome/Firefox/Edge/Safari -> 0.0
                    known_agents = {"chrome", "firefox", "safari", "edge", "opera", "brave", "ie", "0", "false", "normal"}
                    def _encode_agent(val: Any) -> float:
                        if pd.isna(val):
                            return 1.0
                        s = str(val).strip().lower()
                        if s in known_agents:
                            return 0.0
                        return 1.0

                    encoded = series.map(_encode_agent).to_numpy(dtype=np.float64)
                    canonical_data[cid] = encoded
                    provenance[cid] = FeatureProvenance(
                        canonical_id=cid,
                        cicids_name=spec.cicids_name,
                        provenance_type="original",
                        source_detail=f"Encoded client user-agent anomaly from '{entry.source_column}' (1=Unknown/Suspicious, 0=Known Browser)",
                        confidence=entry.confidence,
                        finite_ratio=1.0,
                    )
                elif cid == "unusual_time_access":
                    # Encode boolean/categorical Yes/True/1 -> 1.0, No/False/0 -> 0.0
                    pos_flags = {"1", "1.0", "yes", "true", "y", "t", "tor", "anomalous"}
                    encoded = (
                        series.astype(str)
                        .str.strip()
                        .str.lower()
                        .map(lambda x: 1.0 if x in pos_flags else 0.0)
                        .to_numpy(dtype=np.float64)
                    )
                    canonical_data[cid] = encoded
                    provenance[cid] = FeatureProvenance(
                        canonical_id=cid,
                        cicids_name=spec.cicids_name,
                        provenance_type="original",
                        source_detail=f"Encoded binary security flag from '{entry.source_column}'",
                        confidence=entry.confidence,
                        finite_ratio=1.0,
                    )

        # Helper to check and record a derived array
        def _record_derived(cid: str, arr: np.ndarray, formula_desc: str, parents: List[str]) -> None:
            spec = CANONICAL_BY_ID[cid]
            cleaned = self._clean_numeric_array(arr, non_negative=spec.non_negative)
            canonical_data[cid] = cleaned
            parent_confs = [provenance[p].confidence for p in parents if p in provenance]
            conf = round(min(parent_confs) * 0.90, 3) if parent_confs else 0.80
            provenance[cid] = FeatureProvenance(
                canonical_id=cid,
                cicids_name=spec.cicids_name,
                provenance_type="derived",
                source_detail=formula_desc,
                confidence=conf,
                finite_ratio=1.0,
            )
            derived_list.append(cid)

        # Step 2: Execute mathematical derivations in topological dependency order
        # 2a. Total packets & bytes
        if "flow_packet_count" not in canonical_data and "total_fwd_packets" in canonical_data and "total_bwd_packets" in canonical_data:
            _record_derived(
                "flow_packet_count",
                canonical_data["total_fwd_packets"] + canonical_data["total_bwd_packets"],
                "total_fwd_packets + total_bwd_packets",
                ["total_fwd_packets", "total_bwd_packets"],
            )

        if "flow_byte_count" not in canonical_data and "total_fwd_bytes" in canonical_data and "total_bwd_bytes" in canonical_data:
            _record_derived(
                "flow_byte_count",
                canonical_data["total_fwd_bytes"] + canonical_data["total_bwd_bytes"],
                "total_fwd_bytes + total_bwd_bytes",
                ["total_fwd_bytes", "total_bwd_bytes"],
            )

        # 2b. Directional packet length means
        if "fwd_packet_length_mean" not in canonical_data and "total_fwd_bytes" in canonical_data and "total_fwd_packets" in canonical_data:
            _record_derived(
                "fwd_packet_length_mean",
                self._safe_divide(canonical_data["total_fwd_bytes"], canonical_data["total_fwd_packets"], min_denom=1.0),
                "total_fwd_bytes / max(total_fwd_packets, 1)",
                ["total_fwd_bytes", "total_fwd_packets"],
            )

        if "bwd_packet_length_mean" not in canonical_data and "total_bwd_bytes" in canonical_data and "total_bwd_packets" in canonical_data:
            _record_derived(
                "bwd_packet_length_mean",
                self._safe_divide(canonical_data["total_bwd_bytes"], canonical_data["total_bwd_packets"], min_denom=1.0),
                "total_bwd_bytes / max(total_bwd_packets, 1)",
                ["total_bwd_bytes", "total_bwd_packets"],
            )

        # 2c. Min / Max / Mean / Average Packet Size
        if "min_packet_length" not in canonical_data and "fwd_packet_length_min" in canonical_data and "bwd_packet_length_min" in canonical_data:
            f_min = canonical_data["fwd_packet_length_min"]
            b_min = canonical_data["bwd_packet_length_min"]
            # If backward packets are 0, b_min is 0; take positive min where available
            both_pos = (f_min > 0) & (b_min > 0)
            derived_min = np.where(both_pos, np.minimum(f_min, b_min), np.maximum(f_min, b_min))
            _record_derived(
                "min_packet_length",
                derived_min,
                "min(fwd_packet_length_min, bwd_packet_length_min)",
                ["fwd_packet_length_min", "bwd_packet_length_min"],
            )

        if "max_packet_length" not in canonical_data and "fwd_packet_length_max" in canonical_data and "bwd_packet_length_max" in canonical_data:
            _record_derived(
                "max_packet_length",
                np.maximum(canonical_data["fwd_packet_length_max"], canonical_data["bwd_packet_length_max"]),
                "max(fwd_packet_length_max, bwd_packet_length_max)",
                ["fwd_packet_length_max", "bwd_packet_length_max"],
            )

        if "packet_length_mean" not in canonical_data:
            if "flow_byte_count" in canonical_data and "flow_packet_count" in canonical_data:
                _record_derived(
                    "packet_length_mean",
                    self._safe_divide(canonical_data["flow_byte_count"], canonical_data["flow_packet_count"], min_denom=1.0),
                    "(total_fwd_bytes + total_bwd_bytes) / max(total_fwd_packets + total_bwd_packets, 1)",
                    ["flow_byte_count", "flow_packet_count"],
                )
            elif "average_packet_size" in canonical_data:
                _record_derived(
                    "packet_length_mean",
                    canonical_data["average_packet_size"].copy(),
                    "copy(average_packet_size)",
                    ["average_packet_size"],
                )

        if "average_packet_size" not in canonical_data:
            if "packet_length_mean" in canonical_data:
                _record_derived(
                    "average_packet_size",
                    canonical_data["packet_length_mean"].copy(),
                    "copy(packet_length_mean)",
                    ["packet_length_mean"],
                )
            elif "flow_byte_count" in canonical_data and "flow_packet_count" in canonical_data:
                _record_derived(
                    "average_packet_size",
                    self._safe_divide(canonical_data["flow_byte_count"], canonical_data["flow_packet_count"], min_denom=1.0),
                    "(total_fwd_bytes + total_bwd_bytes) / max(total_fwd_packets + total_bwd_packets, 1)",
                    ["flow_byte_count", "flow_packet_count"],
                )

        # 2d. Packet Length Std <-> Variance
        if "packet_length_variance" not in canonical_data and "packet_length_std" in canonical_data:
            _record_derived(
                "packet_length_variance",
                np.square(canonical_data["packet_length_std"]),
                "packet_length_std ** 2",
                ["packet_length_std"],
            )
        elif "packet_length_std" not in canonical_data and "packet_length_variance" in canonical_data:
            _record_derived(
                "packet_length_std",
                np.sqrt(np.maximum(canonical_data["packet_length_variance"], 0.0)),
                "sqrt(packet_length_variance)",
                ["packet_length_variance"],
            )

        # 2e. Segment size averages (identical to directional packet length means in CICFlowMeter)
        if "avg_fwd_segment_size" not in canonical_data and "fwd_packet_length_mean" in canonical_data:
            _record_derived(
                "avg_fwd_segment_size",
                canonical_data["fwd_packet_length_mean"].copy(),
                "copy(fwd_packet_length_mean)",
                ["fwd_packet_length_mean"],
            )

        if "avg_bwd_segment_size" not in canonical_data and "bwd_packet_length_mean" in canonical_data:
            _record_derived(
                "avg_bwd_segment_size",
                canonical_data["bwd_packet_length_mean"].copy(),
                "copy(bwd_packet_length_mean)",
                ["bwd_packet_length_mean"],
            )

        # 2f. Flow & directional rates (duration is in microseconds; convert to seconds = dur / 1e6)
        if "flow_duration" in canonical_data:
            dur_sec = canonical_data["flow_duration"] / 1e6
            if "flow_bytes_per_second" not in canonical_data and "flow_byte_count" in canonical_data:
                _record_derived(
                    "flow_bytes_per_second",
                    self._safe_divide(canonical_data["flow_byte_count"], dur_sec, min_denom=1e-6),
                    "(total_fwd_bytes + total_bwd_bytes) / (flow_duration / 1e6)",
                    ["flow_byte_count", "flow_duration"],
                )
            if "flow_packets_per_second" not in canonical_data and "flow_packet_count" in canonical_data:
                _record_derived(
                    "flow_packets_per_second",
                    self._safe_divide(canonical_data["flow_packet_count"], dur_sec, min_denom=1e-6),
                    "(total_fwd_packets + total_bwd_packets) / (flow_duration / 1e6)",
                    ["flow_packet_count", "flow_duration"],
                )
            if "fwd_packets_per_second" not in canonical_data and "total_fwd_packets" in canonical_data:
                _record_derived(
                    "fwd_packets_per_second",
                    self._safe_divide(canonical_data["total_fwd_packets"], dur_sec, min_denom=1e-6),
                    "total_fwd_packets / (flow_duration / 1e6)",
                    ["total_fwd_packets", "flow_duration"],
                )
            if "bwd_packets_per_second" not in canonical_data and "total_bwd_packets" in canonical_data:
                _record_derived(
                    "bwd_packets_per_second",
                    self._safe_divide(canonical_data["total_bwd_packets"], dur_sec, min_denom=1e-6),
                    "total_bwd_packets / (flow_duration / 1e6)",
                    ["total_bwd_packets", "flow_duration"],
                )
            if "flow_iat_mean" not in canonical_data and "flow_packet_count" in canonical_data:
                intervals = np.maximum(canonical_data["flow_packet_count"] - 1.0, 1.0)
                _record_derived(
                    "flow_iat_mean",
                    self._safe_divide(canonical_data["flow_duration"], intervals, min_denom=1.0),
                    "flow_duration / max(flow_packet_count - 1, 1)",
                    ["flow_duration", "flow_packet_count"],
                )

        # 2g. Down/Up ratio, duplicate Fwd Header Length, and Subflow totals
        if "down_up_ratio" not in canonical_data and "total_bwd_packets" in canonical_data and "total_fwd_packets" in canonical_data:
            _record_derived(
                "down_up_ratio",
                np.floor(self._safe_divide(canonical_data["total_bwd_packets"], canonical_data["total_fwd_packets"], min_denom=1.0)),
                "floor(total_bwd_packets / max(total_fwd_packets, 1))",
                ["total_bwd_packets", "total_fwd_packets"],
            )

        if "fwd_header_length_dup" not in canonical_data and "fwd_header_length" in canonical_data:
            _record_derived(
                "fwd_header_length_dup",
                canonical_data["fwd_header_length"].copy(),
                "copy(fwd_header_length)",
                ["fwd_header_length"],
            )

        if "subflow_fwd_packets" not in canonical_data and "total_fwd_packets" in canonical_data:
            _record_derived(
                "subflow_fwd_packets",
                canonical_data["total_fwd_packets"].copy(),
                "copy(total_fwd_packets) [single-subflow]",
                ["total_fwd_packets"],
            )
        if "subflow_fwd_bytes" not in canonical_data and "total_fwd_bytes" in canonical_data:
            _record_derived(
                "subflow_fwd_bytes",
                canonical_data["total_fwd_bytes"].copy(),
                "copy(total_fwd_bytes) [single-subflow]",
                ["total_fwd_bytes"],
            )
        if "subflow_bwd_packets" not in canonical_data and "total_bwd_packets" in canonical_data:
            _record_derived(
                "subflow_bwd_packets",
                canonical_data["total_bwd_packets"].copy(),
                "copy(total_bwd_packets) [single-subflow]",
                ["total_bwd_packets"],
            )
        if "subflow_bwd_bytes" not in canonical_data and "total_bwd_bytes" in canonical_data:
            _record_derived(
                "subflow_bwd_bytes",
                canonical_data["total_bwd_bytes"].copy(),
                "copy(total_bwd_bytes) [single-subflow]",
                ["total_bwd_bytes"],
            )

        # 2h. Universal Threat Intelligence, Vulnerability & Categorical Security Feature Extraction
        # Ensures threat feeds (OTX, VirusTotal), CVE catalogs, and categorical/text security logs produce
        # actionable quantitative threat indicators without fabricating network flow measurements.
        import re
        col_lower_map = {str(c).strip().lower(): c for c in df.columns}

        # (i) Malware family / Known Ransomware campaign indicator
        malware_flag = np.zeros(n_rows, dtype=np.float64)
        has_malware_col = False
        for cand in ("knownransomwarecampaignuse", "malware_families", "threat_category"):
            if cand in col_lower_map:
                has_malware_col = True
                s = df[col_lower_map[cand]].fillna("").astype(str).str.strip().str.lower()
                if cand == "knownransomwarecampaignuse":
                    malware_flag = np.maximum(malware_flag, (s == "known").astype(np.float64).to_numpy())
                else:
                    benign_vals = {"", "unknown", "none", "null", "nan", "clean", "unrated", "harmless", "benign", "normal", "0", "false"}
                    malware_flag = np.maximum(malware_flag, (~s.isin(benign_vals)).astype(np.float64).to_numpy())
        if has_malware_col and "malware_or_ransomware_indicator" not in canonical_data:
            canonical_data["malware_or_ransomware_indicator"] = malware_flag
            derived_list.append("malware_or_ransomware_indicator")

        # (ii) MITRE ATT&CK technique count / CWE weakness count
        for cand in ("attack_ids", "cwes"):
            if cand in col_lower_map:
                s = df[col_lower_map[cand]].fillna("").astype(str)
                pattern = r"T\d{4}" if cand == "attack_ids" else r"CWE-\d+"
                counts = s.map(lambda x: float(len(re.findall(pattern, x, flags=re.IGNORECASE)))).to_numpy(dtype=np.float64)
                canonical_data["mitre_or_cwe_count"] = counts
                derived_list.append("mitre_or_cwe_count")
                break

        # (iii) Security Keyword Severity Score from text/description/tag/severity columns
        text_cands = [
            col_lower_map[k]
            for k in ("tags", "shortdescription", "vulnerabilityname", "description", "title", "threat_severity")
            if k in col_lower_map
        ]
        if text_cands:
            high_risk_terms = (
                "ransomware", "remote code execution", "command injection", "zero-day", "0-day",
                "c2 communication", "data exfiltration", "credential theft", "privilege escalation",
                "authentication bypass", "arbitrary code", "arbitrary commands", "backdoor",
                "banking trojan", "information stealer", "rootkit", "sql injection", "buffer overflow",
            )
            combined_text = df[text_cands[0]].fillna("").astype(str).str.lower()
            for tc in text_cands[1:]:
                combined_text = combined_text + " " + df[tc].fillna("").astype(str).str.lower()

            def _score_text(txt: str) -> float:
                hits = sum(1 for term in high_risk_terms if term in txt)
                if "critical" in txt or "high" in txt:
                    hits += 1
                return min(1.0, hits * 0.28)

            kw_scores = combined_text.map(_score_text).to_numpy(dtype=np.float64)
            if np.any(kw_scores > 0):
                canonical_data["security_keyword_severity_score"] = kw_scores
                derived_list.append("security_keyword_severity_score")

        # (iv) Empirical Categorical Rarity Index fallback when dataset has < 2 non-constant numeric columns
        non_const_num = sum(
            1 for arr in canonical_data.values() if len(arr) > 1 and float(np.std(arr)) > 1e-9
        )
        if non_const_num < 2 and n_rows > 1:
            cat_cols = [
                c for c in df.columns
                if str(c).strip().lower() not in ("label", "attack_detected", "threat_label", "cveid", "pulse_id", "session_id", "id", "ip")
                and not pd.api.types.is_numeric_dtype(df[c])
            ]
            for idx_c, c in enumerate(cat_cols[:3]):
                s = df[c].fillna("__MISSING__").astype(str)
                freq = s.value_counts(normalize=True)
                rarity = (1.0 - s.map(freq).to_numpy(dtype=np.float64))
                if float(np.std(rarity)) > 1e-9:
                    cid_r = f"categorical_rarity_{idx_c + 1}"
                    canonical_data[cid_r] = rarity
                    derived_list.append(cid_r)

        # Step 3: Record missing provenance for remaining unmapped CIC-IDS2017 features
        missing_cicids: List[str] = []
        for spec in CICIDS70_FEATURES:
            if spec.canonical_id not in canonical_data:
                if spec.cicids_name:
                    missing_cicids.append(spec.cicids_name)
                provenance[spec.canonical_id] = FeatureProvenance(
                    canonical_id=spec.canonical_id,
                    cicids_name=spec.cicids_name,
                    provenance_type="missing",
                    source_detail="Cannot be derived from available columns without fabrication.",
                    confidence=0.0,
                    finite_ratio=0.0,
                )

        canonical_df = pd.DataFrame(canonical_data, index=df.index)
        return DerivationResult(
            canonical_df=canonical_df,
            provenance=provenance,
            derived_features_list=derived_list,
            missing_cicids_features=missing_cicids,
            available_canonical_features=list(canonical_df.columns),
            warnings=warnings,
        )

    def build_cicids70_scaled_matrix(
        self,
        derivation: DerivationResult,
        scaler: Any,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Construct the 70-feature scaled matrix for CICIDS2017_SUPERVISED_MODEL.
        For any unavailable feature in a high-coverage dataset (e.g. missing TCP window
        features in hybrid mode), uses `scaler.mean_[idx]` in raw space so that after
        `scaler.transform`, the standardized value is `0.0` (the neutral mean of the
        training distribution) rather than raw `0.0` (which distorts to -mean/scale).
        """
        n_rows = len(derivation.canonical_df)
        X_raw = np.zeros((n_rows, 70), dtype=np.float64)
        imputation_notes: Dict[str, str] = {}

        scaler_means = getattr(scaler, "mean_", np.zeros(70, dtype=np.float64))

        for spec in CICIDS70_FEATURES:
            idx = spec.cicids_index
            if idx is None:
                continue
            cid = spec.canonical_id
            if cid in derivation.canonical_df.columns:
                X_raw[:, idx] = derivation.canonical_df[cid].to_numpy(dtype=np.float64)
            else:
                neutral_mean = float(scaler_means[idx]) if idx < len(scaler_means) else 0.0
                X_raw[:, idx] = neutral_mean
                imputation_notes[spec.cicids_name or cid] = (
                    f"Set to training distribution mean ({neutral_mean:.4f}) -> 0.0 after StandardScaler"
                )
                derivation.provenance[cid] = FeatureProvenance(
                    canonical_id=cid,
                    cicids_name=spec.cicids_name,
                    provenance_type="neutral_training_mean_baseline",
                    source_detail=imputation_notes[spec.cicids_name or cid],
                    confidence=0.0,
                    finite_ratio=0.0,
                )

        # Detect if the input matrix was ALREADY standardized (e.g. test_inference.csv with mean~0, std~1)
        col_means = np.abs(np.mean(X_raw, axis=0))
        col_stds = np.std(X_raw, axis=0)
        is_already_standardized = bool(
            np.median(col_means) < 2.0 and 0.2 < np.median(col_stds) < 5.0 and np.max(np.abs(X_raw)) < 1000.0
        )

        if is_already_standardized and all(
            p.provenance_type == "approximated"
            for p in derivation.provenance.values()
            if p.cicids_name is not None
        ):
            X_scaled = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            imputation_notes["__scaler__"] = "Input detected as pre-standardized matrix; preserved standardized scale."
        else:
            X_scaled = scaler.transform(X_raw).astype(np.float32)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled, imputation_notes


_derivation_engine: Optional[FeatureDerivationEngine] = None


def get_derivation_engine() -> FeatureDerivationEngine:
    global _derivation_engine
    if _derivation_engine is None:
        _derivation_engine = FeatureDerivationEngine()
    return _derivation_engine
