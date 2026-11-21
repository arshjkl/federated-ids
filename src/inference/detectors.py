"""
Phase 7 (Part 2) & Phase 9: Multi-Mode Detectors & Transparent Risk Scorer.

Implements:
  - Mode 1: SupervisedCICIDSDetector (CICIDS2017_SUPERVISED_MODEL)
  - Mode 2: UnsupervisedAnomalyDetector (IsolationForest + Robust Modified Z-Score)
  - Mode 3: NetworkRuleEngine (Configurable Evidence-Based Network Heuristic Rules)
  - Mode 4: AdaptiveHybridScorer (Transparent combination of signals, severity,
            honest categorization, and per-record explanations)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.dashboard.services.model_loader import get_model_loader
from src.dashboard.services.preprocessing import get_preprocessing_loader
from src.inference.feature_derivation import DerivationResult, get_derivation_engine
from src.inference.model_router import CompatibilityEvaluation, DEFAULT_HYBRID_CONFIG


@dataclass
class TriggeredRule:
    rule_id: str
    name: str
    weight: float
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecordDetectionOutcome:
    record_index: int
    prediction: int  # 1 if flagged (attack / suspicious / anomaly / rule alert), 0 if normal
    predicted_category: str
    supervised_probability: Optional[float]
    anomaly_score: Optional[float]
    rule_score: float
    risk_score: float
    confidence: float
    severity: str  # informational, low, medium, high, critical
    detection_mode: str
    triggered_rules: List[Dict[str, Any]]
    explanation: str
    top_anomaly_features: List[str] = field(default_factory=list)
    true_label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NetworkRuleEngine:
    """Evidence-based configurable network intrusion rule engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            "high_pkt_rate_pps": 5000.0,
            "high_byte_rate_bps": 5_000_000.0,
            "syn_flood_min_syns": 2.0,
            "port_scan_max_dur_us": 10000.0,
            "port_scan_max_pkts": 4.0,
            "asymmetric_fwd_min_pkts": 30.0,
            "short_burst_min_pkts": 20.0,
            "short_burst_max_dur_us": 100000.0,
            "rst_spike_min_cnt": 2.0,
            "conn_freq_spike": 80.0,
            "failed_logins_min": 3.0,
            "login_attempts_min": 7.0,
            "ip_reputation_min": 0.60,
            **(config or {}),
        }

    def evaluate(
        self,
        canonical_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[List[TriggeredRule]]]:
        """
        Evaluate configurable network rules across all rows of `canonical_df`.
        Returns (rule_scores[N], triggered_rules_per_row[N]).
        """
        n_rows = len(canonical_df)
        if n_rows == 0:
            return np.array([], dtype=np.float64), []

        cols = set(canonical_df.columns)
        scores = np.zeros(n_rows, dtype=np.float64)
        triggered: List[List[TriggeredRule]] = [[] for _ in range(n_rows)]

        def _col(name: str) -> Optional[np.ndarray]:
            return canonical_df[name].to_numpy(dtype=np.float64) if name in cols else None

        syn = _col("syn_flag_count")
        ack = _col("ack_flag_count")
        rst = _col("rst_flag_count")
        fwd_pkts = _col("total_fwd_packets")
        bwd_pkts = _col("total_bwd_packets")
        fwd_bytes = _col("total_fwd_bytes")
        bwd_bytes = _col("total_bwd_bytes")
        dur = _col("flow_duration")
        pps = _col("flow_packets_per_second")
        fwd_pps = _col("fwd_packets_per_second")
        bps = _col("flow_bytes_per_second")
        dst_port = _col("destination_port")
        failed_conn = _col("failed_connections")
        port_div = _col("port_diversity")
        conn_freq = _col("connection_frequency")

        # Rule 1: SYN Flood / Unacknowledged SYN Pattern
        if syn is not None:
            ack_arr = ack if ack is not None else np.zeros(n_rows)
            bwd_arr = bwd_pkts if bwd_pkts is not None else np.zeros(n_rows)
            mask = (syn >= self.config["syn_flood_min_syns"]) & (ack_arr == 0) & (bwd_arr <= 1)
            for idx in np.where(mask)[0]:
                scores[idx] += 0.55
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_SYN_FLOOD",
                        name="Unacknowledged SYN Flood Pattern",
                        weight=0.55,
                        evidence=f"SYN={syn[idx]:.0f}, ACK={ack_arr[idx]:.0f}, BwdPkts={bwd_arr[idx]:.0f}",
                    )
                )
        if failed_conn is not None:
            mask = failed_conn >= 0.5
            for idx in np.where(mask)[0]:
                scores[idx] += 0.50
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_HIGH_SYN_ERROR_RATE",
                        name="High Connection SYN/REJ Error Rate",
                        weight=0.50,
                        evidence=f"ErrorRate={failed_conn[idx]:.2f}",
                    )
                )

        # Rule 2: Port Scan Probe Pattern
        if dur is not None and fwd_pkts is not None and bwd_bytes is not None:
            mask = (
                (dur >= 0)
                & (dur <= self.config["port_scan_max_dur_us"])
                & (fwd_pkts >= 1)
                & (fwd_pkts <= self.config["port_scan_max_pkts"])
                & (bwd_bytes == 0)
            )
            if fwd_bytes is not None:
                mask = mask & (fwd_bytes <= 120)
            for idx in np.where(mask)[0]:
                scores[idx] += 0.45
                port_str = f", DstPort={dst_port[idx]:.0f}" if dst_port is not None else ""
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_PORT_SCAN_PROBE",
                        name="Rapid Zero-Payload Port Probe",
                        weight=0.45,
                        evidence=f"Dur={dur[idx]:.0f}us, FwdPkts={fwd_pkts[idx]:.0f}, BwdBytes=0{port_str}",
                    )
                )
        if port_div is not None:
            mask = port_div >= 0.5
            for idx in np.where(mask)[0]:
                scores[idx] += 0.45
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_HIGH_PORT_DIVERSITY",
                        name="Elevated Destination Port / Service Diversity",
                        weight=0.45,
                        evidence=f"PortDiversity={port_div[idx]:.2f}",
                    )
                )

        # Rule 3: Excessive Packet Rate Burst
        rate_arr = pps if pps is not None else fwd_pps
        if rate_arr is not None:
            mask = rate_arr >= self.config["high_pkt_rate_pps"]
            for idx in np.where(mask)[0]:
                w = 0.55 if rate_arr[idx] >= 25000.0 else 0.40
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_EXCESSIVE_PACKET_RATE",
                        name="Excessive Packet Rate Spike",
                        weight=w,
                        evidence=f"PacketRate={rate_arr[idx]:.1f} pkt/s",
                    )
                )

        # Rule 4: Excessive Byte Throughput Flood
        if bps is not None:
            mask = bps >= self.config["high_byte_rate_bps"]
            for idx in np.where(mask)[0]:
                scores[idx] += 0.50
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_HIGH_BYTE_THROUGHPUT",
                        name="Volumetric Byte Rate Spike",
                        weight=0.50,
                        evidence=f"ByteRate={bps[idx] / 1e6:.2f} MB/s",
                    )
                )

        # Rule 5: Asymmetric Unidirectional Flood Ratio
        if fwd_pkts is not None and bwd_pkts is not None:
            mask = (fwd_pkts >= self.config["asymmetric_fwd_min_pkts"]) & (bwd_pkts == 0)
            for idx in np.where(mask)[0]:
                scores[idx] += 0.50
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_UNIDIRECTIONAL_FLOOD",
                        name="Unidirectional Forward Packet Flood (0 Responses)",
                        weight=0.50,
                        evidence=f"FwdPkts={fwd_pkts[idx]:.0f}, BwdPkts=0",
                    )
                )

        # Rule 6: Short-Duration High-Packet Burst
        if dur is not None and fwd_pkts is not None:
            tot_p = fwd_pkts + (bwd_pkts if bwd_pkts is not None else 0.0)
            mask = (dur > 0) & (dur <= self.config["short_burst_max_dur_us"]) & (tot_p >= self.config["short_burst_min_pkts"])
            for idx in np.where(mask)[0]:
                scores[idx] += 0.45
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_SHORT_DURATION_BURST",
                        name="High-Density Micro-Burst Traffic",
                        weight=0.45,
                        evidence=f"TotalPkts={tot_p[idx]:.0f} in {dur[idx]:.0f}us",
                    )
                )

        # Rule 7: TCP RST Rejection Spike
        if rst is not None:
            mask = rst >= self.config["rst_spike_min_cnt"]
            for idx in np.where(mask)[0]:
                scores[idx] += 0.40
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_TCP_RST_SPIKE",
                        name="Repeated TCP Reset (RST) Rejections",
                        weight=0.40,
                        evidence=f"RST_Count={rst[idx]:.0f}",
                    )
                )

        # Rule 8: Connection Frequency Spike (NSL-KDD / UNSW-NB15 behavioral count)
        if conn_freq is not None:
            mask = conn_freq >= self.config["conn_freq_spike"]
            for idx in np.where(mask)[0]:
                scores[idx] += 0.45
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_HIGH_CONNECTION_FREQUENCY",
                        name="High Repeated Host/Service Connection Frequency",
                        weight=0.45,
                        evidence=f"ConnCount={conn_freq[idx]:.0f}",
                    )
                )

        # Session & Behavioral Intrusion Rules (Rules 9-13)
        failed_logins = _col("failed_logins")
        login_attempts = _col("login_attempts")
        ip_rep = _col("ip_reputation_score")
        unusual_time = _col("unusual_time_access")
        enc_strength = _col("encryption_strength")
        client_agent = _col("client_agent_anomaly")

        # Rule 9: Brute-Force Failed Login Spike
        if failed_logins is not None:
            min_fl = float(self.config["failed_logins_min"])
            mask = failed_logins >= min_fl
            for idx in np.where(mask)[0]:
                w = round(float(np.clip(0.70 + 0.08 * (failed_logins[idx] - min_fl), 0.70, 0.96)), 3)
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_BRUTE_FORCE_FAILED_LOGINS",
                        name="Brute-Force Authentication Failures",
                        weight=w,
                        evidence=f"FailedLogins={failed_logins[idx]:.0f} (>= {min_fl:.0f})",
                    )
                )

        # Rule 10: Excessive Login / Credential Stuffing Attempts
        if login_attempts is not None:
            min_la = float(self.config["login_attempts_min"])
            mask = login_attempts >= min_la
            for idx in np.where(mask)[0]:
                w = round(float(np.clip(0.70 + 0.05 * (login_attempts[idx] - min_la), 0.70, 0.96)), 3)
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_CREDENTIAL_STUFFING_LOGINS",
                        name="Excessive Login Attempts (Credential Stuffing)",
                        weight=w,
                        evidence=f"LoginAttempts={login_attempts[idx]:.0f} (>= {min_la:.0f})",
                    )
                )

        # Rule 11: High-Risk IP Reputation / Threat Intelligence Score
        mal_votes = _col("malicious_vote_count")
        if mal_votes is not None:
            mask = mal_votes >= 1.0
            for idx in np.where(mask)[0]:
                w = round(float(np.clip(0.65 + 0.06 * mal_votes[idx], 0.65, 0.96)), 3)
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_MALICIOUS_THREAT_VOTES",
                        name="Positive Threat Intelligence Malicious Votes",
                        weight=w,
                        evidence=f"MaliciousVotes={mal_votes[idx]:.0f} (>= 1)",
                    )
                )

        if ip_rep is not None:
            max_rep = float(np.nanmax(ip_rep)) if len(ip_rep) > 0 else 0.0
            min_rep = float(np.nanmin(ip_rep)) if len(ip_rep) > 0 else 0.0
            if min_rep < 0.0:
                # VirusTotal style reputation score where negative values (< 0) indicate malicious reputation
                mask = ip_rep < 0.0
                for idx in np.where(mask)[0]:
                    w = round(float(np.clip(0.65 + 0.015 * abs(ip_rep[idx]), 0.65, 0.96)), 3)
                    scores[idx] += w
                    triggered[idx].append(
                        TriggeredRule(
                            rule_id="RULE_NEGATIVE_IP_REPUTATION",
                            name="Negative IP Reputation Score (Threat Feed)",
                            weight=w,
                            evidence=f"ReputationScore={ip_rep[idx]:.0f} (< 0)",
                        )
                    )
            elif max_rep <= 1.05 and min_rep >= 0.0:
                thresh = float(self.config["ip_reputation_min"])
                mask = ip_rep >= thresh
                for idx in np.where(mask)[0]:
                    w = round(float(np.clip(0.68 + 0.80 * (ip_rep[idx] - thresh), 0.68, 0.96)), 3)
                    scores[idx] += w
                    triggered[idx].append(
                        TriggeredRule(
                            rule_id="RULE_HIGH_RISK_IP_REPUTATION",
                            name="High-Risk IP Reputation Score",
                            weight=w,
                            evidence=f"IPReputation={ip_rep[idx]:.3f} (>= {thresh:.2f})",
                        )
                    )
            elif min_rep >= 0.0 and max_rep > 1.05:
                thresh = 50.0 if max_rep > 25.0 else 2.0
                mask = ip_rep >= thresh
                for idx in np.where(mask)[0]:
                    w = 0.75
                    scores[idx] += w
                    triggered[idx].append(
                        TriggeredRule(
                            rule_id="RULE_HIGH_RISK_IP_REPUTATION",
                            name="Malicious IP Threat Intelligence Indicator",
                            weight=w,
                            evidence=f"ThreatScore={ip_rep[idx]:.1f} (>= {thresh:.1f})",
                        )
                    )

        # Rule 14: Threat Intelligence Malware / Ransomware / Critical CVE Indicators
        malware_ind = _col("malware_or_ransomware_indicator")
        mitre_cnt = _col("mitre_or_cwe_count")
        kw_sev = _col("security_keyword_severity_score")
        if malware_ind is not None:
            mask = malware_ind >= 1.0
            for idx in np.where(mask)[0]:
                w = 0.72
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_KNOWN_MALWARE_OR_RANSOMWARE",
                        name="Known Malware Family / Ransomware Campaign Attribution",
                        weight=w,
                        evidence="MalwareOrRansomware=Confirmed",
                    )
                )
        if mitre_cnt is not None:
            mask = mitre_cnt >= 6.0
            for idx in np.where(mask)[0]:
                w = round(float(np.clip(0.55 + 0.03 * (mitre_cnt[idx] - 6.0), 0.55, 0.90)), 3)
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_MULTI_STAGE_MITRE_CHAIN",
                        name="Multi-Technique MITRE ATT&CK / CWE Exploitation Chain",
                        weight=w,
                        evidence=f"TechniquesCount={mitre_cnt[idx]:.0f} (>= 6)",
                    )
                )
        if kw_sev is not None:
            mask = kw_sev >= 0.55
            for idx in np.where(mask)[0]:
                w = round(float(np.clip(kw_sev[idx], 0.55, 0.92)), 3)
                scores[idx] += w
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_CRITICAL_EXPLOIT_SIGNATURE",
                        name="High-Severity Exploit / Threat Signature in Advisory",
                        weight=w,
                        evidence=f"ExploitSignatureScore={kw_sev[idx]:.2f}",
                    )
                )

        # Rule 12: Unidentified / Anomalous Client Browser / User-Agent with Active Login Attempts
        if client_agent is not None:
            if login_attempts is not None:
                mask = (client_agent >= 1.0) & (login_attempts >= 3.0)
            else:
                mask = client_agent >= 1.0
            for idx in np.where(mask)[0]:
                w = 0.68
                scores[idx] += w
                la_str = f", LoginAttempts={login_attempts[idx]:.0f}" if login_attempts is not None else ""
                fl_str = f", FailedLogins={failed_logins[idx]:.0f}" if failed_logins is not None else ""
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_ANOMALOUS_CLIENT_AGENT",
                        name="Unidentified Client User-Agent Activity",
                        weight=w,
                        evidence=f"ClientAgent=Unknown{la_str}{fl_str}",
                    )
                )

        # Rule 13: Corroborating Weak/Unencrypted Cipher or Off-Hours Access on Active Alerts
        if unusual_time is not None or enc_strength is not None:
            ut_arr = unusual_time if unusual_time is not None else np.zeros(n_rows)
            enc_arr = enc_strength if enc_strength is not None else np.full(n_rows, 2.0)
            corrob_mask = (scores >= 0.45) & ((ut_arr >= 1.0) | (enc_arr <= 1.0))
            for idx in np.where(corrob_mask)[0]:
                w = 0.18
                scores[idx] += w
                ev_parts = []
                if ut_arr[idx] >= 1.0:
                    ev_parts.append("UnusualTimeAccess=1")
                if enc_arr[idx] == 0.0:
                    ev_parts.append("Encryption=None")
                elif enc_arr[idx] == 1.0:
                    ev_parts.append("Encryption=Weak(DES)")
                triggered[idx].append(
                    TriggeredRule(
                        rule_id="RULE_OFF_HOURS_OR_WEAK_CRYPTO",
                        name="Off-Hours / Weak-Encryption Session Context",
                        weight=w,
                        evidence=", ".join(ev_parts),
                    )
                )

        normalized_scores = np.clip(scores, 0.0, 1.0)
        return normalized_scores, triggered


class UnsupervisedAnomalyDetector:
    """
    Schema-agnostic unsupervised anomaly detector combining:
      1. Isolation Forest (tree-based isolation depth on robust-scaled numeric features)
      2. Robust Modified Z-Score / IQR multivariate statistical outlier detector
    """

    def score_dataset(
        self,
        df: pd.DataFrame,
        canonical_df: pd.DataFrame,
        column_types: Dict[str, str],
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Compute per-record anomaly scores in [0.0, 1.0] and top contributing feature names.
        """
        n_rows = len(df)
        if n_rows == 0:
            return np.array([], dtype=np.float64), []

        # Prefer canonical numeric features if >= 4 are available; else use all clean numeric columns from df
        if len(canonical_df.columns) >= 4:
            work_df = canonical_df.copy()
        else:
            num_cols = [
                c for c in df.columns
                if column_types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not num_cols:
                return np.zeros(n_rows, dtype=np.float64), [[] for _ in range(n_rows)]
            work_df = df[num_cols].copy()

        feature_names = list(work_df.columns)
        X = np.array(work_df.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True)

        # Clean NaN/Inf per column using median
        for j in range(X.shape[1]):
            col_arr = X[:, j]
            finite = np.isfinite(col_arr)
            med = float(np.median(col_arr[finite])) if np.any(finite) else 0.0
            col_arr[~finite] = med
            X[:, j] = np.clip(col_arr, -1e15, 1e15)

        # Remove zero-variance columns
        stds = np.std(X, axis=0)
        non_const_idx = np.where(stds > 1e-12)[0]
        if len(non_const_idx) == 0:
            return np.zeros(n_rows, dtype=np.float64), [[] for _ in range(n_rows)]

        X_active = X[:, non_const_idx]
        active_names = [feature_names[j] for j in non_const_idx]

        # 1. Robust Modified Z-Score per feature: 0.6745 * |x - median| / max(MAD, IQR/1.35, std)
        medians = np.median(X_active, axis=0)
        abs_dev = np.abs(X_active - medians)
        mad = np.median(abs_dev, axis=0)
        q75, q25 = np.percentile(X_active, [75, 25], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = np.std(X_active, axis=0)
        robust_scale = np.where(mad > 1e-9, mad / 0.6745, np.where(iqr_scale > 1e-9, iqr_scale, np.maximum(std_scale, 1e-6)))

        z_matrix = abs_dev / robust_scale  # (N, D)
        # RMS of top-3 feature z-scores per record mapped smoothly to [0, 1] via 1 - exp(-(z / 4.5)^2)
        k_top = min(3, z_matrix.shape[1])
        part_sorted = np.sort(z_matrix, axis=1)[:, -k_top:]
        rms_top_z = np.sqrt(np.mean(np.square(part_sorted), axis=1))
        stat_scores = 1.0 - np.exp(-np.square(rms_top_z / 4.5))

        # Top contributing features per record
        top_contributors: List[List[str]] = []
        top_indices = np.argsort(z_matrix, axis=1)[:, ::-1][:, :3]
        for i in range(n_rows):
            feats = [
                f"{active_names[j]} (z={z_matrix[i, j]:.1f})"
                for j in top_indices[i]
                if z_matrix[i, j] >= 2.5
            ]
            top_contributors.append(feats)

        # 2. Isolation Forest when n_rows >= 10
        if n_rows >= 10:
            scaler = RobustScaler()
            X_robust = scaler.fit_transform(X_active)
            X_robust = np.nan_to_num(X_robust, nan=0.0, posinf=0.0, neginf=0.0)

            iso = IsolationForest(
                n_estimators=100,
                max_samples=min(256, n_rows),
                contamination="auto",
                random_state=42,
                n_jobs=1,
            )
            iso.fit(X_robust)
            # score_samples returns negative values; more negative = more anomalous
            raw_iso = -iso.score_samples(X_robust)  # typically in [0.35, 0.85]
            # Calibrate raw_iso into [0, 1] where 0.50 is the decision boundary (offset_ is typically ~ -0.5)
            iso_min = float(np.min(raw_iso))
            iso_max = float(np.max(raw_iso))
            if iso_max - iso_min > 1e-6:
                # Decision function: < 0 means anomaly
                dec = -iso.decision_function(X_robust)  # > 0 means anomaly
                iso_scores = 1.0 / (1.0 + np.exp(-dec * 8.0))
            else:
                iso_scores = np.zeros(n_rows, dtype=np.float64)

            combined_anomaly = np.clip(0.60 * iso_scores + 0.40 * stat_scores, 0.0, 1.0)
        else:
            combined_anomaly = np.clip(stat_scores, 0.0, 1.0)

        return combined_anomaly, top_contributors


class SupervisedCICIDSDetector:
    """Wrapper for existing trained CICIDS2017_SUPERVISED_MODEL checkpoints."""

    def __init__(self, device: str = "cpu"):
        self.model_loader = get_model_loader(device)
        self.preprocessing_loader = get_preprocessing_loader()
        self.derivation_engine = get_derivation_engine()

    def predict_probabilities(
        self,
        derivation: DerivationResult,
        model_key: str = "centralized_improved",
        batch_size: int = 512,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Build the scaled feature matrix and run batched PyTorch inference.
        Supports both 70-feature CIC-IDS2017 checkpoints and dynamically trained
        custom-schema models from the Experiments engine.
        """
        # Check if model_key or raw_df matches a custom trained model in models/custom_trained/
        try:
            from src.dashboard.services.experiment_trainer import ExperimentTrainerService
            target_df = raw_df if raw_df is not None else derivation.canonical_df
            ckpt_match = ExperimentTrainerService.find_matching_custom_checkpoint(
                target_df,
                preferred_checkpoint_path=model_key,
            )
            if ckpt_match:
                ckpt_p, payload, overlap = ckpt_match
                probs, _, matched, missing = ExperimentTrainerService.predict_with_custom_checkpoint(
                    target_df,
                    payload,
                    canonical_df=derivation.canonical_df,
                    batch_size=batch_size,
                )
                notes = {f"custom_feature_{c}": "neutralized" for c in missing}
                return probs, notes
        except Exception:
            pass

        scaler = self.preprocessing_loader.scaler
        X_scaled, imputation_notes = self.derivation_engine.build_cicids70_scaled_matrix(
            derivation, scaler
        )
        n_rows = len(X_scaled)
        if n_rows == 0:
            return np.array([], dtype=np.float64), imputation_notes

        all_probs: List[np.ndarray] = []
        for start in range(0, n_rows, batch_size):
            batch = torch.FloatTensor(X_scaled[start : start + batch_size])
            probs = self.model_loader.predict(model_key, batch).numpy().flatten()
            all_probs.append(probs)

        return np.concatenate(all_probs).astype(np.float64), imputation_notes


class AdaptiveHybridScorer:
    """
    Combines compatible detection signals into transparent per-record risk scores,
    documented severity tiers, honest categories, and human-readable explanations.
    """

    @staticmethod
    def _derive_severity(risk_score: float, prediction: int) -> str:
        """
        Documented severity derivation:
          - critical:      risk_score >= 0.85
          - high:          0.70 <= risk_score < 0.85
          - medium:        0.50 <= risk_score < 0.70
          - low:           0.30 <= risk_score < 0.50
          - informational: risk_score < 0.30
        """
        if risk_score >= 0.85:
            return "critical"
        if risk_score >= 0.70:
            return "high"
        if risk_score >= 0.50 or prediction == 1:
            return "medium"
        if risk_score >= 0.30:
            return "low"
        return "informational"

    def synthesize_results(
        self,
        n_rows: int,
        compatibility: CompatibilityEvaluation,
        supervised_probs: Optional[np.ndarray],
        anomaly_scores: Optional[np.ndarray],
        rule_scores: np.ndarray,
        triggered_rules: List[List[TriggeredRule]],
        top_anomaly_features: List[List[str]],
        threshold: float = 0.50,
        true_labels: Optional[List[Optional[int]]] = None,
    ) -> List[RecordDetectionOutcome]:
        """Produce rich per-record detection outcomes."""
        mode = compatibility.selected_mode
        weights = compatibility.scoring_weights
        w_sup = weights.get("supervised_weight", 0.0)
        w_anom = weights.get("anomaly_weight", 0.0)
        w_rule = weights.get("rule_weight", 0.0)
        cov_factor = max(0.5, compatibility.feature_coverage_score / 100.0)

        outcomes: List[RecordDetectionOutcome] = []
        anom_threshold = max(0.62, threshold)

        for idx in range(n_rows):
            p_sup = float(supervised_probs[idx]) if supervised_probs is not None else None
            s_anom = float(anomaly_scores[idx]) if anomaly_scores is not None else None
            s_rule = float(rule_scores[idx]) if idx < len(rule_scores) else 0.0
            rules_i = triggered_rules[idx] if idx < len(triggered_rules) else []
            anom_feats_i = top_anomaly_features[idx] if idx < len(top_anomaly_features) else []
            t_label = true_labels[idx] if true_labels and idx < len(true_labels) else None

            if mode == "supervised" and p_sup is not None:
                # Primary signal is the supervised model; strong rules or extreme anomalies boost risk
                base_risk = w_sup * p_sup + w_anom * (s_anom or 0.0) + w_rule * s_rule
                risk_score = float(np.clip(max(p_sup, base_risk), 0.0, 1.0))
                prediction = 1 if p_sup >= threshold else 0
                raw_conf = abs(p_sup - threshold) / max(threshold, 1.0 - threshold)
                confidence = float(np.clip(0.5 + 0.5 * raw_conf, 0.0, 1.0))

                if prediction == 1:
                    if rules_i:
                        predicted_category = f"Confirmed Attack ({rules_i[0].name})"
                    else:
                        predicted_category = "Confirmed Attack (Supervised)"
                    explanation = (
                        f"CICIDS2017_SUPERVISED_MODEL classified record as malicious with probability "
                        f"{p_sup:.4f} (>= threshold {threshold:.2f})."
                    )
                    if rules_i:
                        explanation += f" Corroborated by rule(s): {', '.join(r.name for r in rules_i)}."
                else:
                    if s_rule >= 0.55:
                        predicted_category = "Rule Alert (Benign Supervised Score)"
                        explanation = (
                            f"Supervised model probability is low ({p_sup:.4f}), but heuristic rule(s) triggered: "
                            f"{', '.join(r.name + ' [' + r.evidence + ']' for r in rules_i)}."
                        )
                    else:
                        predicted_category = "Normal Traffic"
                        explanation = (
                            f"CICIDS2017_SUPERVISED_MODEL classified record as normal traffic "
                            f"(attack probability {p_sup:.4f} < {threshold:.2f})."
                        )

            elif mode == "hybrid":
                p_s = p_sup if p_sup is not None else 0.0
                s_a = s_anom if s_anom is not None else 0.0
                weighted_sum = w_sup * p_s + w_anom * s_a + w_rule * s_rule
                # Allow strong consensus or high-confidence single signal to surface
                consensus_boost = 0.12 if (p_s >= threshold and (s_a >= 0.55 or s_rule >= 0.45)) else 0.0
                risk_score = float(np.clip(max(weighted_sum + consensus_boost, 0.85 * p_s * cov_factor, 0.85 * s_rule), 0.0, 1.0))
                prediction = 1 if (risk_score >= threshold or p_s >= max(threshold, 0.60)) else 0

                # Signal agreement increases confidence
                signals = [p_s >= threshold, s_a >= 0.60, s_rule >= 0.45]
                agreement = sum(1 for x in signals if x) / 3.0 if prediction == 1 else sum(1 for x in signals if not x) / 3.0
                confidence = float(np.clip(agreement * (0.65 + 0.35 * cov_factor), 0.25, 0.98))

                if p_s >= threshold and (s_a >= 0.55 or s_rule >= 0.45):
                    predicted_category = "Confirmed Attack (Hybrid Consensus)"
                    explanation = (
                        f"Multi-signal consensus: supervised probability={p_s:.3f}, "
                        f"anomaly score={s_a:.3f}, rule score={s_rule:.3f} (risk={risk_score:.3f})."
                    )
                elif prediction == 1 and p_s >= threshold:
                    predicted_category = "Probable Suspicious Traffic (Hybrid Supervised)"
                    explanation = (
                        f"Partial-schema supervised probability ({p_s:.3f}) exceeded threshold "
                        f"with {compatibility.feature_coverage_score:.1f}% feature coverage (risk={risk_score:.3f})."
                    )
                elif prediction == 1 and s_rule >= 0.50:
                    predicted_category = f"Rule Alert ({rules_i[0].name})" if rules_i else "Rule Alert (Hybrid)"
                    explanation = (
                        f"Heuristic rule alert triggered ({', '.join(r.name for r in rules_i)}) "
                        f"with anomaly score={s_a:.3f} (risk={risk_score:.3f})."
                    )
                elif prediction == 1:
                    predicted_category = "Statistical Anomaly (Hybrid)"
                    explanation = (
                        f"Elevated hybrid risk ({risk_score:.3f}) driven by statistical anomaly score ({s_a:.3f})."
                    )
                else:
                    predicted_category = "Normal Traffic"
                    explanation = (
                        f"Hybrid evaluation within normal bounds (supervised={p_s:.3f}, "
                        f"anomaly={s_a:.3f}, rules={s_rule:.3f}, risk={risk_score:.3f})."
                    )

            elif mode == "anomaly":
                s_a = s_anom if s_anom is not None else 0.0
                risk_score = float(np.clip(max(w_anom * s_a + w_rule * s_rule, 0.85 * s_a, 0.90 * s_rule), 0.0, 1.0))
                is_anom = s_a >= anom_threshold
                is_rule = s_rule >= 0.50
                prediction = 1 if (is_anom or is_rule) else 0
                confidence = float(np.clip(abs(risk_score - 0.5) * 1.6 + 0.2, 0.20, 0.90))

                if is_anom and is_rule:
                    predicted_category = "Suspicious Anomaly + Rule Alert"
                    explanation = (
                        f"Statistical outlier (anomaly score={s_a:.3f}) corroborated by network rule(s): "
                        f"{', '.join(r.name + ' [' + r.evidence + ']' for r in rules_i)}. "
                        f"Note: Unsupervised anomaly detection identifies unusual traffic, not a confirmed supervised attack class."
                    )
                elif is_anom:
                    predicted_category = "Statistical Anomaly"
                    feat_note = f" Top deviating features: {', '.join(anom_feats_i)}." if anom_feats_i else ""
                    explanation = (
                        f"IsolationForest & Robust Modified Z-Score flagged flow as a statistical anomaly "
                        f"(anomaly score={s_a:.3f} >= {anom_threshold:.2f}).{feat_note} "
                        f"Note: Statistical unusualness does not alone confirm malicious intent."
                    )
                elif is_rule:
                    predicted_category = f"Rule Alert ({rules_i[0].name})" if rules_i else "Rule Alert"
                    explanation = (
                        f"Network rule triggered: {', '.join(r.name + ' [' + r.evidence + ']' for r in rules_i)}."
                    )
                else:
                    predicted_category = "Normal Baseline Traffic"
                    explanation = (
                        f"Flow conforms to dataset statistical baseline (anomaly score={s_a:.3f}, rule score={s_rule:.3f})."
                    )

            else:
                # Mode 4: rule_based
                risk_score = float(np.clip(s_rule, 0.0, 1.0))
                prediction = 1 if s_rule >= 0.45 else 0
                confidence = float(np.clip(0.50 + 0.40 * s_rule if prediction == 1 else 0.70, 0.30, 0.90))
                if prediction == 1:
                    predicted_category = f"Rule Alert ({rules_i[0].name})" if rules_i else "Rule Alert"
                    explanation = (
                        f"Heuristic rule alert: {', '.join(r.name + ' [' + r.evidence + ']' for r in rules_i)}."
                    )
                else:
                    predicted_category = "Normal Baseline Traffic"
                    explanation = "No heuristic network intrusion rules triggered."

            severity = self._derive_severity(risk_score, prediction)

            outcomes.append(
                RecordDetectionOutcome(
                    record_index=idx,
                    prediction=prediction,
                    predicted_category=predicted_category,
                    supervised_probability=round(p_sup, 6) if p_sup is not None else None,
                    anomaly_score=round(s_anom, 6) if s_anom is not None else None,
                    rule_score=round(s_rule, 6),
                    risk_score=round(risk_score, 6),
                    confidence=round(confidence, 6),
                    severity=severity,
                    detection_mode=mode,
                    triggered_rules=[r.to_dict() for r in rules_i],
                    explanation=explanation,
                    top_anomaly_features=anom_feats_i,
                    true_label=t_label,
                )
            )

        return outcomes
