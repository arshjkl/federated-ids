"""
Laya Security Evidence Builder.
Converts IDS predictions and network features into a compact state for Laya decision-making.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
import json


class SecurityEvidenceBuilder:
    """
    Builds security evidence for Laya from IDS predictions and network flow features.
    """

    def __init__(self, feature_names: List[str] = None):
        """
        Initialize the evidence builder.

        Args:
            feature_names: List of feature names (in order) for mapping
        """
        self.feature_names = feature_names or []

        # Define which features to extract for network evidence
        # These are key indicators from CIC-IDS2017 that are useful for security decisions
        self.network_feature_mapping = {
            'flow_duration_ms': 'Flow Duration',
            'packets_per_second': 'Flow Packets/s',
            'forward_packets': 'Total Fwd Packets',
            'backward_packets': 'Total Backward Packets',
            'flow_bytes_per_second': 'Flow Bytes/s',
            'destination_port': 'Destination Port',
            'syn_flag_count': 'SYN Flag Count',
            'anomaly_score': None  # Will be computed
        }

        # Reverse mapping for quick lookup
        self.feature_to_index = {}
        if self.feature_names:
            for idx, name in enumerate(self.feature_names):
                self.feature_to_index[name.strip()] = idx

    def build_evidence(
        self,
        ids_prediction: Dict[str, Any],
        network_features: Union[np.ndarray, List],
        model_context: Dict[str, Any] = None,
        flow_packet_timings: np.ndarray = None,
        standardized_features: Union[np.ndarray, List] = None,
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build security evidence from IDS prediction and network features.

        Args:
            ids_prediction: Dictionary containing IDS prediction results
            network_features: Feature vector for the network flow
            model_context: Context about the model's performance
            flow_packet_timings: Timings of packets for anomaly detection

        Returns:
            Dictionary containing structured evidence for Laya
        """
        # Handle tensor input - check if it's a tensor-like object with numpy method
        if hasattr(network_features, 'cpu') and hasattr(network_features, 'numpy'):
            network_features = network_features.cpu().numpy()

        # Ensure we have a 1D array
        if network_features.ndim > 1:
            network_features = network_features.flatten()

        if standardized_features is not None:
            if hasattr(standardized_features, 'cpu') and hasattr(standardized_features, 'numpy'):
                standardized_features = standardized_features.cpu().numpy()
            if standardized_features.ndim > 1:
                standardized_features = standardized_features.flatten()
        else:
            standardized_features = network_features

        # Initialize evidence structure
        evidence = {
            "event_type": "network_intrusion_detection",
            "ids_prediction": self._build_ids_evidence(ids_prediction),
            "network_evidence": self._build_network_evidence(
                network_features,
                flow_packet_timings,
                standardized_features=standardized_features,
                anomaly_score=anomaly_score,
            ),
            "model_context": model_context or {}
        }

        return evidence

    def _build_ids_evidence(self, ids_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build IDS prediction evidence.

        Args:
            ids_prediction: IDS prediction results

        Returns:
            Structured IDS evidence
        """
        # Extract key information from IDS prediction
        # Expected keys: class, attack_probability, benign_probability, confidence
        evidence = {
            "class": ids_prediction.get("class", "UNKNOWN"),
            "attack_probability": float(ids_prediction.get("attack_probability", 0.0)),
            "benign_probability": float(ids_prediction.get("benign_probability", 0.0)),
            "confidence": float(ids_prediction.get("confidence", 0.0))
        }

        # Ensure probabilities sum to approximately 1.0
        total_prob = evidence["attack_probability"] + evidence["benign_probability"]
        if abs(total_prob - 1.0) > 0.01 and total_prob > 0:
            # Normalize if needed
            evidence["attack_probability"] /= total_prob
            evidence["benign_probability"] /= total_prob

        return evidence

    def _build_network_evidence(
        self,
        network_features: np.ndarray,
        flow_packet_timings: np.ndarray = None,
        standardized_features: np.ndarray = None,
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build network evidence from features.

        Args:
            network_features: RAW feature vector (used for human-readable named fields)
            flow_packet_timings: Packet timings for anomaly detection
            standardized_features: standardized (z-scored) feature vector used to
                compute the anomaly score. Defaults to ``network_features``.
            anomaly_score: pre-computed anomaly score (0-1). Takes precedence.

        Returns:
            Structured network evidence
        """
        evidence = {}

        # Extract mapped features from the RAW vector so named fields are
        # human-interpretable (e.g. "Flow Duration" in real units, real port numbers).
        for evidence_key, feature_name in self.network_feature_mapping.items():
            if feature_name is None:
                # Special case like anomaly_score that needs computation
                continue

            if feature_name in self.feature_to_index:
                idx = self.feature_to_index[feature_name]
                if idx < len(network_features):
                    evidence[evidence_key] = float(network_features[idx])
                else:
                    evidence[evidence_key] = 0.0
            else:
                # Feature not found, try to find similar name
                found = False
                for fname, findex in self.feature_to_index.items():
                    if feature_name.lower() in fname.lower() or fname.lower() in feature_name.lower():
                        if findex < len(network_features):
                            evidence[evidence_key] = float(network_features[findex])
                            found = True
                            break
                if not found:
                    evidence[evidence_key] = 0.0

        # Compute anomaly score from the standardized (z-scored) vector so the
        # |z| > 2 heuristic stays meaningful.
        if anomaly_score is not None:
            evidence["anomaly_score"] = float(anomaly_score)
        else:
            feats_for_anomaly = standardized_features if standardized_features is not None else network_features
            evidence["anomaly_score"] = self._compute_anomaly_score(feats_for_anomaly, flow_packet_timings)

        # Add temporal features if packet timings available
        if flow_packet_timings is not None and len(flow_packet_timings) > 1:
            evidence.update(self._compute_temporal_features(flow_packet_timings))

        # Add historical/repeated activity indicators (placeholder - would need history in practice)
        evidence["previous_alerts"] = 0  # Would come from security history
        evidence["repeated_suspicious_activity"] = 0  # Would come from behavior analysis

        return evidence

    def _compute_anomaly_score(
        self,
        network_features: np.ndarray,
        flow_packet_timings: np.ndarray = None
    ) -> float:
        """
        Compute an anomaly score from network features.

        Args:
            network_features: Network flow feature vector
            flow_packet_timings: Packet timings

        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        # Simple anomaly score based on feature magnitudes
        # In practice, this would use a trained anomaly detector or statistical methods

        # Normalize features to 0-1 range (assuming they're already somewhat normalized)
        # For now, use a simple heuristic based on extreme values
        if len(network_features) == 0:
            return 0.0

        # Use percentile-based approach to avoid outliers dominating
        try:
            # Compute how many features are in extreme ranges
            abs_features = np.abs(network_features)
            high_values = np.sum(abs_features > 2.0)  # Features > 2 std devs from mean (assuming normalized)
            anomaly_ratio = min(high_values / len(network_features), 1.0)

            # Add temporal anomaly if packet timings available
            temporal_anomaly = 0.0
            if flow_packet_timings is not None and len(flow_packet_timings) > 2:
                # Check for bursty traffic or irregular intervals
                intervals = np.diff(flow_packet_timings)
                if len(intervals) > 0:
                    cv = np.std(intervals) / (np.mean(intervals) + 1e-8)  # Coefficient of variation
                    temporal_anomaly = min(cv / 2.0, 1.0)  # Normalize

            # Combine scores
            anomaly_score = (anomaly_ratio + temporal_anomaly) / 2.0
            return float(anomaly_score)
        except:
            return 0.0

    def _compute_temporal_features(self, flow_packet_timings: np.ndarray) -> Dict[str, float]:
        """
        Compute temporal features from packet timings.

        Args:
            flow_packet_timings: Packet timings

        Returns:
            Dictionary of temporal features
        """
        features = {}

        if len(flow_packet_timings) < 2:
            return features

        # Compute inter-packet intervals
        intervals = np.diff(flow_packet_timings)

        if len(intervals) > 0:
            features["packet_interval_mean"] = float(np.mean(intervals))
            features["packet_interval_std"] = float(np.std(intervals))
            features["packet_interval_cv"] = float(
                np.std(intervals) / (np.mean(intervals) + 1e-8)
            )

            # Detect bursts (short intervals)
            if len(intervals) > 0:
                burst_threshold = np.percentile(intervals, 10)  # Fastest 10% of intervals
                burst_count = np.sum(intervals < burst_threshold)
                features["burst_packet_ratio"] = float(burst_count / len(intervals))

        return features

    def create_default_evidence(self) -> Dict[str, Any]:
        """
        Create a default evidence structure for testing.

        Returns:
            Default evidence structure
        """
        return {
            "event_type": "network_intrusion_detection",
            "ids_prediction": {
                "class": "BENIGN",
                "attack_probability": 0.3,
                "benign_probability": 0.7,
                "confidence": 0.7
            },
            "network_evidence": {
                "flow_duration_ms": 50.0,
                "packets_per_second": 100.0,
                "forward_packets": 10,
                "backward_packets": 8,
                "flow_bytes_per_second": 1000.0,
                "destination_port": 80,
                "syn_flag_count": 0,
                "anomaly_score": 0.1,
                "previous_alerts": 0,
                "repeated_suspicious_activity": 0
            },
            "model_context": {
                "accuracy": 0.95,
                "precision": 0.90,
                "recall": 0.85,
                "f1": 0.87,
                "false_positive_rate": 0.05,
                "false_negative_rate": 0.15
            }
        }


def build_evidence_from_ids_output(
    ids_class: str,
    ids_confidence: float,
    network_features: np.ndarray,
    feature_names: List[str] = None,
    flow_packet_timings: np.ndarray = None,
    model_context: Dict[str, Any] = None,
    standardized_features: np.ndarray = None,
    anomaly_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function to build evidence from IDS output.

    Args:
        ids_class: IDS prediction class ("BENIGN" or "ATTACK")
        ids_confidence: IDS confidence in prediction
        network_features: RAW feature vector (used for human-readable named fields)
        feature_names: Feature names for mapping
        flow_packet_timings: Packet timings for anomaly detection
        model_context: Model performance context passed through to Laya
        standardized_features: standardized (z-scored) vector for anomaly scoring
        anomaly_score: pre-computed anomaly score (0-1)

    Returns:
        Security evidence dictionary
    """
    # Convert class to probabilities
    if ids_class.upper() == "BENIGN":
        attack_prob = 1.0 - ids_confidence
        benign_prob = ids_confidence
    else:  # ATTACK
        attack_prob = ids_confidence
        benign_prob = 1.0 - ids_confidence

    ids_prediction = {
        "class": ids_class.upper(),
        "attack_probability": float(attack_prob),
        "benign_probability": float(benign_prob),
        "confidence": float(ids_confidence)
    }

    builder = SecurityEvidenceBuilder(feature_names)
    return builder.build_evidence(
        ids_prediction=ids_prediction,
        network_features=network_features,
        model_context=model_context,
        flow_packet_timings=flow_packet_timings,
        standardized_features=standardized_features,
        anomaly_score=anomaly_score,
    )


if __name__ == "__main__":
    # Test the evidence builder
    print("Testing SecurityEvidenceBuilder...")

    # Create builder
    builder = SecurityEvidenceBuilder()

    # Create test evidence
    evidence = builder.create_default_evidence()

    print("Default evidence:")
    print(json.dumps(evidence, indent=2))

    # Test with sample data
    print("\nTesting with sample data...")
    sample_features = np.array([80, 100000, 10, 5, 5000, 80, 0, 0.2])  # [port, duration, fwd_pkts, bwd_pkts, bytes, port, syn, anomaly]
    sample_timings = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    evidence = build_evidence_from_ids_output(
        ids_class="ATTACK",
        ids_confidence=0.85,
        network_features=sample_features,
        flow_packet_timings=sample_timings
    )

    print("Sample evidence:")
    print(json.dumps(evidence, indent=2))

    print("\nSecurityEvidenceBuilder test completed!")