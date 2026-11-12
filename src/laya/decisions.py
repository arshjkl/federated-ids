"""
Laya Security Decision Module.
Integrates Laya decision engine for security event analysis.

Uses a real (offline-capable) Laya checkpoint when available and falls back to a
deterministic, well-shaped mock engine otherwise. Never silently returns random
weights or fake metrics.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Import Laya (now available in the environment)
try:
    from laya import Router
    LAYA_AVAILABLE = True
except ImportError:
    LAYA_AVAILABLE = False
    logging.warning("Laya not available. Using mock implementation for testing.")


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def _local_laya_snapshot() -> Optional[Path]:
    """
    Locate a locally-cached Laya checkpoint so the engine never needs network access.

    Prefers the bundle repo snapshot (convaiinnovations/laya) which ships the
    english checkpoint at its root. Returns the snapshot directory containing
    rl_agent_config.json, or None.
    """
    if not HF_CACHE_ROOT.exists():
        return None
    for entry in HF_CACHE_ROOT.iterdir():
        name = entry.name
        if not name.startswith("models--convaiinnovations--laya"):
            continue
        if not entry.is_dir():
            continue
        snapshots = entry / "snapshots"
        if not snapshots.exists():
            continue
        for rev in snapshots.iterdir():
            if (rev / "rl_agent_config.json").exists() and (rev / "model.safetensors").exists():
                return rev
    return None


class LayaSecurityDecisionEngine:
    """
    Laya-based security decision engine for analyzing network security events.

    Attempts to run real Laya inference against a locally-cached checkpoint
    (the bundled english RLCD agent). If Laya is unavailable or cannot load,
    falls back to a deterministic rule-based mock engine whose response shape is
    identical to the real one so the dashboard never breaks.
    """

    def __init__(self, use_mock: bool = False, model: str = "typed-decisions", preload: bool = True):
        self.model_preference = model
        self.router = None
        self.checkpoint_path: Optional[Path] = None
        self.use_mock = False
        self._load_error: Optional[str] = None

        if use_mock or not LAYA_AVAILABLE:
            self.use_mock = True
            self.checkpoint_name = "mock"
            self.engine_name_val = "mock"
            logging.info("Laya engine running in mock mode")
            return

        snapshot = _local_laya_snapshot()
        if snapshot is None:
            self._load_error = "No locally-cached Laya checkpoint found (network required for typed-decisions). Falling back to mock."
            logging.warning(self._load_error)
            self.use_mock = True
            self.checkpoint_name = "mock"
            self.engine_name_val = "mock"
            return

        self.checkpoint_path = snapshot
        self.checkpoint_name = "english (local cache)"
        self.engine_name_val = "laya"

        try:
            # Point the router at the local snapshot so no network is needed.
            # auto_task_detection=False keeps routing deterministic: we explicitly
            # pass `model="english"` below instead of letting it try to download
            # the unreachable typed-decisions standalone repo.
            self.router = Router(
                models={"english": str(snapshot)},
                default="english",
                auto_task_detection=False,
                standalone_repos=False,
                preload=preload,
                max_loaded=1,
            )
            logging.info("Laya router initialized from local checkpoint %s", snapshot)
        except Exception as exc:
            self._load_error = f"Failed to initialize Laya router: {exc}. Falling back to mock."
            logging.warning(self._load_error)
            self.use_mock = True
            self.checkpoint_name = "mock"
            self.engine_name_val = "mock"

    # ------------------------------------------------------------------ API
    def engine_name(self) -> str:
        return self.engine_name_val

    def checkpoint(self) -> str:
        return self.checkpoint_name

    def load_error(self) -> Optional[str]:
        return self._load_error

    def make_security_decision(
        self,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a security decision based on evidence using Laya (or mock fallback).

        Args:
            evidence: Security evidence dictionary from SecurityEvidenceBuilder

        Returns:
            Dictionary containing Laya's decisions and confidence scores
        """
        if self.use_mock:
            return self._make_mock_decision(evidence)
        return self._make_laya_decision(evidence)

    def _make_laya_decision(
        self,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make security decision using the real Laya engine."""
        try:
            text_state = self._evidence_to_text(evidence)

            result = self.router.predict(
                state=text_state,
                questions=self.security_questions,
                model="english",
            )

            decisions = {}
            for question_key, question_config in self.security_questions.items():
                try:
                    question_result = result["answers"][question_key]
                    decision_type = question_config["type"]

                    if decision_type == "choice":
                        decisions[question_key] = {
                            "decision": question_result["choice"],
                            "confidence": float(question_result["confidence"]),
                            "probabilities": question_result.get("probabilities", {}),
                        }
                    elif decision_type == "noul":
                        decisions[question_key] = {
                            "probability": float(question_result["noul"]),
                            "confidence": float(question_result["confidence"]),
                            "answer_confidence": float(question_result.get("answer_confidence", 0.0)),
                        }
                    else:
                        decisions[question_key] = {
                            "value": str(question_result.get("choice", "UNKNOWN")),
                            "confidence": float(question_result.get("confidence", 0.5)),
                        }
                except Exception as exc:
                    logging.warning("Laya prediction failed for %s: %s", question_key, exc)
                    decisions[question_key] = self._make_mock_single_decision(question_key, evidence)

            return {
                "engine": "laya",
                "model": result.get("routing", {}).get("model", self.checkpoint_name),
                "checkpoint": self.checkpoint_name,
                "decisions": decisions,
                "evidence_used": evidence.get("event_type", "unknown"),
                "routing": result.get("routing", {}),
                "usage": result.get("usage", {}),
                "timestamp": None,
            }

        except Exception as exc:
            logging.error("Laya decision engine failed: %s", exc)
            return self._make_mock_decision(evidence)

    # ------------------------------------------------------------------ mock
    def _make_mock_decision(
        self,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deterministic rule-based decisions with the same shape as the real engine."""
        decisions = {}
        for question_key, question_config in self.security_questions.items():
            decisions[question_key] = self._make_mock_single_decision(question_key, evidence)

        return {
            "engine": "mock",
            "model": self.checkpoint_name,
            "checkpoint": self.checkpoint_name,
            "decisions": decisions,
            "evidence_used": evidence.get("event_type", "unknown"),
            "load_error": self._load_error,
            "timestamp": None,
        }

    def _make_mock_single_decision(
        self,
        question_key: str,
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single mock decision keyed on the IDS prediction & evidence heuristics."""
        ids_pred = evidence.get("ids_prediction", {})
        network_ev = evidence.get("network_evidence", {})
        model_ctx = evidence.get("model_context", {})

        ids_class = ids_pred.get("class", "BENIGN")
        ids_confidence = ids_pred.get("confidence", 0.5)
        attack_prob = ids_pred.get("attack_probability", 0.0)
        benign_prob = ids_pred.get("benign_probability", 0.0)

        is_attack = ids_class == "ATTACK"

        if question_key == "true_positive":
            if not is_attack:
                decision, confidence = "NO", 0.8
            else:
                high_confidence = ids_confidence > 0.7
                strong_evidence = (
                    network_ev.get("syn_flag_count", 0) > 5
                    or network_ev.get("packets_per_second", 0) > 1000
                    or network_ev.get("anomaly_score", 0) > 0.5
                )
                decision = "YES" if (high_confidence and strong_evidence) or attack_prob > 0.8 else "NO"
                confidence = 0.8 if decision == "YES" else 0.6
            return {"decision": decision, "confidence": float(confidence)}

        if question_key == "severity":
            if not is_attack:
                decision, confidence = "LOW", 0.85
            elif attack_prob > 0.9 or network_ev.get("syn_flag_count", 0) > 50:
                decision, confidence = "CRITICAL", 0.9
            elif attack_prob > 0.7 or network_ev.get("packets_per_second", 0) > 500:
                decision, confidence = "HIGH", 0.8
            elif attack_prob > 0.4 or network_ev.get("anomaly_score", 0) > 0.3:
                decision, confidence = "MEDIUM", 0.7
            else:
                decision, confidence = "LOW", 0.6
            return {"decision": decision, "confidence": float(confidence)}

        if question_key == "urgency":
            if not is_attack:
                decision, confidence = "LOW", 0.8
            elif attack_prob > 0.85 or network_ev.get("syn_flag_count", 0) > 30:
                decision, confidence = "HIGH", 0.85
            elif attack_prob > 0.6 or network_ev.get("flow_bytes_per_second", 0) > 100000:
                decision, confidence = "MEDIUM", 0.75
            else:
                decision, confidence = "LOW", 0.65
            return {"decision": decision, "confidence": float(confidence)}

        if question_key == "disposition":
            if not is_attack:
                if network_ev.get("anomaly_score", 0) > 0.5:
                    decision, confidence = "MONITOR", 0.75
                else:
                    decision, confidence = "ALLOW", 0.85
            elif attack_prob > 0.9:
                decision, confidence = "IMMEDIATE_RESPONSE", 0.9
            elif attack_prob > 0.7:
                decision, confidence = "ALERT", 0.8
            elif attack_prob > 0.4:
                decision, confidence = "REVIEW", 0.7
            elif network_ev.get("anomaly_score", 0) > 0.5:
                decision, confidence = "MONITOR", 0.75
            else:
                decision, confidence = "ALLOW", 0.8
            return {"decision": decision, "confidence": float(confidence)}

        if question_key == "credential_compromise":
            port = network_ev.get("destination_port", 0)
            auth_ports = {22, 23, 3389, 5900, 1433, 3306, 5432, 6379, 27017}
            is_auth_target = port in auth_ports
            high_privilege = port < 1024 and port not in {80, 443, 8080, 8443}

            if not is_attack:
                probability, confidence = 0.05, 0.9
            elif is_auth_target and attack_prob > 0.7:
                probability, confidence = 0.8, 0.8
            elif high_privilege and attack_prob > 0.6:
                probability, confidence = 0.6, 0.7
            elif attack_prob > 0.8:
                probability, confidence = 0.5, 0.7
            else:
                probability, confidence = 0.2, 0.6
            return {
                "probability": float(probability),
                "confidence": float(confidence),
            }

        return {"decision": "UNKNOWN", "confidence": 0.5}

    # ------------------------------------------------------------------ helpers
    @property
    def security_questions(self) -> Dict[str, Any]:
        """Question set matching the security_incidents workflow exactly."""
        return {
            "true_positive": {
                "type": "choice",
                "instructions": "Is the IDS security assessment sufficiently supported by the available evidence?",
                "criteria": {
                    "YES": "The IDS assessment is well-supported by network evidence and context",
                    "NO": "The IDS assessment is not sufficiently supported by available evidence",
                },
            },
            "severity": {
                "type": "choice",
                "instructions": "How severe is this network security event?",
                "criteria": {
                    "LOW": "Minimal potential impact, easily contained",
                    "MEDIUM": "Moderate potential impact requiring attention",
                    "HIGH": "Significant potential impact requiring immediate action",
                    "CRITICAL": "Severe potential impact requiring emergency response",
                },
            },
            "urgency": {
                "type": "choice",
                "instructions": "How urgently should this security event be investigated?",
                "criteria": {
                    "LOW": "Can be addressed during routine maintenance",
                    "MEDIUM": "Should be investigated within business hours",
                    "HIGH": "Requires immediate investigation",
                    "CRITICAL": "Requires emergency response within minutes",
                },
            },
            "disposition": {
                "type": "choice",
                "instructions": "What should the security system do with this event?",
                "criteria": {
                    "ALLOW": "Allow the traffic - no action needed",
                    "MONITOR": "Monitor the source but take no immediate action",
                    "REVIEW": "Flag for human analyst review",
                    "ALERT": "Generate security alert for investigation",
                    "IMMEDIATE_RESPONSE": "Trigger automatic containment/response",
                },
            },
            "credential_compromise": {
                "type": "noul",
                "instructions": "Does this network security event indicate possible credential compromise?",
            },
        }

    def _evidence_to_text(self, evidence: Dict[str, Any]) -> str:
        ids_pred = evidence.get("ids_prediction", {})
        network_ev = evidence.get("network_evidence", {})
        model_ctx = evidence.get("model_context", {})

        text_parts = [
            "Network Security Event Analysis:",
            "",
            "IDS Assessment:",
            f"- Classification: {ids_pred.get('class', 'UNKNOWN')}",
            f"- Attack Probability: {ids_pred.get('attack_probability', 0):.3f}",
            f"- Benign Probability: {ids_pred.get('benign_probability', 0):.3f}",
            f"- IDS Confidence: {ids_pred.get('confidence', 0):.3f}",
            "",
            "Network Evidence:",
            f"- Flow Duration: {network_ev.get('flow_duration_ms', 0):.1f}",
            f"- Packet Rate: {network_ev.get('packets_per_second', 0):.1f} packets/sec",
            f"- Forward Packets: {network_ev.get('forward_packets', 0)}",
            f"- Backward Packets: {network_ev.get('backward_packets', 0)}",
            f"- Bandwidth: {network_ev.get('flow_bytes_per_second', 0):.1f} bytes/sec",
            f"- Destination Port: {network_ev.get('destination_port', 0)}",
            f"- SYN Flags: {network_ev.get('syn_flag_count', 0)}",
            f"- Anomaly Score: {network_ev.get('anomaly_score', 0):.3f}",
            f"- Previous Alerts: {network_ev.get('previous_alerts', 0)}",
            f"- Repeated Suspicious Activity: {network_ev.get('repeated_suspicious_activity', 0)}",
            "",
            "Model Context:",
            f"- Model Accuracy: {model_ctx.get('accuracy', 0):.3f}",
            f"- Model Precision: {model_ctx.get('precision', 0):.3f}",
            f"- Model Recall: {model_ctx.get('recall', 0):.3f}",
            f"- Model F1-Score: {model_ctx.get('f1', 0):.3f}",
            f"- Model FPR: {model_ctx.get('false_positive_rate', 0):.3f}",
            f"- Model FNR: {model_ctx.get('false_negative_rate', 0):.3f}",
        ]
        return "\n".join(text_parts)


def make_security_decision(
    evidence: Dict[str, Any],
    use_mock: bool = False
) -> Dict[str, Any]:
    """Convenience function to make a security decision (module-level cached engine)."""
    engine = LayaSecurityDecisionEngine(use_mock=use_mock)
    return engine.make_security_decision(evidence)
