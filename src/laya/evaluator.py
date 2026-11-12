"""
Laya Security Decision Evaluator.
Evaluates the performance of the Laya security decision layer,
particularly for false positive and false negative cases.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import time

# Import from the same package
from .evidence import SecurityEvidenceBuilder, build_evidence_from_ids_output
from .decisions import LayaSecurityDecisionEngine, make_security_decision


class LayaSecurityEvaluator:
    """
    Evaluates Laya security decisions, especially for FP/FN analysis.
    """

    def __init__(self, feature_names: List[str] = None):
        """
        Initialize the Laya security evaluator.

        Args:
            feature_names: List of feature names for evidence building
        """
        self.feature_names = feature_names
        self.evidence_builder = SecurityEvidenceBuilder(feature_names)
        self.decision_engine = LayaSecurityDecisionEngine(use_mock=True)  # Use mock for consistency in testing

    def evaluate_ids_prediction(
        self,
        ids_class: str,
        ids_confidence: float,
        true_label: str,
        network_features: np.ndarray,
        flow_packet_timings: np.ndarray = None,
        model_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single IDS prediction with Laya security decision layer.

        Args:
            ids_class: IDS prediction class ("BENIGN" or "ATTACK")
            ids_confidence: IDS confidence in prediction
            true_label: Actual label ("BENIGN" or "ATTACK")
            network_features: Network feature vector
            flow_packet_timings: Packet timings for anomaly detection
            model_context: Model performance context

        Returns:
            Evaluation results including IDS and Laya decisions
        """
        # Determine if IDS prediction is correct
        ids_correct = (ids_class.upper() == true_label.upper())

        # Build security evidence
        evidence = build_evidence_from_ids_output(
            ids_class=ids_class,
            ids_confidence=ids_confidence,
            network_features=network_features,
            feature_names=self.feature_names,
            flow_packet_timings=flow_packet_timings
        )

        # Add model context if provided
        if model_context:
            evidence["model_context"] = model_context

        # Get Laya security decision
        laya_decision = self.decision_engine.make_security_decision(evidence)

        # Determine if this is a false positive or false negative
        is_fp = (ids_class.upper() == "ATTACK" and true_label.upper() == "BENIGN")
        is_fn = (ids_class.upper() == "BENIGN" and true_label.upper() == "ATTACK")
        is_tn = (ids_class.upper() == "BENIGN" and true_label.upper() == "BENIGN")
        is_tp = (ids_class.upper() == "ATTACK" and true_label.upper() == "ATTACK")

        # Prepare result
        result = {
            "ids_prediction": {
                "class": ids_class.upper(),
                "confidence": ids_confidence,
                "correct": ids_correct
            },
            "true_label": true_label.upper(),
            "is_false_positive": is_fp,
            "is_false_negative": is_fn,
            "is_true_negative": is_tn,
            "is_true_positive": is_tp,
            "evidence": evidence,
            "laya_decision": laya_decision,
            "timestamp": time.time()
        }

        return result

    def evaluate_batch(
        self,
        ids_predictions: List[Dict[str, Any]],
        true_labels: List[str],
        network_features_list: List[np.ndarray],
        flow_packet_timings_list: List[np.ndarray] = None,
        model_contexts: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of IDS predictions.

        Args:
            ids_predictions: List of IDS prediction dictionaries
            true_labels: List of true labels
            network_features_list: List of network feature vectors
            flow_packet_timings_list: List of packet timings (optional)
            model_contexts: List of model contexts (optional)

        Returns:
            List of evaluation results
        """
        if flow_packet_timings_list is None:
            flow_packet_timings_list = [None] * len(ids_predictions)
        if model_contexts is None:
            model_contexts = [None] * len(ids_predictions)

        results = []
        for i, (ids_pred, true_label, features, timings, context) in enumerate(
            zip(ids_predictions, true_labels, network_features_list,
                flow_packet_timings_list, model_contexts)
        ):
            result = self.evaluate_ids_prediction(
                ids_class=ids_pred["class"],
                ids_confidence=ids_pred["confidence"],
                true_label=true_label,
                network_features=features,
                flow_packet_timings=timings,
                model_context=context
            )
            results.append(result)

        return results

    def analyze_false_positives(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze false positive cases to see how Laya improves decisions.

        Args:
            results: List of evaluation results from evaluate_ids_prediction or evaluate_batch

        Returns:
            Analysis of false positive cases
        """
        # Filter for false positives
        fp_results = [r for r in results if r["is_false_positive"]]

        if not fp_results:
            return {
                "total_false_positives": 0,
                "analysis": "No false positives found"
            }

        # Analyze Laya decisions for FP cases
        escalated_to_alert = 0
        escalated_to_review = 0
        escalated_to_monitor = 0
        allowed_incorrectly = 0

        confidence_scores = []

        for result in fp_results:
            laya_dec = result["laya_decision"]["decisions"]

            # Check disposition decision
            disposition = laya_dec.get("disposition", {}).get("decision", "ALLOW")
            disposition_conf = laya_dec.get("disposition", {}).get("confidence", 0.0)

            # Check true positive assessment (should be NO for FPs)
            tp_assessment = laya_dec.get("true_positive", {}).get("decision", "YES")
            tp_conf = laya_dec.get("true_positive", {}).get("confidence", 0.0)

            confidence_scores.append({
                "disposition_conf": disposition_conf,
                "tp_assessment_conf": tp_conf
            })

            # Count escalations
            if disposition in ["ALERT", "IMMEDIATE_RESPONSE"]:
                escalated_to_alert += 1
            elif disposition == "REVIEW":
                escalated_to_review += 1
            elif disposition == "MONITOR":
                escalated_to_monitor += 1
            elif disposition == "ALLOW":
                allowed_incorrectly += 1

        # Calculate rates
        total_fp = len(fp_results)
        escalation_rate = (escalated_to_alert + escalated_to_review + escalated_to_monitor) / total_fp
        alert_rate = escalated_to_alert / total_fp
        review_rate = escalated_to_review / total_fp
        monitor_rate = escalated_to_monitor / total_fp
        false_escalation_rate = allowed_incorrectly / total_fp  # FPs that were incorrectly allowed

        # Average confidences
        avg_disposition_conf = np.mean([c["disposition_conf"] for c in confidence_scores]) if confidence_scores else 0.0
        avg_tp_assessment_conf = np.mean([c["tp_assessment_conf"] for c in confidence_scores]) if confidence_scores else 0.0

        return {
            "total_false_positives": total_fp,
            "escalation_analysis": {
                "escalated_to_alert": escalated_to_alert,
                "escalated_to_review": escalated_to_review,
                "escalated_to_monitor": escalated_to_monitor,
                "allowed_incorrectly": allowed_incorrectly
            },
            "rates": {
                "escalation_rate": escalation_rate,
                "alert_rate": alert_rate,
                "review_rate": review_rate,
                "monitor_rate": monitor_rate,
                "false_escalation_rate": false_escalation_rate
            },
            "average_confidences": {
                "disposition_confidence": avg_disposition_conf,
                "tp_assessment_confidence": avg_tp_assessment_conf
            }
        }

    def analyze_false_negatives(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze false negative cases to see how Laya improves decisions.

        Args:
            results: List of evaluation results from evaluate_ids_prediction or evaluate_batch

        Returns:
            Analysis of false negative cases
        """
        # Filter for false negatives
        fn_results = [r for r in results if r["is_false_negative"]]

        if not fn_results:
            return {
                "total_false_negatives": 0,
                "analysis": "No false negatives found"
            }

        # Analyze Laya decisions for FN cases
        escalated_to_alert = 0
        escalated_to_review = 0
        escalated_to_monitor = 0
        allowed_incorrectly = 0  # FNs that were allowed (should be escalated)

        confidence_scores = []

        for result in fn_results:
            laya_dec = result["laya_decision"]["decisions"]

            # Check disposition decision
            disposition = laya_dec.get("disposition", {}).get("decision", "ALLOW")
            disposition_conf = laya_dec.get("disposition", {}).get("confidence", 0.0)

            # Check true positive assessment (should be YES for FNs)
            tp_assessment = laya_dec.get("true_positive", {}).get("decision", "NO")
            tp_conf = laya_dec.get("true_positive", {}).get("confidence", 0.0)

            confidence_scores.append({
                "disposition_conf": disposition_conf,
                "tp_assessment_conf": tp_conf
            })

            # Count escalations (what Laya recommended for FNs)
            if disposition in ["ALERT", "IMMEDIATE_RESPONSE"]:
                escalated_to_alert += 1
            elif disposition == "REVIEW":
                escalated_to_review += 1
            elif disposition == "MONITOR":
                escalated_to_monitor += 1
            elif disposition == "ALLOW":
                allowed_incorrectly += 1  # These are dangerous - FNs that were allowed

        # Calculate rates
        total_fn = len(fn_results)
        escalation_rate = (escalated_to_alert + escalated_to_review + escalated_to_monitor) / total_fn
        alert_rate = escalated_to_alert / total_fn
        review_rate = escalated_to_review / total_fn
        monitor_rate = escalated_to_monitor / total_fn
        missed_escalation_rate = allowed_incorrectly / total_fn  # FNs that were incorrectly allowed (dangerous!)

        # Average confidences
        avg_disposition_conf = np.mean([c["disposition_conf"] for c in confidence_scores]) if confidence_scores else 0.0
        avg_tp_assessment_conf = np.mean([c["tp_assessment_conf"] for c in confidence_scores]) if confidence_scores else 0.0

        return {
            "total_false_negatives": total_fn,
            "escalation_analysis": {
                "escalated_to_alert": escalated_to_alert,
                "escalated_to_review": escalated_to_review,
                "escalated_to_monitor": escalated_to_monitor,
                "allowed_incorrectly": allowed_incorrectly  # Dangerous cases
            },
            "rates": {
                "escalation_rate": escalation_rate,
                "alert_rate": alert_rate,
                "review_rate": review_rate,
                "monitor_rate": monitor_rate,
                "missed_escalation_rate": missed_escalation_rate
            },
            "average_confidences": {
                "disposition_confidence": avg_disposition_conf,
                "tp_assessment_confidence": avg_tp_assessment_conf
            }
        }

    def get_overall_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get overall statistics about the evaluation.

        Args:
            results: List of evaluation results

        Returns:
            Overall statistics
        """
        if not results:
            return {"total_evaluated": 0}

        total = len(results)
        tps = sum(1 for r in results if r["is_true_positive"])
        tns = sum(1 for r in results if r["is_true_negative"])
        fps = sum(1 for r in results if r["is_false_positive"])
        fns = sum(1 for r in results if r["is_false_negative"])

        # IDS metrics
        ids_accuracy = (tps + tns) / total if total > 0 else 0.0
        ids_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        ids_recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        ids_f1 = 2 * (ids_precision * ids_recall) / (ids_precision + ids_recall) if (ids_precision + ids_recall) > 0 else 0.0

        # Laya-enhanced metrics (considering escalations as "catching" errors)
        # For simplicity, we'll consider that any escalation (ALERT, REVIEW, MONITOR, IMMEDIATE_RESPONSE)
        # represents the system taking action on the event

        # Count events where Laya recommended action (not ALLOW)
        laya_action_taken = 0
        laya_action_on_fp = 0  # Good: Laya acted on FP
        laya_action_on_fn = 0  # Good: Laya acted on FN
        laya_no_action_on_fp = 0  # Bad: Laya didn't act on FP
        laya_no_action_on_fn = 0  # Very bad: Laya didn't act on FN

        for result in results:
            laya_dec = result["laya_decision"]["decisions"]
            disposition = laya_dec.get("disposition", {}).get("decision", "ALLOW")

            took_action = disposition != "ALLOW"
            if took_action:
                laya_action_taken += 1
                if result["is_false_positive"]:
                    laya_action_on_fp += 1
                elif result["is_false_negative"]:
                    laya_action_on_fn += 1
            else:
                if result["is_false_positive"]:
                    laya_no_action_on_fp += 1
                elif result["is_false_negative"]:
                    laya_no_action_on_fn += 1

        # Enhanced metrics considering Laya's second-layer review.
        # Laya reviews every IDS decision. For each case:
        #   * IDS said ATTACK but it's BENIGN (FP):
        #       - Laya takes action (escalates)  -> still a false alarm, FP remains
        #       - Laya does NOT act (ALLOW)      -> FP resolved, becomes a TN
        #   * IDS said BENIGN but it's ATTACK (FN):
        #       - Laya takes action (escalates)  -> FN caught, becomes a TP
        #       - Laya does NOT act (ALLOW)      -> FN remains, dangerous miss
        enhanced_tp = tps + laya_action_on_fn
        enhanced_fn = fns - laya_action_on_fn
        enhanced_fp = fps - laya_no_action_on_fp
        enhanced_tn = tns + laya_no_action_on_fp

        enhanced_total = enhanced_tp + enhanced_fp + enhanced_tn + enhanced_fn
        if enhanced_total > 0:
            enhanced_accuracy = (enhanced_tp + enhanced_tn) / enhanced_total
            enhanced_precision = enhanced_tp / (enhanced_tp + enhanced_fp) if (enhanced_tp + enhanced_fp) > 0 else 0.0
            enhanced_recall = enhanced_tp / (enhanced_tp + enhanced_fn) if (enhanced_tp + enhanced_fn) > 0 else 0.0
            enhanced_f1 = 2 * (enhanced_precision * enhanced_recall) / (enhanced_precision + enhanced_recall) if (enhanced_precision + enhanced_recall) > 0 else 0.0
        else:
            enhanced_accuracy = enhanced_precision = enhanced_recall = enhanced_f1 = 0.0

        return {
            "total_evaluated": total,
            "ids_performance": {
                "true_positives": tps,
                "true_negatives": tns,
                "false_positives": fps,
                "false_negatives": fns,
                "accuracy": ids_accuracy,
                "precision": ids_precision,
                "recall": ids_recall,
                "f1": ids_f1
            },
            "laya_enhanced_performance": {
                "true_positives": enhanced_tp,
                "true_negatives": enhanced_tn,
                "false_positives": enhanced_fp,
                "false_negatives": enhanced_fn,
                "accuracy": enhanced_accuracy,
                "precision": enhanced_precision,
                "recall": enhanced_recall,
                "f1": enhanced_f1,
                "laya_actions_taken": laya_action_taken,
                "laya_actions_on_fp": laya_action_on_fp,
                "laya_actions_on_fn": laya_action_on_fn,
                "laya_no_action_on_fp": laya_no_action_on_fp,
                "laya_no_action_on_fn": laya_no_action_on_fn
            }
        }


def evaluate_laya_on_fp_fn(
    fps_results: List[Dict[str, Any]],
    fns_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function to evaluate Laya performance on FP and FN cases.

    Args:
        fps_results: List of false positive evaluation results
        fns_results: List of false negative evaluation results

    Returns:
        Combined evaluation results
    """
    evaluator = LayaSecurityEvaluator()

    fp_analysis = evaluator.analyze_false_positives(fps_results) if fps_results else {
        "total_false_positives": 0,
        "analysis": "No false positives provided"
    }

    fn_analysis = evaluator.analyze_false_negatives(fns_results) if fns_results else {
        "total_false_negatives": 0,
        "analysis": "No false negatives provided"
    }

    # Combine results for overall stats
    all_results = fps_results + fns_results
    overall_stats = evaluator.get_overall_statistics(all_results) if all_results else {
        "total_evaluated": 0
    }

    return {
        "false_positive_analysis": fp_analysis,
        "false_negative_analysis": fn_analysis,
        "overall_statistics": overall_stats
    }


if __name__ == "__main__":
    # Test the Laya evaluator
    print("Testing LayaSecurityEvaluator...")

    # Create evaluator instance
    evaluator = LayaSecurityEvaluator()

    # Create test data
    import numpy as np

    # Test case 1: False Positive (IDS says ATTACK, actually BENIGN)
    fp_evidence = {
        "ids_class": "ATTACK",
        "ids_confidence": 0.75,
        "true_label": "BENIGN",
        "network_features": np.array([80, 50000, 5, 3, 1000, 80, 2, 0.3]),  # Low anomaly
        "model_context": {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1": 0.87,
            "false_positive_rate": 0.04,
            "false_negative_rate": 0.10
        }
    }

    # Test case 2: False Negative (IDS says BENIGN, actually ATTACK)
    fn_evidence = {
        "ids_class": "BENIGN",
        "ids_confidence": 0.60,
        "true_label": "ATTACK",
        "network_features": np.array([22, 200000, 50, 40, 500000, 22, 25, 0.9]),  # High anomaly, SSH
        "model_context": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90,
            "false_positive_rate": 0.03,
            "false_negative_rate": 0.07
        }
    }

    # Test case 3: True Positive (IDS says ATTACK, actually ATTACK)
    tp_evidence = {
        "ids_class": "ATTACK",
        "ids_confidence": 0.90,
        "true_label": "ATTACK",
        "network_features": np.array([443, 150000, 20, 15, 300000, 443, 5, 0.7]),  # HTTPS attack
        "model_context": {
            "accuracy": 0.96,
            "precision": 0.94,
            "recall": 0.91,
            "f1": 0.92,
            "false_positive_rate": 0.02,
            "false_negative_rate": 0.05
        }
    }

    # Test case 4: True Negative (IDS says BENIGN, actually BENIGN)
    tn_evidence = {
        "ids_class": "BENIGN",
        "ids_confidence": 0.80,
        "true_label": "BENIGN",
        "network_features": np.array([80, 10000, 2, 1, 50000, 80, 0, 0.1]),  # Normal web traffic
        "model_context": {
            "accuracy": 0.97,
            "precision": 0.96,
            "recall": 0.93,
            "f1": 0.94,
            "false_positive_rate": 0.01,
            "false_negative_rate": 0.02
        }
    }

    test_cases = [fp_evidence, fn_evidence, tp_evidence, tn_evidence]

    # Evaluate each case
    results = []
    for i, case in enumerate(test_cases):
        print(f"\nEvaluating test case {i+1}: {case['ids_class']} -> {case['true_label']}")
        result = evaluator.evaluate_ids_prediction(
            ids_class=case["ids_class"],
            ids_confidence=case["ids_confidence"],
            true_label=case["true_label"],
            network_features=case["network_features"],
            model_context=case["model_context"]
        )
        results.append(result)

        # Print brief result
        ids_pred = result["ids_prediction"]
        laya_dec = result["laya_decision"]["decisions"]
        print(f"  IDS: {ids_pred['class']} (conf: {ids_pred['confidence']:.2f}), Correct: {ids_pred['correct']}")
        print(f"  Laya Disposition: {laya_dec.get('disposition', {}).get('decision', 'N/A')} "
              f"(conf: {laya_dec.get('disposition', {}).get('confidence', 0):.2f})")
        print(f"  Laya TP Assessment: {laya_dec.get('true_positive', {}).get('decision', 'N/A')} "
              f"(conf: {laya_dec.get('true_positive', {}).get('confidence', 0):.2f})")

    # Analyze FP and FN cases
    fps_only = [r for r in results if r["is_false_positive"]]
    fns_only = [r for r in results if r["is_false_negative"]]

    print(f"\n{'='*60}")
    print("FALSE POSITIVE ANALYSIS")
    print(f"{'='*60}")
    if fps_only:
        fp_analysis = evaluator.analyze_false_positives(fps_only)
        print(f"Total False Positives: {fp_analysis['total_false_positives']}")
        if fp_analysis['total_false_positives'] > 0:
            print(f"Escalation Rate: {fp_analysis['rates']['escalation_rate']:.3f}")
            print(f"Alert Rate: {fp_analysis['rates']['alert_rate']:.3f}")
            print(f"Review Rate: {fp_analysis['rates']['review_rate']:.3f}")
            print(f"Monitor Rate: {fp_analysis['rates']['monitor_rate']:.3f}")
            print(f"False Escalation Rate: {fp_analysis['rates']['false_escalation_rate']:.3f}")
    else:
        print("No false positives to analyze")

    print(f"\n{'='*60}")
    print("FALSE NEGATIVE ANALYSIS")
    print(f"{'='*60}")
    if fns_only:
        fn_analysis = evaluator.analyze_false_negatives(fns_only)
        print(f"Total False Negatives: {fn_analysis['total_false_negatives']}")
        if fn_analysis['total_false_negatives'] > 0:
            print(f"Escalation Rate: {fn_analysis['rates']['escalation_rate']:.3f}")
            print(f"Alert Rate: {fn_analysis['rates']['alert_rate']:.3f}")
            print(f"Review Rate: {fn_analysis['rates']['review_rate']:.3f}")
            print(f"Monitor Rate: {fn_analysis['rates']['monitor_rate']:.3f}")
            print(f"Missed Escalation Rate: {fn_analysis['rates']['missed_escalation_rate']:.3f}")
    else:
        print("No false negatives to analyze")

    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    overall = evaluator.get_overall_statistics(results)
    print(f"Total Evaluated: {overall['total_evaluated']}")
    print(f"\nIDS Performance:")
    ids_perf = overall['ids_performance']
    print(f"  Accuracy: {ids_perf['accuracy']:.3f}")
    print(f"  Precision: {ids_perf['precision']:.3f}")
    print(f"  Recall: {ids_perf['recall']:.3f}")
    print(f"  F1-Score: {ids_perf['f1']:.3f}")
    print(f"  TP: {ids_perf['true_positives']}, TN: {ids_perf['true_negatives']}")
    print(f"  FP: {ids_perf['false_positives']}, FN: {ids_perf['false_negatives']}")

    print(f"\nLaya-Enhanced Performance:")
    laya_perf = overall['laya_enhanced_performance']
    print(f"  Accuracy: {laya_perf['accuracy']:.3f}")
    print(f"  Precision: {laya_perf['precision']:.3f}")
    print(f"  Recall: {laya_perf['recall']:.3f}")
    print(f"  F1-Score: {laya_perf['f1']:.3f}")
    print(f"  TP: {laya_perf['true_positives']}, TN: {laya_perf['true_negatives']}")
    print(f"  FP: {laya_perf['false_positives']}, FN: {laya_perf['false_negatives']}")
    print(f"  Laya Actions Taken: {laya_perf['laya_actions_taken']}")
    print(f"  Laya Actions on FP: {laya_perf['laya_actions_on_fp']}")
    print(f"  Laya Actions on FN: {laya_perf['laya_actions_on_fn']}")
    print(f"  Laya No Action on FP: {laya_perf['laya_no_action_on_fp']}")
    print(f"  Laya No Action on FN: {laya_perf['laya_no_action_on_fn']}")

    print(f"\nLayaSecurityEvaluator test completed!")