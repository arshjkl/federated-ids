"""
Unit tests for the FastAPI dashboard backend (dashboard/backend/main.py).

Exercises component loading (MLP-only fallback included when Laya is
unavailable) and the HTTP endpoints via FastAPI's TestClient.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main

PROJECT_ROOT = "/Users/admin/Desktop/federated-ids"


class TestDashboardLoading(unittest.TestCase):
    """Component loading must succeed and degrade gracefully."""

    @classmethod
    def setUpClass(cls):
        cls.ok = main.load_components()

    def test_load_components_succeeds(self):
        self.assertTrue(self.ok, "load_components should return True")

    def test_mlp_model_loaded(self):
        self.assertIsNotNone(main.mlp_model, "MLP model must be loaded")

    def test_preprocessing_loaded(self):
        self.assertIsNotNone(main.scaler, "Scaler must be loaded")
        self.assertGreater(len(main.feature_names), 0, "Feature names must be loaded")

    def test_component_status_honest(self):
        self.assertTrue(main.component_status["api"])
        self.assertTrue(main.component_status["ids_model"])
        self.assertTrue(main.component_status["preprocessing"])


class TestDashboardEndpoints(unittest.TestCase):
    """HTTP endpoint behaviour."""

    @classmethod
    def setUpClass(cls):
        main.load_components()
        cls.client = TestClient(main.app)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "healthy")

    def test_system_status(self):
        r = self.client.get("/api/system-status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["api"])
        self.assertTrue(body["ids_model"])
        self.assertTrue(body["preprocessing"])

    def test_model_metrics(self):
        r = self.client.get("/api/model-metrics")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("active_model", body)
        self.assertIn(body["active_model"], ("federated", "centralized", "none"))

    def test_analyze(self):
        r = self.client.post("/api/analyze", json={"features": [0.0] * 70})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("ids", body)
        self.assertIn("pipeline", body)
        self.assertIn("laya", body)
        self.assertIn(body["ids"]["prediction"], ("BENIGN", "ATTACK"))

    def test_analyze_wrong_feature_count(self):
        r = self.client.post("/api/analyze", json={"features": [0.0] * 10})
        self.assertEqual(r.status_code, 400)

    def test_demo_events(self):
        r = self.client.get("/api/demo-events")
        self.assertEqual(r.status_code, 200)
        events = r.json().get("demo_events", [])
        self.assertGreater(len(events), 0)

    def test_history(self):
        r = self.client.get("/api/history")
        self.assertEqual(r.status_code, 200)
        self.assertIn("events", r.json())

    def test_metrics(self):
        r = self.client.get("/api/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("total_events", r.json())


class TestMLPFallback(unittest.TestCase):
    """When Laya cannot be imported/initialized, the dashboard still serves MLP-only."""

    @patch("main.IDS_LAYA_AVAILABLE", False)
    @patch("main.LayaSecurityDecisionEngine", None)
    @patch("main.SecurityEvidenceBuilder", None)
    def test_load_components_mlp_only(self):
        # Reload module-level globals are patched; call load_components and verify
        # it does not crash and still loads the MLP model + scaler.
        main.laya_engine = None
        main.evidence_builder = None
        ok = main.load_components()
        self.assertTrue(ok)
        self.assertIsNotNone(main.mlp_model)
        self.assertIsNotNone(main.scaler)
        self.assertIsNone(main.laya_engine)
        self.assertEqual(main.component_status["laya_engine"], "unavailable")

    def test_run_laya_analysis_degrades(self):
        saved = main.laya_engine
        main.laya_engine = None
        try:
            result = main.run_laya_analysis({})
            self.assertFalse(result.get("available", True))
            self.assertEqual(result["final_decision"]["action"], "REVIEW")
        finally:
            main.laya_engine = saved


if __name__ == "__main__":
    unittest.main()
