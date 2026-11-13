"""
Unit tests for the manual federated learning path (src.federated.run_manual_federated).

Uses small synthetic models/data so tests run fast. Covers FedAvg averaging,
parameter get/set round-trips, and evaluation on synthetic data.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.mlp import MLPIDS
from src.federated.run_manual_federated import (
    get_model_parameters,
    set_model_parameters,
    average_parameters,
    evaluate_model,
)


def _make_model(input_dim: int = 8) -> MLPIDS:
    return MLPIDS(input_dim=input_dim, hidden_dims=[8, 4], dropout_rate=0.0)


class TestFederatedHelpers(unittest.TestCase):
    """Pure federation helpers."""

    def test_get_set_parameters_round_trip(self):
        model = _make_model()
        params_before = get_model_parameters(model)
        set_model_parameters(model, params_before)
        params_after = get_model_parameters(model)
        for pb, pa in zip(params_before, params_after):
            np.testing.assert_array_equal(pb, pa)

    def test_average_parameters_unweighted(self):
        m1 = _make_model()
        m2 = _make_model()
        p1 = get_model_parameters(m1)
        p2 = get_model_parameters(m2)
        avg = average_parameters([p1, p2], weights=None)
        self.assertEqual(len(avg), len(p1))
        for a, a1, a2 in zip(avg, p1, p2):
            np.testing.assert_allclose(a, (a1 + a2) / 2.0)

    def test_average_parameters_weighted(self):
        m1 = _make_model()
        m2 = _make_model()
        p1 = get_model_parameters(m1)
        p2 = get_model_parameters(m2)
        # weight 1.0 on m1, 0.0 on m2 -> avg == p1
        avg = average_parameters([p1, p2], weights=[1.0, 0.0])
        for a, a1 in zip(avg, p1):
            np.testing.assert_allclose(a, a1)
        # weight 0.0 on m1, 1.0 on m2 -> avg == p2
        avg = average_parameters([p1, p2], weights=[0.0, 1.0])
        for a, a2 in zip(avg, p2):
            np.testing.assert_allclose(a, a2)

    def test_evaluate_model_shapes_and_keys(self):
        model = _make_model(input_dim=8)
        model.eval()
        rng = np.random.RandomState(0)
        X_test = rng.randn(20, 8).astype(np.float32)
        y_test = rng.randint(0, 2, size=20)
        metrics = evaluate_model(model, X_test, y_test, device="cpu")
        required = {
            "loss", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
            "tp", "tn", "fp", "fn",
        }
        self.assertTrue(required.issubset(set(metrics.keys())))
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"], 20)


class TestLocalTrainingChangesParameters(unittest.TestCase):
    """A client training step must actually modify parameters."""

    def test_training_changes_parameters(self):
        model = _make_model(input_dim=8)
        before = get_model_parameters(model)
        rng = np.random.RandomState(1)
        X = rng.randn(32, 8).astype(np.float32)
        y = rng.randint(0, 2, size=32)

        network = model.network
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        for _ in range(10):
            optimizer.zero_grad()
            out = network(X_t).view(-1)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        after = get_model_parameters(model)
        max_diff = max(
            float(np.abs(b - a).max()) for b, a in zip(before, after)
        )
        self.assertGreater(max_diff, 1e-6, "Training should modify at least one parameter")


if __name__ == "__main__":
    unittest.main()
