"""
Unit tests for the MLPIDS model (src.models.mlp).

Uses a small slice of the processed data so tests run quickly without
retraining. Verifies forward pass, predict_proba, predict, parameter counting,
and loading a saved checkpoint.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from src.models.mlp import MLPIDS, count_parameters, create_mlp_model

PROJECT_ROOT = "/Users/admin/Desktop/federated-ids"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class _SmallData:
    """Lazy-load a small slice of the processed arrays."""

    _data = None

    @classmethod
    def data(cls):
        if cls._data is None:
            X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
            y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
            cls._data = (X_train[:500], y_train[:500])
        return cls._data


class TestMLPModel(unittest.TestCase):
    """Model creation, forward pass, prediction."""

    def setUp(self):
        self.input_dim = 70
        self.model = MLPIDS(
            input_dim=self.input_dim,
            hidden_dims=[64, 32],
            dropout_rate=0.3,
        )
        self.model.eval()

    def test_model_creation(self):
        self.assertIsNotNone(self.model.network)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dims, [64, 32])

    def test_forward_pass_shape(self):
        X, _ = _SmallData.data()
        X_tensor = torch.FloatTensor(X[:32])
        with torch.no_grad():
            logits = self.model(X_tensor)
        self.assertEqual(tuple(logits.shape), (32, 1))

    def test_predict_proba_shape(self):
        X, _ = _SmallData.data()
        X_tensor = torch.FloatTensor(X[:32])
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        self.assertEqual(tuple(probs.shape), (32, 2))
        # Probabilities sum to 1 per row
        row_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(32), atol=1e-5))
        # Values in [0, 1]
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_predict_shape(self):
        X, _ = _SmallData.data()
        X_tensor = torch.FloatTensor(X[:32])
        with torch.no_grad():
            preds = self.model.predict(X_tensor)
        self.assertEqual(tuple(preds.shape), (32,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())

    def test_count_parameters(self):
        n = count_parameters(self.model)
        self.assertGreater(n, 0)
        # input_dim=70, hidden=[64,32], batchnorm on both hidden layers, out=1:
        # Linear(70,64)=4544 + BN(64)=128 + Linear(64,32)=2080 + BN(32)=64 + Linear(32,1)=33 = 6849
        expected = (70 * 64 + 64) + (64 * 2) + (64 * 32 + 32) + (32 * 2) + (32 * 1 + 1)
        self.assertEqual(n, expected)

    def test_create_mlp_model_factory(self):
        model = create_mlp_model(input_dim=70, hidden_dims=[128, 64, 32])
        self.assertIsInstance(model, MLPIDS)


class TestMLPCheckpointLoading(unittest.TestCase):
    """Saved checkpoints must load into a matching architecture."""

    def test_load_centralized_checkpoint(self):
        path = os.path.join(MODELS_DIR, "centralized_mlp.pth")
        self.assertTrue(os.path.exists(path), "centralized_mlp.pth must exist")
        checkpoint = torch.load(path, map_location="cpu")
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["input_dim"], 70)
        self.assertEqual(checkpoint["hidden_dims"], [128, 64, 32])

        model = MLPIDS(input_dim=70, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        state_dict = checkpoint["model_state_dict"]
        network_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("network."):
                network_state_dict[k[8:]] = v
            else:
                network_state_dict[k] = v
        model.network.load_state_dict(network_state_dict, strict=True)
        model.eval()

        # Forward pass should work after loading
        X, _ = _SmallData.data()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X[:16]))
        self.assertEqual(tuple(logits.shape), (16, 1))

    def test_load_federated_checkpoint(self):
        path = os.path.join(MODELS_DIR, "federated_mlp_manual.pth")
        self.assertTrue(os.path.exists(path), "federated_mlp_manual.pth must exist")
        checkpoint = torch.load(path, map_location="cpu")
        self.assertIn("model_state_dict", checkpoint)
        model = MLPIDS(input_dim=70, hidden_dims=[128, 64, 32], dropout_rate=0.3)
        state_dict = checkpoint["model_state_dict"]
        network_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("network."):
                network_state_dict[k[8:]] = v
            else:
                network_state_dict[k] = v
        model.network.load_state_dict(network_state_dict, strict=True)


if __name__ == "__main__":
    unittest.main()
