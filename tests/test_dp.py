"""
Unit tests for differential privacy (src.privacy.dp_sgd).

Tests gradient clipping, Gaussian noise, the privacy accountant, and the
DPSGDTrainer (both Opacus and manual fallback paths) on small synthetic data.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLPIDS
from src.privacy.dp_sgd import (
    DPSGDTrainer,
    clip_gradients,
    add_gaussian_noise,
    compute_epsilon,
    privacy_report,
)


def _tiny_model(input_dim: int = 4) -> MLPIDS:
    return MLPIDS(input_dim=input_dim, hidden_dims=[4, 2], dropout_rate=0.0)


def _tiny_loader(n_samples: int = 16, batch_size: int = 8, input_dim: int = 4):
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, input_dim).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples)
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class TestGradientClipping(unittest.TestCase):
    def test_clip_gradients_scales_down_large_norm(self):
        model = _tiny_model()
        # Manually assign large gradients (avoids autograd/numpy quirks)
        with torch.no_grad():
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 10.0

        before_norm = clip_gradients(model.parameters(), max_grad_norm=1.0)
        after_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]),
            2,
        ).item()
        self.assertLessEqual(after_norm, 1.0 + 1e-5)
        # clipping should have reduced a large norm
        self.assertLess(after_norm, before_norm + 1e-6)

    def test_clip_gradients_no_op_when_small_norm(self):
        model = _tiny_model()
        # Zero grads -> norm 0 -> no clipping
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        norm = clip_gradients(model.parameters(), max_grad_norm=1.0)
        self.assertEqual(norm, 0.0)


class TestGaussianNoise(unittest.TestCase):
    def test_add_gaussian_noise_changes_gradients(self):
        model = _tiny_model()
        X = torch.randn(8, 4)
        y = torch.randn(8)
        out = model(X).view(-1)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        before = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

        add_gaussian_noise(model.parameters(), noise_multiplier=1.0, max_grad_norm=1.0)
        after = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "Gaussian noise should modify gradients")


class TestPrivacyAccountant(unittest.TestCase):
    def test_compute_epsilon_positive_finite(self):
        eps = compute_epsilon(
            steps=100, num_examples=1000, batch_size=32,
            noise_multiplier=1.0, delta=1e-5,
        )
        self.assertGreater(eps, 0.0)
        self.assertTrue(np.isfinite(eps))

    def test_compute_epsilon_infinite_when_no_noise(self):
        eps = compute_epsilon(
            steps=100, num_examples=1000, batch_size=32,
            noise_multiplier=0.0, delta=1e-5,
        )
        self.assertEqual(eps, float("inf"))

    def test_compute_epsilon_decreases_with_larger_noise(self):
        eps_low = compute_epsilon(steps=100, num_examples=1000, batch_size=32,
                                  noise_multiplier=0.5, delta=1e-5)
        eps_high = compute_epsilon(steps=100, num_examples=1000, batch_size=32,
                                   noise_multiplier=2.0, delta=1e-5)
        self.assertLess(eps_high, eps_low)

    def test_privacy_report_structure(self):
        report = privacy_report(
            steps=100, num_examples=1000, batch_size=32,
            noise_multiplier=1.0, delta=1e-5,
            target_epsilons=[1.0, 2.0],
        )
        self.assertIn("epsilon", report)
        self.assertIn("delta", report)
        self.assertIn("noise_multiplier", report)
        self.assertTrue(report["dp_enabled"])
        self.assertIn("required_noise_for_targets", report)
        self.assertEqual(len(report["required_noise_for_targets"]), 2)


class TestDPSGDTrainer(unittest.TestCase):
    def test_train_epoch_manual_fallback(self):
        loader = _tiny_loader(n_samples=16, batch_size=8)
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = DPSGDTrainer(
            model=model, optimizer=optimizer, dataloader=loader,
            max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5,
            use_opacus=False,  # force manual path for deterministic behaviour
        )
        info = trainer.train_epoch()
        self.assertIn("loss", info)
        self.assertIn("epsilon", info)
        self.assertIn("delta", info)
        self.assertGreaterEqual(trainer.steps, 1)
        self.assertFalse(info["opacus_used"])
        self.assertTrue(np.isfinite(info["epsilon"]))

    def test_get_privacy_spent_returns_epsilon_delta(self):
        loader = _tiny_loader(n_samples=16, batch_size=8)
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = DPSGDTrainer(
            model=model, optimizer=optimizer, dataloader=loader,
            max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5,
            use_opacus=False,
        )
        trainer.train_epoch()
        eps, delta = trainer.get_privacy_spent()
        self.assertEqual(delta, 1e-5)
        self.assertGreaterEqual(eps, 0.0)

    def test_dp_enabled_flag(self):
        loader = _tiny_loader()
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer_noisy = DPSGDTrainer(
            model=model, optimizer=optimizer, dataloader=loader,
            max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5, use_opacus=False,
        )
        self.assertTrue(trainer_noisy.is_dp())

        trainer_private = DPSGDTrainer(
            model=_tiny_model(), optimizer=torch.optim.Adam(_tiny_model().parameters(), lr=0.01),
            dataloader=loader, max_grad_norm=1.0, noise_multiplier=0.0, delta=1e-5,
            use_opacus=False,
        )
        self.assertFalse(trainer_private.is_dp())

    @unittest.skipIf(not __import__("src.privacy.dp_sgd", fromlist=["OPACUS_AVAILABLE"]).OPACUS_AVAILABLE,
                     "Opacus not available")
    def test_train_epoch_opacus_path(self):
        loader = _tiny_loader(n_samples=32, batch_size=8)
        # Opacus cannot train BatchNorm under DP; use a BatchNorm-free model.
        model = MLPIDS(input_dim=4, hidden_dims=[4, 2], dropout_rate=0.0, use_batchnorm=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = DPSGDTrainer(
            model=model, optimizer=optimizer, dataloader=loader,
            max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5,
            use_opacus=True,
        )
        info = trainer.train_epoch()
        self.assertTrue(info["opacus_used"])
        self.assertGreaterEqual(trainer.steps, 1)


if __name__ == "__main__":
    unittest.main()
